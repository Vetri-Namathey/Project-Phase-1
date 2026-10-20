"""Record a CARLA clip of the truck-drop scenario for the demo and the report
(DISCUSSION_NOTES.md, sections 7 and 7b).

An ego car on CARLA's autopilot follows a lead truck. After --pre-s seconds of
truck-only driving, an object is released from the truck's tailgate with the
truck's velocity plus a tilt and spin, falls to the road and slides to rest.
The ego car then slows down so the object stays in view while it gets closer.
The clip stops when the object is --stop-dist metres away (the ego car is never
allowed to drive over it).

Needs a running CARLA 0.9.15 server (CarlaUE4.exe). Never run this while model
inference uses the GPU: record, close CARLA, then infer.

Ground truth per frame comes from CARLA's instance-segmentation camera. Its ids
are NOT actor ids (first test: ids like 29740 for truck actor 24), so each id is
learned from the image: the object's bounding-box centre is projected into the
frame, the id under that pixel is read, and it is accepted only if its region has
a plausible size and position. The mask is then every pixel with that id.
  * the truck's id must be learned within 10 frames, else the script exits;
  * the dropped object's id must be learned after the drop (and seen in at least
    5 frames), else gt_ok = false in clip.json and a WARNING is printed. Frames
    between the drop and the first time the object is visible have an empty mask.
Always run ONE test clip first and open out/debug/*.png: red = ground truth
for the dropped object, green = the truck / other vehicles.

Output layout (--out), compatible with eval_spatial.py --dataset carla:
    rgb/000000.png        mask/000000.npy       (anomaly pixels; all False before the drop
                                                 and in non-OOD clips)
    mask_vehicle/000000.npy                      (truck + extra vehicle pixels: must NOT be flagged)
    meta.jsonl  clip.json  clip.mp4  debug/*.png

Test clip:
    python record_scenario.py --out video_truck_test --object box02 --gap 15
Non-OOD clip (nothing falls; an ordinary car sits between ego and truck):
    python record_scenario.py --out video_truck_normal --object none --extra vehicle.toyota.prius
"""

import argparse
import json
import math
import os
import queue
import random
import sys

import carla
import numpy as np
from PIL import Image

WIDTH, HEIGHT, FOV = 1024, 512, 50.0
SIM_DT = 0.05            # 20 Hz simulation
SAVE_EVERY = 2           # -> 10 frames per second in the clip
TRUCK_BP = "vehicle.carlamotors.european_hgv"
EGO_BP = "vehicle.tesla.model3"
CAR_HALF_LENGTH_M = 2.4  # rough half length used only to place the cars
BED_HEIGHT_M = 1.4       # height of the tailgate above the road
MIN_ID_PIXELS = 15
# object -> blueprint, tilt (deg, at release) and spin (deg/s)
PRESETS = {
    "box02":         dict(bp="static.prop.box02",         tilt=25, spin=60),
    "trashbag":      dict(bp="static.prop.trashbag",      tilt=20, spin=50),
    "barrel":        dict(bp="static.prop.barrel",        tilt=75, spin=90),
    "briefcase":     dict(bp="static.prop.briefcase",     tilt=30, spin=120),
    "trafficcone02": dict(bp="static.prop.trafficcone02", tilt=60, spin=100),
    "haybale":       dict(bp="static.prop.haybale",       tilt=15, spin=30),
}


def get_for_frame(q, frame, timeout=10.0):
    while True:
        image = q.get(timeout=timeout)
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"sensor skipped frame {frame} (got {image.frame})")


def bgra(image):
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)


def rgb_of(image):
    return bgra(image)[:, :, :3][:, :, ::-1].copy()


def inst_ids(image):
    """Both byte orders of the (G, B) instance id; the right one is learned from the truck."""
    a = bgra(image)
    b = a[:, :, 0].astype(np.int32)
    g = a[:, :, 1].astype(np.int32)
    return {"gb": g | (b << 8), "bg": b | (g << 8)}


def project(cam, loc, fx):
    """World location -> (u, v) pixel, or None if behind the camera."""
    inv = np.array(cam.get_transform().get_inverse_matrix())
    x, y, z = (inv @ np.array([loc.x, loc.y, loc.z, 1.0]))[:3]
    if x <= 0.1:
        return None
    return WIDTH / 2.0 + fx * y / x, HEIGHT / 2.0 - fx * z / x


def learn_id(ids, uv, exclude, max_frac):
    """Instance id of the object whose bounding-box centre projects to pixel uv.

    CARLA 0.9.15's instance ids are NOT actor ids (the first test clip showed ids like 29740
    against truck actor 24), so the id is read off the image at the projected centre and
    accepted only if its region has a sensible size and sits around that pixel."""
    if uv is None:
        return None
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (3 <= u < WIDTH - 3 and 3 <= v < HEIGHT - 3):
        return None
    vals, counts = np.unique(ids[v - 3:v + 4, u - 3:u + 4].ravel(), return_counts=True)
    for idx in np.argsort(-counts):
        cand = int(vals[idx])
        if cand in exclude:
            continue
        region = ids == cand
        area = int(region.sum())
        if area < MIN_ID_PIXELS or area > max_frac * WIDTH * HEIGHT:
            continue
        ys, xs = np.nonzero(region)
        if math.hypot(xs.mean() - u, ys.mean() - v) > max(60.0, 2.0 * math.sqrt(area)):
            continue
        return cand, area
    return None


def to_world(tf, loc):
    """Local -> world point. Some CARLA versions return the point, others edit it in place."""
    p = carla.Location(loc.x, loc.y, loc.z)
    r = tf.transform(p)
    r = r if r is not None else p
    return carla.Location(r.x, r.y, r.z)  # 0.9.15 returns a Vector3D; Transform() and distance() want a Location


def transform_dict(t):
    return {"x": t.location.x, "y": t.location.y, "z": t.location.z,
            "pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll}


def straight_start(world_map, rng, ahead_m, behind_m):
    """A driving-lane waypoint with a junction-free, near-straight road ahead and behind.
    Tries progressively shorter requirements (the lead truck needs roughly ahead_m of road
    to stay in view; Town10 is a dense city with few long straights)."""
    cands = [world_map.get_waypoint(sp.location) for sp in world_map.get_spawn_points()]
    cands += list(world_map.generate_waypoints(15.0))
    cands = [w for w in cands if w.lane_type == carla.LaneType.Driving and not w.is_junction]
    rng.shuffle(cands)

    def straight(wp, step, need, direction):
        yaw0, cur, d = wp.transform.rotation.yaw, wp, 0.0
        while d < need:
            nxt = cur.next(step) if direction > 0 else cur.previous(step)
            if not nxt:
                return False
            cur, d = nxt[0], d + step
            dyaw = (cur.transform.rotation.yaw - yaw0 + 180.0) % 360.0 - 180.0
            if cur.is_junction or abs(dyaw) > 12.0:
                return False
        return True

    for scale in (1.0, 0.75, 0.55, 0.4):
        need_a, need_b = ahead_m * scale, behind_m * scale
        for wp in cands:
            if straight(wp, 5.0, need_a, +1) and straight(wp, 5.0, need_b, -1):
                print(f"road: {need_a:.0f} m straight ahead, {need_b:.0f} m behind "
                      f"(requirement scale {scale})")
                return wp
    raise SystemExit("no straight road found even with relaxed limits -- try --town Town06 or Town04")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--object", default="box02", help=f"{', '.join(PRESETS)}, or none (non-OOD clip)")
    parser.add_argument("--gap", type=float, default=15.0, help="metres between ego and the vehicle ahead before the drop")
    parser.add_argument("--extra", default=None, help="optional ordinary vehicle between ego and truck, e.g. vehicle.toyota.prius")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--town", default=None, help="e.g. Town10HD_Opt; default = keep loaded map")
    parser.add_argument("--warmup-s", type=float, default=5.0, help="unrecorded seconds so the cars settle")
    parser.add_argument("--pre-s", type=float, default=3.0, help="recorded truck-only seconds before the drop")
    parser.add_argument("--max-post-s", type=float, default=16.0, help="longest recording after the drop")
    parser.add_argument("--stop-dist", type=float, default=5.5, help="stop when the object is this close to the camera (m)")
    parser.add_argument("--truck-speed-diff", type=float, default=50.0, help="percent BELOW the speed limit")
    parser.add_argument("--ego-speed-diff", type=float, default=30.0)
    parser.add_argument("--ego-after-diff", type=float, default=75.0, help="ego slows to this percent below the limit after the drop")
    args = parser.parse_args()

    if args.object != "none" and args.object not in PRESETS:
        raise SystemExit(f"unknown --object {args.object}; choose from {list(PRESETS)} or none")
    drop = args.object != "none"
    rng = random.Random(args.seed)
    random.seed(args.seed)
    for sub in ("rgb", "mask", "mask_vehicle", "debug"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        print(f"loading {args.town} ...")
        world = client.load_world(args.town)
    print(f"map: {world.get_map().name}")

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DT
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    actors = []
    try:
        world_map = world.get_map()
        bp_lib = world.get_blueprint_library()

        # ----- vehicles, in a straight lane: truck, [extra], ego -----
        wp = straight_start(world_map, rng, ahead_m=140.0, behind_m=args.gap + 45.0)
        truck = world.try_spawn_actor(bp_lib.find(TRUCK_BP), carla.Transform(
            carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z + 0.3),
            wp.transform.rotation))
        if truck is None:
            raise SystemExit("could not spawn the truck here -- try another --seed")
        actors.append(truck)
        truck_half = truck.bounding_box.extent.x

        def spawn_behind(bp_id, centre_dist):
            back = wp.previous(centre_dist)
            if not back:
                raise SystemExit("not enough road behind the truck -- try another --seed")
            t = back[0].transform
            a = world.try_spawn_actor(bp_lib.find(bp_id), carla.Transform(
                carla.Location(t.location.x, t.location.y, t.location.z + 0.3), t.rotation))
            if a is None:
                raise SystemExit(f"could not spawn {bp_id} -- try another --seed")
            actors.append(a)
            return a

        extra = None
        d_prev_half = truck_half
        if args.extra:
            extra = spawn_behind(args.extra, truck_half + CAR_HALF_LENGTH_M + 7.0)
            d_prev_half = truck_half + 2 * CAR_HALF_LENGTH_M + 7.0
        ego = spawn_behind(EGO_BP, d_prev_half + CAR_HALF_LENGTH_M + args.gap)
        print(f"ego gap {args.gap} m behind the {'extra car' if extra else 'truck'}")

        for v, diff, lead_gap in ((truck, args.truck_speed_diff, 8.0),
                                  (extra, 40.0, 7.0), (ego, args.ego_speed_diff, args.gap)):
            if v is None:
                continue
            v.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(v, False)
            tm.ignore_lights_percentage(v, 100)
            tm.ignore_signs_percentage(v, 100)
            tm.vehicle_percentage_speed_difference(v, diff)
            tm.distance_to_leading_vehicle(v, lead_gap)

        # ----- cameras on the ego car -----
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        inst_bp = bp_lib.find("sensor.camera.instance_segmentation")
        for bp in (rgb_bp, inst_bp):
            bp.set_attribute("image_size_x", str(WIDTH))
            bp.set_attribute("image_size_y", str(HEIGHT))
            bp.set_attribute("fov", str(FOV))
        rgb_bp.set_attribute("gamma", "2.2")
        rgb_bp.set_attribute("exposure_mode", "histogram")
        cam_tf = carla.Transform(carla.Location(x=1.0, z=1.5), carla.Rotation(pitch=-2.0))
        rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        inst_cam = world.spawn_actor(inst_bp, cam_tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        actors += [rgb_cam, inst_cam]
        rgb_q, inst_q = queue.Queue(), queue.Queue()
        rgb_cam.listen(rgb_q.put)
        inst_cam.listen(inst_q.put)
        fx = WIDTH / (2.0 * math.tan(math.radians(FOV) / 2.0))
        intrinsics = {"fx": fx, "fy": fx, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
                      "width": WIDTH, "height": HEIGHT, "fov": FOV}

        def tick():
            f = world.tick()
            return get_for_frame(rgb_q, f), get_for_frame(inst_q, f), f

        print(f"warming up {args.warmup_s:.0f} s ...")
        for _ in range(int(args.warmup_s / SIM_DT)):
            tick()

        # ----- recording loop -----
        pre_ticks = int(args.pre_s / SIM_DT)
        max_post_ticks = int(args.max_post_s / SIM_DT)
        prop, preset = None, PRESETS.get(args.object)
        learned = {}             # name -> instance id read off the image
        prop_seen = 0            # saved frames in which the prop's id matched
        drop_tick = None
        saved, frames_with_gt = 0, 0
        veh_actors = [("truck", truck)] + ([("extra", extra)] if extra else [])
        lead = extra if extra else truck
        k = 0
        with open(os.path.join(args.out, "meta.jsonl"), "w") as meta:
            while True:
                if drop and prop is None and k == pre_ticks:
                    rear_local = carla.Location(x=-truck_half - 0.6, y=0.0, z=BED_HEIGHT_M)
                    loc = to_world(truck.get_transform(), rear_local)
                    yaw = truck.get_transform().rotation.yaw
                    tf = carla.Transform(loc, carla.Rotation(pitch=preset["tilt"], yaw=yaw, roll=rng.uniform(-15, 15)))
                    prop = world.try_spawn_actor(bp_lib.find(preset["bp"]), tf)
                    if prop is None:
                        raise SystemExit("could not spawn the prop behind the truck -- try another --seed")
                    actors.append(prop)
                    world.tick()
                    prop.set_simulate_physics(True)
                    world.tick()
                    fwd = truck.get_transform().get_forward_vector()
                    v = truck.get_velocity()
                    prop.set_target_velocity(carla.Vector3D(v.x - 0.5 * fwd.x, v.y - 0.5 * fwd.y, v.z))
                    prop.set_target_angular_velocity(carla.Vector3D(0.0, preset["spin"], preset["spin"] / 2.0))
                    tm.vehicle_percentage_speed_difference(ego, args.ego_after_diff)
                    drop_tick = k
                    print(f"dropped {preset['bp']} (actor {prop.id}) at t={k * SIM_DT:.1f} s")

                rgb_img, inst_img, sim_frame = tick()
                k += 1
                if prop is not None and prop.get_location().z < truck.get_location().z - 1.5:
                    raise SystemExit("the prop fell through the road -- unusable object, pick another")

                if (k - 1) % SAVE_EVERY == 0:
                    ids = inst_ids(inst_img)["gb"]
                    for name, actor in veh_actors:
                        if name not in learned:
                            uv = project(rgb_cam, to_world(actor.get_transform(), actor.bounding_box.location), fx)
                            hit = learn_id(ids, uv, set(learned.values()), 0.6)
                            if hit:
                                learned[name] = hit[0]
                                print(f"instance id of the {name}: {hit[0]} ({hit[1]} px)")
                    if "truck" not in learned and saved >= 10:
                        uv = project(rgb_cam, to_world(truck.get_transform(), truck.bounding_box.location), fx)
                        raise SystemExit(f"could not read the truck's instance id off the image in 10 frames "
                                         f"(centre pixel {uv}); debug/ shows the frames. Ground truth not possible.")
                    veh = np.isin(ids, [learned[n] for n, _ in veh_actors if n in learned])
                    gt = np.zeros((HEIGHT, WIDTH), bool)
                    dist = None
                    if prop is not None:
                        centre = to_world(prop.get_transform(), prop.bounding_box.location)
                        dist = float(rgb_cam.get_transform().location.distance(centre))
                        if "prop" not in learned and dist < 40.0:
                            hit = learn_id(ids, project(rgb_cam, centre, fx), set(learned.values()), 0.15)
                            if hit:
                                learned["prop"] = hit[0]
                                print(f"instance id of the dropped object: {hit[0]} ({hit[1]} px) at {dist:.1f} m")
                        if "prop" in learned:
                            gt = ids == learned["prop"]
                        if gt.sum() >= MIN_ID_PIXELS:
                            prop_seen += 1
                    Image.fromarray(rgb_of(rgb_img)).save(os.path.join(args.out, "rgb", f"{saved:06d}.png"))
                    np.save(os.path.join(args.out, "mask", f"{saved:06d}.npy"), gt)
                    np.save(os.path.join(args.out, "mask_vehicle", f"{saved:06d}.npy"), veh)
                    frames_with_gt += int(gt.any())
                    meta.write(json.dumps({
                        "frame": saved, "sim_frame": sim_frame,
                        "t_since_drop_s": None if drop_tick is None else round((k - 1 - drop_tick) * SIM_DT, 2),
                        "dropped": prop is not None, "camera": transform_dict(rgb_cam.get_transform()),
                        "intrinsics": intrinsics, "mask_px": int(gt.sum()), "vehicle_px": int(veh.sum()),
                        "object_distance_m": None if dist is None else round(dist, 2),
                        "ego_speed_mps": round(ego.get_velocity().length(), 2),
                        "lead_distance_m": round(float(ego.get_location().distance(lead.get_location())), 2),
                    }) + "\n")
                    if saved % 5 == 0:
                        shot = rgb_of(rgb_img).astype(np.float32)
                        shot[gt] = 0.4 * shot[gt] + 0.6 * np.array([255, 0, 0])
                        shot[veh] = 0.7 * shot[veh] + 0.3 * np.array([0, 255, 0])
                        Image.fromarray(shot.astype(np.uint8)).save(os.path.join(args.out, "debug", f"{saved:06d}.png"))
                    if saved % 20 == 0:
                        print(f"frame {saved}  object_px={int(gt.sum())}  vehicle_px={int(veh.sum())}  dist={dist}")
                    saved += 1

                if drop_tick is not None and prop is not None:
                    if dist is not None and dist < args.stop_dist and k - drop_tick > 20:
                        print(f"object is {dist:.1f} m away: stopping the clip")
                        break
                    if k - drop_tick >= max_post_ticks:
                        print("reached --max-post-s: stopping the clip")
                        break
                if not drop and k >= pre_ticks + max_post_ticks:
                    break

        gt_ok = (not drop) or prop_seen >= 5
        if drop and not gt_ok:
            print("WARNING: the dropped object's instance id was seen in fewer than 5 frames. "
                  "mask/*.npy for this clip is NOT trustworthy -- open debug/*.png (red = object).")

        import av
        out = av.open(os.path.join(args.out, "clip.mp4"), mode="w")
        stream = out.add_stream("libx264", rate=int(round(1.0 / (SIM_DT * SAVE_EVERY))))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20"}
        for i in range(saved):
            arr = np.array(Image.open(os.path.join(args.out, "rgb", f"{i:06d}.png")).convert("RGB"))
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()

        summary = {"object": args.object, "preset": preset, "gap_m": args.gap, "extra": args.extra,
                   "seed": args.seed, "map": world_map.name, "truck": TRUCK_BP, "ego": EGO_BP,
                   "frames": saved, "fps": int(round(1.0 / (SIM_DT * SAVE_EVERY))),
                   "drop_frame": None if drop_tick is None else drop_tick // SAVE_EVERY,
                   "frames_with_object_pixels": frames_with_gt, "gt_method": "instance id read at the projected bounding-box centre", "learned_ids": learned,
                   "gt_ok": bool(gt_ok)}
        with open(os.path.join(args.out, "clip.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"done: {saved} frames, {frames_with_gt} with object pixels -> {args.out}  (gt_ok={gt_ok})")
    finally:
        for a in actors:
            if a.type_id.startswith("sensor."):
                a.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        print("cleaned up, world settings restored")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"CARLA error: {e}")
        sys.exit(1)
