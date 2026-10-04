"""Record a CARLA driving sequence with native anomaly props, for the demo
video (PLAN.md "Real-time CARLA video proof", Option 1; V0 = the pilot).

Needs a running CARLA 0.9.15 server (CarlaUE4.exe). Never run this while
model inference is using the GPU -- record, close CARLA, then infer.

Design (PLAN.md "Recording design"):
- Synchronous mode, so every saved RGB frame and its mask come from the same
  simulator tick.
- Kinematic camera along map waypoints (set_transform each frame) -- no
  autopilot, so props are never hit and the same --seed replays the same route.
- 1024x512 at FOV 50: exactly the model input size, close to Cityscapes optics.
- Ground truth per frame = semantic-tag diff between two ticks at the SAME
  camera pose: props hidden underground, then props in place. The same
  before/after technique generate_anomalies.py already uses, which is proven
  on this install; instance-segmentation ids were not used because their
  mapping to spawned actor ids is unverified for static props in 0.9.15.
- Saves per-frame camera pose + intrinsics (meta.jsonl), which the later
  post-render paste step needs to world-anchor COCO/CARLA cutouts.

Output layout (--out):
    rgb/000000.png  mask/000000.npy  meta.jsonl  props.json

Pilot (V0):
    python record_route.py --out video_pilot --frames 40 --step-m 1.0 --props 4
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
from scipy import ndimage

WIDTH, HEIGHT, FOV = 1024, 512, 50.0
CAMERA_HEIGHT_M = 1.5
HIDDEN_Z = -500.0
MIN_COMPONENT_PIXELS = 20  # smaller than generate_anomalies.py's 50: half the resolution


def get_for_frame(q, frame, timeout=5.0):
    """Drain a sensor queue until the image for this exact tick arrives."""
    while True:
        image = q.get(timeout=timeout)
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"sensor skipped frame {frame} (got {image.frame})")


def tags_of(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return arr[:, :, 2].copy()  # red channel = semantic tag (BGRA layout)


def rgb_of(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return arr[:, :, :3][:, :, ::-1].copy()


def diff_mask(before, after):
    changed = before != after
    labels, n = ndimage.label(changed)
    if n == 0:
        return changed
    sizes = ndimage.sum(changed, labels, range(1, n + 1))
    keep = np.isin(labels, np.nonzero(sizes >= MIN_COMPONENT_PIXELS)[0] + 1)
    return keep


def build_route(world_map, start, n, step_m):
    route = [world_map.get_waypoint(start.location)]
    while len(route) < n:
        nxt = route[-1].next(step_m)
        if not nxt:
            break
        route.append(nxt[0])  # first branch: deterministic for a given seed
    return route


def camera_transform(wp):
    t = wp.transform
    return carla.Transform(
        carla.Location(t.location.x, t.location.y, t.location.z + CAMERA_HEIGHT_M),
        carla.Rotation(pitch=-2.0, yaw=t.rotation.yaw, roll=0.0))


def transform_dict(t):
    return {"x": t.location.x, "y": t.location.y, "z": t.location.z,
            "pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="video_pilot")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--step-m", type=float, default=1.0,
                        help="metres travelled between saved frames (0.4 = 8 m/s at 20 fps)")
    parser.add_argument("--props", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--town", default=None, help="e.g. Town10HD_Opt; default = keep loaded map")
    args = parser.parse_args()

    random.seed(args.seed)
    for sub in ("rgb", "mask"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        print(f"loading {args.town} ...")
        world = client.load_world(args.town)
    print(f"map: {world.get_map().name}")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    actors = []
    try:
        world_map = world.get_map()
        bp_lib = world.get_blueprint_library()
        start = random.choice(world_map.get_spawn_points())

        # Route for the camera, plus look-ahead so props near the end stay in view.
        lookahead = int(30 / args.step_m)
        route_full = build_route(world_map, start, args.frames + lookahead, args.step_m)
        if len(route_full) < args.frames:
            raise SystemExit(f"route ended after {len(route_full)} waypoints -- try another --seed")
        route = route_full[:args.frames]

        cam_tf = camera_transform(route[0])
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        for bp in (rgb_bp, sem_bp):
            bp.set_attribute("image_size_x", str(WIDTH))
            bp.set_attribute("image_size_y", str(HEIGHT))
            bp.set_attribute("fov", str(FOV))
        # Same look as the training crops from generate_anomalies.py.
        rgb_bp.set_attribute("gamma", "2.2")
        rgb_bp.set_attribute("exposure_mode", "histogram")
        rgb_cam = world.spawn_actor(rgb_bp, cam_tf)
        sem_cam = world.spawn_actor(sem_bp, cam_tf)
        actors += [rgb_cam, sem_cam]
        rgb_q, sem_q = queue.Queue(), queue.Queue()
        rgb_cam.listen(rgb_q.put)
        sem_cam.listen(sem_q.put)

        # Props: ahead of the start, beside the camera's path (the camera moves
        # kinematically, so a prop dead-centre in its lane would be driven through).
        prop_bps = [bp for bp in bp_lib.filter("static.prop.*")
                    if "box" not in bp.id and "shopping_cart" not in bp.id]
        lo = int(10 / args.step_m)
        hi = len(route_full) - 1
        props = []
        for k in range(args.props):
            idx = lo + (hi - lo) * (k + 1) // (args.props + 1)
            wp = route_full[idx]
            right = wp.transform.get_right_vector()
            side = 1 if k % 2 == 0 else -1
            offset = side * random.uniform(1.6, 2.4)
            loc = wp.transform.location + carla.Location(right.x * offset, right.y * offset, 0.05)
            tf = carla.Transform(loc, carla.Rotation(yaw=random.uniform(0, 360)))
            bp = random.choice(prop_bps)
            actor = world.try_spawn_actor(bp, tf)
            if actor is None:
                print(f"prop {k}: spawn failed ({bp.id}), skipping")
                continue
            actor.set_simulate_physics(False)
            actors.append(actor)
            props.append({"actor": actor, "transform": tf, "blueprint": bp.id,
                          "route_index": idx, "mechanism": "native_spawn"})
            print(f"prop {k}: {bp.id} at route index {idx}, lateral {offset:+.2f} m")
        if not props:
            raise SystemExit("no props spawned -- try another --seed")

        hidden = [carla.Transform(carla.Location(p["transform"].location.x,
                                                 p["transform"].location.y, HIDDEN_Z))
                  for p in props]

        def place(visible):
            for p, h in zip(props, hidden):
                p["actor"].set_transform(p["transform"] if visible else h)

        world.tick()  # let spawns settle

        fx = WIDTH / (2.0 * math.tan(math.radians(FOV) / 2.0))
        intrinsics = {"fx": fx, "fy": fx, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
                      "width": WIDTH, "height": HEIGHT, "fov": FOV}
        frames_with_object = 0
        with open(os.path.join(args.out, "meta.jsonl"), "w") as meta:
            for i, wp in enumerate(route):
                tf = camera_transform(wp)
                rgb_cam.set_transform(tf)
                sem_cam.set_transform(tf)

                place(visible=False)
                f_before = world.tick()
                before = tags_of(get_for_frame(sem_q, f_before))
                get_for_frame(rgb_q, f_before)  # discard

                place(visible=True)
                f_after = world.tick()
                after_sem = get_for_frame(sem_q, f_after)
                after_rgb = get_for_frame(rgb_q, f_after)

                mask = diff_mask(before, tags_of(after_sem))
                Image.fromarray(rgb_of(after_rgb)).save(os.path.join(args.out, "rgb", f"{i:06d}.png"))
                np.save(os.path.join(args.out, "mask", f"{i:06d}.npy"), mask)
                frames_with_object += int(mask.any())
                meta.write(json.dumps({"frame": i, "sim_frame": f_after,
                                       "camera": transform_dict(tf),
                                       "intrinsics": intrinsics,
                                       "mask_px": int(mask.sum())}) + "\n")
                if i % 10 == 0:
                    print(f"frame {i}/{len(route)}  mask_px={int(mask.sum())}")

        with open(os.path.join(args.out, "props.json"), "w") as f:
            json.dump([{k: (transform_dict(v) if k == "transform" else v)
                        for k, v in p.items() if k != "actor"} for p in props], f, indent=2)
        print(f"done: {len(route)} frames, {frames_with_object} with a visible prop -> {args.out}")
        if frames_with_object == 0:
            print("WARNING: no prop pixels in any frame -- check props.json placement before using this")
    finally:
        for a in actors:
            if a.type_id.startswith("sensor."):
                a.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        world.apply_settings(original_settings)
        print("cleaned up, world settings restored")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"CARLA error: {e}")
        sys.exit(1)
