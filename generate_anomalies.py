import argparse
import carla
import random
import numpy as np
from PIL import Image
from scipy import ndimage
import sys
import time
import queue
import os

MIN_COMPONENT_PIXELS = 50


def extract_new_object_mask(after_tags, baseline_tags):
    """actor.semantic_tags is unreliable for static.prop.* actors in this
    CARLA build -- it comes back empty even after a settle delay, across
    many different blueprint types. So instead of filtering by a known tag
    value, diff every pixel directly: nothing else moves in an otherwise
    static, traffic-free scene, so any pixel whose class changed between
    baseline and after IS the newly spawned object. Semantic segmentation
    outputs discrete class IDs (not noisy continuous values), so this diff
    is clean -- group changed pixels into connected blobs and keep the
    largest one, which discards any stray single-pixel render noise.
    """
    changed = after_tags != baseline_tags

    if not changed.any():
        return np.zeros_like(changed, dtype=np.uint8)

    labeled, num_components = ndimage.label(changed)
    best_label, best_size = None, 0
    for label_id in range(1, num_components + 1):
        size = (labeled == label_id).sum()
        if size < MIN_COMPONENT_PIXELS:
            continue
        if size > best_size:
            best_label, best_size = label_id, size

    if best_label is None:
        return np.zeros_like(changed, dtype=np.uint8)
    return (labeled == best_label).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--out_images", default="data/images")
    parser.add_argument("--out_masks", default="data/masks")
    args = parser.parse_args()

    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_masks, exist_ok=True)

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
    except RuntimeError as e:
        print(f"Error connecting to CARLA server: {e}")
        sys.exit(1)

    print("Connected to Port 2000 successfully! Letting geometry settle...")
    time.sleep(2.0)

    weather = carla.WeatherParameters.ClearSunset
    world.set_weather(weather)

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    safe_transform = random.choice(spawn_points)

    camera_transform = carla.Transform(
        carla.Location(
            x=safe_transform.location.x,
            y=safe_transform.location.y,
            z=safe_transform.location.z + 2.0
        ),
        carla.Rotation(pitch=-10.0, yaw=safe_transform.rotation.yaw, roll=0.0)
    )

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '1024')
    rgb_bp.set_attribute('image_size_y', '1024')
    rgb_bp.set_attribute('fov', '90')
    rgb_bp.set_attribute('gamma', '2.2')
    rgb_bp.set_attribute('exposure_mode', 'histogram')
    rgb_bp.set_attribute('chromatic_aberration_intensity', '0.5')
    rgb_bp.set_attribute('chromatic_aberration_offset', '0.1')
    rgb_bp.set_attribute('lens_circle_falloff', '5.0')
    rgb_bp.set_attribute('lens_circle_multiplier', '0.0')
    rgb_bp.set_attribute('lens_k', '-0.02')
    rgb_bp.set_attribute('blur_amount', '0.3')
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform)

    sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '1024')
    sem_bp.set_attribute('image_size_y', '1024')
    sem_bp.set_attribute('fov', '90')
    sem_camera = world.spawn_actor(sem_bp, camera_transform)

    rgb_queue = queue.Queue()
    sem_queue = queue.Queue()
    rgb_camera.listen(rgb_queue.put)
    sem_camera.listen(sem_queue.put)

    anomaly_blueprints = [bp for bp in blueprint_library.filter('static.prop.*')
                           if 'box' not in bp.id and 'shopping_cart' not in bp.id]

    if not anomaly_blueprints:
        anomaly_blueprints = blueprint_library.filter('static.prop.*')

    DestroyActor = carla.command.DestroyActor

    print("Testing camera data streams...")
    try:
        rgb_queue.get(timeout=5.0)
        print("SUCCESS: Image stream received!")
    except queue.Empty:
        print("FATAL ERROR: Sensor streaming port blocked.")
        sys.exit(1)

    print(f"Starting generation of {args.num_frames} anomaly frames...")
    try:
        valid_frames = 0
        attempt = 0
        t_start = time.time()
        while valid_frames < args.num_frames:
            attempt += 1
            elapsed = time.time() - t_start
            print(f"[attempt {attempt}, {elapsed:.1f}s elapsed] starting iteration...")

            forward_vector = safe_transform.get_forward_vector()
            anomaly_bp = random.choice(anomaly_blueprints)
            print(f"[attempt {attempt}] chose blueprint: {anomaly_bp.id}")

            distance = random.uniform(3.0, 6.0)
            prop_x = safe_transform.location.x + (forward_vector.x * distance)
            prop_y = safe_transform.location.y + (forward_vector.y * distance)
            prop_z = safe_transform.location.z + 1.5

            # Baseline semantic frame BEFORE spawning -- captured fresh every
            # iteration (camera may have been repositioned on a failed spawn).
            # Used to diff out ambient scene clutter that happens to share the
            # spawned prop's semantic tag, so only pixels that are genuinely
            # NEW end up in the mask.
            with sem_queue.mutex:
                sem_queue.queue.clear()
            try:
                baseline_sem_image = sem_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"[attempt {attempt}] RETRY: baseline semantic frame timed out (2s) -- "
                      f"camera stream is lagging, likely GPU-bound")
                continue
            print(f"[attempt {attempt}] baseline captured OK")
            baseline_tags = np.frombuffer(baseline_sem_image.raw_data, dtype=np.uint8)
            baseline_tags = np.reshape(baseline_tags, (1024, 1024, 4))[:, :, 2]

            spawn_loc = carla.Location(x=prop_x, y=prop_y, z=prop_z)
            anomaly_actor = world.try_spawn_actor(anomaly_bp, carla.Transform(spawn_loc))

            if anomaly_actor is None:
                print(f"[attempt {attempt}] RETRY: spawn failed (collision/blocked) -- "
                      f"repositioning camera to a new spawn point")
                safe_transform = random.choice(spawn_points)
                new_cam_transform = carla.Transform(
                    carla.Location(
                        x=safe_transform.location.x,
                        y=safe_transform.location.y,
                        z=safe_transform.location.z + 2.0
                    ),
                    carla.Rotation(pitch=-10.0, yaw=safe_transform.rotation.yaw, roll=0.0)
                )
                rgb_camera.set_transform(new_cam_transform)
                sem_camera.set_transform(new_cam_transform)
                time.sleep(0.5)
                continue
            print(f"[attempt {attempt}] spawned {anomaly_bp.id} OK")

            anomaly_actor.set_simulate_physics(True)
            time.sleep(1.0)

            with rgb_queue.mutex:
                rgb_queue.queue.clear()
            with sem_queue.mutex:
                sem_queue.queue.clear()

            try:
                rgb_image = rgb_queue.get(timeout=2.0)
                sem_image = sem_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"[attempt {attempt}] RETRY: post-spawn frame capture timed out (2s)")
                client.apply_batch([DestroyActor(anomaly_actor)])
                continue
            print(f"[attempt {attempt}] post-spawn frames captured OK, computing mask...")

            img_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
            img_array = np.reshape(img_array, (1024, 1024, 4))
            rgb_out = img_array[:, :, :3][:, :, ::-1]

            sem_array = np.frombuffer(sem_image.raw_data, dtype=np.uint8)
            sem_array = np.reshape(sem_array, (1024, 1024, 4))
            after_tags = sem_array[:, :, 2]

            binary_mask = extract_new_object_mask(after_tags, baseline_tags)
            print(f"[attempt {attempt}] mask pixel count = {binary_mask.sum()}")

            if binary_mask.sum() == 0:
                # No connected blob of changed pixels found -- actor likely
                # fully occluded, or the diff picked up nothing at all.
                print(f"[attempt {attempt}] RETRY: no changed blob found (nothing new detected)")
                client.apply_batch([DestroyActor(anomaly_actor)])
                continue

            detected_tags = np.unique(after_tags[binary_mask == 1]).tolist()

            Image.fromarray(rgb_out).save(f"{args.out_images}/anomaly_{valid_frames:03d}.png")
            np.save(f"{args.out_masks}/anomaly_mask_{valid_frames:03d}.npy", binary_mask)

            client.apply_batch([DestroyActor(anomaly_actor)])

            valid_frames += 1
            print(f"Generated frame {valid_frames}/{args.num_frames} "
                  f"(detected_tags={detected_tags}, mask_px={binary_mask.sum()})")

    finally:
        print("Cleaning up world...")
        try:
            rgb_camera.stop()
            sem_camera.stop()
            client.apply_batch([DestroyActor(rgb_camera), DestroyActor(sem_camera)])
        except Exception:
            pass
        print("Dataset generation completed and safely shut down.")

if __name__ == '__main__':
    main()
