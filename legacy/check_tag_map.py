"""One-off CARLA probe, kept for reference -- not part of the pipeline.

The sweep version of `check_tag_diff.py`: spawns every `static.prop.*`
blueprint in turn and reports which semantic tag each one lands under, by
pixel-count difference against a baseline frame. Used to pick the props
`generate_anomalies.py` spawns. Needs the `carla` package and a running
CARLA server on localhost:2000.
"""

import carla
import time
import numpy as np
import queue

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
transform = spawn_points[0]

camera_transform = carla.Transform(
    carla.Location(x=transform.location.x, y=transform.location.y, z=transform.location.z + 2.0),
    carla.Rotation(pitch=-10.0, yaw=transform.rotation.yaw, roll=0.0)
)

sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
sem_bp.set_attribute('image_size_x', '1024')
sem_bp.set_attribute('image_size_y', '1024')
sem_bp.set_attribute('fov', '90')
sem_camera = world.spawn_actor(sem_bp, camera_transform)

sem_queue = queue.Queue()
sem_camera.listen(sem_queue.put)
time.sleep(1.0)

def capture():
    with sem_queue.mutex:
        sem_queue.queue.clear()
    img = sem_queue.get(timeout=5.0)
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape((1024, 1024, 4))[:, :, 2]

base_vals, base_counts = np.unique(capture(), return_counts=True)
base_map = dict(zip(base_vals.tolist(), base_counts.tolist()))

props = [bp for bp in bp_lib.filter('static.prop.*')
         if 'box' not in bp.id and 'shopping_cart' not in bp.id]

forward = transform.get_forward_vector()
loc = carla.Location(
    x=transform.location.x + forward.x * 4.0,
    y=transform.location.y + forward.y * 4.0,
    z=transform.location.z + 1.0
)

print(f"Testing {len(props)} prop types...\n")
for bp in props:
    actor = world.try_spawn_actor(bp, carla.Transform(loc))
    if actor is None:
        print(f"{bp.id}: FAILED TO SPAWN")
        continue
    actor.set_simulate_physics(True)
    time.sleep(1.0)
    vals, counts = np.unique(capture(), return_counts=True)
    cur_map = dict(zip(vals.tolist(), counts.tolist()))
    best_tag, best_diff = None, 0
    for t, c in cur_map.items():
        diff = c - base_map.get(t, 0)
        if diff > best_diff:
            best_diff = diff
            best_tag = t
    print(f"{bp.id}: tag={best_tag} (+{best_diff} px)")
    actor.destroy()
    time.sleep(0.3)

sem_camera.stop()
sem_camera.destroy()
