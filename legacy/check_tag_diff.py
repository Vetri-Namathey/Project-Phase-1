"""One-off CARLA probe, kept for reference -- not part of the pipeline.

Captures a semantic-segmentation frame before and after spawning one prop,
to confirm a spawned object actually shows up as its own tag in the
semantic camera. That technique is what `generate_anomalies.py` now uses to
build exact masks. Needs the `carla` package and a running CARLA server on
localhost:2000.
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

with sem_queue.mutex:
    sem_queue.queue.clear()
img1 = sem_queue.get(timeout=5.0)
tags1 = np.frombuffer(img1.raw_data, dtype=np.uint8).reshape((1024, 1024, 4))[:, :, 2]
vals1, counts1 = np.unique(tags1, return_counts=True)
print("BASELINE:", dict(zip(vals1.tolist(), counts1.tolist())))

forward = transform.get_forward_vector()
loc = carla.Location(
    x=transform.location.x + forward.x * 4.0,
    y=transform.location.y + forward.y * 4.0,
    z=transform.location.z + 1.0
)
actor = world.try_spawn_actor(bp_lib.find('static.prop.atm'), carla.Transform(loc))
print("Spawned ATM:", actor is not None)
time.sleep(1.5)

with sem_queue.mutex:
    sem_queue.queue.clear()
img2 = sem_queue.get(timeout=5.0)
tags2 = np.frombuffer(img2.raw_data, dtype=np.uint8).reshape((1024, 1024, 4))[:, :, 2]
vals2, counts2 = np.unique(tags2, return_counts=True)
print("WITH PROP:", dict(zip(vals2.tolist(), counts2.tolist())))

if actor:
    actor.destroy()
sem_camera.stop()
sem_camera.destroy()
