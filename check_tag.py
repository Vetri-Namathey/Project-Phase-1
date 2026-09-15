import carla
import random
import time

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
transform = random.choice(spawn_points)

props = [bp for bp in bp_lib.filter('static.prop.*')
         if 'box' not in bp.id and 'shopping_cart' not in bp.id]

for bp in props[:5]:
    loc = carla.Location(transform.location.x, transform.location.y, transform.location.z + 1.5)
    actor = world.try_spawn_actor(bp, carla.Transform(loc))
    if actor:
        time.sleep(1.0)
        print(bp.id, "-> semantic_tags:", actor.semantic_tags)
        actor.destroy()
        time.sleep(0.2)
    else:
        print(bp.id, "-> failed to spawn")
