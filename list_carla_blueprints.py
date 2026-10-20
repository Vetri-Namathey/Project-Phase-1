"""List the CARLA 0.9.15 blueprints that matter for the truck-drop scenario.

Needs a running CARLA server (CarlaUE4.exe). Read-only apart from spawning and
destroying temporary actors. Never run it while model inference uses the GPU.

For every static.prop.* it reports the size (bounding-box extent) and whether it
really falls and moves when physics is switched on: spawned 3 m above the road,
80 ticks at 0.05 s. It also lists truck-like and car-like vehicle blueprints.

    python list_carla_blueprints.py
    python list_carla_blueprints.py --town Town02 > carla_blueprints.txt
"""

import argparse

import carla

HOVER_M = 3.0
TICKS = 80
TRUCK_WORDS = ("truck", "carlamotors", "van", "ambulance", "firetruck", "sprinter", "hgv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", default=None, help="e.g. Town02; default = keep the loaded map")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    old = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn = world.get_map().get_spawn_points()[0]
    base = carla.Transform(carla.Location(spawn.location.x, spawn.location.y, spawn.location.z + HOVER_M),
                           carla.Rotation())
    try:
        ground_z = spawn.location.z  # the spawn point lies on the road surface
        print("=== static.prop.* (size = bounding-box half-extent in metres) ===")
        print("result: LANDED = ended within -0.3..+1.2 m of the road; AIR = never left its hover height;")
        print("        THROUGH = ended more than 0.3 m below the road (no collision, unusable);")
        print("        turn_deg = pitch/roll change (tumbling); drift_m = real horizontal movement")
        print(f"{'blueprint':36s} {'half-extent x,y,z':20s} {'result':>8s} {'end_dz_m':>9s} {'turn_deg':>9s} {'drift_m':>8s}")
        for bp in sorted(bp_lib.filter("static.prop.*"), key=lambda b: b.id):
            actor = world.try_spawn_actor(bp, base)
            if actor is None:
                print(f"{bp.id:36s} {'spawn failed':20s}")
                continue
            try:
                ext = actor.bounding_box.extent
                world.tick()  # a spawned actor reports its pose only after the first tick
                start = actor.get_transform()
                actor.set_simulate_physics(True)
                for _ in range(TICKS):
                    world.tick()
                end = actor.get_transform()
                dz = end.location.z - ground_z
                if dz < -0.3:
                    result = "THROUGH"
                elif dz <= 1.2:
                    result = "LANDED"
                elif abs(end.location.z - start.location.z) < 0.5:
                    result = "AIR"
                else:
                    result = "MID-AIR"
                drift = ((end.location.x - start.location.x) ** 2 + (end.location.y - start.location.y) ** 2) ** 0.5
                turn = abs(end.rotation.pitch - start.rotation.pitch) + abs(end.rotation.roll - start.rotation.roll)
                print(f"{bp.id:36s} {ext.x:6.2f},{ext.y:5.2f},{ext.z:5.2f}    {result:>8s} {dz:9.2f} {turn:9.1f} {drift:8.2f}")
            finally:
                actor.destroy()
                world.tick()

        print("\n=== vehicle.* that could be the lead truck ===")
        for bp in sorted(bp_lib.filter("vehicle.*"), key=lambda b: b.id):
            if any(w in bp.id for w in TRUCK_WORDS):
                print(bp.id)
        print("\n=== all other vehicle.* (non-OOD traffic) ===")
        for bp in sorted(bp_lib.filter("vehicle.*"), key=lambda b: b.id):
            if not any(w in bp.id for w in TRUCK_WORDS):
                print(bp.id)
    finally:
        world.apply_settings(old)


if __name__ == "__main__":
    main()
