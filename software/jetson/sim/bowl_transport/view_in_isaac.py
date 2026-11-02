"""Open the fixture in Isaac Sim. This is a scene smoke check, not a trained task."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,default=Path(__file__).parent/'output'/'bowl_platform_fixture.usda')
p.add_argument('--headless',action='store_true')
p.add_argument('--rebuild',action='store_true',help='Rebuild fixture from scenario.json inside the Isaac runtime')
p.add_argument('--frames',type=int,default=0,help='0 keeps GUI open; headless defaults to 240 updates')
args=p.parse_args()
if not args.scene.is_file() and not args.rebuild:p.error('Scene missing; pass --rebuild')
model=Path('/proc/device-tree/model')
if model.exists() and 'Jetson' in model.read_text():
    p.error('Run Isaac Sim on the rented RTX workstation, not this Jetson. demo.py works here.')
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless})
try:
    if args.rebuild:
        from build_scene import build
        args.scene=args.scene.resolve();args.scene.parent.mkdir(parents=True,exist_ok=True)
        build(json.loads(Path(__file__).with_name('scenario.json').read_text()),args.scene)
    import omni.usd
    import omni.timeline
    if not omni.usd.get_context().open_stage(str(args.scene.resolve())):
        raise RuntimeError('Failed to open stage')
    for _ in range(10):app.update()
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport:viewport.camera_path='/World/OverviewCamera'
    omni.timeline.get_timeline_interface().play()
    limit=args.frames or (240 if args.headless else 0)
    i=0
    while app.is_running() and (not limit or i<limit):
        app.update();i+=1
    print(f'Fixture scene updated {i} times. No articulated robot or physical fluid is present.')
finally:
    app.close()
