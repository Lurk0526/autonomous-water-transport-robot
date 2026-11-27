"""Export the portable Gazebo scene and compare its closed corridor with the reference model."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ring_training import world


def corridor_stats(points):
    if points.ndim != 2 or points.shape[1] != 6 or not np.isfinite(points).all():
        raise ValueError("Expected finite pairs of (x,y) for center, inner and outer boundaries")
    if not np.allclose(points[0], points[-1], atol=1e-6, rtol=0):
        raise ValueError("Center or boundary does not close at the original start gate")
    center = points[:-1, :2]
    tangent = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    moving = lengths > 1e-9
    inner = points[:-1, 2:4] - center
    outer = points[:-1, 4:6] - center
    def side(offset):
        return tangent[:,0] * offset[:,1] - tangent[:,1] * offset[:,0]
    if not moving.any() or not np.all((side(inner)*side(outer))[moving] < 0):
        raise ValueError("Both boundaries must stay on opposite sides of the direction of travel")
    widths = np.linalg.norm(inner - outer, axis=1)
    return {"waypoint_rows_including_closure": len(points), "zero_length_segments_skipped": int((~moving).sum()),
            "reference_polyline_length_m": float(lengths.sum()),
            "min_boundary_spacing_m": float(widths.min()), "mean_boundary_spacing_m": float(widths.mean()),
            "max_boundary_spacing_m": float(widths.max()), "all_three_curves_close": True,
            "boundaries_on_opposite_sides": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"worlds")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--layout", choices=world.LAYOUTS, default=world.DEFAULT_LAYOUT)
    parser.add_argument("--cone-count", choices=(3, 4, 5), type=int)
    opts = parser.parse_args()
    directory = opts.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sdf = world.write_world(directory/"ring.sdf", seed=opts.seed, layout=opts.layout, cone_count=opts.cone_count)
    tree = ET.parse(sdf)
    for element in (*tree.iter("uri"), *tree.iter("albedo_map")):
        file = Path(element.text)
        if file.is_absolute():
            relative = file.relative_to(directory)
            if not file.is_file():
                raise FileNotFoundError(file)
            element.text = relative.as_posix()
    tree.write(sdf, encoding="utf-8", xml_declaration=True)
    reference_dir = ROOT/"references/deepracer-oval"
    reference = np.load(reference_dir/"bundle/routes/Oval_track.npy", allow_pickle=False)
    # Use the same six columns as the upstream waypoints, starting our loop at the timing gate.
    offsets = (0, world.YELLOW_LINE_OFFSET, -world.YELLOW_LINE_OFFSET)
    distances = np.linspace(0, world.TRACK_LENGTH, math.ceil(world.TRACK_LENGTH/0.25)+1)
    points = np.array([[coord for d in offsets for coord in world.centreline(world.START_S+s, d)[:2]]
                       for s in distances])
    with (directory/"ring-waypoints.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["center_x_m","center_y_m","inner_line_x_m","inner_line_y_m","outer_line_x_m","outer_line_y_m"])
        writer.writerows(points)
    cone_rows = world.cone_stations(opts.seed, opts.layout, opts.cone_count)
    report = {"world_schema": world.WORLD_SCHEMA, "seed": opts.seed, "layout": world.resolve_layout(opts.seed, opts.layout),
              "reference_model": "AWS DeepRacer Oval Track", "source_revision": json.loads((reference_dir/"source.json").read_text())["revision"],
              "upstream_waypoints": corridor_stats(reference), "gazebo_oval": corridor_stats(points),
              "yellow_line_width_m": world.YELLOW_LINE_WIDTH,
              "usable_width_between_inner_edges_m": 2*world.YELLOW_LINE_OFFSET-world.YELLOW_LINE_WIDTH,
              "paved_width_m": 2 * world.ROAD_HALF_WIDTH,
              "bypass_shoulder_outside_each_yellow_line_m": world.ROAD_HALF_WIDTH - world.YELLOW_LINE_OFFSET - world.YELLOW_LINE_WIDTH / 2,
              "cart_cad_width_m": world.VEHICLE_CAD["display_mesh_width_m"],
              "cart_scoring_width_m": 2 * world.BODY_HALF_WIDTH, "cart_wheelbase_m": world.WHEELBASE,
              "cone_count": len(cone_rows), "cone_offsets_from_center_m": [offset for _,offset in cone_rows],
              "cone_forward_spacing_m": [b[0]-a[0] for a,b in zip(cone_rows,cone_rows[1:])],
              "one_start_finish_center_xy_m": points[0,:2].tolist(), "analytical_lap_length_m": world.TRACK_LENGTH}
    (directory/"geometry-check.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
