"""Bounded deterministic detour search in a known-width road corridor.

This is a first local-planning baseline, not an arbitrary-world global planner.
Candidates are smooth lateral offsets; geometry, curvature and footprint checks
can reject all of them, in which case the independent stop monitor still applies.
"""
import math
from .model import Route, State, clamp, footprint_corners, obstacle_clearance, wrap


def quintic(u):
    u = clamp(u, 0.0, 1.0)
    return u*u*u*(10+u*(-15+6*u))


def corridor_detour(road, pose, obstacles, vehicle, navigation, road_progress=None):
    metadata = {"attempted": True, "candidate_count": 0, "feasible_count": 0,
                "accepted": False, "scope": "known_road_corridor"}
    if road_progress is None:
        road_progress=road.project(pose.x,pose.y)[0]
    projected=[(o,*road.project(o.x,o.y,road_progress,backward=.2,forward=navigation["sensor_range_m"]+1)) for o in obstacles]
    relevant=[o for o,s,e in projected if s>road_progress and abs(e)<road.half_width+o.radius]
    if not relevant:
        metadata["reason"] = "no_forward_obstacles"
        return None, metadata
    first_s=min(s-o.radius for o,s,e in projected if o in relevant)
    last_s=max(s+o.radius for o,s,e in projected if o in relevant)
    start=max(0,road_progress+.35,first_s-2.8)
    rise_end,fall_start=first_s-.5,last_s+.5
    end=min(road.length,fall_start+2.2)
    if rise_end-start < 1.2 or end-fall_start < 1.2:
        metadata["reason"] = "not_enough_distance_to_turn"
        return None, metadata
    margin = navigation["footprint_margin_m"]
    base_offset = max(o.radius for o in relevant)+vehicle["width_m"]/2+margin+.15
    feasible = []
    for sign in (1, -1):
        for extra in (0.0, .15, .30):
            offset = sign*(base_offset+extra)
            metadata["candidate_count"] += 1
            points = []
            for i,(x,y) in enumerate(road.points):
                heading=road.headings[min(i,len(road.headings)-1)]
                if 0<i<len(road.points)-1:
                    heading=road.headings[i-1]+wrap(road.headings[i]-road.headings[i-1])/2
                rise=quintic((road.s[i]-start)/(rise_end-start))
                fall=quintic((road.s[i]-fall_start)/(end-fall_start))
                shift=offset*rise*(1-fall)
                points.append((x-math.sin(heading)*shift,y+math.cos(heading)*shift))
            route = Route(points, road.half_width)
            max_curvature = route.max_curvature()
            if max_curvature > .90*math.tan(vehicle["max_steer_rad"])/vehicle["wheelbase_m"]:
                continue
            clear = True
            min_clearance = math.inf
            for index,distance in enumerate(route.s):
                x, y, yaw = route.at(distance)
                candidate = State(x, y, yaw)
                if any(abs(road.corridor_error(cx,cy,road.s[index]))+margin>road.half_width
                       for cx,cy in footprint_corners(candidate,vehicle)):
                    clear = False
                    break
                for o in relevant:
                    clearance = obstacle_clearance(candidate, o, vehicle, margin+.06)
                    min_clearance = min(min_clearance, clearance)
                    if clearance <= 0:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                metadata["feasible_count"] += 1
                score = route.length+.3*sum(k*k for k in route.curvatures)/len(route.curvatures)
                feasible.append((score, route, offset, min_clearance))
    if not feasible:
        metadata["reason"] = "all_candidates_violate_footprint_or_curvature"
        return None, metadata
    _, route, offset, clearance = min(feasible, key=lambda item: item[0])
    metadata.update(accepted=True, lateral_offset_m=offset,
                    planned_clearance_after_margin_m=clearance,
                    max_curvature_m_inv=route.max_curvature())
    return route, metadata
