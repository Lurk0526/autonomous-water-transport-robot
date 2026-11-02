# AstraGO P4 content basis — 2026-10-19

Approved structure: 10 slides, 7–8 minutes. English slides, bilingual speaker notes. Warm ivory / graphite / burnt orange, continuing P3's visual identity; no literal template reuse required.

Primary authority: current user answers to the 26 P4 questions and supplied images. These supersede older PROJECT_CONTEXT and P3 facts.

## Evidence and limits

- Vision: team reports OpenMV image capture, YOLO11n-seg on Orin, STM32-controlled arm tracking of a moving black round bowl. Supplied photos document bench setup and preview; they do not independently show segmentation outputs or a quantitative benchmark.
- Dataset: over 500 images, varied positions/lighting/backgrounds, random 80/10/10 split. No exact split counts or mAP/P/R supplied. Never invent test results.
- Drive: STM32 controls motor controller; start/stop, forward/reverse, speed adjustment and actual speed readback demonstrated per user. No speed-loop, steering or whole-vehicle performance claims.
- Chassis: two 6060 longitudinal rails, three 4545 crossmembers; fewer members/connectors and more equipment room per user; parts available and fabrication/assembly underway. No stiffness/weight improvement percentage or FEA claim.
- Gripper: supplier base model with team-modified arm mounting interface; manufactured, not mounted/tested. Supplier image does not show the team's adapter design in detail.
- Navigation camera received, streaming pending. Isaac Lab installed, arm import pending. Remote emergency power-off planned.

## Supplied source image provenance

1. vision_bench.jpg — user's bench photo, attachment 1
2. vision_tracking.jpg — user's closer bench photo, attachment 2
3. chassis_iso.png — user's latest chassis CAD screenshot, attachment 3
4. chassis_plan.png — user's latest chassis CAD screenshot, attachment 4
5. gripper_supplier.jpg — user-supplied supplier CAD image, attachment 5; supplier base model, not team-original geometry

No AI-generated photographs or fabricated detection overlays. Native diagrams describe functional control relationships rather than a verified electrical wiring schematic. Planned pathways are dashed and labelled.
