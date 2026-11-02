"""Synthetic route geometry and independently varied disturbances."""
from dataclasses import replace
import math
from .model import Obstacle, Route
from .navigation import Experiment


def straight(length=9.0, start_x=0.0, half_width=0.9):
    n = math.ceil(length/0.08)
    return Route([(start_x+length*i/n, 0.0) for i in range(n+1)], half_width)


def bend(sign=1, radius=2.0):
    points = [(3*i/40, 0) for i in range(41)]
    for i in range(1, 51):
        angle = math.pi/2*i/50
        points.append((3+radius*math.sin(angle), sign*radius*(1-math.cos(angle))))
    for i in range(1, 31):
        points.append((3+radius, sign*(radius+2*i/30)))
    return Route(points)


def s_curve():
    return Route([(12*i/160, math.sin(2*math.pi*i/160)) for i in range(161)], 1.0)


def hairpin():
    radius=1.4
    points=[(3*i/40,0) for i in range(41)]
    points += [(3+radius*math.sin(math.pi*i/90),radius*(1-math.cos(math.pi*i/90))) for i in range(1,91)]
    points += [(3-3*i/40,2*radius) for i in range(1,41)]
    return Route(points,.75)


def crossing_loop():
    return Route([(4*math.sin(-math.pi/2+2*math.pi*i/400),
                   2*math.sin(2*(-math.pi/2+2*math.pi*i/400))) for i in range(401)],.85)


def cases():
    return [
        Experiment("straight", "无标线直线", straight()),
        Experiment("left", "无标线左转", bend(1)),
        Experiment("right", "无标线右转", bend(-1)),
        Experiment("s_curve", "无标线 S 弯", s_curve()),
        Experiment("noise_delay", "S 弯＋噪声＋双向延迟", s_curve(), position_noise_m=.025,
                   yaw_noise_rad=.015, localization_delay_s=.18, command_delay_s=.16),
        Experiment("persistent_obstacle", "固定障碍停车", straight(), expected="obstacle_stop",
                   obstacles=(Obstacle("box", 4, 0, .25),), stop_on_block=True),
        Experiment("temporary_obstacle", "障碍撤走后重新行驶", straight(),
                   obstacles=(Obstacle("box", 3, 0, .25, 0, 13),),
                   notes="障碍消失且观测连续清晰 0.6 秒后恢复，故障锁存不会自动恢复。"),
        Experiment("sudden_obstacle", "前方突然出现障碍", straight(), expected="obstacle_stop",
                   obstacles=(Obstacle("intruder", 4, 0, .25, 4),), stop_on_block=True),
        Experiment("offpath_obstacle", "路旁障碍无需停车", straight(),
                   obstacles=(Obstacle("roadside", 4, 1.1, .20),)),
        Experiment("detour", "已知宽走廊内平滑绕障", straight(10.0, half_width=1.3),
                   obstacles=(Obstacle("box", 4.5, 0, .25),), replan=True,
                   notes="只在已知直走廊内搜索平滑候选路径，原始道路边界保持不变。"),
        Experiment("detour_side", "一侧受阻时选择另一侧绕行", straight(10.0, half_width=1.3),
                   obstacles=(Obstacle("box", 4.5, 0, .25), Obstacle("side", 4.5, .85, .20)), replan=True),
        Experiment("detour_rejected", "窄走廊无法绕行时停车", straight(9.0, half_width=.55),
                   expected="obstacle_stop", obstacles=(Obstacle("box", 4.5, 0, .28),),
                   replan=True, stop_on_block=True),
        Experiment("unknown_area", "前方未知区域按阻塞处理", straight(), expected="obstacle_stop",
                   obstacles=(Obstacle("unknown", 4, 0, .35, kind="unknown"),), stop_on_block=True),
        Experiment("blocked_corridor", "整条走廊受阻停车", straight(), expected="obstacle_stop",
                   obstacles=(Obstacle("barrier", 4, 0, 1.0),), stop_on_block=True),
        Experiment("localization_dropout", "定位断流后停车", straight(), expected="stale_localization",
                   fault="localization_dropout"),
        Experiment("obstacle_dropout", "障碍观测断流后停车", straight(), expected="stale_obstacles",
                   fault="obstacle_dropout"),
        Experiment("pose_jump", "定位突跳拒绝继续", straight(), expected="localization_jump", fault="pose_jump"),
        Experiment("frozen_timestamp", "定位重复旧时间戳", straight(), expected="stale_localization",
                   fault="frozen_timestamp"),
        Experiment("nonfinite_pose", "非法定位数值", straight(), expected="nonfinite_pose", fault="nonfinite_pose"),
        Experiment("uncertain_pose", "定位不确定度过大", straight(), expected="localization_uncertain",
                   position_noise_m=.20),
        Experiment("tight_turn", "转弯半径不足拒绝出发", bend(radius=.35), expected="infeasible_route"),
        Experiment("outside_corridor", "初始车位安全余量不足", straight(), expected="outside_corridor",
                   initial_offset_m=.72),
        Experiment("hairpin", "前轮转向掉头路线", hairpin()),
        Experiment("crossing_loop", "交叉闭环路线保持进度", crossing_loop(), max_time_s=130),
        Experiment("long_winding", "长距离连续弯道", Route([(24*i/320,1.2*math.sin(4*math.pi*i/320)) for i in range(321)],1.0), max_time_s=130),
        Experiment("narrow_s", "窄路连续弯道", Route(s_curve().points,.48), initial_offset_m=.05),
        Experiment("steering_lag", "转向响应慢于控制模型", s_curve(), plant_overrides={"steer_time_constant_s":.40,"steer_rate_rad_s":.45}),
        Experiment("wheelbase_mismatch", "实际轴距比控制参数大25%", s_curve(), plant_overrides={"wheelbase_m":.40,"width_m":.36}),
        Experiment("reduced_braking", "制动能力下降时停车", straight(), expected="obstacle_stop",
                   obstacles=(Obstacle("box",4,0,.25),), stop_on_block=True,
                   plant_overrides={"emergency_decel_m_s2":.30}),
        Experiment("delayed_obstacles", "障碍观测与指令都有延迟", straight(),expected="obstacle_stop",
                   obstacles=(Obstacle("box",4,0,.25),), obstacle_delay_s=.30,command_delay_s=.18,stop_on_block=True),
        Experiment("future_timestamp", "输入时间戳异常超前", straight(),expected="future_timestamp",fault="future_timestamp"),
        Experiment("frozen_pose", "位姿不变但速度持续报告运动", straight(),expected="localization_inconsistent",fault="frozen_pose"),
        Experiment("growing_uncertainty", "无绝对校正时上报不确定度增长", straight(),expected="localization_uncertain",
                   position_noise_m=.01,drift_y_m_s=.015,reported_sigma_growth_m_s=.025),
        Experiment("startup_missing", "启动时传感器迟迟未就绪", straight(),expected="missing_inputs",localization_delay_s=1.4),
        Experiment("false_obstacle", "短时误检停止后重新确认", straight(),obstacles=(Obstacle("false_positive",2.0,0,.20,2,5),)),
        Experiment("curved_detour", "弯道路廊内平滑绕障", Route(bend().points,1.3),
                   obstacles=(Obstacle("curve_box",*bend().at(4.2)[:2],.25),),replan=True),
        Experiment("s_detour", "S弯内绕障后回到参考路线", Route(s_curve().points,1.3),
                   obstacles=(Obstacle("s_box",*s_curve().at(6.0)[:2],.25),),replan=True),
        Experiment("curved_no_space", "弯道净空不足拒绝绕行", Route(bend().points,.55),expected="obstacle_stop",
                   obstacles=(Obstacle("curve_box",*bend().at(4.2)[:2],.32),),replan=True,stop_on_block=True),
    ]


def sweep_cases():
    base = Experiment("sweep", "参数敏感性", s_curve(), yaw_noise_rad=.01)
    for noise in (0.0, .025, .05):
        for delay in (0.0, .12, .25):
            for seed in range(3):
                yield replace(base, name=f"noise{noise:g}_delay{delay:g}_seed{seed}",
                              position_noise_m=noise, localization_delay_s=delay,
                              command_delay_s=delay, seed=seed)
