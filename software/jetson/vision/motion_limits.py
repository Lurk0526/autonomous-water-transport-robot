"""Shared startup, calibration and tracking ranges."""
import json

# Elbow lower target: calibration-20261022-200630, commanded 1437 /
# measured 1473, followed by the accepted five-axis trial. Upper target:
# calibration-20261023-204203, commanded 2447 / measured 2456.
CURRENT_POSE_ENVELOPES = {0: (1100, 1900), 2: (1437, 2447), 3: (650, 1800)}
CURRENT_POSE_MAX_WINDOW = max(high-low for low,high in CURRENT_POSE_ENVELOPES.values())
FULL_RANGE_POLICY = 'operator_confirmed_full_range'
FULL_POSE_ENVELOPES = {axis: (500, 2500) for axis in range(6)}


def pose_envelopes(config=None):
    policy = (config or {}).get('motion_range_policy', 'commissioned')
    if policy == FULL_RANGE_POLICY:
        return dict(FULL_POSE_ENVELOPES)
    if policy != 'commissioned':
        raise ValueError(f'Unknown motion range policy: {policy}')
    return dict(CURRENT_POSE_ENVELOPES)


def pose_max_window(config):
    return max(high-low for low, high in pose_envelopes(config).values())


def runtime_profile(root):
    """Apply the persistent user-selected policy to old and newly built sessions."""
    path = root / 'config/bowl-follow.runtime.json'
    if not path.exists():
        return {}
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or set(profile)-{
            'motion_range_policy', 'profile_version', 'range_authorization', 'response_profile'}:
        raise ValueError('Invalid runtime follow profile')
    pose_envelopes(profile)
    return profile


def validate_current_pose(positions, axes, *, tolerance_p=10, config=None):
    """Reject unsupported starts with every offending measured joint listed."""
    if type(tolerance_p) is not int or not 0 <= tolerance_p <= 12:
        raise ValueError('Invalid bounded actuator setting: feedback_tolerance_p')
    names = {0: '底座', 1: '肩部', 2: '肘部', 3: '腕部', 4: '末端旋转', 5: '夹爪'}
    envelopes = pose_envelopes(config)
    if (config or {}).get('motion_range_policy') == FULL_RANGE_POLICY:
        axes = range(6)
    outside = []
    for axis in axes:
        if axis not in envelopes:
            continue
        low, high = envelopes[axis]
        actual = positions.get(axis)
        if type(actual) is not int or not max(500,low-tolerance_p) <= actual <= min(2500,high+tolerance_p):
            outside.append(f'{axis:03d} {names[axis]} P{actual}（允许 P{low}～P{high}）')
    if outside:
        raise ValueError('当前位置超出所选跟随范围：' + '；'.join(outside)
                         + '。请检查关节读数和当前范围配置。')
