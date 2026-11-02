#!/usr/bin/env python3
"""Jetson 抓碗项目环境诊断。仅依赖标准库；不安装软件、不控制机械臂。

默认不打开相机或串口。GPU 检查会分配少量显存并运行小规模计算。
示例：python3 check_jetson_env.py --json jetson_check.json
"""

import argparse
import collections
import datetime
import glob
import grp
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time


RESULTS = []
MARKER = "__JETSON_CHECK_RESULT__"


def report(section, name, status, detail, hint=""):
    item = dict(section=section, name=name, status=status, detail=detail, hint=hint)
    RESULTS.append(item)
    print("\n[{}] {} / {}".format(status, section, name), flush=True)
    print(detail, flush=True)
    if hint:
        print("  提示：" + hint, flush=True)


def read(path):
    try:
        return Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return ""


def run(argv, timeout=15):
    """每项检查独立超时；终止整个子进程组，防止原生库挂起。"""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return 124, out, err + "\n检查超时（{} 秒）".format(timeout)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise


def command(section, name, argv, hint="", timeout=15):
    if not shutil.which(argv[0]) and not Path(argv[0]).is_file():
        report(section, name, "MISSING", "命令不可见：" + argv[0], hint)
        return
    rc, out, err = run(argv, timeout)
    detail = (out + ("\n" + err if err else "")).strip()
    report(section, name, "INFO" if rc == 0 else "WARN",
           detail or "无输出；退出码 {}".format(rc), hint)


def probe(name, module, body, hint="", timeout=45):
    # 使用当前解释器，并隔离 import、GPU 运算、崩溃和 SDK 日志。
    source = """
import importlib, importlib.util, json, traceback
def emit(status, detail):
    print(%r + json.dumps(dict(status=status, detail=detail), ensure_ascii=False), flush=True)
try:
    if importlib.util.find_spec(%r) is None:
        emit('MISSING', '当前 Python 环境中未找到模块：' + %r)
    else:
        m = importlib.import_module(%r)
%s
except BaseException:
    emit('ERROR', traceback.format_exc())
""" % (MARKER, module, module, module, textwrap.indent(textwrap.dedent(body), "        "))
    rc, out, err = run([sys.executable, "-c", source], timeout)
    lines = [line[len(MARKER):] for line in out.splitlines() if line.startswith(MARKER)]
    if lines and rc == 0:
        try:
            item = json.loads(lines[-1])
            detail = item["detail"]
            if not isinstance(detail, str):
                detail = json.dumps(detail, ensure_ascii=False, indent=2)
            report("Python/运行", name, item["status"], detail, hint)
            return
        except (ValueError, KeyError):
            pass
    report("Python/运行", name, "ERROR", "子进程退出码 {}\n{}\n{}".format(
        rc, out[-4000:], err[-4000:]), hint)


def system_checks():
    model = read("/proc/device-tree/model") or read("/sys/firmware/devicetree/base/model")
    report("系统", "板卡", "INFO" if model else "WARN", model or "未读取到 Jetson 设备树型号。",
           "以实际型号为准；若是容器，设备树不可见不等于宿主机不是 Jetson。")
    report("系统", "操作系统", "INFO", dict_text({
        "kernel": platform.release(), "arch": platform.machine(),
        "os_release": read("/etc/os-release"),
        "L4T": read("/etc/nv_tegra_release") or "未找到 /etc/nv_tegra_release"}))
    report("系统", "当前解释器", "INFO", dict_text({
        "python": sys.version, "executable": sys.executable,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")}))
    mem = read("/proc/meminfo")
    selected = [line for line in mem.splitlines() if line.split(":")[0] in
                ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")]
    disks = {}
    for path in dict.fromkeys([str(Path.cwd()), "/tmp"]):
        usage = shutil.disk_usage(path)
        disks[path] = {"total_GiB": round(usage.total / 2**30, 2),
                       "free_GiB": round(usage.free / 2**30, 2)}
    report("系统", "内存与存储", "INFO", "\n".join(selected) + "\n" + dict_text(disks),
           "Orin 使用共享内存；这里不是模型推理容量或实时性能测试。")
    command("系统", "JetPack/推理包", ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n",
            "nvidia-jetpack*", "nvidia-l4t-core", "cuda-cudart-*", "libcudnn*", "libnvinfer*"],
            "某些包未安装可导致非零退出码。JetPack 元包缺失不代表运行组件全部缺失。")
    command("系统", "当前功耗模式", ["nvpmodel", "-q"],
            "只查询，不切换模式。权限不足时先保留结果；不要为检查脚本整体使用 sudo。")
    temps = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        temps.append({"type": read(zone + "/type"), "raw_millidegrees_C": read(zone + "/temp")})
    report("系统", "温度传感器快照", "INFO" if temps else "WARN", dict_text(temps),
           "单次温度读数不能证明持续负载下不会降频。")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
        nvcc = candidates[-1] if candidates else "nvcc"
    command("GPU", "CUDA 编译器", [nvcc, "--version"],
            "nvcc 是编译工具；缺失不直接判定 CUDA 推理不可用。")
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    if Path(ldconfig).is_file() or shutil.which(ldconfig):
        rc, out, err = run([ldconfig, "-p"])
        selected = [line.strip() for line in out.splitlines() if re.search(
            r"lib(cuda|cudart|cudnn|nvinfer|nvonnxparser|Orbbec|obsensor|OpenNI)", line, re.I)]
        report("GPU", "动态库缓存", "INFO" if rc == 0 else "WARN",
               "\n".join(selected) or err or "缓存中没有匹配项。",
               "库文件可见不等于能初始化或完成推理；pip/自定义路径的库可能不在缓存中。")


def dict_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def python_checks(skip_gpu):
    if skip_gpu:
        report("GPU", "CUDA 驱动初始化", "SKIP", "按 --skip-gpu 跳过")
    else:
        probe("CUDA 驱动初始化", "ctypes", """
lib = m.CDLL('libcuda.so.1')
lib.cuInit.argtypes = [m.c_uint]
lib.cuInit.restype = m.c_int
lib.cuDriverGetVersion.argtypes = [m.POINTER(m.c_int)]
lib.cuDriverGetVersion.restype = m.c_int
lib.cuDeviceGetCount.argtypes = [m.POINTER(m.c_int)]
lib.cuDeviceGetCount.restype = m.c_int
rc = lib.cuInit(0)
if rc:
    emit('ERROR', dict(cuInit_result=rc))
else:
    version, count = m.c_int(), m.c_int()
    vr = lib.cuDriverGetVersion(m.byref(version))
    cr = lib.cuDeviceGetCount(m.byref(count))
    emit('PASS' if vr == 0 and cr == 0 and count.value > 0 else 'ERROR',
         dict(driver_api_version=version.value, device_count=count.value,
              version_result=vr, count_result=cr,
              limitation='CUDA Driver API 版本并非已安装 CUDA Toolkit 版本'))
""", "独立于 PyTorch 检查 libcuda；初始化不等于模型推理成功。")
    probe("NumPy", "numpy", """
emit('PASS', dict(version=m.__version__, module=m.__file__))
""")
    torch_body = """
info = dict(version=m.__version__, module=m.__file__, built_cuda=m.version.cuda,
            cudnn_version=m.backends.cudnn.version())
if SKIP_GPU:
    emit('INFO', dict(info, gpu_test='已跳过 GPU 检查'))
elif not m.cuda.is_available():
    emit('ERROR', dict(info, cuda_available=False))
else:
    info['gpu'] = m.cuda.get_device_name(0)
    info['capability'] = m.cuda.get_device_capability(0)
    a = m.ones((64, 64), device='cuda')
    result = a @ a
    m.cuda.synchronize()
    assert m.allclose(result, m.full_like(result, 64)), 'GPU 矩阵计算结果错误'
    with m.no_grad():
        conv = m.nn.Conv2d(3, 4, 3).cuda()
        y = conv(m.zeros((1, 3, 32, 32), device='cuda'))
        m.cuda.synchronize()
        assert tuple(y.shape) == (1, 4, 30, 30) and bool(m.isfinite(y).all()), '卷积结果异常'
    info['gpu_matmul_and_conv'] = '通过；不是性能基准或 cuDNN 独立验证'
    info['cudnn_available'] = m.backends.cudnn.is_available()
    emit('PASS', info)
""".replace("SKIP_GPU", repr(skip_gpu))
    probe("PyTorch + GPU 计算", "torch", torch_body,
          "仅检测当前解释器。导入失败、CPU 版和 GPU 运行失败需要分别处理；不要直接用通用 pip 包覆盖 Jetson 环境。", 75)
    probe("torchvision NMS 算子", "torchvision", """
import torch
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
scores = torch.tensor([0.9, 0.8])
info = dict(version=m.__version__, cpu_nms=m.ops.nms(boxes, scores, 0.5).tolist())
assert info['cpu_nms'] == [0], 'CPU NMS 输出异常'
if not SKIP_GPU and torch.cuda.is_available():
    info['gpu_nms'] = m.ops.nms(boxes.cuda(), scores.cuda(), 0.5).cpu().tolist()
    assert info['gpu_nms'] == [0], 'GPU NMS 输出异常'
emit('PASS', info)
""".replace("SKIP_GPU", repr(skip_gpu)), "用于发现 torch/torchvision 二进制或 CUDA 算子不匹配。", 75)
    probe("TensorRT", "tensorrt", """
info = dict(version=m.__version__, module=m.__file__)
if SKIP_GPU:
    emit('INFO', dict(info, builder='已跳过 GPU 初始化'))
else:
    logger = m.Logger(m.Logger.ERROR)
    plugins = m.init_libnvinfer_plugins(logger, '')
    builder = m.Builder(logger)
    if builder is None or not plugins:
        emit('ERROR', dict(info, plugins=plugins, builder=builder is not None))
    else:
        emit('PASS', dict(info, plugins=True, builder_initialized=True,
                          limitation='尚未构建引擎或执行模型推理'))
""".replace("SKIP_GPU", repr(skip_gpu)), "Builder 初始化通过也不能证明具体模型或外部 engine 可运行。", 60)
    probe("OpenCV 图像运算", "cv2", """
import numpy as np
im = np.zeros((32, 32, 3), dtype=np.uint8)
assert m.cvtColor(im, m.COLOR_BGR2GRAY).shape == (32, 32)
build = m.getBuildInformation()
features = [s.strip() for s in build.splitlines() if any(k in s for k in
            ['GStreamer', 'FFMPEG', 'v4l/v4l2', 'NVIDIA CUDA'])]
emit('PASS', dict(version=m.__version__, module=m.__file__, build_features=features))
""", "OpenCV 没有 CUDA 不妨碍 TensorRT；取流能力还需实际验证。")
    for name, module in [("ONNX", "onnx"), ("Orbbec Python SDK", "pyorbbecsdk"),
                         ("串口 Python 库", "serial"), ("YAML", "yaml")]:
        probe(name, module, """
emit('PASS', dict(version=getattr(m, '__version__', '未提供版本属性'),
                  module=getattr(m, '__file__', 'unknown'), limitation='仅验证导入'))
""", "此项缺失是否需要安装，取决于最终部署链路。")
    probe("ONNX Runtime", "onnxruntime", """
emit('INFO', dict(version=m.__version__, providers=m.get_available_providers(),
                 limitation='provider 列表不代表具体模型已成功使用 GPU'))
""", "直接使用 TensorRT 时不要求 ONNX Runtime GPU。")
    # 不导入 Ultralytics，避免首次导入时创建配置或触发额外行为。
    probe("Ultralytics 发行包", "importlib.metadata", """
try:
    emit('INFO', dict(version=m.version('ultralytics'), limitation='只读取包元数据；不下载或加载模型'))
except m.PackageNotFoundError:
    emit('MISSING', '未安装 ultralytics 发行包；TensorRT 独立部署不一定需要它')
""")
    command("多媒体", "GStreamer", ["gst-inspect-1.0", "--version"])
    if shutil.which("gst-inspect-1.0"):
        for plugin in ["v4l2src", "appsink", "nvarguscamerasrc"]:
            rc, _, err = run(["gst-inspect-1.0", "--exists", plugin])
            report("多媒体", plugin, "PASS" if rc == 0 else "MISSING",
                   "插件可见" if rc == 0 else err.strip() or "插件不可见",
                   "CSI Argus 插件与 USB 相机是不同链路；按实际相机选用。")


def device_checks():
    command("设备", "USB 拓扑与速率", ["lsusb", "-t"],
            "深度相机应核对实际 USB 链路速率；接口外形不能证明已经使用 USB 3。")
    command("设备", "V4L2 设备清单", ["v4l2-ctl", "--list-devices"],
            "不打开视频流；设备节点存在不代表已成功取流。")
    devices = []
    for pattern in ["/dev/video*", "/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyTHS*"]:
        for name in sorted(glob.glob(pattern)):
            st = os.stat(name)
            try:
                group = grp.getgrgid(st.st_gid).gr_name
            except KeyError:
                group = str(st.st_gid)
            devices.append(dict(path=name, group=group, mode=oct(st.st_mode & 0o777),
                                read_access=os.access(name, os.R_OK),
                                write_access=os.access(name, os.W_OK),
                                name=read("/sys/class/video4linux/" + Path(name).name + "/name")))
    report("设备", "相机/串口节点与权限", "INFO" if devices else "WARN", dict_text(devices),
           "不会打开串口或发送指令。权限可见也不证明设备空闲、协议兼容或机械臂反馈可用。")
    usb = []
    for path in sorted(glob.glob("/sys/bus/usb/devices/*")):
        if read(path + "/idVendor"):
            usb.append(dict(vendor=read(path + "/idVendor"), product_id=read(path + "/idProduct"),
                            product=read(path + "/product"), speed_Mbps=read(path + "/speed")))
    report("设备", "USB 产品信息（不采集序列号）", "INFO", dict_text(usb),
           "不能据此认定 Gemini Pro Plus 的准确型号、固件或 SDK 支持范围；需要商品/厂商资料核对。")


def ros_checks():
    report("ROS", "已安装目录与当前环境", "INFO", dict_text({
        "directories": sorted(glob.glob("/opt/ros/*")),
        "ROS_DISTRO": os.environ.get("ROS_DISTRO", "未设置"),
        "ROS_VERSION": os.environ.get("ROS_VERSION", "未设置"),
        "AMENT_PREFIX_PATH": os.environ.get("AMENT_PREFIX_PATH", "未设置"),
        "ros2_command": shutil.which("ros2"), "roscore_command": shutil.which("roscore")}),
        "未 source 的 ROS 可能已安装但在当前环境中不可用；脚本不会自动选择版本或连接 ROS 图。")
    probe("ROS 2 Python", "rclpy", """
emit('PASS', dict(module=m.__file__, limitation='仅导入，不初始化节点或连接设备'))
""")
    probe("ROS 2 项目包索引", "ament_index_python", """
from ament_index_python.packages import get_packages_with_prefixes
packages = get_packages_with_prefixes()
names = ['cv_bridge', 'image_transport', 'sensor_msgs', 'geometry_msgs', 'tf2_ros',
         'robot_state_publisher', 'joint_state_publisher', 'rosbag2', 'rosbag2_transport',
         'moveit_ros_move_group', 'moveit_servo', 'controller_manager',
         'joint_trajectory_controller', 'nav2_bringup', 'nav2_mppi_controller',
         'ackermann_msgs', 'orbbec_camera']
emit('INFO', {name: ('可见' if name in packages else '当前索引不可见') for name in names})
""", "这些是候选组件，不要求全部安装；包索引可见也不代表驱动或节点运行成功。")


def camera_check(device):
    if not re.fullmatch(r"/dev/video[0-9]+", device):
        raise ValueError("--camera 只接受 /dev/video数字，例如 /dev/video0")
    if not Path(device).exists():
        report("相机实测", device, "ERROR", "设备不存在")
        return
    probe("指定 USB/V4L2 相机单帧", "cv2", """
cap = m.VideoCapture(DEVICE, m.CAP_V4L2)
try:
    opened = cap.isOpened()
    ok, frame = cap.read() if opened else (False, None)
    if ok and frame is not None and frame.size:
        emit('PASS', dict(device=DEVICE, shape=list(frame.shape), saved_image=False,
                          limitation='仅一次取流，未验证连续帧率、深度或 RGB/深度对齐'))
    else:
        emit('ERROR', dict(device=DEVICE, opened=opened, frame_received=False))
finally:
    cap.release()
""".replace("DEVICE", repr(device)), "此项会短暂打开指定相机；不会保存图像。", 25)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, metavar="PATH", help="另外保存结构化报告，不覆盖已有文件")
    parser.add_argument("--skip-gpu", action="store_true", help="跳过 GPU 运算及 TensorRT Builder 初始化")
    parser.add_argument("--camera", metavar="/dev/videoX", help="可选：打开指定 V4L2 相机并读取一帧，不存图")
    parser.add_argument("--strict", action="store_true", help="有 ERROR 项时返回退出码 2；缺少可选项不算 ERROR")
    args = parser.parse_args()
    if args.camera and not re.fullmatch(r"/dev/video[0-9]+", args.camera):
        parser.error("--camera 只接受 /dev/video数字")
    if args.json and args.json.exists():
        parser.error("报告已存在，请选择新文件名：{}".format(args.json))
    if platform.system() != "Linux":
        parser.error("请在 Jetson 的 Linux 系统运行此脚本；Windows 仿真电脑需另行检查。")
    start = time.monotonic()
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print("Jetson 抓碗项目环境检查 v1.0 | " + timestamp, flush=True)
    print("PASS=本项检查通过 INFO=信息 WARN=需核对 MISSING=当前不可见 ERROR=运行失败 SKIP=跳过", flush=True)
    print("默认不打开相机/串口；不安装、不改功耗、不启动 ROS、不下载模型。", flush=True)
    if os.geteuid() == 0:
        report("系统", "运行身份", "WARN", "正在以 root 运行，权限结果不能代表普通用户。",
               "建议退出 sudo，在实际项目 Python 环境中以普通用户重新运行。")
    for name, check in [("系统", system_checks), ("Python", lambda: python_checks(args.skip_gpu)),
                        ("设备", device_checks), ("ROS", ros_checks)]:
        try:
            check()
        except Exception as exc:
            report(name, "检查过程异常", "ERROR", "{}: {}".format(type(exc).__name__, exc))
    if args.camera:
        camera_check(args.camera)
    else:
        report("相机实测", "视频帧读取", "SKIP", "默认仅列举设备，未验证 RGB/深度帧。",
               "普通 UVC 相机可用 --camera /dev/video0；Orbbec 深度链路需在准确型号确认后专项验证。")
    counts = dict(collections.Counter(item["status"] for item in RESULTS))
    print("\n汇总：" + dict_text(counts), flush=True)
    print("检查完成不等于整车或机械臂可用。未覆盖：实际模型推理、连续相机流、机械臂协议、承载、急停与控制时延。", flush=True)
    if args.json:
        try:
            with args.json.open("x", encoding="utf-8") as output:
                json.dump(dict(schema_version=1, timestamp=timestamp,
                               duration_seconds=round(time.monotonic() - start, 2),
                               counts=counts, checks=RESULTS), output, ensure_ascii=False, indent=2)
                output.write("\n")
            print("结构化报告：" + str(args.json.resolve()))
        except OSError as exc:
            print("无法保存 JSON 报告：{}".format(exc), file=sys.stderr)
            return 1
    return 2 if args.strict and counts.get("ERROR", 0) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断检查。", file=sys.stderr)
        sys.exit(130)
