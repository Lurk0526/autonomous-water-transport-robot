#!/usr/bin/env python3
"""Jetson software inventory. No installs, actuator commands or config changes.

Run using the SAME Python environment intended for the robot application:
    python3 check_jetson_env.py | tee jetson_check.txt
Only stdout is written unless the caller redirects it. Imports can initialize
library caches. CUDA smoke testing allocates a small tensor, not a benchmark.
Missing optional tools do not mean the entire system is broken.
"""
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def heading(label):
    print("\n=== " + label + " ===", flush=True)


def run(args, timeout=15, keep=None):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True,
                                timeout=timeout)
        lines = result.stdout.splitlines()
        if keep:
            lines = [line for line in lines if keep(line)]
        print("\n".join(lines) or "(no matching output)", flush=True)
        if result.returncode:
            print("[CHECK] exit code:", result.returncode, flush=True)
    except FileNotFoundError:
        print("[MISSING / not in PATH]", args[0], flush=True)
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", args[0], flush=True)
    except OSError as exc:
        print("[CHECK]", exc, flush=True)


def read_file(path):
    try:
        print(path + ": " + Path(path).read_text().replace("\x00", "").strip())
    except OSError:
        print("[UNAVAILABLE]", path)


def main():
    if platform.system() != "Linux":
        print("Run this script on Jetson Ubuntu, not on macOS/Windows.")
        return 2
    print("Read-only inventory; no serial port opens, no ROS launches, no installs.")
    heading("Hardware / OS / L4T")
    print("architecture:", platform.machine(), "kernel:", platform.release())
    for name in ("/proc/device-tree/model", "/etc/os-release", "/etc/nv_tegra_release"):
        read_file(name)
    run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n",
         "nvidia-l4t-core", "nvidia-jetpack"])
    print("nvidia-jetpack is a metapackage; its absence alone does not prove SDK components are absent.")
    heading("Memory / storage / current power mode")
    run(["free", "-h"])
    run(["df", "-h", "/"])
    run(["nvpmodel", "-q"])
    if shutil.which("tegrastats") and shutil.which("timeout"):
        run(["timeout", "3", "tegrastats", "--interval", "1000"], timeout=5)
        print("Exit 124 above is expected when the 3-second sample ends.")
    print("A permission error is inconclusive. This script does not elevate privileges or change power modes.")
    heading("Installed CUDA / TensorRT / cuDNN components")
    run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"],
        keep=lambda line: any(x in line.lower() for x in
                              ("cuda", "cudnn", "nvinfer", "tensorrt", "orbbec", "openni")))
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    run([nvcc, "--version"])
    print("nvcc tests the compiler only; missing nvcc does not prove the runtime is broken.")
    print("trtexec:", shutil.which("trtexec") or
          ("/usr/src/tensorrt/bin/trtexec" if Path("/usr/src/tensorrt/bin/trtexec").exists()
           else "not found in standard locations"))
    heading("Active Python environment")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV", "(none)"))
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "(none)"))
    run([sys.executable, "-m", "pip", "--version"])
    # Each import runs independently, so a broken binary module cannot stop all checks.
    for mod in ("numpy", "cv2", "torch", "torchvision", "tensorrt", "onnx",
                "onnxruntime", "pyorbbecsdk", "rclpy", "serial"):
        heading("Python module: " + mod)
        code = (
            "import importlib\n"
            "m = importlib.import_module(" + repr(mod) + ")\n"
            "print('version:', getattr(m, '__version__', 'not exposed'))\n"
            "print('path:', getattr(m, '__file__', 'not exposed'))\n"
        )
        if mod == "cv2":
            code += "print('\\n'.join(x for x in m.getBuildInformation().splitlines() if any(k in x for k in ('GStreamer', 'NVIDIA CUDA', 'Video I/O'))))\n"
        elif mod == "torch":
            code += (
                "print('compiled CUDA:', m.version.cuda)\n"
                "print('CUDA available:', m.cuda.is_available())\n"
                "if m.cuda.is_available():\n"
                "    print('GPU:', m.cuda.get_device_name(0))\n"
                "    a = m.ones((32, 32), device='cuda')\n"
                "    b = a @ a\n"
                "    m.cuda.synchronize()\n"
                "    assert b[0, 0].item() == 32.0\n"
                "    print('small CUDA matmul: PASS')\n"
            )
        elif mod == "torchvision":
            code += (
                "import torch\n"
                "boxes = torch.tensor([[0.,0.,1.,1.]])\n"
                "scores = torch.tensor([0.9])\n"
                "print('CPU NMS:', m.ops.nms(boxes, scores, 0.5).tolist())\n"
                "if torch.cuda.is_available():\n"
                "    print('CUDA NMS:', m.ops.nms(boxes.cuda(), scores.cuda(), 0.5).tolist())\n"
            )
        elif mod == "tensorrt":
            code += (
                "logger = m.Logger(m.Logger.ERROR)\n"
                "builder = m.Builder(logger)\n"
                "print('TensorRT builder:', 'created' if builder is not None else 'FAILED')\n"
            )
        elif mod == "onnxruntime":
            code += "print('providers:', m.get_available_providers())\n"
        run([sys.executable, "-B", "-c", code], timeout=40)
    heading("USB / camera nodes / serial nodes (inventory only)")
    run(["lsusb"])
    run(["lsusb", "-t"])
    run(["v4l2-ctl", "--list-devices"])
    for pattern in ("/dev/video*", "/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*"):
        print(pattern + ":", ", ".join(sorted(glob.glob(pattern))) or "(none)")
    run(["id", "-Gn"])
    print("No images were captured; no serial device was opened. Node presence is not a functional test.")
    heading("ROS installations and active environment")
    print("Installed ROS folders:", ", ".join(sorted(glob.glob("/opt/ros/*"))) or "none")
    for name in ("ROS_DISTRO", "ROS_VERSION", "RMW_IMPLEMENTATION"):
        print(name + ":", os.environ.get(name, "(not set)"))
    if shutil.which("ros2"):
        run(["ros2", "pkg", "list"], timeout=20,
            keep=lambda line: any(x in line for x in
                                  ("nav2", "moveit", "robot_localization", "cv_bridge",
                                   "image_transport", "orbbec", "rosbag2", "tf2",
                                   "ros2_control", "controller_manager")))
    else:
        print("ros2 is not in PATH: it may be absent or its setup file has not been sourced.")
    heading("Interpretation")
    print("Missing optional packages are not a global FAIL. Report matches only this Python/ROS environment.")
    print("No throughput, camera depth quality, driver protocol or closed-loop hardware test has been performed.")
    print("Before sharing the log, review local paths and device names/identifiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
