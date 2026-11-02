#!/usr/bin/env python3
"""Check the installed project stack without accessing cameras or actuators.

Run after sourcing activate_project.sh. Uses synthetic data and random model
weights; this is an installation smoke test, not model accuracy validation.
"""
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time

results = []


def check(name, function):
    start = time.monotonic()
    try:
        detail = function()
        row = dict(name=name, status="PASS", detail=detail)
    except Exception as exc:
        row = dict(name=name, status="FAIL", detail="{}: {}".format(type(exc).__name__, exc))
    row["seconds"] = round(time.monotonic() - start, 2)
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def modules():
    info = {}
    for name in ["numpy", "cv2", "scipy", "matplotlib", "serial", "yaml", "onnx",
                 "onnxruntime", "pyorbbecsdk", "tensorrt", "cuda.bindings.driver"]:
        module = importlib.import_module(name)
        info[name] = dict(version=getattr(module, "__version__", "n/a"),
                          path=getattr(module, "__file__", "n/a"))
    return info


def gpu():
    import torch
    import torchvision
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), "CUDA unavailable"
    if torch.cuda.get_device_capability(0) == (8, 7):
        assert "sm_87" in torch.cuda.get_arch_list(), "Orin requires a native sm_87 build"
    torch.cuda.set_per_process_memory_fraction(0.08)
    a = torch.ones((64, 64), device="cuda")
    assert torch.allclose(a @ a, torch.full_like(a, 64))
    with torch.no_grad():
        conv = torch.nn.Conv2d(3, 4, 3).cuda()
        y = conv(torch.zeros((1, 3, 32, 32), device="cuda"))
        assert torch.isfinite(y).all()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    kept = torchvision.ops.nms(boxes, scores, 0.5).tolist()
    assert kept == [0], kept
    cpu_layer = torch.nn.Conv2d(3, 4, 3)
    gpu_layer = torch.nn.Conv2d(3, 4, 3).cuda()
    gpu_layer.load_state_dict(cpu_layer.state_dict())
    data = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
    reference = cpu_layer(data)
    actual = gpu_layer(data.cuda())
    assert torch.allclose(actual.detach().cpu(), reference.detach(), rtol=1e-3, atol=1e-4)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu_layer.parameters())
    matrix = torch.tensor([[3., 1.], [1., 2.]], device="cuda")
    rhs = torch.tensor([9., 8.], device="cuda")
    solution = torch.linalg.solve(matrix, rhs)
    assert torch.allclose(solution, torch.tensor([2., 3.], device="cuda"), atol=1e-4)
    torch.cuda.synchronize()
    return dict(torch=torch.__version__, torchvision=torchvision.__version__,
                cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                capability=torch.cuda.get_device_capability(0),
                compiled_arches=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version(),
                matmul="PASS", conv="PASS", gpu_nms="PASS", cpu_gpu_agreement="PASS",
                backward="PASS", linear_solve="PASS",
                native_sm87_listed="sm_87" in torch.cuda.get_arch_list())


def trt():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    assert trt.Builder(logger) is not None
    return dict(version=trt.__version__, plugins="PASS", builder="PASS")


def onnx_roundtrip():
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = np.ones((1, 3), dtype=np.float32)
    output = session.run(None, {"x": data})[0]
    assert np.allclose(output, 2 * data)
    return dict(providers=ort.get_available_providers(), synthetic_inference="PASS",
                note="ONNX Runtime CPU; GPU deployment uses TensorRT/PyTorch")


def ros():
    import rclpy
    import cv_bridge
    import numpy as np
    from sensor_msgs.msg import Image, JointState
    from geometry_msgs.msg import PoseStamped
    for msg in [Image, JointState, PoseStamped]:
        msg.__class__.__import_type_support__()
        assert msg.__class__._TYPE_SUPPORT is not None
    bridge = cv_bridge.CvBridge()
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    message = bridge.cv2_to_imgmsg(data, encoding="bgr8")
    assert np.array_equal(data, bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
    return dict(rclpy=rclpy.__file__, cv_bridge=cv_bridge.__file__,
                messages="PASS", image_roundtrip="PASS", nodes_started=False)


def yolo():
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.08)
    # YAML is bundled with the installed package. No pretrained weights download.
    model = YOLO("yolo11n-seg.yaml")
    prediction = model.predict(np.zeros((160, 160, 3), dtype=np.uint8),
                               imgsz=160, device=0, verbose=False, save=False)
    torch.cuda.synchronize()
    assert len(prediction) == 1
    with torch.no_grad():
        raw = model.model.eval().to("cuda")(torch.zeros((1, 3, 160, 160), device="cuda"))
    def finite(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), "Non-finite YOLO tensor"
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
    finite(raw)
    return dict(version=importlib.metadata.version("ultralytics"),
                random_weight_segmentation_forward="PASS", size=160,
                raw_outputs_finite=True, pretrained_weights=False, accuracy_tested=False)


if __name__ == "__main__":
    checks = [("imports", modules), ("gpu", gpu), ("TensorRT", trt),
              ("ONNX", onnx_roundtrip), ("ROS", ros), ("YOLO", yolo)]
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        check(sys.argv[2], dict(checks)[sys.argv[2]])
        sys.exit(0 if results[-1]["status"] == "PASS" else 1)
    # Release native libraries and GPU memory between checks on the 8 GB board.
    for name, function in checks:
        try:
            proc = subprocess.run([sys.executable, __file__, "--only", name],
                                  capture_output=True, text=True, timeout=120)
            print(proc.stdout, end="", flush=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="", flush=True)
            rows = []
            for line in proc.stdout.splitlines():
                try:
                    row = json.loads(line)
                    if row.get("name") == name:
                        rows.append(row)
                except (ValueError, AttributeError):
                    pass
            if rows and proc.returncode == 0:
                results.append(rows[-1])
            else:
                results.append(dict(name=name, status="FAIL", detail="exit {}".format(proc.returncode)))
        except subprocess.TimeoutExpired:
            row = dict(name=name, status="FAIL", detail="120s timeout")
            results.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps(dict(summary={"passed": sum(x["status"] == "PASS" for x in results),
                                   "failed": sum(x["status"] == "FAIL" for x in results)})))
    sys.exit(1 if any(x["status"] == "FAIL" for x in results) else 0)
