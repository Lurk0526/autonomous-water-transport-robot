#!/usr/bin/env python3
"""Run a tiny TensorRT GPU engine on PyTorch-owned input/output buffers.

Checks the project's CUDA runtime interoperability using synthetic data.
Does not access cameras, serial ports, or robot controllers.
"""
import json
import numpy as np
import torch
import tensorrt as trt


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.08)
    assert "sm_87" in torch.cuda.get_arch_list()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * 1024 * 1024)
    config.builder_optimization_level = 0
    source = network.add_input("input", trt.float32, (1, 4))
    one = network.add_constant((1, 4), np.ones((1, 4), dtype=np.float32))
    result = network.add_elementwise(source, one.get_output(0), trt.ElementWiseOperation.SUM)
    target = result.get_output(0)
    target.name = "output"
    network.mark_output(target)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "TensorRT engine build failed"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(4, dtype=torch.float32, device="cuda").reshape(1, 4)
        y = torch.full_like(x, float("nan"))
        assert context.set_tensor_address("input", x.data_ptr())
        assert context.set_tensor_address("output", y.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert torch.allclose(y, x + 1)
    print(json.dumps(dict(status="PASS", torch=torch.__version__,
                          torch_cuda=torch.version.cuda, native_arches=torch.cuda.get_arch_list(),
                          tensorrt=trt.__version__, engine_build=True,
                          gpu_inference=True, output=y.cpu().tolist())))


if __name__ == "__main__":
    main()
