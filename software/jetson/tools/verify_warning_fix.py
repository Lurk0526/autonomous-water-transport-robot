#!/usr/bin/env python3
"""Fail if the project's dependency or Orin architecture warnings return."""
import json
import subprocess
import sys
import warnings


def main():
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    assert result.returncode == 0, "pip dependency check failed"
    assert "not supported on this platform" not in output.lower(), "Unsupported wheel platform tag"
    # Turn user warnings into errors; never suppress the compatibility warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        import torch
        import torchvision
        torch.cuda.init()
        assert torch.cuda.get_device_capability(0) == (8, 7)
        assert "sm_87" in torch.cuda.get_arch_list()
        image = torch.zeros((3, 8, 8), dtype=torch.uint8)
        encoded = torchvision.io.encode_jpeg(image)
        decoded = torchvision.io.decode_jpeg(encoded)
        assert decoded.shape == image.shape and torch.equal(decoded, image)
    print(json.dumps(dict(status="PASS", torch=torch.__version__,
                          torchvision=torchvision.__version__, cuda_build=torch.version.cuda,
                          native_arches=torch.cuda.get_arch_list(),
                          dependency_check="PASS", jpeg_roundtrip="PASS",
                          user_warnings_as_errors=True)))


if __name__ == "__main__":
    main()
