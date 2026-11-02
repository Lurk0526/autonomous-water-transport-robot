#!/usr/bin/env python3
"""Expose pip-installed CUDA runtime libraries when activating a Bash venv.

Changes only the generated activation script. deactivate restores the previous
LD_LIBRARY_PATH, including its original unset state. No global linker changes.
"""
import argparse
from pathlib import Path

MARKER = "# Jetson project CUDA runtime activation"
RESTORE = '''    # Restore the linker path saved by this project's runtime hook.
    if [ "${_JETSON_CUDA_LD_ACTIVE:-}" = 1 ]; then
        if [ "${_JETSON_CUDA_LD_WAS_SET:-}" = x ]; then
            export LD_LIBRARY_PATH="${_JETSON_CUDA_LD_OLD-}"
        else
            unset LD_LIBRARY_PATH
        fi
        unset _JETSON_CUDA_LD_ACTIVE _JETSON_CUDA_LD_WAS_SET _JETSON_CUDA_LD_OLD
    fi
'''
ACTIVATE = '''
# Jetson project CUDA runtime activation
_JETSON_CUDA_LD_OLD="${LD_LIBRARY_PATH-}"
_JETSON_CUDA_LD_WAS_SET="${LD_LIBRARY_PATH+x}"
_JETSON_CUDA_LD_ACTIVE=1
for _jetson_runtime_dir in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$_jetson_runtime_dir" ]; then
        case ":${LD_LIBRARY_PATH-}:" in
            *":${_jetson_runtime_dir}:"*) ;;
            *) LD_LIBRARY_PATH="${_jetson_runtime_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
export LD_LIBRARY_PATH
unset _jetson_runtime_dir
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args()
    activate = args.venv.resolve() / "bin" / "activate"
    text = activate.read_text()
    if MARKER in text:
        print("Runtime activation already configured:", activate)
        return
    anchor = "deactivate () {\n"
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected venv activation script; no changes made")
    text = text.replace(anchor, anchor + RESTORE, 1)
    activate.write_text(text + ACTIVATE)
    print("Configured scoped CUDA runtime activation:", activate)


if __name__ == "__main__":
    main()
