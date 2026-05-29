# Development Environment: Verified Setup & Toolchain

This document records the exact, verified development environment setup on macOS for the EdgeScoot embedded ML pipeline in `scooter-imu-analytics`.

## 1. Host Machine Specifications

- **Operating System:** macOS (Version 27.0.1, Build 26A434)
- **Host Architecture:** Apple Silicon (`arm64`)
- **Available Disk Space:** ~23 GiB on `/System/Volumes/Data` (measured via `df -h ~`)
- **Xcode Command Line Tools:** `/Library/Developer/CommandLineTools`
- **Compiler:** Apple Clang version 21.0.0 (`clang-2100.3.34.2`, target `arm64-apple-darwin27.0.0`)
- **CMake:** Missing / Not installed initially (required for Stage 6; installable via `brew install cmake`)

## 2. Python Package & Environment Manager (`uv`)

All Python dependency and environment management in this repository is strictly governed by `uv`.

- **Installed Version:** `uv 0.8.22 (ade2bdbd2 2025-09-23)`
- **Binary Location:** `/Users/pulkit/.local/bin/uv` (installed via pipx)
- **Python Version Policy:** Python 3.11 (pinned in `.python-version` as `3.11`, utilizing `cpython-3.11.13-macos-aarch64-none`)
- **Lockfile Policy:** `uv.lock` is version-controlled and acts as the reproducible single source of truth for resolved dependencies.

### Verified Setup & Execution Commands

```bash
# 1. Verify uv installation and runtime
uv --version

# 2. Sync root environment with locked dependencies (including dev and ml optional groups)
uv sync --locked --extra dev --extra ml

# 3. Verify Python version managed by uv
uv run --locked python --version

# 4. Run full existing test suite
uv run --locked pytest
```

## 3. Dependency Management Rules

1. **No pip/Conda:** Do not run `pip install`, `python -m pip install`, or manually manage virtual environments.
2. **Adding Dependencies:** Use `uv add` for project dependencies and `uv add --dev` (or `--extra <group>`) for development/optional tooling.
3. **Lockfile Updates:** Whenever dependencies are updated in `pyproject.toml`, run `uv lock`, review the diff, and commit `uv.lock`.
4. **Isolated Experiments:** GPU-only / TensorRT dependencies must never be added to the root Mac project environment; they belong in an isolated `experiments/tensorrt/` workspace.

