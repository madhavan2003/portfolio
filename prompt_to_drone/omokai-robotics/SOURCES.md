# Sources

Every external repository, library, snippet, or reference used in this project.

## Libraries (via pip)

| Library | Version | License | Purpose | URL |
|---|---|---|---|---|
| pydantic | 2.9.2 | MIT | Mission schema definition, field validation, JSON serialisation. Bumped from an earlier 2.7.4 pin after a real conflict surfaced: `ollama==0.6.2` requires `pydantic>=2.9`. | https://github.com/pydantic/pydantic |
| pydantic-settings | 2.3.4 | MIT | Environment variable loading | https://github.com/pydantic/pydantic-settings |
| ollama | 0.6.2 | MIT | Client for a locally-running Ollama server (no API key/cost) | https://github.com/ollama/ollama-python |
| mavsdk | 3.10.2 | BSD-3-Clause | Python wrapper for MAVLink/MAVSDK; drone control API | https://github.com/mavlink/MAVSDK-Python |
| click | 8.1.7 | BSD-3-Clause | CLI argument parsing | https://github.com/pallets/click |
| rich | 13.7.1 | MIT | Terminal pretty-printing (panels, syntax highlighting) | https://github.com/Textualize/rich |
| pytest | 8.2.2 | MIT | Test runner | https://github.com/pytest-dev/pytest |
| pytest-asyncio | 0.23.7 | Apache-2.0 | Async test support | https://github.com/pytest-dev/pytest-asyncio |
| pytest-mock | 3.14.0 | MIT | Mock helpers for pytest | https://github.com/pytest-dev/pytest-mock |

## Autopilot / Simulator

Both the native setup and the Docker image run **Gazebo Harmonic** (gz-sim
8.14.0) against PX4's `gz_x500` target — they are the same simulator
generation, not two different ones. The difference between them is
reproducibility, not the underlying engine: native was built directly on
the host with whatever PX4/Gazebo versions `Tools/setup/ubuntu.sh` pulled
at the time, with nothing pinned; Docker pins one exact PX4 commit and one
exact Gazebo package version so the build is reproducible regardless of
when or where it's run. See ARCHITECTURE.md for the full reasoning.

| Component | Version | License | What was taken | URL |
|---|---|---|---|---|
| PX4-Autopilot (native) | cloned via `git clone --recursive`, no commit pinned | BSD-3-Clause | SITL binary, `gz_x500` bridge, MAVLink UDP interface. Not modified. | https://github.com/PX4/PX4-Autopilot |
| Gazebo Harmonic (native) | gz-sim 8.14.0 | Apache-2.0 | Simulation physics and rendering. Installed via `Tools/setup/ubuntu.sh`. Not modified. | https://github.com/gazebosim/gz-sim |
| PX4-Autopilot (Docker) | pinned commit `e0137fe7a7a1be179db7d04b8703ef77c71a656e` | BSD-3-Clause | Full source tree, built from scratch in `docker/Dockerfile.px4` for the `gz_x500` SITL target. Not modified. | https://github.com/PX4/PX4-Autopilot |
| Gazebo Harmonic (Docker) | 1.0.0-1~jammy (gz-sim 8.14.0) | Apache-2.0 | Installed via OSRF's apt repo inside `docker/Dockerfile.px4`, pinned to an exact package version. Not modified. | https://github.com/gazebosim/gz-sim |

## Reference Documentation

| Source | URL | What was referenced |
|---|---|---|
| PX4 SITL + Gazebo (gz) docs | https://docs.px4.io/main/en/sim_gazebo_gz/ | `gz_x500` target, `Tools/setup/ubuntu.sh`, headless env var — used for both native and Docker setups |
| Ollama install/releases | https://github.com/ollama/ollama/releases | `install.sh`, and the redistributable `linux-amd64.tar.zst` tarball used for the no-root install path |
| MAVSDK-Python docs | https://mavsdk.mavlink.io/main/en/python/ | `System`, `Action`, `Telemetry` API usage |
| MAVSDK-Python examples | https://github.com/mavlink/MAVSDK-Python/tree/main/examples | `takeoff_and_land.py` referenced for arm/takeoff/land pattern |
| Pydantic v2 docs — discriminated unions | https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions | Used for `MissionAction` union type |
| Pydantic v2 docs — model_validator | https://docs.pydantic.dev/latest/concepts/validators/#model-validators | Used for cross-field safety checks in `Mission` |
| Ollama API docs — /api/chat, structured outputs | https://github.com/ollama/ollama/blob/main/docs/api.md | `format="json"` + retry-loop pattern in `_plan_with_ollama()` |
| Haversine formula | https://en.wikipedia.org/wiki/Haversine_formula | `_haversine_m()` in executor.py |
| asyncio.wait_for (Python 3.10) | https://docs.python.org/3.10/library/asyncio-task.html#asyncio.wait_for | Used instead of asyncio.timeout (Python 3.11+) |

## Code Patterns Adapted

| Pattern | Source | Where adapted | Nature of adaptation |
|---|---|---|---|
| Pydantic discriminated union for heterogeneous action list | Pydantic v2 docs | `src/schemas/mission.py` — `MissionAction` type alias | Applied standard pattern to the specific action type set |
| MAVSDK arm→takeoff→goto→RTL sequence | MAVSDK-Python `takeoff_and_land.py` example | `src/executor/executor.py` | Extended with loop resolution, logging, and per-waypoint arrival polling |
| `asyncio.wait_for` instead of `asyncio.timeout` | Python 3.10 stdlib docs | `src/executor/executor.py` — `_exec_takeoff()` | Compatibility fix: `asyncio.timeout` is Python 3.11+; replaced with the 3.10-compatible `wait_for` |

## Use of AI Coding Assistants

This codebase was built with the help of AI coding assistants — Antigravity
in the early planning and scaffolding stage, then Claude Code for the
majority of implementation, debugging, and the Docker setup. This is
disclosed here rather than hidden: the task explicitly allows AI assistance
provided the author understands and can explain and modify the resulting
code, which I can. No AI tool was given access to fly a real vehicle or any
network beyond this local development environment, and every design
decision — the module boundary between planner/validator/executor, the
validation rules, the retry-loop approach for the local LLM's JSON
reliability — was reviewed and is understood well enough to defend and
modify live.
