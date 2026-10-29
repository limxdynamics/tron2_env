<div align="center">

<img src="https://raw.githubusercontent.com/limxdynamics/tron2_env/main/docs/images/banner.png" alt="tron2_env — runtime for LimX TRON2 manipulation" width="100%">

**English** | [中文](https://github.com/limxdynamics/tron2_env/blob/main/README_CN.md)

[![PyPI](https://img.shields.io/pypi/v/tron2-env?color=0A7BBB)](https://pypi.org/project/tron2-env/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/tron2-env/)
[![CI](https://github.com/limxdynamics/tron2_env/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/limxdynamics/tron2_env/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/limxdynamics/tron2_env/blob/main/LICENSE)

</div>

`tron2_env` is the public TRON2 runtime package used by the TRON2 OpenPI
deployment example. It provides robot communication, motion execution,
observation collection, and RTC helper utilities for real-robot deployment.

Only the WebSocket robot transport is exposed; the low-level path is not part of
this release.

## Highlights

- **Gym-style runtime.** `Tron2Env` wraps the robot behind `reset()` / `step()` so a policy client only deals in observation and action vectors.
- **Smooth, high-rate motion.** `MotionController` interpolates each policy target and publishes commands from a background thread (300 Hz by default), so policy FPS and control rate stay decoupled.
- **Modular hardware layouts.** ServoJ or ServoP arms, grippers or BrainCo2 hands, head, chassis and lifter are composed into explicit 16D to 31D state/action vectors.
- **Multi-camera observation.** Images and state come from TRON2 Bridge, or from locally attached RealSense cameras.
- **Real-Time Chunking helpers.** `ActionQueue` and `LatencyTracker` for smooth chunked-action execution.
- **Testable without a robot.** A mock-transport Quick Start and a software-only test suite run anywhere.

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/limxdynamics/tron2_env/main/docs/images/architecture.png" alt="tron2_env architecture: policy client calls reset/step on Tron2Env, which drives MotionController and WebsocketTransport towards the TRON2 robot, and reads observations back from TRON2 Bridge" width="100%">
</p>

`Tron2Env` reads observations through `BridgeObservationProvider` (TRON2 Bridge)
or `MultiCameraManager` (local RealSense, legacy), and sends each policy target
through `MotionController` to `WebsocketTransport` as WebSocket JSON. Each target
is interpolated over `eta = 1/fps` on a background thread, so the policy can step
at any rate while the arm still receives commands at `publish_rate`.

<p align="center">
  <img src="https://raw.githubusercontent.com/limxdynamics/tron2_env/main/docs/images/timing.png" alt="Timing: each policy target is interpolated over eta = 1/fps while a background thread publishes at 300 Hz" width="100%">
</p>

## Installation

From PyPI:

```bash
python -m pip install "tron2-env[bridge,openpi]"
```

Or from source as an editable package:

```bash
git clone https://github.com/limxdynamics/tron2_env.git
cd tron2_env
python -m pip install -e ".[bridge,openpi]"
```

Optional extras:

| Extra | Purpose |
| --- | --- |
| `bridge` | WebSocket bridge observation support. |
| `camera` | Local RealSense camera support through `pyrealsense2`. |
| `openpi` | Image formatting helper dependency used by the optional policy wrapper. |
| `replay` | Parquet and YAML dependencies for `examples/replay_data.py`. |
| `dev` | Test runner dependencies. |
| `all` | Bridge, camera, OpenPI helper, and replay dependencies. |

## No-Hardware Quick Start

Install the development dependencies and run the mock transport example:

```bash
python -m pip install -e ".[dev]"
python examples/mock_quickstart.py
python -m pytest -q
```

The example uses an in-memory transport. It does not open a network connection,
discover hardware, or send commands to a real robot.

## Scope

The public package covers the runtime layer of a deployment: `Tron2Env` and
`Tron2Config` for the robot loop, `WebsocketTransport` and `MotionController` for
command delivery, `BridgeObservationProvider` and the legacy `MultiCameraManager`
for observation, explicit layout objects for arms, hands, head and chassis, and
`ActionQueue` / `LatencyTracker` plus joint index helpers for RTC.

Not included: low-level robot transport, private robot safety controllers, robot
calibration or customer profiles, and model weights, datasets or training code.

## Using with tron2_openpi

For policy deployment, keep this package next to
[`tron2_openpi`](https://github.com/limxdynamics/tron2_openpi):

```text
parent-directory/
├── tron2_openpi/
└── tron2_env/
```

`tron2_openpi/examples/tron2/pi_client.py` adds `../tron2_env/src` to
`sys.path`, so it can import `tron2_env` directly during local development.

Clone `tron2_openpi` as a sibling, sync it and check the import resolves back to
this checkout:

```bash
cd ..
git clone https://github.com/limxdynamics/tron2_openpi.git
cd tron2_openpi
uv sync
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run python -c "import tron2_env; print(tron2_env.__file__)"
```

The printed path should point to `../tron2_env/src/tron2_env/__init__.py`.

## Real-Robot Usage

The following example connects to the configured robot and may send commands.
Only run it in an authorised environment with the normal robot bring-up and
emergency-stop procedures in place.

```python
from tron2_env import Tron2Config, create_motion_controller

config = Tron2Config(
    robot_ip="ROBOT_IP",
    port=5000,
    init_joints=None,
    init_head=None,
)

controller = create_motion_controller(
    config,
    backend="websocket",
    publish_rate=300.0,
)

try:
    state = controller.get_joint_states(timeout=1.0)
    print(state)
finally:
    controller.disconnect()
```

For policy deployment, use `tron2_openpi/examples/tron2/pi_client.py`; it wires
the policy client, `Tron2Env`, observations, and action playback together.

## Replay Recorded Data

`examples/replay_data.py` replays a recorded parquet trajectory directly through
`tron2_env`. It uses the deployment YAML from the sibling `tron2_openpi`
repository for robot IP, initialization pose, backend, and playback rate.

Always dry-run first, and drop `--dry-run` only after confirming the file, replay
range, robot address, initial pose and workspace safety:

```bash
python -m pip install -e ".[replay]"
python examples/replay_data.py \
  --file /path/to/trajectory.parquet \
  --deploy-config ../tron2_openpi/configs/deploy/tron2_deploy.local.yaml \
  --data-key action \
  --dry-run
```

## Configuration Concepts

`Tron2Config` contains robot-level settings:

| Field | Description |
| --- | --- |
| `robot_ip` | TRON2 robot WebSocket controller address. |
| `port` | Robot controller WebSocket port. |
| `init_joints` | Optional 14-value arm initialization pose. |
| `init_head` | Optional 2-value head initialization pose. |
| `init_ee_z_min` | Minimum end-effector Z height before initialization routes through the intermediate joint pose; set to `None` to disable this check. |
| `state_queue_maxlen` | Robot state feedback queue length. |
| `polling_rate` | Robot state polling rate. |
| `connection_timeout` | WebSocket connection timeout. |

`EnvConfig` is used by `Tron2Env` and the TRON2 client to select observation
mode, action playback rate, debug recording, bridge settings, and camera
settings. The public deployment templates in `tron2_openpi/configs/deploy/`
show the recommended values.

### Modular Physical Layouts

`RobotModules` selects one arm mode, one paired end-effector type, the head and
the mobile base; `PhysicalLayout` flattens them into the policy state/action
vector in a fixed order.

<p align="center">
  <img src="https://raw.githubusercontent.com/limxdynamics/tron2_env/main/docs/images/state-action-layout.png" alt="State and action vector layout: 7 arm + 1 gripper + 7 arm + 1 gripper = 16; six-value BrainCo2 hands bring it to 26; head and chassis add 2 and 3 to reach 31" width="100%">
</p>

| Module combination | Physical dimension |
| --- | ---: |
| ServoJ/ServoP + paired grippers | 16 |
| ServoJ/ServoP + paired grippers + head | 18 |
| ServoJ/ServoP + paired grippers + chassis/lifter | 19 |
| ServoJ/ServoP + paired grippers + head + chassis/lifter | 21 |
| ServoJ/ServoP + paired BrainCo2 hands | 26 |
| ServoJ/ServoP + paired BrainCo2 hands + head | 28 |
| ServoJ/ServoP + paired BrainCo2 hands + chassis/lifter | 29 |
| ServoJ/ServoP + paired BrainCo2 hands + head + chassis/lifter | 31 |

Legacy configs that specify only `state_dim: 16` or `18` remain supported.
Other dimensions require `RobotModules` or an explicit `state_layout` so the
runtime does not guess physical semantics from vector width. When
`RobotModules` is used, any supplied `state_dim` and `state_layout` must exactly
match the derived layout.

Modular real-robot configurations also have these constraints:

- ServoP represents each arm as `[x, y, z, qw, qx, qy, qz]`; quaternion values
  must not be linearly interpolated or blended.
- BrainCo2 requires a valid `BrainCo2Config.command_time`, and both arms use
  the same end-effector type.
- When Bridge supplies images, ServoP and BrainCo2 physical state must come
  from the robot WebSocket.
- Lifter control requires millimetre position state; raw joint values are
  observation-only.
- The public package supports the WebSocket control backend only.

See `configs/deploy/tron2_deploy.client.example.yaml` in the sibling
`tron2_openpi` repository for a complete deployment YAML example.

## Observation Modes

Bridge mode:

- Reads images from TRON2 Bridge.
- Can read bridge-aligned joint state.
- Does not require camera USB cables to be attached to the host computer.

Legacy RealSense mode:

- Reads images from locally attached RealSense cameras.
- Reads robot state from the WebSocket robot transport.
- Requires camera serial numbers to be mapped to policy camera names.

The policy image names are `cam_high`, `cam_left_wrist` and `cam_right_wrist`.

## Safety Notes

- Confirm bring-up poses before commanding a real robot.
- Start motion tests at low speed and with short action chunks.
- Keep emergency stop access available while using real-robot clients.
- Treat local configuration files as private deployment assets.
- This package exposes runtime plumbing; it does not replace robot-level safety
  procedures or private low-level safety systems.
- The public CI and `examples/mock_quickstart.py` cover software-only paths. They
  do not validate robot safety, calibration, Bridge, RealSense, or real-hardware
  behaviour.

## Network Security Boundary

- All runtime network interfaces—the robot-control WebSocket, TRON2 Bridge
  observation WebSockets, and OpenPI policy WebSocket—are supported only on a
  controlled robot LAN restricted to authorised systems.
- Do not expose these interfaces directly to the Internet or to an untrusted or
  shared network. Public port forwarding, cross-site links, and cloud deployment
  require a separate security review and appropriate transport protection.
- The current robot-control interface uses `ws://` and does not establish
  transport encryption or client authentication. Bridge deployments may also be
  configured to skip server-certificate verification. Network segmentation,
  firewalling, and access control are therefore part of the required deployment
  trust boundary.
- Treat camera images, robot state and metadata, and policy actions as protected
  runtime data. Do not capture, forward, or publish them without the applicable
  data and deployment approval.

## Development Checks

Beyond the Quick Start, CI also runs:

```bash
python -m compileall -q src tests examples
python -m build
```

See `.github/workflows/ci.yml` for the exact job, `CONTRIBUTING.md` for
contribution boundaries and `SECURITY.md` for private vulnerability reporting.

## Troubleshooting

- If imports fail, make sure `tron2_env` and `tron2_openpi` are siblings and
  that `PYTHONPATH` includes the parent directory.
- If WebSocket connection fails, verify `robot_ip`, `port`, network reachability,
  and robot controller state.
- If bridge observations time out, verify the Bridge host, WebSocket path, TLS
  setting, and topic availability.
- If legacy cameras are missing, verify RealSense permissions and serial number
  mappings.
- If actions look discontinuous, check policy FPS, publish rate, RTC settings,
  and action normalization stats in the policy checkpoint.

## Related Repositories

| Repository | Description |
| --- | --- |
| [tron2_openpi](https://github.com/limxdynamics/tron2_openpi) | OpenPI derivative for TRON2: pi0 / pi0.5 policy serving, TRON2 transforms, deployment configs and robot clients. |
| [tron2-robot-description](https://github.com/limxdynamics/tron2-robot-description) | URDF / Xacro, MuJoCo and USD models for TRON2 variants. |

## Third-Party Origins

Most of this package is LimX-developed TRON2 runtime code. The RTC
`ActionQueue` is modified vendored source derived from the LeRobot
implementation at commit `ca87ccd9413c59c30f524967222d2e3f1b7bb549`. It was
ported from PyTorch to NumPy and adapted for TRON2. See `NOTICE` for details.

## License

Unless otherwise noted, this package is distributed under the Apache License
2.0 in `LICENSE`. Notices for vendored third-party source are listed in
`NOTICE`; licenses for external direct dependencies are inventoried in
`THIRD_PARTY_DEPENDENCIES.md`.
