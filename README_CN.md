<div align="center">

<img src="https://raw.githubusercontent.com/limxdynamics/tron2_env/main/docs/images/banner.png" alt="tron2_env — LimX TRON2 操作任务运行时" width="100%">

[English](https://github.com/limxdynamics/tron2_env/blob/main/README.md) | **中文**

[![PyPI](https://img.shields.io/pypi/v/tron2-env?color=0A7BBB)](https://pypi.org/project/tron2-env/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/tron2-env/)
[![CI](https://github.com/limxdynamics/tron2_env/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/limxdynamics/tron2_env/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/limxdynamics/tron2_env/blob/main/LICENSE)

</div>

`tron2_env` 是 TRON2 OpenPI 部署示例使用的公开 TRON2 运行时包，负责机器人通信、
运动执行、观测采集和 RTC 辅助逻辑。

当前公开版本只提供 WebSocket 机器人 transport。尚未开发完成的 low-level
transport 路径已主动排除，不在本次开源范围内。

## 亮点

- **Gym 风格运行时。** `Tron2Env` 通过 `reset()` / `step()` 封装机器人，policy 客户端只需处理观测和动作向量。
- **平滑的高频运动。** `MotionController` 对每个 policy 目标做插值，并由后台线程发布指令（默认 300 Hz），policy 帧率与控制频率解耦。
- **模块化硬件布局。** ServoJ 或 ServoP 双臂、夹爪或 BrainCo2 灵巧手、头部、底盘和升降台，组合为显式的 16 至 31 维 state/action 向量。
- **多相机观测。** 图像和状态可来自 TRON2 Bridge，也可来自本机直连的 RealSense 相机。
- **Real-Time Chunking 辅助工具。** `ActionQueue` 和 `LatencyTracker` 用于平滑执行分块动作。
- **无需真机即可测试。** 提供 mock transport Quick Start 和纯软件测试集，任何环境都能运行。

## 架构

```text
  OpenPI policy server  (tron2_openpi)
          │  WebSocket
          ▼
  Policy client  (tron2_openpi/examples/tron2/pi_client.py)
  RTC helpers: ActionQueue, LatencyTracker
          │  reset() / step(action)               ▲  observation
          ▼                                       │
  ┌─────────────────────────── tron2_env ─────────┴────────────────────────────┐
  │  Tron2Env ──► MotionController ──► WebsocketTransport ──► TRON2 robot      │
  │               interpolation +      WebSocket JSON         controller       │
  │               publish thread                                               │
  │                                                                            │
  │  Tron2Env ◄── BridgeObservationProvider ◄── TRON2 Bridge                   │
  │           ◄── MultiCameraManager        ◄── local RealSense (legacy)       │
  └────────────────────────────────────────────────────────────────────────────┘
```

## 安装

从 PyPI 安装：

```bash
python -m pip install "tron2-env[bridge,openpi]"
```

或从源码以 editable 方式安装：

```bash
git clone https://github.com/limxdynamics/tron2_env.git
cd tron2_env
python -m pip install -e ".[bridge,openpi]"
```

可选 extras：
| Extra | 用途 |
| --- | --- |
| `bridge` | Bridge WebSocket 观测支持。 |
| `camera` | 通过 `pyrealsense2` 使用本机 RealSense 相机。 |
| `openpi` | 可选 policy wrapper 使用的图像格式化 helper 依赖。 |
| `replay` | `examples/replay_data.py` 使用的 parquet 和 YAML 依赖。 |
| `dev` | 测试运行依赖。 |
| `all` | Bridge、camera、OpenPI helper 和 replay 依赖。 |

## 无硬件 Quick Start

安装开发依赖并运行 mock transport 示例：

```bash
python -m pip install -e ".[dev]"
python examples/mock_quickstart.py
python -m pytest -q
```

该示例只使用内存 transport，不会建立网络连接、发现硬件或向真实机器人发送指令。

## 本包包含什么

- `Tron2Config`：机器人连接和初始化配置。
- `WebsocketTransport`：用于 TRON2 机器人状态和命令的 WebSocket JSON transport。
- `MotionController`：插值和高频指令发布。
- `Tron2Env`：面向 policy 执行的 gym-style 运行时封装。
- `BridgeObservationProvider`：通过 TRON2 Bridge 获取图像和状态。
- `MultiCameraManager`：legacy 本机 RealSense 相机观测。
- 显式物理布局：支持 ServoJ/ServoP、夹爪/BrainCo2 双灵巧手、头部、底盘和升降台。
- `ActionQueue` 和 `LatencyTracker`：RTC 辅助工具。
- `examples/replay_data.py`：通过 runtime controller 回放录制动作。
- 关节索引工具和包级异常类型。

## 本包不包含什么

- low-level 机器人 transport。
- 私有机器人安全控制器。
- 机器人标定、客户配置或本地网络设置。
- 模型权重、数据集或 policy 训练代码。

## 配合 tron2_openpi 使用

进行 policy 部署时，请将本包和 [`tron2_openpi`](https://github.com/limxdynamics/tron2_openpi) 放在同级：

```text
同级目录/
├── tron2_openpi/
└── tron2_env/
    ├── examples/
    ├── pyproject.toml
    ├── src/
    │   └── tron2_env/
    └── tests/
```

`tron2_openpi/examples/tron2/pi_client.py` 会把 `../tron2_env/src` 加入
`sys.path`，因此本地开发时可以直接导入 `tron2_env`。

请单独 clone 并安装/sync 同级的 `tron2_openpi`：

```bash
cd ..
git clone https://github.com/limxdynamics/tron2_openpi.git
cd tron2_openpi
uv sync
```

导入检查，从 `tron2_openpi/` 中运行：

```bash
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run python -c "import tron2_env; print(tron2_env.__file__)"
```

输出路径应该指向 `../tron2_env/src/tron2_env/__init__.py`。

## 真机用法

以下示例会连接配置的机器人，并可能发送指令。只能在已授权环境中运行，并应遵循
正常的机器人初始化和急停流程。

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

实际 policy 部署建议直接使用 `tron2_openpi/examples/tron2/pi_client.py`，它已经把
policy client、`Tron2Env`、观测和 action 播放串联好。

## 回放录制数据

`examples/replay_data.py` 会通过 `tron2_env` 直接回放 parquet 轨迹。它使用同级
`tron2_openpi` 仓库中的部署 YAML 读取机器人 IP、初始化姿态、backend 和播放频率。

必须先 dry-run：

```bash
python -m pip install -e ".[replay]"
python examples/replay_data.py \
  --file /path/to/trajectory.parquet \
  --deploy-config ../tron2_openpi/configs/deploy/tron2_deploy.local.yaml \
  --data-key action \
  --dry-run
```

只有在确认文件、回放范围、机器人地址、初始化姿态和工作空间安全后，才移除
`--dry-run`。

## 配置概念

`Tron2Config` 包含机器人级设置：

| 字段 | 说明 |
| --- | --- |
| `robot_ip` | TRON2 机器人 WebSocket 控制器地址。 |
| `port` | 机器人控制器 WebSocket 端口。 |
| `init_joints` | 可选 14 维双臂初始化姿态。 |
| `init_head` | 可选 2 维头部初始化姿态。 |
| `init_ee_z_min` | 初始化前的末端 Z 高度下限；低于该值时先经过中间关节姿态，设为 `None` 可禁用此检查。 |
| `state_queue_maxlen` | 机器人状态反馈队列长度。 |
| `polling_rate` | 机器人状态轮询频率。 |
| `connection_timeout` | WebSocket 连接超时时间。 |

`EnvConfig` 由 `Tron2Env` 和 TRON2 客户端使用，用于选择观测模式、action 播放频率、
调试记录、Bridge 设置和相机设置。推荐值可以参考
`tron2_openpi/configs/deploy/` 中的公开部署模板。

### 模块化物理布局

`RobotModules` 统一选择双臂模式、双侧末端执行器、头部和移动底盘；
`PhysicalLayout` 按固定顺序将模块展开为 policy state/action 向量。ServoJ 与 ServoP
每条手臂都占 7 维，夹爪每侧占 1 维，BrainCo2 每侧占 6 维，头部占 2 维，
底盘/升降台占 3 维。

| 模块组合 | 物理维度 |
| --- | ---: |
| ServoJ/ServoP + 双夹爪 | 16 |
| ServoJ/ServoP + 双夹爪 + 头部 | 18 |
| ServoJ/ServoP + 双夹爪 + 底盘/升降台 | 19 |
| ServoJ/ServoP + 双夹爪 + 头部 + 底盘/升降台 | 21 |
| ServoJ/ServoP + BrainCo2 双灵巧手 | 26 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 头部 | 28 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 底盘/升降台 | 29 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 头部 + 底盘/升降台 | 31 |

兼容旧配置时，只写 `state_dim: 16` 或 `18` 仍可解析；其他维度必须通过
`RobotModules` 或显式 `state_layout` 给出物理语义。使用 `RobotModules` 时，
`state_dim` 和 `state_layout` 如有填写，必须与推导结果完全一致。

模块化真机配置还需遵守以下限制：

- ServoP 每臂表示 `[x, y, z, qw, qx, qy, qz]`，四元数不能线性插值或混合。
- BrainCo2 必须提供有效的 `BrainCo2Config.command_time`，双手使用同一种末端执行器。
- Bridge 提供图像时，ServoP 和 BrainCo2 的物理状态必须从机器人 WebSocket 读取。
- 启用升降台控制时必须使用毫米位置状态；原始关节量仅支持观测。
- 公开包只支持 WebSocket 控制后端。

部署 YAML 的完整示例见同级 `tron2_openpi` 仓库中的
`configs/deploy/tron2_deploy.client.example_CN.yaml`。

## 观测模式

Bridge 模式：

- 从 TRON2 Bridge 读取图像。
- 可以读取 Bridge 对齐后的关节状态。
- 不需要将相机 USB 线连接到主机电脑。

Legacy RealSense 模式：

- 从本机直连 RealSense 相机读取图像。
- 从 WebSocket 机器人 transport 读取状态。
- 需要把相机序列号映射到 policy 期望的相机名。

policy 图像名为：

- `cam_high`
- `cam_left_wrist`
- `cam_right_wrist`

## 安全注意事项

- 指令真实机器人前先确认初始化姿态安全可达。
- 首次运动测试使用较低速度和较短 action chunk。
- 真机运行时保持急停可用。
- 本地配置文件属于私有部署资产，不应提交。
- 本包提供运行时管线，不替代机器人级安全流程或私有 low-level 安全系统。
- 公共 CI 和 `examples/mock_quickstart.py` 只覆盖纯软件路径，不验证机器人安全、
  标定、Bridge、RealSense 或真机行为。

## 网络安全边界

- 当前全部运行时网络接口——机器人控制 WebSocket、TRON2 Bridge 观测
  WebSocket 和 OpenPI 策略 WebSocket——仅支持在只允许授权系统接入的受控机器人
  局域网内使用。
- 不要将这些接口直接暴露到互联网或不受信任、共享网络。公网端口映射、跨站点连接
  和云端部署需要单独进行安全评审，并配置相应的传输保护。
- 当前机器人控制接口使用 `ws://`，自身不提供传输加密或客户端认证；Bridge 部署
  也可能配置为跳过服务器证书验证。因此，网络分段、防火墙和访问控制属于部署所需
  的信任边界。
- 相机图像、机器人状态及 metadata、策略动作均应作为受保护的运行数据；未经相应
  数据和部署审批，不要采集、转发或公开。

## 开发检查

运行与 CI 相同的纯软件检查：

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests examples
python -m pytest -q
python examples/mock_quickstart.py
python -m build
```

贡献边界见 `CONTRIBUTING.md`，安全漏洞的私密报告方式见 `SECURITY.md`。

## 常见问题

- 如果导入失败，确认 `tron2_env` 和 `tron2_openpi` 是同级目录，并且 `PYTHONPATH`
  包含父目录。
- 如果 WebSocket 连接失败，检查 `robot_ip`、`port`、网络可达性和机器人控制器状态。
- 如果 Bridge 观测超时，检查 Bridge 地址、WebSocket path、TLS 设置和 topic 状态。
- 如果 legacy 相机缺失，检查 RealSense 权限和相机序列号映射。
- 如果动作不连续，检查 policy FPS、publish rate、RTC 设置和 checkpoint 中的 action
  归一化统计。

## 相关仓库

| 仓库 | 说明 |
| --- | --- |
| [tron2_openpi](https://github.com/limxdynamics/tron2_openpi) | 面向 TRON2 的 OpenPI 衍生仓库：pi0 / pi0.5 策略服务、TRON2 transforms、部署配置和真机客户端。 |
| [tron2-robot-description](https://github.com/limxdynamics/tron2-robot-description) | TRON2 各型号的 URDF / Xacro、MuJoCo 和 USD 模型。 |

## 第三方来源

本包大部分为 LimX 自研 TRON2 运行时代码。RTC `ActionQueue` 是基于 LeRobot
commit `ca87ccd9413c59c30f524967222d2e3f1b7bb549` 中实现修改的第三方内嵌源码，
已由 PyTorch 移植到 NumPy 并适配 TRON2。详细信息见 `NOTICE`。

## 许可证

除非另有说明，本包使用 `LICENSE` 中的 Apache License 2.0。内嵌第三方源码的
声明见 `NOTICE`，外部直接依赖的许可证清单见 `THIRD_PARTY_DEPENDENCIES.md`。
