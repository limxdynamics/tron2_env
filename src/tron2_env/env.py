"""
Tron2 Real Robot Environment

提供 Tron2 机器人的环境封装:
- 机器人控制委托给 ``MotionController`` (transport + interpolator + publish loop)
- 观测来源切换:bridge WebSocket(图像+关节由 bridge 对齐)或 legacy(RealSense 直连)
- ``step(action)`` 非阻塞:按显式 physical layout 拆分目标,关节部分交给 publish 线程,后台以 ``publish_rate``
  (默认 300 Hz)持续 send_joint_cmd,在 ``eta = 1/fps`` 时间内平滑过渡
"""

import copy
import logging
import threading
import time
from dataclasses import InitVar, asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tron2_env.bridge import BridgeConfig
from tron2_env.config import Tron2Config
from tron2_env.debug_images import AsyncDebugImageWriter
from tron2_env.joints import JointIndex
from tron2_env.layout import BrainCo2Config, PhysicalLayout, RobotModules
from tron2_env.motion import MotionController, create_motion_controller

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


def _positive_config_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass
class CameraConfig:
    """相机配置"""
    # 相机名称 (统一使用 obs 输出名称)
    camera_names: List[str] = field(default_factory=lambda: [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist"
    ])
    
    # 相机分辨率 (H, W, C)
    resolution: Tuple[int, int, int] = (480, 640, 3)
    
    # 最大队列大小
    max_queue_size: int = 10
    
    # 是否保存调试图像
    save_debug_images: bool = True
    debug_image_dir: str = "./debug_images"

    


@dataclass
class EnvConfig:
    """环境配置"""
    # 机器人配置
    robot_config: Tron2Config = field(default_factory=Tron2Config)

    # 相机配置
    camera_config: CameraConfig = field(default_factory=CameraConfig)

    # 当前公开包仅支持 high-level WebSocket 控制后端。
    control_backend: str = "websocket"

    # MotionController 后台 publish 频率 (Hz)。两个后端推荐 300Hz。
    publish_rate: float = 300.0

    # consumer 节拍 / 命令到达目标的预期耗时 = 1/fps,MotionController 的
    # LinearInterpolator 用这个 ETA 在两次 command_joints 之间平滑过渡。
    fps: float = 30.0

    # 时间同步容差 (秒)
    time_sync_tolerance: float = 0.01
    time_sync_max_retries: int = 3
    legacy_use_time_sync: bool = True

    # 夹爪初始化开口度 (0-1)
    init_gripper_opening: float = 0.9

    # 原始配置字典（用于透传给其他组件）
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # 观测来源: "legacy" (RealSense 直连) | "bridge" (WebSocket bridge)
    observation_source: str = "legacy"

    # 平铺 policy state/action 布局。旧配置只写 state_dim=16/18 仍兼容；
    # 其它维度必须显式写 state_layout，避免从维度猜测物理含义。
    state_dim: Optional[int] = None
    state_layout: Optional[List[str]] = None

    # chassis/lifter 状态轮询和 lifter stream 刷新节流。
    auxiliary_polling_rate: float = 20.0
    lifter_state_source: str = "raw_q"
    lifter_control_enabled: bool = False
    lifter_command_interval: float = 0.1
    lifter_command_epsilon_mm: InitVar[Optional[float]] = None
    lifter_command_duration_ms: InitVar[Optional[int]] = None

    # Bridge 模式下 state 来源: "bridge" 使用 bridge 对齐 state，"legacy" 使用机器人直连 state
    bridge_state_source: str = "bridge"

    # Bridge WebSocket 配置（observation_source="bridge" 时生效）
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)

    # Explicit modular selection. ``None`` preserves the legacy layout path.
    modules: Optional[RobotModules] = None
    brainco2_config: Optional[BrainCo2Config] = None
    state_max_age: float = 0.5
    head_command_interval: Optional[float] = None
    head_move_duration: Optional[float] = None
    head_ack_timeout: float = 1.0
    lifter_stream_velocity_mm_s: float = 80.0

    physical_layout: PhysicalLayout = field(init=False)

    def __post_init__(
        self,
        lifter_command_epsilon_mm: Optional[float],
        lifter_command_duration_ms: Optional[int],
    ) -> None:
        removed_lifter_options = (
            ("lifter_command_epsilon_mm", lifter_command_epsilon_mm),
            ("lifter_command_duration_ms", lifter_command_duration_ms),
        )
        for option_name, option_value in removed_lifter_options:
            if option_value is not None:
                raise ValueError(
                    f"{option_name} was removed; lifter streaming uses "
                    "lifter_command_interval and lifter_stream_velocity_mm_s"
                )

        if self.modules is None:
            self.physical_layout = PhysicalLayout.resolve(self.state_layout, self.state_dim)
        elif isinstance(self.modules, RobotModules):
            module_layout = PhysicalLayout.from_modules(self.modules)
            if (
                self.state_layout is not None
                and tuple(self.state_layout) != module_layout.components
            ):
                raise ValueError(
                    "state_layout must exactly match the layout derived from modules"
                )
            if self.state_dim is not None:
                if isinstance(self.state_dim, (bool, np.bool_)) or not isinstance(
                    self.state_dim,
                    (int, np.integer),
                ):
                    raise ValueError("state_dim must be an integer")
                if int(self.state_dim) != module_layout.dim:
                    raise ValueError(
                        "state_dim must exactly match the dimension derived from modules"
                    )
            self.physical_layout = module_layout
        else:
            raise TypeError("modules must be a RobotModules instance or None")
        self.state_dim = self.physical_layout.dim

        if self.control_backend != "websocket":
            raise ValueError(
                "The lowlevel backend is not available; the public package supports only websocket"
            )

        if self.modules is not None:
            if self.modules.end_effector == "brainco2":
                if not isinstance(self.brainco2_config, BrainCo2Config):
                    raise ValueError("BrainCo2 modules require BrainCo2Config")
            elif self.brainco2_config is not None:
                raise ValueError(
                    "BrainCo2Config is only valid when the BrainCo2 end effector is selected"
                )
            if (
                self.observation_source == "bridge"
                and self.bridge_state_source != "legacy"
                and (
                    self.modules.arm != "servoj"
                    or self.modules.end_effector != "gripper"
                )
            ):
                raise ValueError(
                    "Bridge ServoP or non-gripper physical state requires "
                    "bridge_state_source='legacy'"
                )
        elif self.brainco2_config is not None:
            raise ValueError(
                "BrainCo2Config is only valid when the BrainCo2 end effector is selected"
            )

        self.state_max_age = _positive_config_number(
            self.state_max_age,
            name="state_max_age",
        )
        for timing_name in ("head_command_interval", "head_move_duration"):
            timing = getattr(self, timing_name)
            if timing is not None:
                setattr(
                    self,
                    timing_name,
                    _positive_config_number(timing, name=timing_name),
                )
        self.head_ack_timeout = _positive_config_number(
            self.head_ack_timeout,
            name="head_ack_timeout",
        )
        self.lifter_stream_velocity_mm_s = _positive_config_number(
            self.lifter_stream_velocity_mm_s,
            name="lifter_stream_velocity_mm_s",
        )
        self.publish_rate = _positive_config_number(
            self.publish_rate,
            name="publish_rate",
        )
        self.fps = _positive_config_number(self.fps, name="fps")
        self.auxiliary_polling_rate = _positive_config_number(
            self.auxiliary_polling_rate,
            name="auxiliary_polling_rate",
        )
        if (
            self.modules is None
            and self.physical_layout.end_effector_name != "gripper"
        ):
            raise ValueError(
                "non-gripper state_layout requires explicit RobotModules so "
                "the matching end-effector provider is selected"
            )
        if self.physical_layout.has("chassis") and self.control_backend != "websocket":
            raise ValueError("The chassis/lifter layout currently requires control_backend='websocket'")
        if self.lifter_state_source not in {"raw_q", "position_mm"}:
            raise ValueError("lifter_state_source must be 'raw_q' or 'position_mm'")
        if self.lifter_control_enabled and not self.physical_layout.has("chassis"):
            raise ValueError("lifter_control_enabled requires the 'chassis' state_layout component")
        if self.lifter_control_enabled and self.lifter_state_source != "position_mm":
            raise ValueError(
                "lifter_control_enabled requires lifter_state_source='position_mm'; "
                "raw q actions must never be sent as absolute millimetres"
            )
        try:
            self.lifter_command_interval = float(self.lifter_command_interval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "lifter_command_interval must be finite, non-negative, and shorter "
                "than the default 300 ms watchdog"
            ) from exc
        if (
            not np.isfinite(self.lifter_command_interval)
            or self.lifter_command_interval < 0
            or self.lifter_command_interval >= 0.3
        ):
            raise ValueError(
                "lifter_command_interval must be finite, non-negative, and shorter "
                "than the default 300 ms watchdog"
            )


del EnvConfig.lifter_command_epsilon_mm
del EnvConfig.lifter_command_duration_ms


# ============================================================================
# Tron2 Environment
# ============================================================================

class Tron2Env:
    """Tron2机器人环境
    
    Examples:
        >>> config = EnvConfig(robot_config=Tron2Config(robot_ip="ROBOT_IP"))
        >>> env = Tron2Env(config)
        >>> obs = env.reset()
        >>> action = np.zeros(config.physical_layout.dim)
        >>> env.step(action)
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        """初始化环境

        Args:
            config: 环境配置，如果为None则使用默认配置
        """
        self.config = config or EnvConfig()
        self.layout = self.config.physical_layout
        self.robot: Optional[MotionController] = None
        self.camera_manager = None
        self.bridge_provider = None
        self._debug_image_writer: Optional[AsyncDebugImageWriter] = None
        self._closed = False
        self._pause_complete = False
        self._close_complete = False
        self._robot_pause_complete = False
        self._close_lifter_stopped = False
        self._close_chassis_stopped = False
        self._close_robot_disconnected = False
        self._close_camera_stopped = False
        self._close_bridge_stopped = False
        self._close_writer_stopped = False
        self._control_lock = threading.RLock()
        self._control_enabled = True
        self._blocked_step_count = 0
        self.last_action: Optional[np.ndarray] = None
        self._last_lifter_command: Optional[float] = None
        self._last_lifter_command_time = float("-inf")
        self._lifter_stop_thread: Optional[threading.Thread] = None
        self._lifter_stop_completed: Optional[threading.Event] = None
        self._lifter_stop_error: Optional[BaseException] = None
        self.init_joints = self.config.robot_config.init_joints

        # 设置日志
        self._setup_logger()

        try:
            self._initialize_resources()
        except BaseException:
            try:
                self.close()
            except BaseException:
                self.logger.exception("Resource cleanup also failed after environment initialization failed")
            raise

        self.logger.info("Environment initialized")

    def _initialize_resources(self) -> None:
        """Initialize hardware and background resources under constructor rollback."""
        # 初始化机器人(MotionController = transport + interpolator + publish loop)
        self.logger.info(
            "Initializing robot controller (backend=%s, publish_rate=%.0fHz, fps=%.1f)...",
            self.config.control_backend,
            self.config.publish_rate,
            self.config.fps,
        )
        self.robot = create_motion_controller(
            self.config.robot_config,
            backend=self.config.control_backend,
            publish_rate=self.config.publish_rate,
            eta_default=1.0 / self.config.fps,
            enable_chassis_lifter=self.layout.has("chassis"),
            auxiliary_polling_rate=self.config.auxiliary_polling_rate,
            lifter_state_source=self.config.lifter_state_source,
            modules=self.config.modules,
            brainco2_config=self.config.brainco2_config,
            state_max_age=self.config.state_max_age,
            head_command_interval=self.config.head_command_interval,
            head_move_duration=self.config.head_move_duration,
            head_ack_timeout=self.config.head_ack_timeout,
        )
        if self.layout.has("chassis") and not self.config.lifter_control_enabled:
            self.logger.warning(
                "Lifter control is disabled: the lifter action remains in the policy vector but will not be sent"
            )

        # 初始化观测来源
        if self.config.observation_source == "bridge":
            self.logger.info("Observation source: Bridge WebSocket")
            if self.config.bridge_state_source not in {"bridge", "legacy"}:
                raise ValueError(
                    f"bridge_state_source must be 'bridge' or 'legacy', got {self.config.bridge_state_source!r}"
                )
            if self.config.bridge_state_source == "legacy":
                self.logger.info("Bridge mode state source: legacy robot WebSocket")
                if self.config.bridge_config.joint_topics:
                    self.logger.info(
                        "Bridge legacy-state mode: disabling bridge joint/gripper subscriptions; using bridge images only"
                    )
                    self.config.bridge_config.joint_topics = {}
            self.bridge_provider = self._init_bridge()
        else:
            self.logger.info("Observation source: legacy direct RealSense connection")
            self.camera_manager = self._init_camera()

        if self.config.observation_source == "bridge":
            if self.config.bridge_config.save_debug_images:
                self._debug_image_writer = AsyncDebugImageWriter(
                    self.config.bridge_config.debug_image_dir
                )
        else:
            if self.config.camera_config.save_debug_images:
                self._debug_image_writer = AsyncDebugImageWriter(
                    self.config.camera_config.debug_image_dir
                )
    
    def _setup_logger(self):
        """Use the package logger configured by the hosting application."""
        self.logger = logger
    
    def _init_bridge(self):
        """初始化 Bridge WebSocket 观测提供者"""
        try:
            from tron2_env.bridge import BridgeObservationProvider
        except ImportError:
            self.logger.error("Failed to import BridgeObservationProvider; make sure websockets is installed")
            raise

        provider = BridgeObservationProvider(self.config.bridge_config)
        try:
            provider.start()

            # 等待首次观测到达
            self.logger.info("Waiting for bridge observations...")
            try:
                provider.get_obs(timeout=5.0)
            except TimeoutError:
                self.logger.warning("Timed out waiting for the first bridge observation; continuing")
            self.logger.info("Bridge observations ready")
        except BaseException:
            try:
                provider.stop()
            except BaseException:
                self.logger.exception("Bridge cleanup also failed after initialization failed")
            raise

        return provider

    def _init_camera(self):
        """初始化相机管理器"""
        try:
            from tron2_env.camera import MultiCameraManager
        except ImportError:
            self.logger.error("Failed to import MultiCameraManager; make sure pyrealsense2 is installed")
            raise
        
        # 尝试从 YAML 加载（如果存在配置字典）
        if hasattr(self.config, 'raw_config'):
            camera_manager = MultiCameraManager.from_config(self.config.raw_config)
        else:
            camera_manager = MultiCameraManager(
                max_queue_size=self.config.camera_config.max_queue_size
            )
        
        try:
            camera_manager.start_capture()

            # 等待相机预热
            self.logger.info("Warming up cameras...")
            time.sleep(2.0)
        except BaseException:
            try:
                camera_manager.stop_capture()
            except BaseException:
                self.logger.exception("Camera cleanup also failed after initialization failed")
            raise
        
        return camera_manager
    
    # ========================================================================
    # Environment Interface
    # ========================================================================
    
    def reset(self) -> Dict:
        """重置环境到初始状态
        
        Returns:
            初始观测
        """
        self.logger.info("Resetting environment...")
        
        # 获取当前观测
        obs = self.get_obs()

        # 验证图像尺寸（仅 legacy 模式）
        if self.config.observation_source != "bridge":
            expected_shape = self.config.camera_config.resolution
            for cam_name in self.config.camera_config.camera_names:
                actual_shape = obs['images'][cam_name].shape
                if actual_shape != expected_shape:
                    self.logger.warning(
                        f"{cam_name} resolution mismatch: expected {expected_shape}, got {actual_shape}"
                    )
        
        # ServoP uses measured Cartesian poses and must not apply joint-space
        # reset checks to its policy vector.
        if self.init_joints is not None and (
            self.config.modules is None or self.config.modules.arm == "servoj"
        ):
            components = self.layout.unpack(obs["state"])
            arm_states = np.concatenate(
                [components["left_arm"], components["right_arm"]]
            )
            init_arm = np.array(self.init_joints)
            
            error = np.abs(arm_states - init_arm).max()
            if error > 0.05:
                self.logger.warning(f"Robot is not at the initial position; maximum error: {error:.4f}")
                self.robot.wait_until_reached(self.init_joints,tolerance=0.05)
        
        if self.config.modules is None:
            # Preserve the legacy reset, including its full-state ServoJ hold.
            test_action = obs['state'].copy()
            test_action[JointIndex.LEFT_GRIPPER] = self.config.init_gripper_opening
            test_action[JointIndex.RIGHT_GRIPPER] = self.config.init_gripper_opening
            self.step(test_action)
        elif self.config.modules.end_effector == "gripper":
            # Opening modular grippers must not replay arm/head/mobile values.
            opening = np.array([self.config.init_gripper_opening], dtype=np.float64)
            with self._control_lock:
                if self._control_enabled:
                    self.robot.raise_if_faulted()
                    self.robot.command_end_effector(opening, opening)
        # BrainCo2 has no defined home/open gesture. Its measured state was
        # already read while composing ``obs`` and is never guessed here.
        
        self.logger.info("Environment reset complete")
        return obs

    def pause_control(self, timeout: float = 1.0) -> None:
        """Close the policy command gate and acknowledge stopped publication."""
        with self._control_lock:
            self._control_enabled = False
            self._reset_lifter_command_cache()
            errors: List[BaseException] = []
            self._pause_complete = False
            self._robot_pause_complete = False
            if self.layout.has("chassis") and self.config.lifter_control_enabled:
                stop_error = self._stop_lifter_stream_bounded(timeout=timeout)
                if stop_error is not None:
                    self.logger.error(
                        "Failed to stop lifter stream while pausing control",
                        exc_info=(
                            type(stop_error),
                            stop_error,
                            stop_error.__traceback__,
                        ),
                    )
                    errors.append(stop_error)
            if self.layout.has("chassis"):
                try:
                    self.robot.stop_chassis()
                except BaseException as exc:
                    self.logger.exception("Failed to stop chassis while pausing control")
                    errors.append(exc)
            try:
                self.robot.pause(timeout=timeout)
                self._robot_pause_complete = True
            except BaseException as exc:
                errors.append(exc)
            if errors:
                raise errors[0]
            self._pause_complete = True

    def resume_control(self, measured_state: Optional[np.ndarray] = None) -> None:
        """Rebase motion to measured state before reopening the command gate."""
        with self._control_lock:
            if self._closed:
                raise RuntimeError("cannot resume a closed environment")
            if not self._pause_complete:
                raise RuntimeError("cannot resume before pause is complete")
            if self.config.modules is not None and self.config.modules.arm == "servop":
                self.robot.resume_from_state()
            elif measured_state is None:
                self.robot.resume_from_state()
            else:
                state = np.asarray(measured_state, dtype=np.float64)
                if state.shape != (JointIndex.STATE_DIM,):
                    raise ValueError(
                        f"measured state shape must be ({JointIndex.STATE_DIM},), got {state.shape}"
                    )
                self.robot.resume_from_state(state)
            self._pause_complete = False
            self._robot_pause_complete = False
            self._control_enabled = True

    @property
    def control_enabled(self) -> bool:
        with self._control_lock:
            return self._control_enabled

    @property
    def blocked_step_count(self) -> int:
        with self._control_lock:
            return self._blocked_step_count

    def step(self, action: Union[List[float], np.ndarray]) -> bool:
        """执行一个动作.

        非阻塞:把关节目标更新到 MotionController 的内部 interpolator,
        publish 线程会在 ~1/fps 时间窗内平滑过渡。夹爪、底盘和升降台走单独通路。

        Args:
            action: 与 ``config.physical_layout`` 完全匹配的平铺动作向量。
        """
        with self._control_lock:
            if not self._control_enabled:
                self._blocked_step_count += 1
                return False

            # Validate the caller's original dtype/shape before numeric coercion,
            # then guard the float64 representation used by the controllers.
            self.layout.unpack(action)
            with np.errstate(over="ignore", invalid="ignore"):
                action_array = np.asarray(action, dtype=np.float64)
            if not np.all(np.isfinite(action_array)):
                raise ValueError("Physical action values must all be finite")
            if self.layout.end_effector_name == "gripper":
                action_array = action_array.copy()
                for component in (
                    self.layout.left_end_effector_component,
                    self.layout.right_end_effector_component,
                ):
                    component_slice = self.layout.component_slice(component)
                    action_array[component_slice] = np.clip(
                        action_array[component_slice], 0.0, 1.0
                    )
            components = self.layout.unpack(action_array)
            mobile_command = (
                self._plan_mobile_command(components["chassis"])
                if "chassis" in components
                else None
            )

            # Phase one is side-effect free: every component that this step may
            # dispatch is accepted before any target, scheduler, or transport is
            # touched. Later I/O failures are intentionally not transactional.
            self.robot.raise_if_faulted()
            self.robot.preflight_arm(
                components["left_arm"],
                components["right_arm"],
                head=components.get("head"),
            )
            self.robot.preflight_end_effector(
                components[self.layout.left_end_effector_component],
                components[self.layout.right_end_effector_component],
            )
            if mobile_command is not None:
                linear_x, angular_z, lifter_command = mobile_command
                self.robot.preflight_chassis(linear_x, angular_z)
                if lifter_command is not None:
                    self.robot.preflight_lifter_stream_position(*lifter_command)

            # Phase two performs the already-validated dispatches.
            self.robot.command_arm(
                components["left_arm"],
                components["right_arm"],
                head=components.get("head"),
            )
            self.robot.command_end_effector(
                components[self.layout.left_end_effector_component],
                components[self.layout.right_end_effector_component],
            )

            if mobile_command is not None:
                self._dispatch_mobile_command(mobile_command)
            self.last_action = action_array.copy()
            return True

    def _plan_mobile_command(
        self,
        mobile_action: np.ndarray,
    ) -> Tuple[float, float, Optional[Tuple[float, float]]]:
        linear_x, angular_z, lifter_action = mobile_action
        linear_value = float(linear_x)
        angular_value = float(angular_z)
        lifter_command = None
        if not self.config.lifter_control_enabled:
            return linear_value, angular_value, lifter_command

        lifter_value = float(lifter_action)
        now = time.monotonic()
        if now >= (
            self._last_lifter_command_time
            + self.config.lifter_command_interval
            - 1e-12
        ):
            lifter_command = (
                lifter_value,
                self.config.lifter_stream_velocity_mm_s,
            )
        return linear_value, angular_value, lifter_command

    def _dispatch_mobile_command(
        self,
        command: Tuple[float, float, Optional[Tuple[float, float]]],
    ) -> None:
        linear_x, angular_z, lifter_command = command
        self.robot.command_chassis(linear_x=linear_x, angular_z=angular_z)
        if lifter_command is None:
            return
        position_mm, velocity_mm_s = lifter_command
        self.robot.stream_lifter_position(position_mm, velocity_mm_s=velocity_mm_s)
        self._last_lifter_command = position_mm
        self._last_lifter_command_time = time.monotonic()

    def _command_mobile(self, mobile_action: np.ndarray) -> None:
        """Compatibility wrapper retaining validation-before-dispatch semantics."""
        command = self._plan_mobile_command(mobile_action)
        linear_x, angular_z, lifter_command = command
        self.robot.preflight_chassis(linear_x, angular_z)
        if lifter_command is not None:
            self.robot.preflight_lifter_stream_position(*lifter_command)
        self._dispatch_mobile_command(command)

    def _reset_lifter_command_cache(self) -> None:
        self._last_lifter_command = None
        self._last_lifter_command_time = float("-inf")

    def _stop_lifter_stream_bounded(self, timeout: float) -> Optional[BaseException]:
        """Request a stream stop without letting unresponsive I/O block cleanup."""
        try:
            timeout_value = max(0.0, float(timeout))
        except (TypeError, ValueError, OverflowError):
            timeout_value = 0.0

        thread = getattr(self, "_lifter_stop_thread", None)
        completed = getattr(self, "_lifter_stop_completed", None)
        if thread is None or completed is None:
            completed = threading.Event()
            self._lifter_stop_completed = completed
            self._lifter_stop_error = None

            def stop_lifter_stream() -> None:
                try:
                    self.robot.stop_lifter_stream()
                except BaseException as exc:
                    self._lifter_stop_error = exc
                finally:
                    completed.set()

            thread = threading.Thread(
                target=stop_lifter_stream,
                name="tron2-lifter-stop",
                daemon=True,
            )
            self._lifter_stop_thread = thread
            thread.start()

        if not completed.wait(timeout=timeout_value):
            return TimeoutError("lifter stream stop timed out")

        stop_error = self._lifter_stop_error
        self._lifter_stop_thread = None
        self._lifter_stop_completed = None
        self._lifter_stop_error = None
        return stop_error

    def _join_lifter_stop_worker(self, timeout: float = 0.1) -> bool:
        """Bounded cleanup after disconnect has released a blocked stop request."""
        thread = getattr(self, "_lifter_stop_thread", None)
        completed = getattr(self, "_lifter_stop_completed", None)
        if thread is None or completed is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        if completed.is_set():
            self._lifter_stop_thread = None
            self._lifter_stop_completed = None
            self._lifter_stop_error = None
        return True

    def _lifter_stop_worker_exited(self) -> bool:
        thread = getattr(self, "_lifter_stop_thread", None)
        return thread is None or not thread.is_alive()

    def _compose_policy_state(
        self,
        raw_state: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build the configured flat policy state from canonical robot feedback."""
        if self.config.modules is not None and raw_state is None:
            return self._compose_modular_policy_state()
        if raw_state is None:
            raise ValueError("raw_state is required for legacy policy state composition")
        canonical = np.asarray(raw_state, dtype=np.float32)
        if canonical.shape != (JointIndex.STATE_DIM,):
            raise ValueError(
                f"Canonical robot state must have shape ({JointIndex.STATE_DIM},), got {canonical.shape}"
            )
        values: Dict[str, np.ndarray] = {
            "left_arm": canonical[JointIndex.LEFT_ARM],
            "left_gripper": canonical[JointIndex.LEFT_GRIPPER : JointIndex.LEFT_GRIPPER + 1],
            "right_arm": canonical[JointIndex.RIGHT_ARM],
            "right_gripper": canonical[JointIndex.RIGHT_GRIPPER : JointIndex.RIGHT_GRIPPER + 1],
            "head": canonical[JointIndex.HEAD],
        }
        metadata: Dict[str, Any] = {}
        if self.config.modules is not None:
            metadata.update(
                {
                    "arm_control_mode": self.config.modules.arm,
                    "end_effector": self.config.modules.end_effector,
                    "modules": asdict(self.config.modules),
                    "field_names": self.layout.field_names,
                }
            )
        if self.layout.has("chassis"):
            if self.config.modules is not None:
                max_age = self.config.state_max_age
                auxiliary = self.robot.get_chassis_lifter_state(
                    timeout=max_age,
                    max_age=max_age,
                )
            else:
                auxiliary = self.robot.get_chassis_lifter_state(timeout=0.5)
            values["chassis"] = np.asarray(
                [
                    auxiliary["linear_velocity"],
                    auxiliary["angular_velocity"],
                    auxiliary["lifter_state"],
                ],
                dtype=np.float32,
            )
            metadata.update(
                {
                    "chassis_timestamp_ms": auxiliary.get("chassis_timestamp"),
                    "lifter_timestamp_ms": auxiliary.get("lifter_timestamp"),
                    "lifter_state_source": self.config.lifter_state_source,
                }
            )
            if self.config.modules is not None:
                metadata.update(
                    {
                        "chassis_received_at": auxiliary.get("chassis_received_at"),
                        "lifter_received_at": auxiliary.get("lifter_received_at"),
                        "mobile_metadata": copy.deepcopy(auxiliary),
                    }
                )
        return self.layout.pack(values), metadata

    def _compose_modular_policy_state(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        max_age = self.config.state_max_age
        arm_state, arm_metadata = self.robot.get_arm_state(
            timeout=max_age,
            max_age=max_age,
        )
        effector_state, effector_metadata = self.robot.get_end_effector_state(
            timeout=max_age,
            max_age=max_age,
        )
        arm_state = np.asarray(arm_state, dtype=np.float64)
        effector_state = np.asarray(effector_state, dtype=np.float64)
        width = self.layout.end_effector_width
        values: Dict[str, np.ndarray] = {
            "left_arm": arm_state[:7],
            self.layout.left_end_effector_component: effector_state[:width],
            "right_arm": arm_state[7:14],
            self.layout.right_end_effector_component: effector_state[width:],
        }
        metadata: Dict[str, Any] = {
            "arm_control_mode": self.config.modules.arm,
            "end_effector": self.config.modules.end_effector,
            "modules": asdict(self.config.modules),
            "field_names": self.layout.field_names,
            "arm_metadata": copy.deepcopy(arm_metadata),
            "end_effector_metadata": copy.deepcopy(effector_metadata),
        }
        self._copy_component_timestamp(metadata, "arm", arm_metadata)
        self._copy_component_timestamp(
            metadata,
            "end_effector",
            effector_metadata,
        )
        left_names = effector_metadata.get("left_names")
        right_names = effector_metadata.get("right_names")
        if left_names is not None:
            metadata[f"{self.layout.left_end_effector_component}_names"] = copy.deepcopy(
                left_names
            )
        if right_names is not None:
            metadata[f"{self.layout.right_end_effector_component}_names"] = copy.deepcopy(
                right_names
            )

        if self.layout.has("head"):
            head_state, head_metadata = self.robot.get_head_state(
                timeout=max_age,
                max_age=max_age,
            )
            values["head"] = np.asarray(head_state, dtype=np.float64)
            metadata["head_metadata"] = copy.deepcopy(head_metadata)
            self._copy_component_timestamp(metadata, "head", head_metadata)

        if self.layout.has("chassis"):
            auxiliary = self.robot.get_chassis_lifter_state(
                timeout=max_age,
                max_age=max_age,
            )
            values["chassis"] = np.asarray(
                [
                    auxiliary["linear_velocity"],
                    auxiliary["angular_velocity"],
                    auxiliary["lifter_state"],
                ],
                dtype=np.float64,
            )
            metadata.update(
                {
                    "chassis_timestamp_ms": auxiliary.get("chassis_timestamp"),
                    "lifter_timestamp_ms": auxiliary.get("lifter_timestamp"),
                    "chassis_received_at": auxiliary.get("chassis_received_at"),
                    "lifter_received_at": auxiliary.get("lifter_received_at"),
                    "lifter_state_source": self.config.lifter_state_source,
                    "mobile_metadata": copy.deepcopy(auxiliary),
                }
            )

        return self.layout.pack(values), metadata

    @staticmethod
    def _copy_component_timestamp(
        destination: Dict[str, Any],
        prefix: str,
        source: Dict[str, Any],
    ) -> None:
        if "timestamp" in source:
            destination[f"{prefix}_timestamp_ms"] = copy.deepcopy(source["timestamp"])
        if "robot_timestamp" in source:
            destination[f"{prefix}_robot_timestamp"] = copy.deepcopy(
                source["robot_timestamp"]
            )
        if "client_timestamp" in source:
            destination[f"{prefix}_client_timestamp_ms"] = copy.deepcopy(
                source["client_timestamp"]
            )
        if "received_at" in source:
            destination[f"{prefix}_received_at"] = copy.deepcopy(source["received_at"])
    
    def get_obs(self) -> Dict:
        """获取当前观测

        Returns:
            观测字典: {
                'state': np.ndarray,  # config.physical_layout 定义的物理状态
                'images': Dict[str, np.ndarray]  # 图像字典
            }

        metadata 中关于时间戳的约定：
        - ``joint_timestamp_ms`` / ``gripper_timestamp_ms`` 始终对应 ``state``
          字段实际的来源（"我们推理用的那帧 state 的时间戳"）。
        - ``state_source`` 取值 ``bridge`` 或 ``legacy``，指示 state 来自哪条路径。
        - 若 ``state_source == "legacy"`` 但图像走 bridge，``bridge_joint_timestamp_ms``
          / ``bridge_gripper_timestamp_ms`` 会保留 bridge 自己对齐到的关节时间戳，
          供调试对比，不参与正常推理。
        """
        # Bridge 模式：图像来自 bridge，可选 state 来自 bridge 或机器人直连
        if self.config.observation_source == "bridge":
            obs = self.bridge_provider.get_obs(timeout=1.0)
            metadata = dict(obs.get("metadata", {}))
            if (
                self.config.modules is not None
                and self.config.bridge_state_source == "legacy"
            ):
                bridge_joint_timestamp_ms = metadata.pop("joint_timestamp_ms", None)
                bridge_gripper_timestamp_ms = metadata.pop(
                    "gripper_timestamp_ms",
                    None,
                )
                obs["state"], modular_metadata = self._compose_policy_state()
                metadata.update(
                    {
                        "state_source": "legacy",
                        "bridge_joint_timestamp_ms": bridge_joint_timestamp_ms,
                        "bridge_gripper_timestamp_ms": bridge_gripper_timestamp_ms,
                    }
                )
                metadata.update(modular_metadata)
            elif self.config.bridge_state_source == "legacy":
                # state 走 robot ws 直拉。bridge 已经在 TopicAligner 里对齐过
                # 图像-state,这里直接信任 robot ws 自己的 timestamp,不再做二次
                # 对齐 —— 之前的 _sync_observation 调用会把 bridge ts 与 robot ws
                # ts 强行拼到一起,反而引入不同源时钟的失配,并多出 ~5ms sleep。
                bridge_joint_timestamp_ms = metadata.get("joint_timestamp_ms")
                bridge_gripper_timestamp_ms = metadata.get("gripper_timestamp_ms")
                qpos_dict = self.robot.get_joint_states(timeout=0.5)
                obs["state"], auxiliary_metadata = self._compose_policy_state(qpos_dict["states"])
                metadata.update({
                    "state_source": "legacy",
                    "bridge_joint_timestamp_ms": bridge_joint_timestamp_ms,
                    "bridge_gripper_timestamp_ms": bridge_gripper_timestamp_ms,
                    "joint_timestamp_ms": qpos_dict.get("timestamp"),
                    "gripper_timestamp_ms": qpos_dict.get("timestamp"),
                })
                metadata.update(auxiliary_metadata)
            else:
                obs["state"], auxiliary_metadata = self._compose_policy_state(obs["state"])
                metadata["state_source"] = "bridge"
                metadata.update(auxiliary_metadata)
            obs["metadata"] = metadata
            if self._debug_image_writer is not None:
                self._debug_image_writer.submit(obs["images"])
            return obs

        # Legacy 模式：先拿图像，用三相机中最旧的时间戳作为 obs 参考时刻，
        # 再在 200Hz joint_state_queue 里找与该时刻最近的 joint 帧。
        # 时间基准统一为客户端 time.time()（camera 和 transport 都用同一时钟）。
        obs_start = time.time()
        # 1. 获取图像
        rgb_images = self._get_images()

        images = {
            cam_name: rgb_images[cam_name]
            for cam_name in self.config.camera_config.camera_names
            if cam_name in rgb_images
        }
        if self._debug_image_writer is not None:
            self._debug_image_writer.submit(images)

        if self.config.modules is not None:
            image_timestamps_ms = {
                cam_name: int(rgb_images[f"{cam_name}_timestamp"] * 1000)
                for cam_name in self.config.camera_config.camera_names
                if f"{cam_name}_timestamp" in rgb_images
            }
            if not image_timestamps_ms:
                raise RuntimeError("No camera frames available in legacy mode")
            policy_state, modular_metadata = self._compose_policy_state()
            return {
                "state": policy_state,
                "images": images,
                "metadata": {
                    "state_source": "legacy",
                    "observation_ref_timestamp_ms": min(image_timestamps_ms.values()),
                    "image_timestamp_ms": min(image_timestamps_ms.values()),
                    "image_timestamps_ms": image_timestamps_ms,
                    **modular_metadata,
                },
            }

        # 2. 确定 obs 参考时间戳 = 三相机中最旧的那帧（保证所有图像都 ≥ 该时刻）
        cam_timestamps = []
        for cam_name in self.config.camera_config.camera_names:
            ts_key = f'{cam_name}_timestamp'
            if ts_key in rgb_images:
                cam_timestamps.append(rgb_images[ts_key])
        if not cam_timestamps:
            raise RuntimeError("No camera frames available in legacy mode")
        img_timestamp = min(cam_timestamps)  # 最旧的那帧

        # 3. 获取关节状态——在 joint_state_queue 里找与 img_timestamp 最近的帧
        synced_qpos: Optional[Dict] = None
        if self.config.legacy_use_time_sync:
            synced_qpos = self.robot.find_nearest_state(img_timestamp)
            if synced_qpos is not None:
                joint_timestamp = synced_qpos['timestamp'] / 1000.0
                self.logger.debug(
                    "legacy obs sync: nearest queued joint_img=%.1fms",
                    (joint_timestamp - img_timestamp) * 1000.0,
                )
        # Fallback: 队列空 / 不启用 sync —— 走原始 popleft 取最新帧
        qpos_dict = synced_qpos
        if qpos_dict is None:
            qpos_dict = self.robot.get_joint_states(timeout=0.5)
            synced_qpos = qpos_dict
        joint_timestamp = qpos_dict['timestamp'] / 1000.0

        self.logger.debug(
            "legacy obs: ref_img=%.3fs joint=%.3fs diff=%.1fms",
            img_timestamp, joint_timestamp,
            (joint_timestamp - img_timestamp) * 1000.0,
        )

        # 4. 构建观测
        image_timestamps_ms = {
            cam_name: int(rgb_images[f'{cam_name}_timestamp'] * 1000)
            for cam_name in self.config.camera_config.camera_names
            if f'{cam_name}_timestamp' in rgb_images
        }
        image_timestamp_ms = (
            min(image_timestamps_ms.values()) if image_timestamps_ms else int(img_timestamp * 1000)
        )
        joint_timestamp_ms = synced_qpos.get('timestamp')
        synced_joint_timestamp = (joint_timestamp_ms or 0) / 1000.0
        obs_end = time.time()
        image_span_ms = 0.0
        if image_timestamps_ms:
            image_span_ms = max(image_timestamps_ms.values()) - min(image_timestamps_ms.values())

        self.logger.debug(
            "legacy obs timing: sync=%s raw_joint_img=%.1fms synced_joint_img=%.1fms "
            "img_age=%.1fms joint_age=%.1fms total=%.1fms image_span=%.1fms",
            self.config.legacy_use_time_sync,
            (joint_timestamp - img_timestamp) * 1000.0,
            (synced_joint_timestamp - img_timestamp) * 1000.0,
            (obs_end - img_timestamp) * 1000.0,
            (obs_end - synced_joint_timestamp) * 1000.0 if synced_joint_timestamp > 0 else float("nan"),
            (obs_end - obs_start) * 1000.0,
            image_span_ms,
        )

        policy_state, auxiliary_metadata = self._compose_policy_state(synced_qpos['states'])
        obs = {
            "state": policy_state,
            "images": images,
            "metadata": {
                "state_source": "legacy",
                "observation_ref_timestamp_ms": image_timestamp_ms,
                "bridge_ref_timestamp_ms": image_timestamp_ms,
                "joint_timestamp_ms": joint_timestamp_ms,
                "gripper_timestamp_ms": joint_timestamp_ms,
                "image_timestamp_ms": image_timestamp_ms,
                "image_timestamps_ms": image_timestamps_ms,
                "legacy_initial_joint_timestamp_ms": qpos_dict.get('timestamp'),
                "legacy_time_sync_enabled": self.config.legacy_use_time_sync,
                **auxiliary_metadata,
            },
        }

        return obs
    
    # ========================================================================
    # Private Methods
    # ========================================================================

    def _get_images(self) -> Dict:
        """获取相机图像

        Returns:
            图像字典: {
                'cam_high': np.ndarray,
                'cam_high_timestamp': float,
                ...
            }
        """
        all_frames = self.camera_manager.get_all_latest_frames()
        image_dict = {}
        
        for camera_name, frame_data in all_frames.items():
            if frame_data is not None:
                # BGR转RGB
                image_dict[camera_name] = frame_data['color'][:, :, ::-1]
                image_dict[f'{camera_name}_timestamp'] = frame_data['timestamp']
        
        return image_dict
    
    def _sync_observation(
        self,
        img_timestamp: float,
        initial_qpos: Dict,
        using_sync: bool = False
    ) -> Dict:
        """同步观测时间戳

        策略: 直接在 transport 的 200Hz joint_state_queue (maxlen=7, ~35ms 历史窗口) 中
        查询与 img_timestamp 时间最近的关节帧——非阻塞，不消耗队列，避免重试 sleep。

        Args:
            img_timestamp: 图像时间戳 (秒)
            initial_qpos: get_obs 已 popleft 的最新关节状态 (作为 fallback)

        Returns:
            同步后的关节状态
        """
        joint_timestamp = initial_qpos['timestamp'] / 1000.0
        time_dif = joint_timestamp - img_timestamp

        if not using_sync:
            self.logger.debug("legacy obs sync disabled: joint_img=%.1fms", time_dif * 1000.0)
            return initial_qpos

        # 已经在容差内，直接用
        if abs(time_dif) <= self.config.time_sync_tolerance:
            self.logger.debug("legacy obs sync ok: joint_img=%.1fms", time_dif * 1000.0)
            return initial_qpos

        # 在队列里找与 img_timestamp 最近的帧
        nearest = self.robot.find_nearest_state(img_timestamp)
        if nearest is None:
            self.logger.debug(
                "legacy obs sync: no queued state available; using initial (joint_img=%.1fms)",
                time_dif * 1000.0,
            )
            return initial_qpos

        nearest_ts = nearest['timestamp'] / 1000.0
        nearest_dif = nearest_ts - img_timestamp
        # 只有当队列里的帧比 initial_qpos 更接近 image_ts 才换用
        if abs(nearest_dif) < abs(time_dif):
            self.logger.debug(
                "legacy obs sync: switched to nearer queued state (initial=%.1fms, nearest=%.1fms)",
                time_dif * 1000.0,
                nearest_dif * 1000.0,
            )
            return nearest

        self.logger.debug(
            "legacy obs sync: initial already nearest (initial=%.1fms, queued_best=%.1fms)",
            time_dif * 1000.0,
            nearest_dif * 1000.0,
        )
        return initial_qpos
    
    def close(self):
        """关闭环境并释放资源"""
        with self._control_lock:
            if getattr(self, "_close_complete", False):
                return
            self._closed = True
            self._control_enabled = False
            self._reset_lifter_command_cache()

            instance_logger = getattr(self, "logger", logger)
            instance_logger.info("Closing environment...")

            def attempt(flag_name: str, label: str, action) -> None:
                if getattr(self, flag_name, False):
                    return
                try:
                    action()
                except Exception:
                    instance_logger.exception("Failed to %s while closing the environment", label)
                else:
                    setattr(self, flag_name, True)

            robot = getattr(self, "robot", None)
            if robot is not None:
                lifter_stream_enabled = (
                    getattr(self, "layout", None) is not None
                    and self.layout.has("chassis")
                    and self.config.lifter_control_enabled
                )
                if (
                    lifter_stream_enabled
                    and not getattr(self, "_close_lifter_stopped", False)
                    and self._lifter_stop_worker_exited()
                ):
                    stop_error = self._stop_lifter_stream_bounded(timeout=0.05)
                    if stop_error is None:
                        self._close_lifter_stopped = True
                    else:
                        instance_logger.error(
                            "Failed to stop the lifter stream while closing the environment",
                            exc_info=(
                                type(stop_error),
                                stop_error,
                                stop_error.__traceback__,
                            ),
                        )
                elif not lifter_stream_enabled:
                    self._close_lifter_stopped = True
                if getattr(self, "layout", None) is not None and self.layout.has("chassis"):
                    attempt(
                        "_close_chassis_stopped",
                        "stop the chassis",
                        robot.stop_chassis,
                    )
                else:
                    self._close_chassis_stopped = True
                attempt(
                    "_robot_pause_complete",
                    "pause the robot",
                    lambda: robot.pause(timeout=1.0),
                )
                attempt(
                    "_close_robot_disconnected",
                    "disconnect the robot",
                    robot.disconnect,
                )
                if getattr(self, "_close_robot_disconnected", False):
                    # A completed disconnect has stopped the controller's
                    # publish loop and is the final fallback when the stop
                    # request failed or remained blocked in transport I/O.
                    self._robot_pause_complete = True
                    self._close_lifter_stopped = True
                    self._close_chassis_stopped = True
                    self._join_lifter_stop_worker()
            else:
                self._robot_pause_complete = True
                self._close_lifter_stopped = True
                self._close_chassis_stopped = True
                self._close_robot_disconnected = True

            camera_manager = getattr(self, "camera_manager", None)
            if camera_manager is not None:
                attempt(
                    "_close_camera_stopped",
                    "stop the cameras",
                    camera_manager.stop_capture,
                )
            else:
                self._close_camera_stopped = True

            bridge_provider = getattr(self, "bridge_provider", None)
            if bridge_provider is not None:
                attempt(
                    "_close_bridge_stopped",
                    "stop the bridge",
                    bridge_provider.stop,
                )
            else:
                self._close_bridge_stopped = True

            debug_image_writer = getattr(self, "_debug_image_writer", None)
            if debug_image_writer is not None:
                attempt(
                    "_close_writer_stopped",
                    "stop the debug image writer",
                    lambda: debug_image_writer.close(timeout=0.2),
                )
                if getattr(self, "_close_writer_stopped", False):
                    self._debug_image_writer = None
            else:
                self._close_writer_stopped = True

            self._close_complete = all(
                (
                    getattr(self, "_robot_pause_complete", False),
                    getattr(self, "_close_lifter_stopped", False),
                    getattr(self, "_close_chassis_stopped", False),
                    getattr(self, "_close_robot_disconnected", False),
                    getattr(self, "_close_camera_stopped", False),
                    getattr(self, "_close_bridge_stopped", False),
                    getattr(self, "_close_writer_stopped", False),
                    self._lifter_stop_worker_exited(),
                )
            )
            if self._close_complete:
                instance_logger.info("Environment closed")
            else:
                instance_logger.warning("Environment cleanup is incomplete; a later close() call will retry")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


# ============================================================================
# Policy Wrapper (Example)
# ============================================================================

class PolicyWrapper:
    """策略包装器基类"""
    
    def get_action(self, observation: Dict) -> np.ndarray:
        """获取动作
        
        Args:
            observation: 观测字典
            
        Returns:
            动作数组
        """
        raise NotImplementedError


class WebsocketPolicyWrapper(PolicyWrapper):
    """基于WebSocket的策略客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """初始化WebSocket策略客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        try:
            from openpi_client import image_tools, websocket_client_policy
            self.ws_client = websocket_client_policy.WebsocketClientPolicy(
                host=host, 
                port=port
            )
            self.image_tools = image_tools
        except ImportError as e:
            raise ImportError(f"Failed to import openpi_client: {e}")
        
        self.logger = logging.getLogger("WebsocketPolicy")
    
    def get_action(self, observation: Dict) -> np.ndarray:
        """通过WebSocket获取动作
        
        Args:
            observation: 观测字典
            
        Returns:
            动作序列 (action_horizon, action_dim)
        """
        import einops
        
        # 预处理图像
        obs = observation.copy()
        for cam_name in obs["images"]:
            img = self.image_tools.convert_to_uint8(
                self.image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
        
        # 推理
        result = self.ws_client.infer(obs)
        actions = np.stack(result['actions'], axis=0)

        return actions
