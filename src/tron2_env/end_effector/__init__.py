"""Robot-wide end-effector providers."""

from tron2_env.end_effector.base import (
    END_EFFECTOR_PROVIDERS,
    EndEffectorProvider,
    create_end_effector_provider,
    register_end_effector_provider,
)
from tron2_env.end_effector.brainco2 import BrainCo2Provider
from tron2_env.end_effector.gripper import GripperProvider

__all__ = [
    "END_EFFECTOR_PROVIDERS",
    "EndEffectorProvider",
    "GripperProvider",
    "BrainCo2Provider",
    "create_end_effector_provider",
    "register_end_effector_provider",
]
