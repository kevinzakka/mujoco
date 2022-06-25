"""A tool to convert a MuJoCo model into a serializable format."""

import numpy as np
import dataclasses
import mujoco
from dm_control import mjcf


@dataclasses.dataclass(frozen=True)
class CartesianPose:
  xpos: np.ndarray
  xquat: np.ndarray

  @staticmethod
  def from_mjdata(data: mujoco.MjData) -> "CartesianPose":
    return CartesianPose(data.xpos.copy(), data.xquat.copy())
