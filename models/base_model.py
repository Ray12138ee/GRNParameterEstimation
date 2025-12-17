# models/base_model.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass
class ParameterSet:
    """
    通用参数容器。具体模型可以把真正的参数结构
    （比如 GRNParameters）塞进 `payload` 里。
    """
    payload: Any
    names: Sequence[str] | None = None


class DynamicalModel(ABC):
    """
    抽象基类：所有动力学模型的统一接口。
    你之后的 Immune / PKPD / Epi 都可以继承它。
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    # ---------- 模型结构相关 ----------
    @abstractmethod
    def state_names(self) -> list[str]:
        """返回状态变量名顺序，长度 = state 维度."""
        ...

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """返回需要估计/管理的参数名列表."""
        ...

    # ---------- 参数 & 初始条件 ----------
    @abstractmethod
    def sample_ground_truth_parameters(
        self, rng: np.random.Generator
    ) -> ParameterSet:
        """从先验/范围中采样一套“真参数”."""
        ...

    @abstractmethod
    def initial_state(
        self, params: ParameterSet, rng: np.random.Generator
    ) -> np.ndarray:
        """给定参数生成一组初始状态."""
        ...

    # ---------- ODE 右端 ----------
    @abstractmethod
    def rhs(
        self, t: float, y: np.ndarray, params: ParameterSet
    ) -> np.ndarray:
        """
        ODE 右端：dy/dt = f(t, y; params)
        返回和 y 同维度的导数向量。
        """
        ...