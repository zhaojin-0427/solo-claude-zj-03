"""领域异常。

校验/计算类问题统一抛出 :class:`GlazeError`，
由 FastAPI 映射为 HTTP 422，并返回结构化错误信息。
"""
from __future__ import annotations

from typing import Any


class GlazeError(ValueError):
    """请求可解析但在釉料业务规则上不合法。

    继承 :class:`ValueError`，使其可直接在 Pydantic 校验器中抛出；
    FastAPI 侧统一映射为 HTTP 422。
    """

    def __init__(self, message: str, code: str = "glaze_error", details: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class NotFoundError(Exception):
    """引用的原料或配方版本不存在。"""

    def __init__(self, message: str, code: str = "not_found"):
        super().__init__(message)
        self.message = message
        self.code = code
