"""Pydantic 数据模型：原料、计算请求、搜索请求、响应结构。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import OXIDE_CATALOG
from .errors import GlazeError


# ---------------------------------------------------------------------------
# 原料库
# ---------------------------------------------------------------------------

class MaterialBase(BaseModel):
    name: str = Field(min_length=1, description="原料名称")
    oxides: dict[str, float] = Field(
        default_factory=dict,
        description="氧化物 -> 质量分数（百分数口径，100 g 基准下的克数）",
    )
    loi: float = Field(0.0, ge=0.0, le=100.0, description="灼烧减量 LOI（百分数）")
    price: float = Field(..., ge=0.0, description="每千克单价")
    available: float = Field(..., ge=0.0, description="库存可用量（kg）")
    analysis_tolerance: float = Field(
        2.0, gt=0.0, description="分析合计允许的绝对误差（百分点）"
    )

    @field_validator("oxides")
    @classmethod
    def _check_oxides(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise GlazeError("原料至少要给出一种氧化物质量分数", "empty_analysis")
        unknown = sorted(set(v) - set(OXIDE_CATALOG))
        if unknown:
            raise GlazeError(
                f"未知氧化物: {', '.join(unknown)}",
                "unknown_oxide",
                {"unknown": unknown},
            )
        negatives = {k: x for k, x in v.items() if x < 0.0}
        if negatives:
            raise GlazeError(
                f"氧化物质量分数不得为负: {negatives}",
                "negative_component",
                {"components": negatives},
            )
        return v

    @model_validator(mode="after")
    def _check_analysis_sum(self) -> "MaterialBase":
        total = sum(self.oxides.values()) + self.loi
        if abs(total - 100.0) > self.analysis_tolerance + 1e-9:
            raise GlazeError(
                f"分析合计 {total:.3f} 超出 100 ± {self.analysis_tolerance} 的允许误差",
                "analysis_sum_error",
                {"total": round(total, 4), "allowed": 100.0, "tol": self.analysis_tolerance},
            )
        return self


class MaterialCreate(MaterialBase):
    pass


class MaterialUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    oxides: Optional[dict[str, float]] = None
    loi: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    price: Optional[float] = Field(default=None, ge=0.0)
    available: Optional[float] = Field(default=None, ge=0.0)
    analysis_tolerance: Optional[float] = Field(default=None, gt=0.0)

    @field_validator("oxides")
    @classmethod
    def _check_oxides(cls, v):
        if v is None:
            return v
        if not v:
            raise GlazeError("原料至少要给出一种氧化物质量分数", "empty_analysis")
        unknown = sorted(set(v) - set(OXIDE_CATALOG))
        if unknown:
            raise GlazeError(
                f"未知氧化物: {', '.join(unknown)}",
                "unknown_oxide",
                {"unknown": unknown},
            )
        negatives = {k: x for k, x in v.items() if x < 0.0}
        if negatives:
            raise GlazeError(
                f"氧化物质量分数不得为负: {negatives}",
                "negative_component",
                {"components": negatives},
            )
        return v


class Material(MaterialBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# 直接计算
# ---------------------------------------------------------------------------

class BatchItem(BaseModel):
    material_id: int
    amount: float = Field(..., ge=0.0, description="投料量（kg），不允许为负")


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1)


# ---------------------------------------------------------------------------
# 搜索 / 重配
# ---------------------------------------------------------------------------

class OxideTarget(BaseModel):
    low: Optional[float] = Field(default=None, ge=0.0, description="釉式下限")
    high: Optional[float] = Field(default=None, ge=0.0, description="釉式上限")
    weight: float = Field(default=1.0, gt=0.0, description="偏差权重")

    @model_validator(mode="after")
    def _check_bounds(self) -> "OxideTarget":
        if self.low is None and self.high is None:
            raise GlazeError(
                "目标釉式至少要给出 low 或 high 之一", "empty_target_bounds"
            )
        if self.low is not None and self.high is not None and self.low > self.high:
            raise GlazeError(
                f"目标区间矛盾: low={self.low} > high={self.high}",
                "contradictory_bounds",
                {"low": self.low, "high": self.high},
            )
        return self


class MaterialLimit(BaseModel):
    low: Optional[float] = Field(default=None, ge=0.0, description="投料量下限（kg）")
    high: Optional[float] = Field(default=None, ge=0.0, description="投料量上限（kg）")

    @model_validator(mode="after")
    def _check_bounds(self) -> "MaterialLimit":
        if (
            self.low is not None
            and self.high is not None
            and self.low > self.high + 1e-12
        ):
            raise GlazeError(
                f"单项用量约束矛盾: low={self.low} > high={self.high}",
                "contradictory_material_bounds",
                {"low": self.low, "high": self.high},
            )
        return self


class SearchRequest(BaseModel):
    batch_size: float = Field(..., gt=0.0, description="目标投料批量（kg，生料）")
    batch_tolerance: Optional[float] = Field(
        default=None, ge=0.0, description="批量允许偏差（kg），默认取 step"
    )
    step: float = Field(
        0.01, gt=0.0, description="投料步进精度（kg），各用量为其整数倍"
    )
    targets: dict[str, OxideTarget] = Field(
        default_factory=dict, description="氧化物 -> 釉式目标区间"
    )
    required: list[int] = Field(default_factory=list, description="必用原料 id")
    forbidden: list[int] = Field(default_factory=list, description="禁用原料 id")
    limits: dict[int, MaterialLimit] = Field(
        default_factory=dict, description="单项原料用量上下限"
    )
    locked: dict[int, float] = Field(
        default_factory=dict, description="锁定用量（kg），必须为 step 的整数倍"
    )

    @model_validator(mode="after")
    def _check_request(self) -> "SearchRequest":
        unknown = sorted(set(self.targets) - set(OXIDE_CATALOG))
        if unknown:
            raise GlazeError(
                f"目标中存在未知氧化物: {', '.join(unknown)}",
                "unknown_oxide",
                {"unknown": unknown},
            )
        overlap = sorted(set(self.required) & set(self.forbidden))
        if overlap:
            raise GlazeError(
                f"原料同时被必用与禁用: {overlap}",
                "required_forbidden_conflict",
                {"material_ids": overlap},
            )
        for mid, amount in self.locked.items():
            if amount < 0:
                raise GlazeError(
                    f"锁定用量不得为负: material {mid} = {amount}",
                    "negative_component",
                )
            n = amount / self.step
            if abs(n - round(n)) > 1e-6:
                raise GlazeError(
                    f"锁定用量 {amount} 不是步进 {self.step} 的整数倍",
                    "lock_not_on_grid",
                    {"material_id": mid, "amount": amount, "step": self.step},
                )
        for mid, lim in self.limits.items():
            if mid in self.forbidden:
                if (lim.low or 0.0) > 0.0 or (lim.high is not None and lim.high > 0.0):
                    raise GlazeError(
                        f"禁用原料 {mid} 又被赋予正用量约束",
                        "forbidden_with_positive_limit",
                    )
            if mid in self.locked:
                amount = self.locked[mid]
                if lim.low is not None and amount < lim.low - 1e-9:
                    raise GlazeError(
                        f"锁定用量 {amount} 低于下限 {lim.low}（原料 {mid}）",
                        "lock_violates_limit",
                    )
                if lim.high is not None and amount > lim.high + 1e-9:
                    raise GlazeError(
                        f"锁定用量 {amount} 高于上限 {lim.high}（原料 {mid}）",
                        "lock_violates_limit",
                    )
        return self


# ---------------------------------------------------------------------------
# 版本冻结
# ---------------------------------------------------------------------------

class FreezeItem(BaseModel):
    material_id: int
    amount: float = Field(..., ge=0.0)


class FreezeRequest(BaseModel):
    items: list[FreezeItem] = Field(min_length=1)
    note: Optional[str] = None
