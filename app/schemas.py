"""Pydantic 数据模型：原料、计算请求、搜索请求、响应结构。"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_STUDY_SEED,
    MAX_RESAMPLES,
    MIN_RESAMPLES,
    OXIDE_CATALOG,
)
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
    """原料更新：所有字段可选，但显式传 null 视为无效更新（422）。

    字段缺省表示"不修改"；显式 ``null`` 不被接受，避免误把必填列
    写成 NULL 导致数据库错误。
    """

    name: Optional[str] = Field(default=None, min_length=1)
    oxides: Optional[dict[str, float]] = None
    loi: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    price: Optional[float] = Field(default=None, ge=0.0)
    available: Optional[float] = Field(default=None, ge=0.0)
    analysis_tolerance: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _reject_explicit_null(self) -> "MaterialUpdate":
        null_fields = [
            name for name in self.model_fields_set
            if getattr(self, name) is None
        ]
        if null_fields:
            raise GlazeError(
                f"以下字段不允许显式置为 null（缺省即不修改）: {null_fields}",
                "null_update_field",
                {"fields": null_fields},
            )
        return self

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
# 原料含水测定（不可变版本）
# ---------------------------------------------------------------------------

class MoistureMeasurementCreate(BaseModel):
    """单批原料的含水测定：取样质量、烘干后质量与生效区间。

    湿基含水率由服务端按 ``(取样 - 烘干)/取样`` 计算，调用方不得直接给值。
    """

    lot: str = Field(..., min_length=1, max_length=64, description="适用批号")
    sample_mass_g: float = Field(..., gt=0.0, description="取样（湿料）质量（g）")
    dried_mass_g: float = Field(..., gt=0.0, description="烘干后质量（g）")
    valid_from: date = Field(..., description="生效起始日（含）")
    valid_to: date = Field(..., description="生效截止日（含）")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_measurement(self) -> "MoistureMeasurementCreate":
        if not self.lot.strip():
            raise GlazeError(
                "适用批号不得为空白",
                "moisture_lot_missing",
            )
        self.lot = self.lot.strip()
        if self.dried_mass_g > self.sample_mass_g + 1e-12:
            raise GlazeError(
                f"烘干后质量 {self.dried_mass_g} g 大于取样质量 "
                f"{self.sample_mass_g} g，质量倒置",
                "moisture_mass_inverted",
                {
                    "sample_mass_g": self.sample_mass_g,
                    "dried_mass_g": self.dried_mass_g,
                },
            )
        if self.valid_to < self.valid_from:
            raise GlazeError(
                f"生效区间倒置: {self.valid_from} 晚于 {self.valid_to}",
                "moisture_interval_inverted",
                {"valid_from": str(self.valid_from), "valid_to": str(self.valid_to)},
            )
        return self


# ---------------------------------------------------------------------------
# 直接计算
# ---------------------------------------------------------------------------

class BatchItem(BaseModel):
    material_id: int
    amount: float = Field(..., ge=0.0, description="干料投料量（kg），不允许为负")


def _check_lots(lots: dict[int, str]) -> dict[int, str]:
    blank = sorted(mid for mid, lot in lots.items() if not lot or not lot.strip())
    if blank:
        raise GlazeError(
            f"以下原料的批号为空: {blank}，批号缺失时应省略该原料而非给空串",
            "moisture_lot_missing",
            {"material_ids": blank},
        )
    return {mid: lot.strip() for mid, lot in lots.items()}


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1)
    lots: dict[int, str] = Field(
        default_factory=dict,
        description="各原料选用的到货批号；未列出的原料按干料称量并显式标注",
    )
    moisture_as_of: Optional[date] = Field(
        default=None, description="含水测定生效基准日（默认今天）"
    )

    @model_validator(mode="after")
    def _check_request(self) -> "BatchRequest":
        self.lots = _check_lots(self.lots)
        item_ids = {i.material_id for i in self.items}
        outsiders = sorted(set(self.lots) - item_ids)
        if outsiders:
            raise GlazeError(
                f"批号选择引用了投料表之外的原料: {outsiders}",
                "moisture_lot_not_in_items",
                {"material_ids": outsiders},
            )
        return self


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
    lots: dict[int, str] = Field(
        default_factory=dict,
        description="各原料选用的到货批号；未列出或无有效测定的原料按干料称量并标注",
    )
    moisture_as_of: Optional[date] = Field(
        default=None, description="含水测定生效基准日（默认今天）"
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
        blank = sorted(mid for mid, lot in self.lots.items() if not lot or not lot.strip())
        if blank:
            raise GlazeError(
                f"以下原料的批号为空: {blank}，批号缺失时应省略该原料而非给空串",
                "moisture_lot_missing",
                {"material_ids": blank},
            )
        self.lots = {mid: lot.strip() for mid, lot in self.lots.items()}
        overlap = sorted(set(self.required) & set(self.forbidden))
        if overlap:
            raise GlazeError(
                f"原料同时被必用与禁用: {overlap}",
                "required_forbidden_conflict",
                {"material_ids": overlap},
            )
        forbidden_locked = sorted(set(self.forbidden) & set(self.locked))
        if forbidden_locked:
            raise GlazeError(
                f"原料同时被禁用与锁定用量: {forbidden_locked}",
                "forbidden_locked_conflict",
                {"material_ids": forbidden_locked},
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
    lots: Optional[dict[int, str]] = Field(
        default=None,
        description="各原料选用的到货批号；给出时把湿料称量方案一并冻结进版本",
    )
    moisture_as_of: Optional[date] = Field(
        default=None, description="含水测定生效基准日（默认今天）"
    )

    @model_validator(mode="after")
    def _check_request(self) -> "FreezeRequest":
        if self.lots is None:
            return self
        blank = sorted(
            mid for mid, lot in self.lots.items() if not lot or not lot.strip()
        )
        if blank:
            raise GlazeError(
                f"以下原料的批号为空: {blank}，批号缺失时应省略该原料而非给空串",
                "moisture_lot_missing",
                {"material_ids": blank},
            )
        self.lots = {mid: lot.strip() for mid, lot in self.lots.items()}
        item_ids = {i.material_id for i in self.items}
        outsiders = sorted(set(self.lots) - item_ids)
        if outsiders:
            raise GlazeError(
                f"批号选择引用了投料表之外的原料: {outsiders}",
                "moisture_lot_not_in_items",
                {"material_ids": outsiders},
            )
        return self


# ---------------------------------------------------------------------------
# 原料批次波动研究
# ---------------------------------------------------------------------------

class AssayBatch(BaseModel):
    """单批化验数据：氧化物分析 + LOI + 价格 + 可用量。"""

    material_id: int
    batch: str = Field(min_length=1, max_length=64, description="批号")
    oxides: dict[str, float] = Field(
        description="氧化物 -> 质量分数（百分数口径，100 g 基准下的克数）"
    )
    loi: float = Field(0.0, ge=0.0, le=100.0, description="灼烧减量 LOI（百分数）")
    price: float = Field(..., ge=0.0, description="每千克单价")
    available: float = Field(..., ge=0.0, description="该批可用量（kg）")
    analysis_tolerance: float = Field(
        2.0, gt=0.0, description="分析合计允许的绝对误差（百分点）"
    )

    @field_validator("oxides")
    @classmethod
    def _check_oxides(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise GlazeError("批次分析至少要给出一种氧化物质量分数", "empty_analysis")
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
    def _check_analysis_sum(self) -> "AssayBatch":
        total = sum(self.oxides.values()) + self.loi
        if abs(total - 100.0) > self.analysis_tolerance + 1e-9:
            raise GlazeError(
                f"批次「{self.batch}」分析合计 {total:.3f} 超出 "
                f"100 ± {self.analysis_tolerance} 的允许误差",
                "analysis_sum_error",
                {"total": round(total, 4), "allowed": 100.0,
                 "tol": self.analysis_tolerance},
            )
        return self


class StudyRequest(BaseModel):
    """创建批次波动研究：一份冻结配方 + 一组化验批次构成独立版本。"""

    version_id: str = Field(min_length=1, description="来源冻结配方版本 id")
    batches: list[AssayBatch] = Field(
        min_length=1, description="配方所用原料的多批化验数据"
    )
    linked_groups: list[list[int]] = Field(
        default_factory=list,
        description="同批联动抽样原料组：组内各原料批号集合须一致，抽样时共用批号",
    )
    targets: dict[str, OxideTarget] = Field(
        default_factory=dict,
        description="釉式目标区间；缺省沿用来源版本约束中的 targets",
    )
    n_resamples: int = Field(
        DEFAULT_N_RESAMPLES, ge=MIN_RESAMPLES, le=MAX_RESAMPLES,
        description="bootstrap 重采样次数",
    )
    seed: int = Field(DEFAULT_STUDY_SEED, ge=0, description="重采样随机种子")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "StudyRequest":
        unknown = sorted(set(self.targets) - set(OXIDE_CATALOG))
        if unknown:
            raise GlazeError(
                f"目标中存在未知氧化物: {', '.join(unknown)}",
                "unknown_oxide",
                {"unknown": unknown},
            )
        seen_batches: set[tuple[int, str]] = set()
        for b in self.batches:
            key = (b.material_id, b.batch)
            if key in seen_batches:
                raise GlazeError(
                    f"原料 {b.material_id} 的批号「{b.batch}」重复提交",
                    "duplicate_batch",
                    {"material_id": b.material_id, "batch": b.batch},
                )
            seen_batches.add(key)
        membership: dict[int, bool] = {}
        for group in self.linked_groups:
            if len(group) < 2:
                raise GlazeError(
                    "联动抽样组至少需要两种原料",
                    "linked_group_conflict",
                    {"group": group},
                )
            if len(set(group)) != len(group):
                raise GlazeError(
                    f"联动组内原料重复: {group}",
                    "linked_group_conflict",
                    {"group": group},
                )
            for mid in group:
                if mid in membership:
                    raise GlazeError(
                        f"原料 {mid} 出现在多个联动组中",
                        "linked_group_conflict",
                        {"material_id": mid},
                    )
                membership[mid] = True
        return self


class RobustSearchRequest(BaseModel):
    """稳健配方搜索：在库存、步进与最大改动量内调整来源配方用量。"""

    max_change: float = Field(
        ..., gt=0.0,
        description="相对来源配方的最大总改动量（kg，各原料 |新-旧| 之和）",
    )
    step: float = Field(
        0.5, gt=0.0, description="用量调整步进（kg），改动量为步进整数倍"
    )
    locked: list[int] = Field(
        default_factory=list,
        description="锁定原料 id（保持来源配方用量不变）",
    )
    n_resamples: Optional[int] = Field(
        default=None, ge=MIN_RESAMPLES, le=MAX_RESAMPLES,
        description="重采样次数，缺省沿用研究设定",
    )
    seed: Optional[int] = Field(
        default=None, ge=0, description="随机种子，缺省沿用研究种子"
    )
    max_candidates: int = Field(5, ge=1, le=20, description="返回候选条数上限")

    @model_validator(mode="after")
    def _check_request(self) -> "RobustSearchRequest":
        if len(set(self.locked)) != len(self.locked):
            raise GlazeError(
                f"锁定原料重复: {self.locked}",
                "duplicate_material",
                {"material_ids": self.locked},
            )
        return self


class RobustFreezeRequest(BaseModel):
    """冻结稳健搜索选定结果：来源配方、化验数据、抽样规则与种子整体存档。"""

    items: list[FreezeItem] = Field(min_length=1)
    note: Optional[str] = None
    n_resamples: Optional[int] = Field(
        default=None, ge=MIN_RESAMPLES, le=MAX_RESAMPLES,
        description="重采样次数，缺省沿用研究设定",
    )
    seed: Optional[int] = Field(
        default=None, ge=0, description="随机种子，缺省沿用研究种子"
    )
    search_constraints: Optional[dict] = Field(
        default=None, description="稳健搜索约束，原样存档"
    )

    @model_validator(mode="after")
    def _check_request(self) -> "RobustFreezeRequest":
        ids = [i.material_id for i in self.items]
        if len(set(ids)) != len(ids):
            raise GlazeError(
                "投料表中原料重复", "duplicate_material", {"material_ids": ids}
            )
        return self


# ---------------------------------------------------------------------------
# 配方混合试验
# ---------------------------------------------------------------------------

BlendMode = Literal["linear", "ternary"]


class BlendCell(BaseModel):
    """试片布局中的一个格位：位置标签 + 各来源配方比例。"""

    position: str = Field(min_length=1, max_length=64, description="格位标签，如 A1")
    weights: list[float] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="各来源配方占比（合计为 1），顺序与 sources 一致",
    )

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, v: list[float]) -> list[float]:
        negatives = [w for w in v if w < 0.0]
        if negatives:
            raise GlazeError(
                f"混合比例不得为负: {v}",
                "negative_component",
                {"weights": v},
            )
        return v


class BlendExperimentRequest(BaseModel):
    """创建配方混合试验：2~3 份冻结配方 + 一张试片布局。"""

    sources: list[str] = Field(
        ..., min_length=2, max_length=3,
        description="来源冻结配方版本 id（线性 2 份 / 三元 3 份，不可重复）",
    )
    mode: BlendMode = Field(..., description="linear 线性混合 / ternary 三元三角")
    cells: list[BlendCell] = Field(
        ..., min_length=1, description="试片布局格位，位置不可重复"
    )
    ratio_low: float = Field(
        0.0, ge=0.0, le=1.0, description="单一来源占比下限（含端点）"
    )
    ratio_high: float = Field(
        1.0, ge=0.0, le=1.0, description="单一来源占比上限（含端点）"
    )
    ratio_step: Optional[float] = Field(
        default=None, gt=0.0, le=1.0,
        description="比例步长（给出时每个非零占比须落在其网格上）",
    )
    dry_mass_per_tile_g: float = Field(
        ..., gt=0.0, description="单片干料量（g，生料）"
    )
    scale_division_g: float = Field(
        ..., gt=0.0, description="电子秤分度（g/最小读数）"
    )
    minimum_weighed_g: float = Field(
        ..., gt=0.0, description="电子秤最小称量（g，低于该值不称量）"
    )
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "BlendExperimentRequest":
        n = len(self.sources)
        if self.mode == "linear" and n != 2:
            raise GlazeError(
                f"线性混合需要恰好 2 份来源配方，收到 {n} 份",
                "mode_source_count_mismatch",
                {"mode": self.mode, "n_sources": n, "expected": 2},
            )
        if self.mode == "ternary" and n != 3:
            raise GlazeError(
                f"三元三角混合需要恰好 3 份来源配方，收到 {n} 份",
                "mode_source_count_mismatch",
                {"mode": self.mode, "n_sources": n, "expected": 3},
            )
        if len(set(self.sources)) != n:
            dup = sorted({s for s in self.sources if self.sources.count(s) > 1})
            raise GlazeError(
                f"来源配方重复提交: {dup}",
                "duplicate_source",
                {"version_ids": dup},
            )
        if self.ratio_low > self.ratio_high:
            raise GlazeError(
                f"比例区间矛盾: low={self.ratio_low} > high={self.ratio_high}",
                "contradictory_bounds",
                {"low": self.ratio_low, "high": self.ratio_high},
            )
        if self.minimum_weighed_g < self.scale_division_g:
            raise GlazeError(
                "最小称量不得小于电子秤分度",
                "minimum_below_division",
                {
                    "minimum_weighed_g": self.minimum_weighed_g,
                    "scale_division_g": self.scale_division_g,
                },
            )
        # 单片干料量须落在秤的网格上（保证最大余数法可精确闭合）
        q = self.dry_mass_per_tile_g / self.scale_division_g
        if abs(q - round(q)) > 1e-6:
            raise GlazeError(
                f"单片干料量 {self.dry_mass_per_tile_g} g 不是分度 "
                f"{self.scale_division_g} g 的整数倍",
                "dry_mass_not_on_grid",
                {
                    "dry_mass_per_tile_g": self.dry_mass_per_tile_g,
                    "scale_division_g": self.scale_division_g,
                },
            )

        seen_positions: set[str] = set()
        for cell in self.cells:
            if cell.position in seen_positions:
                raise GlazeError(
                    f"格位标签重复: {cell.position}",
                    "duplicate_cell_position",
                    {"position": cell.position},
                )
            seen_positions.add(cell.position)
            if len(cell.weights) != n:
                raise GlazeError(
                    f"格位 {cell.position} 的比例个数 {len(cell.weights)} "
                    f"与来源数 {n} 不一致",
                    "weights_length_mismatch",
                    {"position": cell.position,
                     "n_weights": len(cell.weights), "expected": n},
                )
            total = sum(cell.weights)
            if abs(total - 1.0) > 1e-6:
                raise GlazeError(
                    f"格位 {cell.position} 比例不闭合：合计 {total}，应为 1",
                    "weights_not_closed",
                    {"position": cell.position, "sum": round(total, 9)},
                )
            for k, w in enumerate(cell.weights):
                if w < self.ratio_low - 1e-9 or w > self.ratio_high + 1e-9:
                    raise GlazeError(
                        f"格位 {cell.position} 第 {k + 1} 份占比 {w} 超出 "
                        f"[{self.ratio_low}, {self.ratio_high}]",
                        "ratio_out_of_range",
                        {"position": cell.position, "source_index": k,
                         "weight": w, "low": self.ratio_low,
                         "high": self.ratio_high},
                    )
            if self.ratio_step is not None:
                for k, w in enumerate(cell.weights):
                    qw = w / self.ratio_step
                    if abs(qw - round(qw)) > 1e-6:
                        raise GlazeError(
                            f"格位 {cell.position} 第 {k + 1} 份占比 {w} "
                            f"不是步长 {self.ratio_step} 的整数倍",
                            "ratio_not_on_grid",
                            {"position": cell.position, "source_index": k,
                             "weight": w, "step": self.ratio_step},
                        )
        return self


class MasterPlanRequest(BaseModel):
    """母料拆分搜索：限定母料批量、最小称量与允许剩余量。"""

    master_batch_g: Optional[float] = Field(
        default=None, gt=0.0,
        description="母料批量（g），须为秤分度整数倍；缺省时按等分需求精确配制",
    )
    master_minimum_weighed_g: Optional[float] = Field(
        default=None, gt=0.0,
        description="母料配料的最小称量（g），缺省沿用试验逐格秤设置",
    )
    allowed_leftover_g: Optional[float] = Field(
        default=None, ge=0.0,
        description="允许剩余料上限（g），仅在限定母料批量时生效，缺省为 0",
    )
    max_candidates: int = Field(
        default=10, ge=1, le=50, description="返回方案条数上限"
    )

    @model_validator(mode="after")
    def _check_request(self) -> "MasterPlanRequest":
        if (
            self.allowed_leftover_g is not None
            and self.master_batch_g is None
        ):
            raise GlazeError(
                "未限定母料批量时不允许设置剩余料（等分配制剩余恒为 0）",
                "leftover_without_batch",
            )
        return self


class BlendFreezeItem(BaseModel):
    """冻结方案中的母料配料项。"""

    material_id: int
    units: int = Field(..., ge=0, description="配料单位数（× 分度为克数）")


class BlendFreezeRequest(BaseModel):
    """冻结选定的母料拆分方案（来源配方、布局、母料拆分与计算常量整体存档）。"""

    plan_index: int = Field(
        ..., ge=0, description="母料搜索返回的候选序号（0 为 best）"
    )
    master_batch_g: Optional[float] = Field(default=None, gt=0.0)
    master_minimum_weighed_g: Optional[float] = Field(default=None, gt=0.0)
    allowed_leftover_g: Optional[float] = Field(default=None, ge=0.0)
    max_candidates: int = Field(default=10, ge=1, le=50)
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "BlendFreezeRequest":
        if (
            self.allowed_leftover_g is not None
            and self.master_batch_g is None
        ):
            raise GlazeError(
                "未限定母料批量时不允许设置剩余料（等分配制剩余恒为 0）",
                "leftover_without_batch",
            )
        return self


# ---------------------------------------------------------------------------
# 釉浆调制批次
# ---------------------------------------------------------------------------

class SlurryBatchCreate(BaseModel):
    """从一份冻结配方创建釉浆调制批次：目标参数与调制常量。"""

    version_id: str = Field(min_length=1, description="来源冻结配方版本 id")
    target_dry_mass_kg: float = Field(..., gt=0.0, description="目标干料量（kg，生料）")
    solids_low: float = Field(
        ..., gt=0.0, lt=1.0, description="目标固含率下限（质量分数，0~1）"
    )
    solids_high: float = Field(
        ..., gt=0.0, lt=1.0, description="目标固含率上限（质量分数，0~1）"
    )
    density_low: float = Field(
        ..., gt=0.0, description="目标比重下限（相对密度，g/mL 数值）"
    )
    density_high: float = Field(
        ..., gt=0.0, description="目标比重上限（相对密度，g/mL 数值）"
    )
    powder_true_density_kg_l: float = Field(
        ..., gt=0.0, description="粉体真密度（kg/L，数值等同 g/mL）"
    )
    water_temp_c: float = Field(..., description="调制水温（°C，0~100）")
    container_capacity_l: float = Field(
        ..., gt=0.0, description="调制容器容量（L）"
    )
    additive_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="添加剂占干料的质量比例（初始称量用，0~1）",
    )
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "SlurryBatchCreate":
        if self.solids_low > self.solids_high:
            raise GlazeError(
                f"固含率区间矛盾: low={self.solids_low} > high={self.solids_high}",
                "contradictory_bounds",
                {"low": self.solids_low, "high": self.solids_high},
            )
        if self.density_low > self.density_high:
            raise GlazeError(
                f"比重区间矛盾: low={self.density_low} > high={self.density_high}",
                "contradictory_bounds",
                {"low": self.density_low, "high": self.density_high},
            )
        if not 0.0 <= self.water_temp_c <= 100.0:
            raise GlazeError(
                f"水温 {self.water_temp_c} °C 超出支持范围 0~100 °C",
                "water_temp_out_of_range",
                {"water_temp_c": self.water_temp_c},
            )
        return self


class SlurryDryAddition(BaseModel):
    """逐笔直接登记的单项干料实际加入量。"""

    material_id: int
    mass_g: float = Field(..., gt=0.0, description="实际加入质量（g）")


class SlurryAdditionRequest(BaseModel):
    """登记一笔实际投料：干料（可多项）、水、添加剂至少一种为正。

    回收浆与预混粉通过各自字段单独登记，互不混用。
    """

    dry_materials: list[SlurryDryAddition] = Field(default_factory=list)
    water_g: float = Field(0.0, ge=0.0, description="实际加入水质量（g）")
    additive_g: float = Field(0.0, ge=0.0, description="实际加入添加剂质量（g）")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "SlurryAdditionRequest":
        ids = [d.material_id for d in self.dry_materials]
        if len(set(ids)) != len(ids):
            raise GlazeError(
                "同一笔投料中干料原料重复，请合并后登记",
                "duplicate_material",
                {"material_ids": ids},
            )
        total = sum(d.mass_g for d in self.dry_materials)
        total += self.water_g + self.additive_g
        if total <= 0.0:
            raise GlazeError(
                "空投料登记：干料、水、添加剂至少需要一项为正",
                "empty_addition",
            )
        return self


class SlurryRecycleRequest(BaseModel):
    """登记一笔同配方回收浆：来源必须为同版本已定稿批次。"""

    source_batch_id: str = Field(min_length=1, description="回收浆来源批次 id")
    slurry_mass_g: float = Field(..., gt=0.0, description="回收浆总质量（g）")
    note: Optional[str] = None


class SlurryPremixRequest(BaseModel):
    """登记一笔同配方预混粉（按冻结配方比例预配的干料）。"""

    premix_mass_g: float = Field(..., gt=0.0, description="预混粉总质量（g）")
    water_g: float = Field(
        0.0, ge=0.0, description="随预混粉同时加入的水质量（g）"
    )
    note: Optional[str] = None


class SlurryReadingRequest(BaseModel):
    """比重杯读数：空杯质量、满杯质量与杯容积。"""

    empty_cup_mass_g: float = Field(..., ge=0.0, description="空杯质量（g）")
    full_cup_mass_g: float = Field(..., ge=0.0, description="满杯质量（g）")
    cup_volume_ml: float = Field(..., gt=0.0, description="杯容积（mL）")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "SlurryReadingRequest":
        if self.full_cup_mass_g < self.empty_cup_mass_g:
            raise GlazeError(
                f"满杯质量 {self.full_cup_mass_g} g 小于空杯质量 "
                f"{self.empty_cup_mass_g} g，读数不闭合",
                "cup_reading_inverted",
                {
                    "empty_cup_mass_g": self.empty_cup_mass_g,
                    "full_cup_mass_g": self.full_cup_mass_g,
                },
            )
        return self


class SlurryCorrectionRequest(BaseModel):
    """纠偏方案搜索：限定加水/预混粉步进与剩余容量。"""

    water_step_g: float = Field(
        10.0, gt=0.0, description="加水步进（g），方案水量为其整数倍"
    )
    premix_step_g: float = Field(
        10.0, gt=0.0, description="同配方预混粉步进（g），粉量为其整数倍"
    )
    remaining_capacity_ml: Optional[float] = Field(
        default=None, ge=0.0,
        description="调用方限定的剩余容量（mL），缺省取容器实际空余",
    )
    max_candidates: int = Field(20, ge=1, le=100, description="返回方案条数上限")


class SlurryAdjustmentRequest(BaseModel):
    """登记一笔已执行的纠偏调整：加水与/或加同配方预混粉。"""

    water_g: float = Field(0.0, ge=0.0, description="实际补加水质量（g）")
    premix_g: float = Field(
        0.0, ge=0.0, description="实际补加同配方预混粉质量（g）"
    )
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_request(self) -> "SlurryAdjustmentRequest":
        if self.water_g + self.premix_g <= 0.0:
            raise GlazeError(
                "空调整记录：补加水与预混粉至少需要一项为正",
                "empty_adjustment",
            )
        return self


class SlurryFinalizeRequest(BaseModel):
    """定稿请求（可选备注）；定稿幂等，重复提交返回同一冻结结果。"""

    note: Optional[str] = None
