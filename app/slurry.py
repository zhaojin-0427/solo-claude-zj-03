"""釉浆调制：质量守恒、比重杯读数闭合与纠偏方案网格搜索。

口径与单位
----------
* 台账（投料、读数、调整）全部以 **g / mL** 计；冻结配方投料量为 kg，
  创建批次时按 1000 倍换算。密度单位 g/mL 与 kg/L 数值相同，理论比重
  （相对密度）直接取 g/mL 数值。
* 釉浆三相质量守恒：

    - 干物 ``dry``      ：直接投入的粉体 + 同配方预混粉 + 回收浆中的干物
    - 水分 ``water``    ：直接加入的水 + 回收浆中的水
    - 添加剂 ``additive``：直接加入的添加剂 + 回收浆中的添加剂（占干料比例
      仅用于创建时的初始称量，过程中逐笔登记实际值）

* 固含率（质量分数）::

      S = dry / (dry + water + additive)

* 体积可加（添加剂视为溶于液相，体积按水温下水的密度折算）::

      V = dry / ρ_powder + (water + additive) / ρ_water

  理论比重 ρ = (dry + water + additive) / V，介于 ρ_water 与 ρ_powder 之间。
* 回收浆来自**同一冻结配方版本**已定稿的调制批次，按其定稿时的
  dry / water / additive 质量构成把回收浆总质量拆成三相。

比重杯
------
空杯质量 m0、满杯质量 m1、杯容积 Vc：

    实测比重 ρ_m = (m1 - m0) / Vc

与质量守恒给出的理论比重之差不超过 ``DENSITY_CLOSURE_TOLERANCE_G_ML``
视为读数闭合。仅凭比重也可反推固含率（忽略添加剂）::

    1/ρ_m = S/ρ_p + (1-S)/ρ_w
    S     = (1/ρ_w - 1/ρ_m) / (1/ρ_w - 1/ρ_p)

纠偏搜索
--------
只允许两种动作：加水 x（降低固含率与比重）、加同配方预混粉 y（同时升高）。
两者按调用方给定步进取整数倍，体积增量不得超过剩余容量。枚举网格后，
固含率与比重同时落入目标区间的方案按（固含率偏差, 比重偏差, 新增质量）
字典序排序；无可行方案时返回字典序最优的 best_effort 供参考。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import CORRECTION_MAX_GRID_COMBOS
from .errors import GlazeError

GRID_EPS = 1e-7

# 纠偏网格枚举的组合数上限：步进相对容量过细时直接拒绝，避免海量枚举
MAX_GRID_COMBOS = CORRECTION_MAX_GRID_COMBOS

# 标准大气压下淡水密度表（°C -> g/mL），区间内线性插值。
# 数值取自 ITS-90 常用淡水密度表（4 °C 附近密度最大）。
WATER_DENSITY_TABLE: tuple[tuple[float, float], ...] = (
    (0.0, 0.99984),
    (4.0, 1.00000),
    (10.0, 0.99970),
    (15.0, 0.99910),
    (20.0, 0.99821),
    (25.0, 0.99705),
    (30.0, 0.99565),
    (35.0, 0.99403),
    (40.0, 0.99222),
    (45.0, 0.99022),
    (50.0, 0.98803),
    (55.0, 0.98570),
    (60.0, 0.98320),
    (65.0, 0.98056),
    (70.0, 0.97778),
    (75.0, 0.97486),
    (80.0, 0.97180),
    (85.0, 0.96862),
    (90.0, 0.96531),
    (95.0, 0.96189),
    (100.0, 0.95840),
)


def water_density(temp_c: float) -> float:
    """按水温线性插值返回淡水密度（g/mL），温度须在 0~100 °C。"""
    if not 0.0 <= temp_c <= 100.0:
        raise GlazeError(
            f"水温 {temp_c} °C 超出支持范围 0~100 °C",
            "water_temp_out_of_range",
            {"water_temp_c": temp_c},
        )
    table = WATER_DENSITY_TABLE
    if temp_c <= table[0][0]:
        return table[0][1]
    if temp_c >= table[-1][0]:
        return table[-1][1]
    for (t0, d0), (t1, d1) in zip(table, table[1:]):
        if t0 <= temp_c <= t1:
            ratio = (temp_c - t0) / (t1 - t0)
            return d0 + (d1 - d0) * ratio
    return table[-1][1]  # 不可达


# ---------------------------------------------------------------------------
# 质量守恒状态
# ---------------------------------------------------------------------------


@dataclass
class SlurryState:
    """调制中釉浆的三相质量累计（g）。

    各相按来源再细分：direct 为直接投料、premix 为同配方预混粉调整、
    recycle 为回收浆带入；水分/添加剂无预混口径，premix 列恒为 0。
    """

    dry_direct_g: float = 0.0
    dry_premix_g: float = 0.0
    dry_recycle_g: float = 0.0
    water_direct_g: float = 0.0
    water_recycle_g: float = 0.0
    additive_direct_g: float = 0.0
    additive_recycle_g: float = 0.0

    @property
    def dry_g(self) -> float:
        return self.dry_direct_g + self.dry_premix_g + self.dry_recycle_g

    @property
    def water_g(self) -> float:
        return self.water_direct_g + self.water_recycle_g

    @property
    def additive_g(self) -> float:
        return self.additive_direct_g + self.additive_recycle_g

    @property
    def total_mass_g(self) -> float:
        return self.dry_g + self.water_g + self.additive_g

    def add_direct(
        self, dry_g: float = 0.0, water_g: float = 0.0, additive_g: float = 0.0
    ) -> None:
        self.dry_direct_g += dry_g
        self.water_direct_g += water_g
        self.additive_direct_g += additive_g

    def add_premix(self, premix_g: float, water_g: float = 0.0) -> None:
        self.dry_premix_g += premix_g
        self.water_direct_g += water_g

    def add_recycle(
        self, dry_g: float, water_g: float, additive_g: float
    ) -> None:
        self.dry_recycle_g += dry_g
        self.water_recycle_g += water_g
        self.additive_recycle_g += additive_g


def slurry_metrics(
    state: SlurryState, powder_density_g_ml: float, water_density_g_ml: float
) -> dict[str, float]:
    """计算当前总质量、固含率、占用体积与理论比重。"""
    dry = state.dry_g
    water = state.water_g
    additive = state.additive_g
    total_mass = dry + water + additive
    solids = dry / total_mass if total_mass > 0.0 else 0.0
    volume_ml = (
        dry / powder_density_g_ml
        + (water + additive) / water_density_g_ml
    )
    density = total_mass / volume_ml if volume_ml > 0.0 else 0.0
    return {
        "dry_mass_g": dry,
        "water_mass_g": water,
        "additive_mass_g": additive,
        "total_mass_g": total_mass,
        "solids_fraction": solids,
        "occupied_volume_ml": volume_ml,
        "theoretical_density_g_ml": density,
    }


def implied_solids_from_density(
    measured_density_g_ml: float,
    powder_density_g_ml: float,
    water_density_g_ml: float,
) -> Optional[float]:
    """由比重杯实测比重反推固含率；读数在物理区间外时返回 None。"""
    rho_m = measured_density_g_ml
    denom = 1.0 / water_density_g_ml - 1.0 / powder_density_g_ml
    if denom <= 0.0:
        return None
    solids = (1.0 / water_density_g_ml - 1.0 / rho_m) / denom
    if solids < -GRID_EPS or solids > 1.0 + GRID_EPS:
        return None
    return max(0.0, min(1.0, solids))


# ---------------------------------------------------------------------------
# 回收浆拆分
# ---------------------------------------------------------------------------


def split_recycle_mass(
    slurry_mass_g: float, source_totals: dict[str, float]
) -> dict[str, float]:
    """按来源批次定稿时的三相质量构成，拆分一笔回收浆总质量。"""
    dry = float(source_totals["dry_mass_g"])
    water = float(source_totals["water_mass_g"])
    additive = float(source_totals["additive_mass_g"])
    source_total = dry + water + additive
    if source_total <= 0.0:
        raise GlazeError(
            "回收浆来源批次总质量为零，无法按构成拆分",
            "empty_recycle_source",
        )
    scale = slurry_mass_g / source_total
    return {
        "dry_g": dry * scale,
        "water_g": water * scale,
        "additive_g": additive * scale,
        "solids_fraction": dry / source_total,
    }


def split_premix(premix_g: float, shares: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同配方预混粉总量按配方份额拆到各原料（最大余数闭合在台账之外，
    这里保留理论小数，实际称量以现场登记的干料投料为准）。"""
    return [
        {
            "material_id": int(s["material_id"]),
            "share": float(s["share"]),
            "mass_g": premix_g * float(s["share"]),
        }
        for s in shares
    ]


# ---------------------------------------------------------------------------
# 纠偏搜索
# ---------------------------------------------------------------------------


def _interval_deviation(value: float, low: float, high: float) -> float:
    """相对目标区间的绝对越出量；落在区间内（含边界）为 0。"""
    return max(low - value, 0.0, value - high)


def search_corrections(
    *,
    current: dict[str, float],
    solids_low: float,
    solids_high: float,
    density_low: float,
    density_high: float,
    powder_density_g_ml: float,
    water_density_g_ml: float,
    container_capacity_ml: float,
    water_step_g: float,
    premix_step_g: float,
    remaining_capacity_ml: Optional[float] = None,
    max_candidates: int = 20,
) -> dict[str, Any]:
    """枚举"加水 + 加预混粉"网格，返回落入目标区间的纠偏方案。

    ``current`` 须含 dry_mass_g / water_mass_g / additive_mass_g /
    occupied_volume_ml。剩余容量取容器实际空余与调用方限定值的较小者。
    """
    d0 = current["dry_mass_g"]
    w0 = current["water_mass_g"]
    a0 = current["additive_mass_g"]
    v0 = current["occupied_volume_ml"]
    rho_p = powder_density_g_ml
    rho_w = water_density_g_ml

    free_ml = max(container_capacity_ml - v0, 0.0)
    if remaining_capacity_ml is not None:
        free_ml = min(free_ml, max(remaining_capacity_ml, 0.0))

    # 各动作单独的体积占用：水 x -> x/ρ_w，干粉 y -> y/ρ_p
    n_water_max = int(free_ml * rho_w / water_step_g + GRID_EPS)
    n_premix_max = int(free_ml * rho_p / premix_step_g + GRID_EPS)
    n_combos = (n_water_max + 1) * (n_premix_max + 1)
    if n_combos > MAX_GRID_COMBOS:
        raise GlazeError(
            f"步进网格组合数 {n_combos} 超过上限 {MAX_GRID_COMBOS}，"
            "请加大加水/预混粉步进或收紧剩余容量",
            "correction_grid_too_fine",
            {
                "n_combos": n_combos,
                "water_step_g": water_step_g,
                "premix_step_g": premix_step_g,
                "remaining_capacity_ml": free_ml,
            },
        )

    feasible: list[dict[str, Any]] = []
    best_effort: Optional[dict[str, Any]] = None
    for nw in range(n_water_max + 1):
        added_water = nw * water_step_g
        for np_ in range(n_premix_max + 1):
            added_premix = np_ * premix_step_g
            added_volume = added_water / rho_w + added_premix / rho_p
            if added_volume > free_ml + GRID_EPS:
                break  # 预混粉继续增大只会更超量
            dry = d0 + added_premix
            water = w0 + added_water
            total = dry + water + a0
            volume = v0 + added_volume
            solids = dry / total if total > 0.0 else 0.0
            density = total / volume if volume > 0.0 else 0.0
            solids_dev = _interval_deviation(solids, solids_low, solids_high)
            density_dev = _interval_deviation(
                density, density_low, density_high
            )
            plan = {
                "water_g": round(added_water, 9),
                "premix_g": round(added_premix, 9),
                "added_mass_g": round(added_water + added_premix, 9),
                "added_volume_ml": round(added_volume, 9),
                "resulting_solids_fraction": round(solids, 9),
                "resulting_density_g_ml": round(density, 9),
                "resulting_volume_ml": round(volume, 9),
                "solids_deviation": round(solids_dev, 12),
                "density_deviation": round(density_dev, 12),
                "targets_met": solids_dev <= 0.0 and density_dev <= 0.0,
            }
            if plan["targets_met"]:
                feasible.append(plan)
            key = (solids_dev, density_dev, added_water + added_premix)
            if best_effort is None or key < (
                best_effort["solids_deviation"],
                best_effort["density_deviation"],
                best_effort["added_mass_g"],
            ):
                best_effort = plan

    feasible.sort(
        key=lambda p: (
            p["solids_deviation"],
            p["density_deviation"],
            p["added_mass_g"],
            p["water_g"],
            p["premix_g"],
        )
    )
    return {
        "status": "ok" if feasible else "target_unreachable",
        "remaining_capacity_ml": round(free_ml, 9),
        "n_combos": n_combos,
        "plans": feasible[:max_candidates],
        "best": feasible[0] if feasible else None,
        "best_effort": best_effort,
    }
