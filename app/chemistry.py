"""釉式化学计算。

质量分数口径（与原料记录一致）：``oxides`` 给出的是 100 g **生料** 折算到
烧后氧化物的克数（即烧后氧化物占生料的百分数），``loi`` 为灼烧减量百分数，
两者合计约 100。因此：

    氧化物质量(kg) = 投料量(kg) * 氧化物百分数 / 100
    烧后质量       = 投料量 * (1 - LOI/100)
    摩尔数         = 氧化物质量 / 摩尔质量
    Seger 釉式     = 各氧化物摩尔数 / 助熔氧化物摩尔数之和
"""
from __future__ import annotations

from typing import Sequence

from .config import MIN_FLUX_MOLES, OXIDE_CATALOG, SEGER_TOLERANCE
from .errors import GlazeError

# 百分数 <-> 小数
PCT = 100.0


def molwt(oxide: str) -> float:
    """返回氧化物摩尔质量，未知则抛错。"""
    try:
        return OXIDE_CATALOG[oxide].molwt
    except KeyError:
        raise GlazeError(f"未知氧化物: {oxide}", "unknown_oxide", {"oxide": oxide})


def oxide_role(oxide: str) -> str:
    return OXIDE_CATALOG[oxide].role


def oxides_by_role(role: str) -> list[str]:
    return [name for name, info in OXIDE_CATALOG.items() if info.role == role]


def validate_analysis(
    oxides: dict[str, float], loi: float, tol: float
) -> tuple[dict[str, float], float]:
    """校验原料分析：未知氧化物 / 负成分 / 合计超差。

    通过后返回归一化后的 ``(氧化物质量分数, LOI)``：
    两者按同一比例缩放到氧化物+LOI=100，消除小幅分析误差。
    """
    if not oxides:
        raise GlazeError("原料分析为空", "empty_analysis")
    unknown = sorted(set(oxides) - set(OXIDE_CATALOG))
    if unknown:
        raise GlazeError(
            f"未知氧化物: {', '.join(unknown)}",
            "unknown_oxide",
            {"unknown": unknown},
        )
    negatives = {k: v for k, v in oxides.items() if v < 0.0}
    if negatives:
        raise GlazeError(
            f"氧化物质量分数不得为负: {negatives}",
            "negative_component",
            {"components": negatives},
        )
    if loi < 0:
        raise GlazeError("灼烧减量不得为负", "negative_component", {"loi": loi})
    total = sum(oxides.values()) + loi
    if abs(total - 100.0) > tol + 1e-9:
        raise GlazeError(
            f"分析合计 {total:.3f} 超出 100 ± {tol} 的允许误差",
            "analysis_sum_error",
            {"total": round(total, 4), "allowed": 100.0, "tol": tol},
        )
    scale = 100.0 / total
    return ({k: v * scale for k, v in oxides.items()}, loi * scale)


def calc_batch(
    items: Sequence[tuple[int, str, dict[str, float], float, float, float]],
    targets: dict | None = None,
) -> dict:
    """按投料量计算完整结果。

    ``items`` 元组顺序为
    ``(material_id, name, oxides_pct, loi_pct, price_per_kg, amount_kg)``。

    返回 Seger 釉式、烧后氧化物质量比、摩尔数、烧失、成本、单项投料分解
    与各氧化物相对目标区间的偏差。
    """
    oxide_mass: dict[str, float] = {}
    batch_mass = 0.0
    fired_mass = 0.0
    cost = 0.0
    breakdown = []

    for mid, name, oxides_pct, loi, price, amount in items:
        if amount < -1e-12:
            raise GlazeError(
                f"投料量不得为负: 原料 {mid} = {amount}", "negative_component"
            )
        batch_mass += amount
        fired = amount * (1.0 - loi / PCT)
        fired_mass += fired
        cost += amount * price
        line_oxide_mass: dict[str, float] = {}
        for oxide, pct in oxides_pct.items():
            m = amount * pct / PCT
            oxide_mass[oxide] = oxide_mass.get(oxide, 0.0) + m
            line_oxide_mass[oxide] = line_oxide_mass.get(oxide, 0.0) + m
        breakdown.append(
            {
                "material_id": mid,
                "name": name,
                "amount": amount,
                "fired_mass": fired,
                "loss_on_ignition": amount - fired,
                "cost": amount * price,
                "oxide_mass": {k: round(v, 9) for k, v in line_oxide_mass.items()},
            }
        )

    # 摩尔数
    moles: dict[str, float] = {
        oxide: mass / OXIDE_CATALOG[oxide].molwt
        for oxide, mass in oxide_mass.items()
    }
    flux_oxides = oxides_by_role("flux")
    flux_moles = sum(moles.get(o, 0.0) for o in flux_oxides)

    if batch_mass <= 0.0:
        raise GlazeError("投料总量为零，无法计算釉式", "empty_batch")
    if flux_moles <= MIN_FLUX_MOLES:
        raise GlazeError(
            "助熔氧化物摩尔数接近零，无法按助熔归一化为 Seger 釉式",
            "no_flux",
            {"flux_moles": flux_moles},
        )

    seger = {oxide: n / flux_moles for oxide, n in moles.items()}
    # 烧后氧化物质量比（烧后总质量为 1）
    fired_mass_ratio = (
        {oxide: mass / fired_mass for oxide, mass in oxide_mass.items()}
        if fired_mass > 0
        else {}
    )

    deviations = _deviations(seger, targets or {})

    return {
        "batch_mass": batch_mass,
        "fired_mass": fired_mass,
        "loss_on_ignition": batch_mass - fired_mass,
        "loss_on_ignition_pct": (batch_mass - fired_mass) / batch_mass * PCT,
        "cost": cost,
        "oxide_mass": {k: round(v, 9) for k, v in oxide_mass.items()},
        "oxide_moles": {k: round(v, 12) for k, v in moles.items()},
        "flux_moles": flux_moles,
        "seger": {k: round(v, 9) for k, v in seger.items()},
        "fired_oxide_mass_ratio": {
            k: round(v, 9) for k, v in fired_mass_ratio.items()
        },
        "deviations": deviations,
        "breakdown": breakdown,
    }


def _deviations(seger: dict[str, float], targets: dict) -> dict:
    """每个目标氧化物的偏差量、是否越界及加权偏差。

    ``in_range`` 与越界计数采用 ``SEGER_TOLERANCE`` 容差，与优化器
    大 M / 越界量约束口径一致；但返回的 ``deviation`` 仍是相对
    区间边界的真实越出量（容差内计 0）。
    """
    out = {}
    for oxide, tgt in targets.items():
        low = getattr(tgt, "low", None)
        high = getattr(tgt, "high", None)
        weight = getattr(tgt, "weight", 1.0)
        val = seger.get(oxide, 0.0)
        if low is None:
            raw = max(0.0, val - high)
        elif high is None:
            raw = max(0.0, low - val)
        else:
            raw = max(low - val, 0.0, val - high)
        in_range = raw <= SEGER_TOLERANCE
        dev = 0.0 if in_range else raw
        out[oxide] = {
            "value": round(val, 9),
            "low": low,
            "high": high,
            "deviation": round(dev, 12),
            "raw_deviation": round(raw, 12),
            "weight": weight,
            "weighted_deviation": round(dev * weight, 12),
            "in_range": bool(in_range),
        }
    return out


def deviation_summary(result: dict, targets: dict) -> dict:
    """供搜索排序使用：越界项数、加权偏差和。"""
    devs = _deviations(
        {o: float(v) for o, v in result["seger"].items()}, targets
    )
    n_violations = sum(1 for d in devs.values() if not d["in_range"])
    weighted = sum(d["weighted_deviation"] for d in devs.values())
    return {
        "n_violations": n_violations,
        "weighted_deviation": weighted,
        "per_oxide": devs,
    }
