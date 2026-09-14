"""原料含水（湿基）测定与现场湿料称量换算。

口径
----
* 湿基含水率：取样（湿料）质量 ``m_w`` 与烘干后质量 ``m_d`` 之差占湿料的比例::

      w = (m_w - m_d) / m_w        （0 <= w < 1）

* 配方计算仍按**干料**进行：釉式、烧失等化学结果只取决于干料量。
  现场为投入干料量 ``D`` 而需称取的湿料质量与原料自带水量::

      M_wet = D / (1 - w)
      M_H2O = M_wet - D = D * w / (1 - w)

* 库存以到货湿料计量，成本按实际称取（湿料）质量乘单价；缺少生效测定
  或未指定批号时，各项明确回退为按干料称量（``basis="dry"``），
  绝不暗用任何"当前含水率"。
"""
from __future__ import annotations

from typing import Any, Optional

# 按干料称量的原因码
LOT_NOT_SPECIFIED = "lot_not_specified"
NO_EFFECTIVE_MEASUREMENT = "no_effective_measurement"

DRY_NOTES = {
    LOT_NOT_SPECIFIED: "未指定原料批号，按干料称量",
    NO_EFFECTIVE_MEASUREMENT: "指定批号在生效基准日无有效含水测定，按干料称量",
}


def wet_basis_fraction(sample_mass_g: float, dried_mass_g: float) -> float:
    """由取样质量与烘干后质量计算湿基含水率。"""
    return (sample_mass_g - dried_mass_g) / sample_mass_g


def wet_for_dry(dry_mass: float, moisture_fraction: float) -> float:
    """为得到给定干料量需要称取的湿料质量。"""
    return dry_mass / (1.0 - moisture_fraction)


def carried_water(wet_mass: float, dry_mass: float) -> float:
    """湿料中额外带入的水分质量。"""
    return wet_mass - dry_mass


def _measurement_brief(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec["id"],
        "lot": rec["lot"],
        "sample_mass_g": rec["sample_mass_g"],
        "dried_mass_g": rec["dried_mass_g"],
        "moisture_fraction": round(rec["moisture_fraction"], 9),
        "valid_from": rec["valid_from"],
        "valid_to": rec["valid_to"],
    }


def weighing_line(
    *,
    material_id: int,
    name: str,
    dry_kg: float,
    price_per_kg: float,
    available_kg: float,
    lot: Optional[str],
    as_of: Optional[str],
    measurement: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """构造单种原料的干/湿称量行。

    ``measurement`` 为该批号在基准日生效的不可变测定记录；为 None 时
    （未指定批号或无有效测定）明确回退按干料称量。
    """
    base = {
        "material_id": material_id,
        "name": name,
        "lot": lot,
        "dry_kg": round(dry_kg, 9),
        "price_per_kg": price_per_kg,
        "stock_available_kg": available_kg,
    }
    if measurement is None:
        reason = LOT_NOT_SPECIFIED if not lot else NO_EFFECTIVE_MEASUREMENT
        base.update(
            {
                "basis": "dry",
                "reason": reason,
                "note": DRY_NOTES[reason],
                "as_of": as_of,
                "moisture_fraction": None,
                "wet_kg": round(dry_kg, 9),
                "carried_water_kg": 0.0,
                "cost": round(dry_kg * price_per_kg, 6),
                "stock_used_kg": round(dry_kg, 9),
                "stock_remaining_kg": round(available_kg - dry_kg, 9),
                "within_stock": dry_kg <= available_kg + 1e-9,
                "measurement": None,
            }
        )
        return base

    w = float(measurement["moisture_fraction"])
    wet_kg = wet_for_dry(dry_kg, w)
    water_kg = carried_water(wet_kg, dry_kg)
    base.update(
        {
            "basis": "wet",
            "reason": None,
            "note": f"批号「{lot}」湿基含水率 {w:.4%}，按湿料称量",
            "as_of": as_of,
            "moisture_fraction": round(w, 9),
            "wet_kg": round(wet_kg, 9),
            "carried_water_kg": round(water_kg, 9),
            "cost": round(wet_kg * price_per_kg, 6),
            "stock_used_kg": round(wet_kg, 9),
            "stock_remaining_kg": round(available_kg - wet_kg, 9),
            "within_stock": wet_kg <= available_kg + 1e-9,
            "measurement": _measurement_brief(measurement),
        }
    )
    return base


def summarize_weighing(
    lines: list[dict[str, Any]],
    *,
    selected_lots: dict[int, str],
    as_of: Optional[str],
) -> dict[str, Any]:
    """汇总逐料称量行为总量、成本、库存扣用与未决批号清单。"""
    unresolved = [
        {
            "material_id": line["material_id"],
            "name": line["name"],
            "lot": line["lot"],
            "reason": line["reason"],
            "note": line["note"],
        }
        for line in lines
        if line["basis"] == "dry" and line["reason"] == NO_EFFECTIVE_MEASUREMENT
    ]
    dry_total = sum(line["dry_kg"] for line in lines)
    wet_total = sum(line["wet_kg"] for line in lines)
    water_total = sum(line["carried_water_kg"] for line in lines)
    cost_total = sum(line["cost"] for line in lines)
    shortfall = [
        {
            "material_id": line["material_id"],
            "name": line["name"],
            "short_kg": round(line["stock_used_kg"] - line["stock_available_kg"], 9),
        }
        for line in lines
        if not line["within_stock"]
    ]
    return {
        "basis": "wet" if any(line["basis"] == "wet" for line in lines) else "dry",
        "as_of": as_of,
        "selected_lots": {str(mid): lot for mid, lot in sorted(selected_lots.items())},
        "lines": lines,
        "totals": {
            "dry_kg": round(dry_total, 9),
            "wet_kg": round(wet_total, 9),
            "carried_water_kg": round(water_total, 9),
            "cost": round(cost_total, 6),
        },
        "unresolved": unresolved,
        "stock_shortfall": shortfall,
    }
