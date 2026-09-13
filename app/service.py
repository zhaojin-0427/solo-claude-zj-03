"""业务编排：直接计算、搜索替代配方、版本冻结。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import db, optimizer
from .chemistry import calc_batch, deviation_summary, validate_analysis
from .config import CONSTANTS_VERSION, constants_snapshot
from .schemas import (
    BatchRequest,
    FreezeRequest,
    MaterialCreate,
    SearchRequest,
)


# ---------------------------------------------------------------------------
# 原料
# ---------------------------------------------------------------------------

def create_material(payload: MaterialCreate):
    normalized_oxides, normalized_loi = validate_analysis(
        payload.oxides, payload.loi, payload.analysis_tolerance
    )
    return db.create_material(payload, normalized_oxides, normalized_loi)


def update_material(material_id: int, payload):
    return db.update_material(material_id, payload)


# ---------------------------------------------------------------------------
# 直接按投料量计算
# ---------------------------------------------------------------------------

def _material_tuples(materials: dict, items) -> list[tuple]:
    tuples = []
    for item in items:
        mat = materials.get(item.material_id)
        if mat is None:
            from .errors import NotFoundError

            raise NotFoundError(
                f"原料不存在: id={item.material_id}", "material_not_found"
            )
        tuples.append(
            (mat.id, mat.name, mat.oxides, mat.loi, mat.price, item.amount)
        )
    return tuples


def compute_batch(req: BatchRequest) -> dict[str, Any]:
    ids = [i.material_id for i in req.items]
    materials = db.get_materials_map(ids)
    tuples = _material_tuples(materials, req.items)
    return calc_batch(tuples, targets={})


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------

def _request_constraints_dict(req: SearchRequest) -> dict[str, Any]:
    return json.loads(req.model_dump_json())


def _snapshot_materials(materials: dict) -> dict[str, Any]:
    snap = {}
    for mid, mat in materials.items():
        snap[str(mid)] = {
            "id": mat.id,
            "name": mat.name,
            "oxides": mat.oxides,
            "loi": mat.loi,
            "price": mat.price,
            "available": mat.available,
            "analysis_tolerance": mat.analysis_tolerance,
        }
    return snap


def _assess(prob: optimizer.Problem, sol: dict, req: SearchRequest,
            materials: dict) -> dict[str, Any]:
    amounts = sol["amounts"]
    tuples = []
    items_payload = []
    for i, m in enumerate(prob.mats):
        amt = round(float(amounts[i]), 9)
        if amt <= 0:
            continue
        mat = materials[m.id]
        tuples.append((mat.id, mat.name, mat.oxides, mat.loi, mat.price, amt))
        items_payload.append({"material_id": mat.id, "amount": amt})
    result = calc_batch(tuples, targets={o: req.targets[o] for o in req.targets})
    summary = deviation_summary(result, req.targets)
    target_results = summary["per_oxide"]
    return {
        "items": items_payload,
        "n_materials": int(sol["used"].sum()),
        "n_violations": summary["n_violations"],
        "weighted_deviation": round(summary["weighted_deviation"], 12),
        # 每个 target 的当前釉式值、区间、绝对偏差、权重与是否达标
        "target_results": target_results,
        "batch_mass": result["batch_mass"],
        "batch_error": round(result["batch_mass"] - req.batch_size, 9),
        "cost": round(result["cost"], 6),
        "loss_on_ignition": round(result["loss_on_ignition"], 9),
        "loss_on_ignition_pct": round(result["loss_on_ignition_pct"], 6),
        "seger": result["seger"],
        "fired_oxide_mass_ratio": result["fired_oxide_mass_ratio"],
        "oxide_moles": result["oxide_moles"],
        "flux_moles": result["flux_moles"],
        "fired_mass": round(result["fired_mass"], 9),
        "breakdown": result["breakdown"],
        "all_targets_met": summary["n_violations"] == 0,
    }


def search_recipes(req: SearchRequest) -> dict[str, Any]:
    # 整个求解生命周期静音 HiGHS C++ 层遗留的 std::cout 调试输出
    # （该缓冲在进程退出才 flush，必须持续重定向 fd1）。
    with optimizer._silence_native_stdout():
        return _search_recipes_impl(req)


def _search_recipes_impl(req: SearchRequest) -> dict[str, Any]:
    all_materials = {m.id: m for m in db.list_materials()}
    mats = optimizer.prepare_materials(all_materials, req)
    targets = optimizer.build_targets(req)

    batch_lo = req.batch_size - (
        req.batch_tolerance if req.batch_tolerance is not None else req.step
    )
    batch_hi = req.batch_size + (
        req.batch_tolerance if req.batch_tolerance is not None else req.step
    )
    batch_lo = max(batch_lo, 0.0)

    prob = optimizer.Problem(
        mats=mats,
        targets=targets,
        step=req.step,
        batch_lo=batch_lo,
        batch_hi=batch_hi,
    )

    status = "optimal"
    diagnosis = None
    candidates: list[dict[str, Any]] = []
    seen_cuts: list = []

    primary = optimizer.lexicographic_search(prob, cuts=None)
    if primary is None:
        status = "infeasible"
        diagnosis = optimizer.diagnose(prob)
    else:
        assessed = _assess(prob, primary, req, all_materials)
        candidates.append(assessed)
        if not assessed["all_targets_met"]:
            status = "target_unreachable"
            diagnosis = optimizer.diagnose(prob)

        # 替代配方：以用料集合 no-good 割反复求次优
        mask = primary["used"].astype(float)
        seen_cuts.append(mask)
        for _ in range(max(req_n_alternatives() - 1, 0)):
            if status == "infeasible":
                break
            alt = optimizer.lexicographic_search(prob, cuts=list(seen_cuts))
            if alt is None:
                break
            alt_assessed = _assess(prob, alt, req, all_materials)
            candidates.append(alt_assessed)
            seen_cuts.append(alt["used"].astype(float))

        candidates = _rank_candidates(candidates, req.batch_size)

    response = {
        "status": status,
        "batch_size": req.batch_size,
        "step": req.step,
        "batch_interval": [batch_lo, batch_hi],
        "candidates": candidates,
        "best": candidates[0] if candidates else None,
        "diagnosis": diagnosis,
        "constants_version": CONSTANTS_VERSION,
    }
    return response


def req_n_alternatives() -> int:
    from .config import settings
    return settings.max_alternatives


def _rank_candidates(candidates: list[dict], batch_size: float) -> list[dict]:
    return sorted(
        candidates,
        key=lambda c: (
            c["n_violations"],
            c["weighted_deviation"],
            c["n_materials"],
            round(c["cost"], 9),
            abs(c["batch_mass"] - batch_size),
        ),
    )


# ---------------------------------------------------------------------------
# 版本冻结
# ---------------------------------------------------------------------------

def _canonical_hash(payload: Any) -> str:
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def freeze_version(req: FreezeRequest, search_constraints: dict | None = None):
    """冻结投料方案：重算并把原料分析、约束、常量、输入哈希整体存档。"""
    ids = [i.material_id for i in req.items]
    materials = db.get_materials_map(ids)
    tuples = _material_tuples(materials, req.items)
    result = calc_batch(tuples, targets={})

    items_payload = [
        {"material_id": i.material_id, "amount": i.amount} for i in req.items
    ]
    canonical_items = sorted(items_payload, key=lambda x: x["material_id"])
    snapshot = _snapshot_materials(materials)
    constraints = search_constraints or {
        "mode": "direct_freeze",
        "items": canonical_items,
    }
    # 约束内嵌的投料表同样规范化排序，保证哈希只取决于语义
    if isinstance(constraints, dict) and isinstance(constraints.get("items"), list):
        constraints = dict(constraints)
        constraints["items"] = sorted(
            constraints["items"], key=lambda x: x["material_id"]
        )
    constants = constants_snapshot()
    hash_payload = {
        "items": canonical_items,
        "materials": json.loads(json.dumps(snapshot, sort_keys=True)),
        "constraints": constraints,
        "constants": constants,
    }
    input_hash = _canonical_hash(hash_payload)

    stored, created = db.save_version(
        input_hash=input_hash,
        items=items_payload,
        note=req.note,
        material_snapshot=snapshot,
        constraints=constraints,
        result=result,
        constants=constants,
    )
    return stored, created
