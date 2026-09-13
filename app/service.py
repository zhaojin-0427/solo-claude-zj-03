"""业务编排：直接计算、搜索替代配方、版本冻结、批次波动研究。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import db, optimizer, variability
from .chemistry import calc_batch, deviation_summary, validate_analysis
from .config import CONSTANTS_VERSION, constants_snapshot
from .errors import GlazeError
from .schemas import (
    BatchRequest,
    FreezeRequest,
    MaterialCreate,
    OxideTarget,
    RobustFreezeRequest,
    RobustSearchRequest,
    SearchRequest,
    StudyRequest,
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


# ---------------------------------------------------------------------------
# 原料批次波动研究
# ---------------------------------------------------------------------------

def _targets_from_version(version: dict) -> dict[str, OxideTarget]:
    """来源版本若由搜索冻结（约束内含 targets），研究缺省沿用其目标。"""
    constraints = version.get("constraints") or {}
    raw = constraints.get("targets") if isinstance(constraints, dict) else None
    targets: dict[str, OxideTarget] = {}
    if isinstance(raw, dict):
        for oxide, spec in raw.items():
            if isinstance(spec, dict):
                targets[oxide] = OxideTarget.model_validate(spec)
    return targets


def _canonical_linked(linked_groups) -> list[list[int]]:
    """联动组规范化：组内排序、组间按首成员排序，保证哈希只取决于语义。"""
    groups = [sorted(g) for g in linked_groups]
    groups.sort(key=lambda g: g[0])
    return groups


def _study_context(study: dict) -> dict[str, Any]:
    """从研究快照还原计算上下文（批次表、来源用量、目标、联动组）。"""
    batches_map: dict[int, dict[str, dict]] = {}
    for row in study["batches"]:
        batches_map.setdefault(row["material_id"], {})[row["batch"]] = {
            "oxides": row["oxides"],
            "loi": row["loi"],
            "price": row["price"],
            "available": row["available"],
        }
    source_amounts = {
        int(i["material_id"]): float(i["amount"]) for i in study["source_items"]
    }
    targets = [
        variability.StudyTarget(o, spec["low"], spec["high"], spec["weight"])
        for o, spec in sorted(study["targets"].items())
    ]
    return {
        "batches_map": batches_map,
        "source_amounts": source_amounts,
        "material_ids": sorted(source_amounts),
        "targets": targets,
        "linked_groups": [list(g) for g in study["linked_groups"]],
    }


def create_study(req: StudyRequest):
    """创建批次波动研究：校验批次与联动组，重采样统计后整体存档。"""
    version = db.get_version(req.version_id)  # 404: 来源配方不存在
    source_items = sorted(
        (
            {"material_id": int(i["material_id"]), "amount": float(i["amount"])}
            for i in version["items"]
            if i["amount"] > 0
        ),
        key=lambda x: x["material_id"],
    )
    if not source_items:
        raise GlazeError("来源配方没有任何正用量原料", "empty_batch")
    recipe_ids = {i["material_id"] for i in source_items}

    batches_map = variability.normalize_batches(req.batches)
    variability.check_coverage(recipe_ids, batches_map)
    linked = _canonical_linked(req.linked_groups)
    variability.check_linked_groups(linked, batches_map, recipe_ids)

    targets = req.targets or _targets_from_version(version)
    target_list = [
        variability.StudyTarget(o, t.low, t.high, t.weight)
        for o, t in sorted(targets.items())
    ]

    material_ids = sorted(recipe_ids)
    plans = variability.draw_plans(
        material_ids, batches_map, linked, req.n_resamples, req.seed
    )
    scenario_set = variability.ScenarioSet(material_ids, batches_map, plans)
    amount_map = {i["material_id"]: i["amount"] for i in source_items}
    amounts = np.array([amount_map[mid] for mid in material_ids])
    ev = scenario_set.evaluate(amounts)
    if not bool(ev["valid"].all()):
        raise GlazeError(
            "部分重采样场景助熔摩尔数接近零，无法按助熔归一化为 Seger 釉式",
            "no_flux",
        )
    names = {
        int(mid): m["name"] for mid, m in version["material_snapshot"].items()
    }
    result = variability.summarize(scenario_set, ev, target_list, names)
    result["seed"] = req.seed

    batches_stored = [
        {
            "material_id": mid,
            "batch": label,
            "oxides": b["oxides"],
            "loi": b["loi"],
            "price": b["price"],
            "available": b["available"],
        }
        for mid in sorted(batches_map)
        for label, b in sorted(batches_map[mid].items())
    ]
    targets_stored = {
        o: {"low": t.low, "high": t.high, "weight": t.weight}
        for o, t in sorted(targets.items())
    }
    constants = constants_snapshot()
    hash_payload = {
        "version_id": version["id"],
        "source_items": source_items,
        "batches": batches_stored,
        "linked_groups": linked,
        "targets": targets_stored,
        "n_resamples": req.n_resamples,
        "seed": req.seed,
        "constants": constants,
    }
    return db.save_study(
        input_hash=_canonical_hash(hash_payload),
        version_id=version["id"],
        source_items=source_items,
        batches=batches_stored,
        linked_groups=linked,
        targets=targets_stored,
        n_resamples=req.n_resamples,
        seed=req.seed,
        note=req.note,
        constants=constants,
        result=result,
    )


def robust_search(study_id: str, req: RobustSearchRequest) -> dict[str, Any]:
    """在库存、步进与最大改动量内搜索稳健配方并排序。"""
    study = db.get_study(study_id)  # 404
    ctx = _study_context(study)
    locked = sorted(set(req.locked))
    outsiders = [m for m in locked if m not in ctx["source_amounts"]]
    if outsiders:
        raise GlazeError(
            f"锁定原料不在来源配方中: {outsiders}",
            "material_not_in_recipe",
            {"material_ids": outsiders},
        )
    n_resamples = (
        req.n_resamples if req.n_resamples is not None else study["n_resamples"]
    )
    seed = req.seed if req.seed is not None else study["seed"]
    material_ids = ctx["material_ids"]
    plans = variability.draw_plans(
        material_ids, ctx["batches_map"], ctx["linked_groups"],
        n_resamples, seed,
    )
    scenario_set = variability.ScenarioSet(material_ids, ctx["batches_map"], plans)
    source = np.array([ctx["source_amounts"][mid] for mid in material_ids])
    # 库存取各批最小可用量（稳健口径：任何批次到货都够用）
    stock = {
        mid: min(b["available"] for b in ctx["batches_map"][mid].values())
        for mid in material_ids
    }
    candidates = variability.robust_search(
        scenario_set, source, stock, req.step, req.max_change,
        locked, ctx["targets"], req.max_candidates,
    )
    assembled = []
    for m in candidates:
        amounts = m["amounts"]
        items = [
            {"material_id": mid, "amount": round(float(amounts[i]), 9)}
            for i, mid in enumerate(material_ids)
            if amounts[i] > 0
        ]
        assembled.append({
            "items": items,
            "is_source": bool(np.allclose(amounts, source, atol=1e-12)),
            "change_amount": m["change_amount"],
            "joint_pass_rate": m["joint_pass_rate"],
            "target_pass_rates": m["target_pass_rates"],
            "worst_quantile_deviation": m["worst_quantile_deviation"],
            "cost_mean": m["cost_mean"],
        })
    return {
        "study_id": study["id"],
        "status": "ok",
        "search_constraints": {
            "step": req.step,
            "max_change": req.max_change,
            "locked": locked,
            "n_resamples": n_resamples,
            "seed": seed,
            "max_candidates": req.max_candidates,
        },
        "stock_basis": stock,
        "candidates": assembled,
        "best": assembled[0] if assembled else None,
        "constants_version": CONSTANTS_VERSION,
    }


def freeze_robust(study_id: str, req: RobustFreezeRequest):
    """冻结稳健搜索选定结果：来源配方、化验数据、抽样规则与种子整体存档。"""
    study = db.get_study(study_id)  # 404
    ctx = _study_context(study)
    recipe_ids = set(ctx["source_amounts"])
    outsiders = sorted({i.material_id for i in req.items} - recipe_ids)
    if outsiders:
        raise GlazeError(
            f"投料表包含来源配方外的原料: {outsiders}",
            "material_not_in_recipe",
            {"material_ids": outsiders},
        )
    n_resamples = (
        req.n_resamples if req.n_resamples is not None else study["n_resamples"]
    )
    seed = req.seed if req.seed is not None else study["seed"]
    material_ids = ctx["material_ids"]
    amount_map = {mid: 0.0 for mid in material_ids}
    for item in req.items:
        amount_map[item.material_id] = item.amount
    amounts = np.array([amount_map[mid] for mid in material_ids])
    if float(amounts.sum()) <= 0.0:
        raise GlazeError("投料总量为零，无法计算釉式", "empty_batch")
    plans = variability.draw_plans(
        material_ids, ctx["batches_map"], ctx["linked_groups"],
        n_resamples, seed,
    )
    scenario_set = variability.ScenarioSet(material_ids, ctx["batches_map"], plans)
    ev = scenario_set.evaluate(amounts)
    if not bool(ev["valid"].all()):
        raise GlazeError(
            "部分重采样场景助熔摩尔数接近零，无法按助熔归一化为 Seger 釉式",
            "no_flux",
        )
    version = db.get_version(study["version_id"])
    names = {
        int(mid): m["name"] for mid, m in version["material_snapshot"].items()
    }
    result = variability.summarize(scenario_set, ev, ctx["targets"], names)
    result["seed"] = seed
    source = np.array([ctx["source_amounts"][mid] for mid in material_ids])
    result["change_amount"] = round(float(np.abs(amounts - source).sum()), 9)

    items_payload = [
        {"material_id": i.material_id, "amount": i.amount} for i in req.items
    ]
    canonical_items = sorted(items_payload, key=lambda x: x["material_id"])
    source_snapshot = {
        "version_id": study["version_id"],
        "source_items": study["source_items"],
        "batches": study["batches"],
        "linked_groups": study["linked_groups"],
        "targets": study["targets"],
        "n_resamples": n_resamples,
        "seed": seed,
        "constants_version": study["constants_version"],
        "constants_snapshot": study["constants_snapshot"],
    }
    constraints = req.search_constraints or {
        "mode": "robust_freeze",
        "items": canonical_items,
    }
    # 约束内嵌的投料表同样规范化排序，保证哈希只取决于语义
    if isinstance(constraints, dict) and isinstance(constraints.get("items"), list):
        constraints = dict(constraints)
        constraints["items"] = sorted(
            constraints["items"], key=lambda x: x["material_id"]
        )
    hash_payload = {
        "study_id": study["id"],
        "items": canonical_items,
        "constraints": constraints,
        "source": source_snapshot,
    }
    return db.save_robust_version(
        input_hash=_canonical_hash(hash_payload),
        study_id=study["id"],
        items=items_payload,
        note=req.note,
        search_constraints=constraints,
        source_snapshot=source_snapshot,
        result=result,
    )
