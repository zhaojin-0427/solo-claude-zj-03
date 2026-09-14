"""业务编排：直接计算、搜索替代配方、版本冻结、批次波动研究、混合试验、釉浆调制。"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import numpy as np

from . import blending, db, optimizer, slurry, variability
from .chemistry import calc_batch, deviation_summary, validate_analysis
from .config import (
    CONSTANTS_VERSION,
    DENSITY_CLOSURE_TOLERANCE_G_ML,
    constants_snapshot,
)
from .errors import GlazeError
from .schemas import (
    BatchRequest,
    BlendExperimentRequest,
    BlendFreezeRequest,
    FreezeRequest,
    MasterPlanRequest,
    MaterialCreate,
    OxideTarget,
    RobustFreezeRequest,
    RobustSearchRequest,
    SearchRequest,
    SlurryAdditionRequest,
    SlurryBatchCreate,
    SlurryCorrectionRequest,
    SlurryFinalizeRequest,
    SlurryPremixRequest,
    SlurryReadingRequest,
    SlurryRecycleRequest,
    SlurryAdjustmentRequest,
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


# ---------------------------------------------------------------------------
# 配方混合试验
# ---------------------------------------------------------------------------

def _collect_blend_sources(version_ids: list[str]) -> tuple[
    list[blending.FrozenSource], dict[int, Any], list[dict[str, Any]]
]:
    """载入来源冻结版本，构造归一化来源与合并原料快照。

    相同原料跨来源出现时，其冻结分析（氧化物 + LOI）必须一致，
    否则两种同名/同 id 原料无法作为同一种料合并称量。**价格允许
    不同**：原料单价可在两次冻结之间更新，合并后每格该原料按各
    来源的实际份额加权计价（见 :meth:`SourceMaterial.effective_price`）。
    快照中的价格取首现来源，仅作展示与哈希兜底；真实成本以各来源
    价格与份额计算。
    """
    versions = [db.get_version(vid) for vid in version_ids]  # 404 由 db 抛出
    sources = blending.build_sources(versions)

    snapshots: dict[int, Any] = {}
    for v in versions:
        for mid_str, snap in v["material_snapshot"].items():
            mid = int(mid_str)
            current = {
                "name": snap["name"],
                "oxides": snap["oxides"],
                "loi": snap["loi"],
                "price": snap["price"],
            }
            if mid in snapshots:
                prev = snapshots[mid]
                if (
                    prev["oxides"] != current["oxides"]
                    or abs(prev["loi"] - current["loi"]) > 1e-12
                ):
                    raise GlazeError(
                        f"原料 {mid}（{current['name']}）在不同来源版本中的冻结"
                        "分析不一致，无法合并",
                        "inconsistent_material_snapshot",
                        {"material_id": mid,
                         "version_ids": [vv["id"] for vv in versions]},
                    )
            else:
                snapshots[mid] = current

    sources_stored = [
        {
            "version_id": src.version_id,
            "note": src.note,
            "shares": [
                {"material_id": mid, "share": round(share, 12),
                 "price_per_kg": src.prices[mid]}
                for mid, share in sorted(src.shares.items())
            ],
            "original_items": [
                {"material_id": mid, "amount": amt,
                 "price_per_kg": src.prices[mid]}
                for mid, amt in sorted(src.original_amounts.items())
            ],
        }
        for src in sources
    ]
    return list(sources), snapshots, sources_stored


def create_blend_experiment(req: BlendExperimentRequest):
    """创建混合试验：校验来源/布局，逐格投料舍入，整体存档为不可变版本。"""
    sources, snapshots, sources_stored = _collect_blend_sources(req.sources)

    cells = [
        {"position": c.position, "weights": [float(w) for w in c.weights]}
        for c in req.cells
    ]
    materials, cell_records = blending.dose_cells(
        sources=sources,
        material_snapshots=snapshots,
        cells=cells,
        dry_mass_g=req.dry_mass_per_tile_g,
        division=req.scale_division_g,
        minimum_weighed=req.minimum_weighed_g,
    )

    material_ids = [m.material_id for m in materials]
    layout_stored = [
        {"position": c["position"], "weights": c["weights"]} for c in cells
    ]
    setup = {
        "dry_mass_per_tile_g": req.dry_mass_per_tile_g,
        "scale_division_g": req.scale_division_g,
        "minimum_weighed_g": req.minimum_weighed_g,
        "ratio_low": req.ratio_low,
        "ratio_high": req.ratio_high,
        "ratio_step": req.ratio_step,
        "n_cells": len(cells),
    }

    # 逐格汇总
    total_dry_g = req.dry_mass_per_tile_g * len(cells)
    total_weighed_g = round(sum(c["total_weighed_g"] for c in cell_records), 6)
    total_fired_g = round(sum(c["fired_mass_g"] for c in cell_records), 6)
    total_cost = round(sum(c["cost"] for c in cell_records), 9)
    per_material: dict[int, dict[str, Any]] = {
        mid: {"material_id": mid, "theoretical_g": 0.0, "weighed_g": 0.0}
        for mid in material_ids
    }
    for c in cell_records:
        for d in c["doses"]:
            per_material[d["material_id"]]["theoretical_g"] += d["theoretical_g"]
            per_material[d["material_id"]]["weighed_g"] += d["weighed_g"]
    material_totals = []
    for mid in material_ids:
        t = per_material[mid]
        t["name"] = snapshots[mid]["name"]
        t["theoretical_g"] = round(t["theoretical_g"], 6)
        t["weighed_g"] = round(t["weighed_g"], 6)
        t["rounding_error_g"] = round(t["weighed_g"] - t["theoretical_g"], 6)
        material_totals.append(t)

    result = {
        "cells": cell_records,
        "summary": {
            "n_cells": len(cells),
            "n_materials": len(material_ids),
            "target_dry_mass_g": req.dry_mass_per_tile_g,
            "total_dry_mass_g": total_dry_g,
            "total_weighed_g": total_weighed_g,
            "total_rounding_error_g": round(total_weighed_g - total_dry_g, 6),
            "total_fired_mass_g": total_fired_g,
            "total_cost": total_cost,
            "max_abs_seger_shift": round(
                max((c["max_abs_seger_shift"] for c in cell_records), default=0.0),
                9,
            ),
        },
        "material_totals": material_totals,
    }

    material_snapshot = {
        str(mid): {
            "id": mid,
            **snapshots[mid],
            "available": None,  # 来自冻结版本，不带可变库存
        }
        for mid in material_ids
    }
    constants = constants_snapshot()
    hash_payload = {
        "sources": sources_stored,
        "mode": req.mode,
        "layout": layout_stored,
        "setup": setup,
        "material_snapshot": material_snapshot,
        "constants": constants,
    }
    return db.save_blend_experiment(
        input_hash=_canonical_hash(hash_payload),
        mode=req.mode,
        sources=sources_stored,
        layout=layout_stored,
        setup=setup,
        material_snapshot=material_snapshot,
        constants=constants,
        note=req.note,
        result=result,
    )


def _master_search_params(req) -> dict[str, Any]:
    return {
        "master_batch_g": req.master_batch_g,
        "master_minimum_weighed_g": req.master_minimum_weighed_g,
        "allowed_leftover_g": req.allowed_leftover_g,
        "max_candidates": req.max_candidates,
    }


def master_plan_search(experiment_id: str, req: MasterPlanRequest) -> dict[str, Any]:
    """对冻结试验搜索"母料 + 逐格补料"方案并排序。"""
    experiment = db.get_blend_experiment(experiment_id)  # 404
    prob = blending.build_master_problem(experiment)
    result = blending.search_master_plans(
        prob,
        master_batch_g=req.master_batch_g,
        master_minimum_weighed=req.master_minimum_weighed_g,
        allowed_leftover_g=req.allowed_leftover_g,
        max_candidates=req.max_candidates,
    )
    result["experiment_id"] = experiment["id"]
    result["constants_version"] = CONSTANTS_VERSION
    return result


def _run_master_search(experiment: dict[str, Any], req: BlendFreezeRequest):
    prob = blending.build_master_problem(experiment)
    return blending.search_master_plans(
        prob,
        master_batch_g=req.master_batch_g,
        master_minimum_weighed=req.master_minimum_weighed_g,
        allowed_leftover_g=req.allowed_leftover_g,
        max_candidates=max(req.plan_index + 1, req.max_candidates),
    )


def freeze_blend_plan(experiment_id: str, req: BlendFreezeRequest):
    """冻结选定母料拆分方案：重跑确定性搜索取 plan_index，整体存档。"""
    experiment = db.get_blend_experiment(experiment_id)  # 404
    result = _run_master_search(experiment, req)
    plans = result["plans"]
    if req.plan_index >= len(plans):
        raise GlazeError(
            f"候选序号 {req.plan_index} 超出范围（共 {len(plans)} 个可行方案）",
            "plan_index_out_of_range",
            {"plan_index": req.plan_index, "n_plans": len(plans)},
        )
    plan = plans[req.plan_index]

    search_constraints = {
        "master_batch_g": req.master_batch_g,
        "master_minimum_weighed_g": req.master_minimum_weighed_g,
        "allowed_leftover_g": req.allowed_leftover_g,
        "max_candidates": req.max_candidates,
        "plan_index": req.plan_index,
        "parameters": result["parameters"],
    }
    source_snapshot = {
        "experiment_id": experiment["id"],
        "mode": experiment["mode"],
        "sources": experiment["sources"],
        "layout": experiment["layout"],
        "setup": experiment["setup"],
        "material_snapshot": experiment["material_snapshot"],
        "cells": experiment["result"]["cells"],
        "constants_version": experiment["constants_version"],
        "constants_snapshot": experiment["constants_snapshot"],
    }
    hash_payload = {
        "experiment_id": experiment["id"],
        "plan_signature": plan["signature"],
        "search_constraints": search_constraints,
        "included": plan["included_material_ids"],
        "bulk": [
            {"material_id": b["material_id"], "prepared_units": b["prepared_units"]}
            for b in plan["bulk"]
        ],
        "source": source_snapshot,
    }
    return db.save_blend_version(
        input_hash=_canonical_hash(hash_payload),
        experiment_id=experiment["id"],
        note=req.note,
        search_constraints=search_constraints,
        plan=plan,
        source_snapshot=source_snapshot,
    )


# ---------------------------------------------------------------------------
# 釉浆调制批次
# ---------------------------------------------------------------------------


def _slurry_recipe_snapshot(version: dict) -> dict[str, Any]:
    """从冻结版本提取釉浆批次需要的配方快照：份额 + 原料名。"""
    totals: dict[int, float] = {}
    for item in version["items"]:
        amt = float(item["amount"])
        if amt > 0.0:
            mid = int(item["material_id"])
            totals[mid] = totals.get(mid, 0.0) + amt
    batch_mass = sum(totals.values())
    if batch_mass <= 0.0:
        raise GlazeError("来源配方没有任何正用量原料", "empty_batch")
    materials = []
    for mid in sorted(totals):
        name = version["material_snapshot"][str(mid)]["name"]
        materials.append(
            {
                "material_id": mid,
                "name": name,
                "amount_kg": round(totals[mid], 9),
                "share": totals[mid] / batch_mass,
            }
        )
    return {"version_id": version["id"], "materials": materials}


def _initial_plan(
    recipe: dict[str, Any], req: SlurryBatchCreate, rho_w: float
) -> dict[str, Any]:
    """按配方份额给各原料、初始用水、添加剂的初始称量值。

    固含率定义 S = 干物 / (干物 + 水 + 添加剂)，初始取水取目标区间中点，
    再按添加剂占干料比例反推初始用水。
    """
    target_solids = (req.solids_low + req.solids_high) / 2.0
    dry_g = req.target_dry_mass_kg * 1000.0
    additive_g = dry_g * req.additive_ratio
    # dry / (dry + water + additive) = S  =>  water = dry/S - dry - additive
    water_g = dry_g / target_solids - dry_g - additive_g
    if water_g < -1e-9:
        raise GlazeError(
            "按目标固含率与添加剂比例反推的初始用水为负：添加剂占比过高，"
            f"目标固含率 {target_solids:.4f} 无法同时容纳 {req.additive_ratio:.4f} "
            "的添加剂比例",
            "additive_ratio_infeasible",
            {
                "target_solids_fraction": target_solids,
                "additive_ratio": req.additive_ratio,
            },
        )
    water_g = max(water_g, 0.0)
    lines = []
    for m in recipe["materials"]:
        mass_g = dry_g * m["share"]
        lines.append(
            {
                "material_id": m["material_id"],
                "name": m["name"],
                "share": m["share"],
                "mass_kg": round(mass_g / 1000.0, 9),
                "mass_g": round(mass_g, 6),
            }
        )
    volume_ml = (
        dry_g / req.powder_true_density_kg_l
        + (water_g + additive_g) / rho_w
    )
    return {
        "target_solids_fraction": target_solids,
        "target_dry_mass_g": dry_g,
        "materials": lines,
        "initial_water_g": round(water_g, 6),
        "initial_water_kg": round(water_g / 1000.0, 9),
        "initial_additive_g": round(additive_g, 6),
        "initial_additive_kg": round(additive_g / 1000.0, 9),
        "planned_volume_ml": round(volume_ml, 6),
        "fits_container": volume_ml <= req.container_capacity_l * 1000.0 + 1e-9,
    }


def create_slurry_batch(req: SlurryBatchCreate):
    version = db.get_version(req.version_id)  # 404
    recipe = _slurry_recipe_snapshot(version)
    rho_w = slurry.water_density(req.water_temp_c)
    plan = _initial_plan(recipe, req, rho_w)
    if not plan["fits_container"]:
        raise GlazeError(
            f"初始计划体积 {plan['planned_volume_ml']:.1f} mL 超出容器容量 "
            f"{req.container_capacity_l * 1000.0:.1f} mL，无法按该干料量调制",
            "planned_over_capacity",
            {
                "planned_volume_ml": plan["planned_volume_ml"],
                "container_capacity_ml": req.container_capacity_l * 1000.0,
            },
        )
    batch_id = "slb_" + uuid.uuid4().hex
    params = {
        "target_dry_mass_kg": req.target_dry_mass_kg,
        "solids_low": req.solids_low,
        "solids_high": req.solids_high,
        "density_low": req.density_low,
        "density_high": req.density_high,
        "powder_true_density_kg_l": req.powder_true_density_kg_l,
        "water_temp_c": req.water_temp_c,
        "water_density_g_ml": rho_w,
        "container_capacity_ml": req.container_capacity_l * 1000.0,
        "additive_ratio": req.additive_ratio,
        "note": req.note,
    }
    batch = db.create_slurry_batch(
        batch_id=batch_id,
        version_id=version["id"],
        params=params,
        initial_plan=plan,
        recipe_snapshot=recipe,
    )
    return assemble_slurry_batch(batch)


def _new_state() -> slurry.SlurryState:
    return slurry.SlurryState()


def _state_breakdown(state: slurry.SlurryState) -> dict[str, dict[str, float]]:
    return {
        "dry_g": {
            "direct": round(state.dry_direct_g, 9),
            "premix": round(state.dry_premix_g, 9),
            "recycle": round(state.dry_recycle_g, 9),
        },
        "water_g": {
            "direct": round(state.water_direct_g, 9),
            "recycle": round(state.water_recycle_g, 9),
        },
        "additive_g": {
            "direct": round(state.additive_direct_g, 9),
            "recycle": round(state.additive_recycle_g, 9),
        },
    }


def _replay(batch: dict, entries: list[dict]) -> slurry.SlurryState:
    """按台账序号重放全部质量动作，重建三相累计。读数不改质量。"""
    state = _new_state()
    for entry in entries:
        p = entry["payload"]
        kind = entry["entry_type"]
        if kind == "addition":
            dry = sum(float(d["mass_g"]) for d in p["dry_materials"])
            state.add_direct(dry, float(p["water_g"]), float(p["additive_g"]))
        elif kind == "recycle":
            state.add_recycle(
                float(p["split"]["dry_g"]),
                float(p["split"]["water_g"]),
                float(p["split"]["additive_g"]),
            )
        elif kind == "premix":
            state.add_premix(float(p["premix_mass_g"]), float(p.get("water_g", 0.0)))
        elif kind == "adjustment":
            state.add_premix(float(p["premix_g"]), float(p["water_g"]))
        # reading 不改变质量
    return state


def _target_status(metrics: dict, batch: dict) -> dict[str, Any]:
    solids = metrics["solids_fraction"]
    density = metrics["theoretical_density_g_ml"]
    solids_dev = max(
        batch["solids_low"] - solids, 0.0, solids - batch["solids_high"]
    )
    density_dev = max(
        batch["density_low"] - density, 0.0, density - batch["density_high"]
    )
    return {
        "solids": {
            "value": round(solids, 9),
            "low": batch["solids_low"],
            "high": batch["solids_high"],
            "deviation": round(solids_dev, 12),
            "in_range": solids_dev <= 0.0,
        },
        "density": {
            "value": round(density, 9),
            "low": batch["density_low"],
            "high": batch["density_high"],
            "deviation": round(density_dev, 12),
            "in_range": density_dev <= 0.0,
        },
        "all_targets_met": solids_dev <= 0.0 and density_dev <= 0.0,
    }


def _build_warnings(
    batch: dict, entries: list[dict], metrics: dict
) -> list[dict[str, Any]]:
    """汇总当前批次的异常：读数不闭合、容器超量、目标越界。"""
    warnings: list[dict[str, Any]] = []
    for entry in entries:
        if entry["entry_type"] != "reading":
            continue
        assessment = entry.get("state_after") or {}
        if not assessment.get("closed"):
            warnings.append(
                {
                    "code": "reading_not_closed",
                    "seq": entry["seq"],
                    "message": assessment.get("reason") or "比重杯读数不闭合",
                    "measured_density_g_ml": assessment.get(
                        "measured_density_g_ml"
                    ),
                    "theoretical_density_g_ml": assessment.get(
                        "theoretical_density_g_ml"
                    ),
                    "difference_g_ml": assessment.get("difference_g_ml"),
                    "tolerance_g_ml": DENSITY_CLOSURE_TOLERANCE_G_ML,
                }
            )
    capacity = batch["container_capacity_ml"]
    occupied = metrics["occupied_volume_ml"]
    if occupied > capacity + 1e-9:
        warnings.append(
            {
                "code": "container_overfill",
                "message": f"占用体积 {occupied:.1f} mL 超出容器容量 {capacity:.1f} mL",
                "occupied_volume_ml": round(occupied, 6),
                "container_capacity_ml": capacity,
                "excess_volume_ml": round(occupied - capacity, 6),
            }
        )
    status = _target_status(metrics, batch)
    if metrics["total_mass_g"] > 0.0 and not status["all_targets_met"]:
        out = []
        if not status["solids"]["in_range"]:
            out.append("solids_fraction")
        if not status["density"]["in_range"]:
            out.append("density")
        warnings.append(
            {
                "code": "target_out_of_range",
                "message": f"当前指标偏离目标区间: {', '.join(out)}",
                "metrics": out,
                "solids": status["solids"],
                "density": status["density"],
            }
        )
    return warnings


def assemble_slurry_batch(batch: dict) -> dict[str, Any]:
    """重放台账并组装批次完整视图（计划参数、守恒状态、读数、告警）。"""
    entries = db.list_slurry_entries(batch["id"])
    state = _replay(batch, entries)
    metrics = slurry.slurry_metrics(
        state,
        batch["powder_true_density_g_ml"],
        batch["water_density_g_ml"],
    )
    metrics = {k: round(v, 9) for k, v in metrics.items()}
    capacity = batch["container_capacity_ml"]
    warnings = _build_warnings(batch, entries, metrics)
    view = {
        "id": batch["id"],
        "version_id": batch["version_id"],
        "status": batch["status"],
        "note": batch["note"],
        "targets": {
            "target_dry_mass_kg": batch["target_dry_mass_kg"],
            "solids_low": batch["solids_low"],
            "solids_high": batch["solids_high"],
            "density_low": batch["density_low"],
            "density_high": batch["density_high"],
            "additive_ratio": batch["additive_ratio"],
        },
        "constants": {
            "powder_true_density_g_ml": batch["powder_true_density_g_ml"],
            "water_temp_c": batch["water_temp_c"],
            "water_density_g_ml": batch["water_density_g_ml"],
            "container_capacity_ml": capacity,
            "density_closure_tolerance_g_ml": DENSITY_CLOSURE_TOLERANCE_G_ML,
            "constants_version": CONSTANTS_VERSION,
        },
        "initial_plan": batch["initial_plan"],
        "recipe": batch["recipe_snapshot"],
        "state": {
            **metrics,
            "free_volume_ml": round(max(capacity - metrics["occupied_volume_ml"], 0.0), 9),
            "breakdown": _state_breakdown(state),
            "target_status": _target_status(metrics, batch),
        },
        "warnings": warnings,
        "entries": entries,
        "n_entries": len(entries),
        "freeze_id": batch["freeze_id"],
        "created_at": batch["created_at"],
        "updated_at": batch["updated_at"],
        "finalized_at": batch["finalized_at"],
    }
    return view


def _require_status(batch: dict, allowed: tuple[str, ...]) -> None:
    if batch["status"] not in allowed:
        raise GlazeError(
            f"批次当前状态为 {batch['status']}，该操作仅允许 {list(allowed)} 状态",
            "invalid_slurry_status",
            {"status": batch["status"], "allowed": list(allowed)},
        )


def _check_recipe_materials(batch: dict, material_ids: list[int]) -> None:
    recipe_ids = {m["material_id"] for m in batch["recipe_snapshot"]["materials"]}
    outsiders = sorted(set(material_ids) - recipe_ids)
    if outsiders:
        raise GlazeError(
            f"干料原料不属于来源冻结配方: {outsiders}",
            "material_not_in_recipe",
            {"material_ids": outsiders},
        )


def _metrics_after(
    batch: dict, state: slurry.SlurryState
) -> dict[str, Any]:
    metrics = slurry.slurry_metrics(
        state,
        batch["powder_true_density_g_ml"],
        batch["water_density_g_ml"],
    )
    overfill = metrics["occupied_volume_ml"] > batch["container_capacity_ml"] + 1e-9
    return {
        **{k: round(v, 9) for k, v in metrics.items()},
        "breakdown": _state_breakdown(state),
        "container_overfill": overfill,
    }


def start_slurry_batch(batch_id: str) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("planned",))
    db.update_slurry_status(batch_id, "mixing")
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def add_slurry_entry(batch_id: str, req: SlurryAdditionRequest) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    dry_items = [
        {"material_id": d.material_id, "mass_g": d.mass_g}
        for d in req.dry_materials
    ]
    _check_recipe_materials(batch, [d["material_id"] for d in dry_items])
    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    dry_g = sum(d["mass_g"] for d in dry_items)
    state.add_direct(dry_g, req.water_g, req.additive_g)
    payload = {
        "dry_materials": dry_items,
        "water_g": req.water_g,
        "additive_g": req.additive_g,
        "note": req.note,
    }
    db.append_slurry_entry(batch_id, "addition", payload, _metrics_after(batch, state))
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def add_slurry_recycle(batch_id: str, req: SlurryRecycleRequest) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    source = db.get_slurry_batch(req.source_batch_id)  # 404
    if source["version_id"] != batch["version_id"]:
        raise GlazeError(
            f"回收浆来源批次 {source['id']} 的配方版本 {source['version_id']} "
            f"与本批次 {batch['version_id']} 不一致，禁止混入",
            "recycle_version_mismatch",
            {
                "source_batch_id": source["id"],
                "source_version_id": source["version_id"],
                "batch_version_id": batch["version_id"],
            },
        )
    if source["status"] != "finalized" or not source.get("freeze_id"):
        raise GlazeError(
            f"回收浆来源批次 {source['id']} 尚未定稿，不能作为回收浆登记",
            "recycle_source_not_finalized",
            {"source_batch_id": source["id"], "status": source["status"]},
        )
    source_entries = db.list_slurry_entries(source["id"])
    source_state = _replay(source, source_entries)
    source_metrics = slurry.slurry_metrics(
        source_state,
        source["powder_true_density_g_ml"],
        source["water_density_g_ml"],
    )
    split = slurry.split_recycle_mass(req.slurry_mass_g, source_metrics)

    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    state.add_recycle(
        split["dry_g"], split["water_g"], split["additive_g"]
    )
    payload = {
        "source_batch_id": source["id"],
        "source_version_id": source["version_id"],
        "slurry_mass_g": req.slurry_mass_g,
        "source_solids_fraction": round(split["solids_fraction"], 9),
        "split": {k: round(v, 9) for k, v in split.items()},
        "note": req.note,
    }
    db.append_slurry_entry(
        batch_id, "recycle", payload, _metrics_after(batch, state)
    )
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def add_slurry_premix(batch_id: str, req: SlurryPremixRequest) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    shares = [
        {"material_id": m["material_id"], "share": m["share"]}
        for m in batch["recipe_snapshot"]["materials"]
    ]
    breakdown = slurry.split_premix(req.premix_mass_g, shares)
    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    state.add_premix(req.premix_mass_g, req.water_g)
    payload = {
        "premix_mass_g": req.premix_mass_g,
        "water_g": req.water_g,
        "breakdown": [
            {
                "material_id": line["material_id"],
                "share": round(line["share"], 9),
                "mass_g": round(line["mass_g"], 6),
            }
            for line in breakdown
        ],
        "note": req.note,
    }
    db.append_slurry_entry(batch_id, "premix", payload, _metrics_after(batch, state))
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def add_slurry_reading(batch_id: str, req: SlurryReadingRequest) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    metrics = slurry.slurry_metrics(
        state,
        batch["powder_true_density_g_ml"],
        batch["water_density_g_ml"],
    )
    net_mass = req.full_cup_mass_g - req.empty_cup_mass_g
    measured = net_mass / req.cup_volume_ml
    rho_p = batch["powder_true_density_g_ml"]
    rho_w = batch["water_density_g_ml"]

    reason = None
    closed = False
    if metrics["total_mass_g"] <= 0.0:
        reason = "批次内尚无任何物料，无法给出理论比重"
    else:
        theoretical = metrics["theoretical_density_g_ml"]
        diff = abs(measured - theoretical)
        closed = diff <= DENSITY_CLOSURE_TOLERANCE_G_ML + 1e-12
        if not closed:
            reason = (
                f"实测比重 {measured:.4f} 与理论比重 {theoretical:.4f} 之差 "
                f"{diff:.4f} 超过闭合容差 {DENSITY_CLOSURE_TOLERANCE_G_ML}"
            )
    implied_solids = slurry.implied_solids_from_density(measured, rho_p, rho_w)
    physically_plausible = rho_w - 1e-6 <= measured <= rho_p + 1e-6
    assessment = {
        "measured_density_g_ml": round(measured, 9),
        "net_cup_mass_g": round(net_mass, 9),
        "theoretical_density_g_ml": round(metrics["theoretical_density_g_ml"], 9)
        if metrics["total_mass_g"] > 0.0
        else None,
        "difference_g_ml": round(
            abs(measured - metrics["theoretical_density_g_ml"]), 9
        )
        if metrics["total_mass_g"] > 0.0
        else None,
        "tolerance_g_ml": DENSITY_CLOSURE_TOLERANCE_G_ML,
        "implied_solids_fraction": round(implied_solids, 9)
        if implied_solids is not None
        else None,
        "physically_plausible": physically_plausible,
        "closed": closed,
        "reason": reason,
    }
    payload = {
        "empty_cup_mass_g": req.empty_cup_mass_g,
        "full_cup_mass_g": req.full_cup_mass_g,
        "cup_volume_ml": req.cup_volume_ml,
        "note": req.note,
    }
    db.append_slurry_entry(batch_id, "reading", payload, assessment)
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def search_slurry_corrections(
    batch_id: str, req: SlurryCorrectionRequest
) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    current = slurry.slurry_metrics(
        state,
        batch["powder_true_density_g_ml"],
        batch["water_density_g_ml"],
    )
    result = slurry.search_corrections(
        current=current,
        solids_low=batch["solids_low"],
        solids_high=batch["solids_high"],
        density_low=batch["density_low"],
        density_high=batch["density_high"],
        powder_density_g_ml=batch["powder_true_density_g_ml"],
        water_density_g_ml=batch["water_density_g_ml"],
        container_capacity_ml=batch["container_capacity_ml"],
        water_step_g=req.water_step_g,
        premix_step_g=req.premix_step_g,
        remaining_capacity_ml=req.remaining_capacity_ml,
        max_candidates=req.max_candidates,
    )
    result["batch_id"] = batch["id"]
    result["search_constraints"] = {
        "water_step_g": req.water_step_g,
        "premix_step_g": req.premix_step_g,
        "remaining_capacity_ml": req.remaining_capacity_ml,
        "max_candidates": req.max_candidates,
    }
    result["premix_breakdown"] = [
        {
            "material_id": m["material_id"],
            "name": m["name"],
            "share": round(m["share"], 9),
        }
        for m in batch["recipe_snapshot"]["materials"]
    ]
    result["constants_version"] = CONSTANTS_VERSION
    return result


def add_slurry_adjustment(batch_id: str, req: SlurryAdjustmentRequest) -> dict[str, Any]:
    batch = db.get_slurry_batch(batch_id)  # 404
    _require_status(batch, ("mixing",))
    shares = [
        {"material_id": m["material_id"], "share": m["share"]}
        for m in batch["recipe_snapshot"]["materials"]
    ]
    breakdown = slurry.split_premix(req.premix_g, shares)
    entries = db.list_slurry_entries(batch_id)
    state = _replay(batch, entries)
    state.add_premix(req.premix_g, req.water_g)
    after = _metrics_after(batch, state)
    payload = {
        "water_g": req.water_g,
        "premix_g": req.premix_g,
        "premix_breakdown": [
            {
                "material_id": line["material_id"],
                "share": round(line["share"], 9),
                "mass_g": round(line["mass_g"], 6),
            }
            for line in breakdown
        ],
        "note": req.note,
    }
    db.append_slurry_entry(batch_id, "adjustment", payload, after)
    return assemble_slurry_batch(db.get_slurry_batch(batch_id))


def finalize_slurry_batch(batch_id: str, req: SlurryFinalizeRequest | None = None):
    """定稿：冻结全部投料、读数、调整记录与计算常量。重复定稿返回同一结果。"""
    batch = db.get_slurry_batch(batch_id)  # 404
    if batch["status"] == "finalized" and batch.get("freeze_id"):
        stored = db.get_slurry_freeze(batch["freeze_id"])
        return stored, False
    _require_status(batch, ("mixing",))

    view = assemble_slurry_batch(batch)
    if view["state"]["dry_mass_g"] <= 0.0:
        raise GlazeError(
            "批次内没有任何干物投入，无法定稿", "empty_slurry_batch"
        )

    constants = {
        "constants_version": CONSTANTS_VERSION,
        "segger_constants": constants_snapshot(),
        "powder_true_density_g_ml": batch["powder_true_density_g_ml"],
        "water_temp_c": batch["water_temp_c"],
        "water_density_g_ml": batch["water_density_g_ml"],
        "water_density_table": [
            {"temp_c": t, "density_g_ml": d}
            for t, d in slurry.WATER_DENSITY_TABLE
        ],
        "container_capacity_ml": batch["container_capacity_ml"],
        "density_closure_tolerance_g_ml": DENSITY_CLOSURE_TOLERANCE_G_ML,
        "additive_ratio": batch["additive_ratio"],
    }
    final_state = {
        **view["state"],
        "warnings": view["warnings"],
    }
    snapshot = {
        "batch_id": batch["id"],
        "version_id": batch["version_id"],
        "targets": view["targets"],
        "initial_plan": batch["initial_plan"],
        "recipe_snapshot": batch["recipe_snapshot"],
        "entries": view["entries"],
        "constants": constants,
        "note": req.note if req is not None else batch["note"],
    }
    # 冻结 id 取决于全部语义内容：同批次台账不变则哈希稳定（幂等兜底）
    freeze_hash = _canonical_hash(
        {
            "batch_id": batch["id"],
            "final_state": final_state,
            "snapshot": snapshot,
        }
    )
    freeze_id = "slf_" + freeze_hash
    stored, created = db.save_slurry_freeze(
        freeze_id=freeze_id,
        batch_id=batch["id"],
        note=req.note if req is not None else batch["note"],
        final_state=final_state,
        snapshot=snapshot,
    )
    db.update_slurry_status(batch["id"], "finalized", freeze_id=stored["id"])
    return stored, created
