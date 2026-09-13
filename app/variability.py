"""原料批次波动研究：固定种子重采样、统计聚合与稳健配方搜索。

研究对象 = 一份冻结配方（来源版本）+ 一组化验批次，二者构成独立版本。
系统按固定种子对各原料的化验批次做 bootstrap 重采样（可声明同批联动组，
组内原料共用同一批号），逐场景重算 Seger 釉式、烧后质量、成本与目标
偏差，聚合出各氧化物 P5/P50/P95、单项/联合达标率、常见越界组合与
原料敏感度（批次选择对联合达标率的影响幅度）。

稳健配方搜索在来源配方基础上按"步进转移"调整投料（一种原料减量、
另一种等量增量，批量总量不变），受库存（取各批最小可用量）、步进与
最大改动量约束，按 (联合达标率, 最差分位偏差, 改动量, 成本) 字典序选优。
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .chemistry import validate_analysis
from .config import (
    MIN_FLUX_MOLES,
    OXIDE_CATALOG,
    ROBUST_MAX_ITERS,
    SEGER_TOLERANCE,
    TOP_VIOLATION_COMBOS,
)
from .errors import GlazeError

PCT = 100.0
# 贪心搜索的转移步长倍数（× step）
_MOVE_MULTIPLIERS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class StudyTarget:
    oxide: str
    low: Optional[float]
    high: Optional[float]
    weight: float


def _r(x: float, ndigits: int = 9) -> float:
    return round(float(x), ndigits)


# ---------------------------------------------------------------------------
# 化验批次校验与结构性检查
# ---------------------------------------------------------------------------

def normalize_batches(batches) -> dict[int, dict[str, dict[str, Any]]]:
    """校验并归一化化验批次，索引为 {material_id: {批号: 批次}}。

    每批分析与原料入库同口径：未知氧化物 / 负成分 / 合计超差拒绝，
    通过后按同一比例归一化到 氧化物+LOI=100。
    """
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for b in batches:
        oxides, loi = validate_analysis(b.oxides, b.loi, b.analysis_tolerance)
        slot = out.setdefault(b.material_id, {})
        if b.batch in slot:
            raise GlazeError(
                f"原料 {b.material_id} 的批号「{b.batch}」重复提交",
                "duplicate_batch",
                {"material_id": b.material_id, "batch": b.batch},
            )
        slot[b.batch] = {
            "oxides": oxides,
            "loi": loi,
            "price": b.price,
            "available": b.available,
        }
    return out


def check_coverage(recipe_ids: set[int], batches_map) -> None:
    """配方每种原料至少一批化验；不得为配方外原料提交批次。"""
    missing = sorted(recipe_ids - set(batches_map))
    if missing:
        raise GlazeError(
            f"缺少配方依赖原料的化验批次: {missing}",
            "missing_batch_data",
            {"material_ids": missing},
        )
    extra = sorted(set(batches_map) - recipe_ids)
    if extra:
        raise GlazeError(
            f"化验批次引用了配方未使用的原料: {extra}",
            "material_not_in_recipe",
            {"material_ids": extra},
        )


def check_linked_groups(linked_groups, batches_map, recipe_ids: set[int]) -> None:
    """联动组批号配对检查：组内各原料的批号集合必须完全一致。"""
    for group in linked_groups:
        for mid in group:
            if mid not in recipe_ids:
                raise GlazeError(
                    f"联动组引用了配方未使用的原料: {mid}",
                    "material_not_in_recipe",
                    {"material_id": mid, "group": list(group)},
                )
        reference = set(batches_map[group[0]])
        for mid in group[1:]:
            labels = set(batches_map[mid])
            if labels != reference:
                raise GlazeError(
                    f"联动组内批号配对不全: 原料 {mid} 与原料 {group[0]} "
                    "的批号集合不一致",
                    "batch_pairing_incomplete",
                    {
                        "group": list(group),
                        "material_id": mid,
                        "missing": sorted(reference - labels),
                        "extra": sorted(labels - reference),
                    },
                )


# ---------------------------------------------------------------------------
# 固定种子重采样
# ---------------------------------------------------------------------------

def draw_plans(
    material_ids: list[int],
    batches_map,
    linked_groups,
    n_resamples: int,
    seed: int,
) -> list[dict[int, str]]:
    """按固定种子抽取每个重采样场景的批次组合。

    非联动原料各自独立均匀抽取批号；联动组内所有原料共用同一次抽取
    的批号（组内批号集合已校验一致）。抽样调用顺序固定（原料 id 升序、
    联动组按首成员升序），保证同种子结果完全可复现。
    """
    rng = random.Random(seed)
    grouped: dict[int, int] = {}
    groups = []
    for group in linked_groups:
        labels = sorted(batches_map[group[0]])
        groups.append((tuple(group), labels))
        for mid in group:
            grouped[mid] = len(groups) - 1
    plans: list[dict[int, str]] = []
    for _ in range(n_resamples):
        plan: dict[int, str] = {}
        for mid in material_ids:
            if mid in grouped:
                continue
            labels = sorted(batches_map[mid])
            plan[mid] = labels[int(rng.random() * len(labels))]
        for group, labels in groups:
            chosen = labels[int(rng.random() * len(labels))]
            for mid in group:
                plan[mid] = chosen
        plans.append(plan)
    return plans


# ---------------------------------------------------------------------------
# 场景集合的向量化求值
# ---------------------------------------------------------------------------

class ScenarioSet:
    """重采样场景的向量化表示：逐场景的原料分析（摩尔/kg、价格、烧后比）。"""

    def __init__(self, material_ids, batches_map, plans):
        self.material_ids = list(material_ids)
        self.plans = plans
        oxide_set: set[str] = set()
        for mid in self.material_ids:
            for batch in batches_map[mid].values():
                oxide_set.update(batch["oxides"])
        self.oxides = sorted(oxide_set)
        self.oxide_index = {o: i for i, o in enumerate(self.oxides)}
        k = len(plans)
        n = len(self.material_ids)
        self.mole_per_kg = np.zeros((k, len(self.oxides), n))
        self.price = np.zeros((k, n))
        self.fired_ratio = np.zeros((k, n))
        for s, plan in enumerate(plans):
            for i, mid in enumerate(self.material_ids):
                batch = batches_map[mid][plan[mid]]
                for oxide, pct in batch["oxides"].items():
                    self.mole_per_kg[s, self.oxide_index[oxide], i] = (
                        pct / PCT / OXIDE_CATALOG[oxide].molwt
                    )
                self.price[s, i] = batch["price"]
                self.fired_ratio[s, i] = 1.0 - batch["loi"] / PCT
        self.flux_roles = np.array([
            1.0 if OXIDE_CATALOG[o].role == "flux" else 0.0
            for o in self.oxides
        ])

    @property
    def n_scenarios(self) -> int:
        return len(self.plans)

    def evaluate(self, amounts) -> dict[str, Any]:
        """对一组投料量逐场景重算釉式、烧后质量与成本。"""
        a = np.asarray(amounts, dtype=float)
        moles = np.einsum("koi,i->ko", self.mole_per_kg, a)
        flux = moles @ self.flux_roles
        valid = flux > MIN_FLUX_MOLES
        seger = np.full(moles.shape, np.nan)
        seger[valid] = moles[valid] / flux[valid, None]
        return {
            "seger": seger,
            "flux": flux,
            "valid": valid,
            "cost": self.price @ a,
            "fired_mass": self.fired_ratio @ a,
            "batch_mass": float(a.sum()),
        }

    def oxide_values(self, ev, oxide: str) -> np.ndarray:
        idx = self.oxide_index.get(oxide)
        if idx is None:
            return np.zeros(self.n_scenarios)
        return ev["seger"][:, idx]

    def pass_matrix(self, ev, targets: list[StudyTarget]) -> np.ndarray:
        """(场景 × 目标) 达标布尔矩阵；无助熔场景全部不达标。"""
        cols = []
        for t in targets:
            v = self.oxide_values(ev, t.oxide)
            ok = ev["valid"].copy()
            if t.low is not None:
                ok &= v >= t.low - SEGER_TOLERANCE
            if t.high is not None:
                ok &= v <= t.high + SEGER_TOLERANCE
            cols.append(ok)
        if not cols:
            return np.ones((self.n_scenarios, 0), dtype=bool)
        return np.column_stack(cols)


# ---------------------------------------------------------------------------
# 统计聚合
# ---------------------------------------------------------------------------

def _stats(values: np.ndarray) -> dict[str, float]:
    p5, p50, p95 = (float(x) for x in np.percentile(values, [5.0, 50.0, 95.0]))
    return {
        "p5": _r(p5),
        "p50": _r(p50),
        "p95": _r(p95),
        "mean": _r(float(np.mean(values))),
    }


def _deviation_array(v: np.ndarray, low, high) -> np.ndarray:
    """逐场景釉式越界量；与 chemistry._deviations 同口径，容差内计 0。"""
    dev = np.zeros_like(v)
    if low is not None:
        dev = np.maximum(dev, low - v)
    if high is not None:
        dev = np.maximum(dev, v - high)
    return np.where(dev <= SEGER_TOLERANCE, 0.0, dev)


def summarize(
    scenarios: ScenarioSet,
    ev: dict[str, Any],
    targets: list[StudyTarget],
    names: dict[int, str],
) -> dict[str, Any]:
    """单配方在多场景下的完整统计（研究创建与结果冻结共用）。"""
    k = scenarios.n_scenarios
    pm = scenarios.pass_matrix(ev, targets)
    joint = pm.all(axis=1) if targets else np.ones(k, dtype=bool)

    oxide_stats = {}
    for oxide in scenarios.oxides:
        oxide_stats[oxide] = _stats(scenarios.oxide_values(ev, oxide))

    target_stats = {}
    for j, t in enumerate(targets):
        v = scenarios.oxide_values(ev, t.oxide)
        stat = _stats(v)
        target_stats[t.oxide] = {
            "low": t.low,
            "high": t.high,
            "weight": t.weight,
            "pass_rate": _r(float(pm[:, j].mean()), 6),
            "p5": stat["p5"],
            "p50": stat["p50"],
            "p95": stat["p95"],
            "mean_deviation": _r(
                float(_deviation_array(v, t.low, t.high).mean()), 12
            ),
        }

    n_violated = int((~joint).sum())
    combos: Counter = Counter()
    for s in range(k):
        if joint[s]:
            continue
        viol = tuple(t.oxide for j, t in enumerate(targets) if not pm[s, j])
        combos[viol] += 1
    violation_combinations = [
        {
            "oxides": list(combo),
            "count": count,
            "share_of_violations": _r(count / n_violated, 6),
        }
        for combo, count in combos.most_common(TOP_VIOLATION_COMBOS)
    ]

    sensitivity = []
    for mid in scenarios.material_ids:
        labels = sorted({plan[mid] for plan in scenarios.plans})
        rates = {}
        for lab in labels:
            mask = np.array([plan[mid] == lab for plan in scenarios.plans])
            rates[lab] = {
                "pass_rate": _r(float(joint[mask].mean()), 6),
                "draws": int(mask.sum()),
            }
        spreads = [r["pass_rate"] for r in rates.values()]
        sensitivity.append({
            "material_id": mid,
            "name": names.get(mid, str(mid)),
            "n_batches": len(labels),
            "spread": _r(max(spreads) - min(spreads), 6),
            "batch_pass_rates": {lab: r["pass_rate"] for lab, r in rates.items()},
        })
    sensitivity.sort(key=lambda x: (-x["spread"], x["material_id"]))

    return {
        "n_resamples": k,
        "batch_mass": _r(ev["batch_mass"]),
        "joint_pass_rate": _r(float(joint.mean()), 6),
        "n_violation_scenarios": n_violated,
        "oxide_stats": oxide_stats,
        "target_stats": target_stats,
        "violation_combinations": violation_combinations,
        "material_sensitivity": sensitivity,
        "fired_mass": _stats(ev["fired_mass"]),
        "cost": _stats(ev["cost"]),
    }


# ---------------------------------------------------------------------------
# 稳健配方搜索
# ---------------------------------------------------------------------------

def worst_quantile_deviation(
    scenarios: ScenarioSet, ev, targets: list[StudyTarget]
) -> Optional[float]:
    """各目标分位偏差（低端看 P5、高端看 P95）按权重取最大。

    存在无助熔场景时返回 None（视为最差，排序垫底）。
    """
    if not bool(ev["valid"].all()):
        return None
    worst = 0.0
    for t in targets:
        v = scenarios.oxide_values(ev, t.oxide)
        dev = 0.0
        if t.low is not None:
            dev = max(dev, t.low - float(np.percentile(v, 5.0)))
        if t.high is not None:
            dev = max(dev, float(np.percentile(v, 95.0)) - t.high)
        worst = max(worst, t.weight * max(dev, 0.0))
    return worst


def candidate_metrics(
    scenarios: ScenarioSet,
    amounts,
    targets: list[StudyTarget],
    source_amounts: np.ndarray,
) -> dict[str, Any]:
    """稳健搜索候选的评估指标（排序键的全部组成）。"""
    amounts = np.asarray(amounts, dtype=float)
    ev = scenarios.evaluate(amounts)
    pm = scenarios.pass_matrix(ev, targets)
    joint = (
        pm.all(axis=1)
        if targets
        else np.ones(scenarios.n_scenarios, dtype=bool)
    )
    wq = worst_quantile_deviation(scenarios, ev, targets)
    return {
        "amounts": amounts,
        "joint_pass_rate": _r(float(joint.mean()), 6),
        "target_pass_rates": {
            t.oxide: _r(float(pm[:, j].mean()), 6)
            for j, t in enumerate(targets)
        },
        "worst_quantile_deviation": _r(wq, 12) if wq is not None else None,
        "change_amount": _r(float(np.abs(amounts - source_amounts).sum())),
        "cost_mean": _r(float(ev["cost"].mean()), 6),
    }


def _rank_key(metrics: dict) -> tuple:
    """排序键：联合达标率降序、最差分位偏差、改动量、成本升序。"""
    wq = metrics["worst_quantile_deviation"]
    return (
        -metrics["joint_pass_rate"],
        wq if wq is not None else float("inf"),
        metrics["change_amount"],
        metrics["cost_mean"],
    )


def _amount_key(amounts) -> tuple:
    return tuple(round(float(x), 9) for x in amounts)


def robust_search(
    scenarios: ScenarioSet,
    source_amounts,
    stock: dict[int, float],
    step: float,
    max_change: float,
    locked_ids,
    targets: list[StudyTarget],
    max_candidates: int,
    max_iters: int = ROBUST_MAX_ITERS,
) -> list[dict[str, Any]]:
    """步进转移式贪心搜索。

    每步把 t×step（t ∈ 1,2,4,8,16）从一种未锁定原料转移到另一种，
    保持批量总量不变；候选须满足库存上限与最大改动量（L1）约束。
    接受使排序键严格变小的最优转移，直至无改进或达到迭代上限。
    返回来源配方与历代改进 incumbent，按排序键排列。
    """
    source = np.asarray(source_amounts, dtype=float)
    n = len(source)
    locked_idx = {
        scenarios.material_ids.index(mid) for mid in locked_ids
    }
    free = [i for i in range(n) if i not in locked_idx]
    stock_arr = np.array([stock[mid] for mid in scenarios.material_ids])

    current = source.copy()
    current_m = candidate_metrics(scenarios, current, targets, source)
    pool: dict[tuple, dict] = {_amount_key(current): current_m}

    for _ in range(max_iters):
        best_move = None
        best_metrics = None
        best_key = _rank_key(current_m)
        for a in free:
            for b in free:
                if b == a:
                    continue
                room = stock_arr[b] - current[b]
                if room < step - 1e-12:
                    continue
                for mult in _MOVE_MULTIPLIERS:
                    d = mult * step
                    if d > current[a] + 1e-12 or d > room + 1e-12:
                        continue
                    trial = current.copy()
                    trial[a] -= d
                    trial[b] += d
                    if float(np.abs(trial - source).sum()) > max_change + 1e-9:
                        continue
                    m = candidate_metrics(scenarios, trial, targets, source)
                    key = _rank_key(m)
                    if key < best_key:
                        best_key = key
                        best_move = trial
                        best_metrics = m
        if best_move is None:
            break
        current = best_move
        current_m = best_metrics
        pool[_amount_key(current)] = current_m

    ordered = sorted(pool.values(), key=_rank_key)
    return ordered[:max_candidates]
