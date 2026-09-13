"""配方混合试验：冻结配方混料网格、逐格秤量修正与母料拆分搜索。

口径与单位
----------
* 冻结配方投料量单位为 kg，试验以"单片干料量"克数（g）计算，
  内部按 ``KG_PER_G = 1/1000`` 换算为 kg 后调用化学计算。
* 电子秤分度 ``division`` 为最小读数（g）；最小称量 ``minimum_weighed``
  为该秤可靠称量下限（g）。所有实际称量值均为分度整数倍且不得小于
  最小称量（零值除外）。

试验创建
--------
两至三份冻结配方按线性（2 份）或三元三角（3 份）比例混合；每个格位
（试片）给出各来源比例（合计为 1）。相同原料跨来源合并用量后，按
单片干料量计算每格投料，再以分度舍入。舍入采用最大余数法把残余分度
补到理论量最大的原料上，保证逐格实际总料量恰为目标干料量；理论量
为正却不足一个分度（被抹零）或达不到最小称量时，拒绝创建并返回
对应格位。

母料拆分
--------
母料只提取**所有格位共用**的原料（"各格共有用量"）。每种入料原料按
分度整数倍 ``j`` 从母料逐格等分（共 n 格，母料配料 n·j 单位）；逐格
差额补料，等分与补料同样受最小称量约束。

若调用方限定母料批量 T（分度整数倍），母料各原料配料量按需求比例
以最大余数法凑足 T：等分 n·j 单位后，多配的 e 单位留在料盆作为
剩余料，要求 0 ≤ 剩余 ≤ allowed_leftover_g。等分保证逐格实际釉式
与冻结方案完全一致（最大成分偏差为 0）；逐格补料越少，称量次数越少。

方案按 (最大成分偏差, 称量次数, 最小称量余量, 剩余料) 字典序排序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .chemistry import calc_batch

KG_PER_G = 1.0 / 1000.0
# 网格（分度整数倍）判定的相对容差
GRID_EPS = 1e-7
# 视为零的理论量（g）：小于该尺度不可能称量
ZERO_EPS = 1e-9


def round_int(x: float) -> int:
    """四舍五入到最近整数（0.5 向上；理论闭合的输入无歧义）。"""
    n = int(x)
    if x - n >= 0.5 - GRID_EPS:
        return n + 1
    return n


def is_on_grid(value: float, division: float) -> bool:
    n = value / division
    return abs(n - round(n)) <= GRID_EPS


# ---------------------------------------------------------------------------
# 来源配方与原料合并
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSource:
    """一份冻结配方在试验创建时的快照（来源版本 id + 干料归一化份额）。"""

    version_id: str
    note: Optional[str]
    # material_id -> kg/kg 干料
    shares: dict[int, float]
    # material_id -> 冻结时的原始投料量（kg）
    original_amounts: dict[int, float]


@dataclass(frozen=True)
class SourceMaterial:
    """合并后的来源原料（相同原料跨来源按比例叠加）。"""

    material_id: int
    name: str
    oxides: dict[str, float]
    loi: float
    price: float
    # 各来源配方归一化后该原料的 kg/kg 干料份额，顺序与 sources 一致
    fractions: tuple[float, ...]


def build_sources(versions: Sequence[dict[str, Any]]) -> tuple[FrozenSource, ...]:
    """从冻结版本记录构造归一化来源。

    冻结版本的 ``items`` 若存在同料多条（历史数据），先按原料合计
    再以该版本批量归一化为干料份额。
    """
    sources: list[FrozenSource] = []
    for v in versions:
        totals: dict[int, float] = {}
        for item in v["items"]:
            amt = float(item["amount"])
            if amt > 0.0:
                mid = int(item["material_id"])
                totals[mid] = totals.get(mid, 0.0) + amt
        batch_mass = sum(totals.values())
        if batch_mass <= 0.0:
            raise ValueError(f"来源配方 {v['id']} 没有正用量原料")
        sources.append(
            FrozenSource(
                version_id=v["id"],
                note=v.get("note"),
                shares={mid: amt / batch_mass for mid, amt in totals.items()},
                original_amounts=dict(sorted(totals.items())),
            )
        )
    return tuple(sources)


def merge_materials(
    sources: Sequence[FrozenSource],
    material_snapshots: dict[int, dict[str, Any]],
) -> list[SourceMaterial]:
    """合并相同原料：叠加各来源份额，分析/价格取冻结快照（首现者）。"""
    merged: dict[int, SourceMaterial] = {}
    n_src = len(sources)
    for s_idx, src in enumerate(sources):
        for mid, share in src.shares.items():
            snap = material_snapshots[mid]
            existing = merged.get(mid)
            if existing is None:
                fractions = [0.0] * n_src
                fractions[s_idx] = share
                merged[mid] = SourceMaterial(
                    material_id=mid,
                    name=snap["name"],
                    oxides=snap["oxides"],
                    loi=snap["loi"],
                    price=snap["price"],
                    fractions=tuple(fractions),
                )
            else:
                fractions = list(existing.fractions)
                fractions[s_idx] = share
                merged[mid] = SourceMaterial(
                    material_id=mid,
                    name=existing.name,
                    oxides=existing.oxides,
                    loi=existing.loi,
                    price=existing.price,
                    fractions=tuple(fractions),
                )
    return [merged[mid] for mid in sorted(merged)]


# ---------------------------------------------------------------------------
# 逐格投料与秤量舍入
# ---------------------------------------------------------------------------


def dose_cells(
    sources: Sequence[FrozenSource],
    material_snapshots: dict[int, dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    dry_mass_g: float,
    division: float,
    minimum_weighed: float,
) -> tuple[list[SourceMaterial], list[dict[str, Any]]]:
    """合并原料、逐格计算投料并按电子秤分度舍入。

    任一格位不可行时抛出 :class:`GlazeError`，details.cells 列出全部
    不可行格位及其原因；否则返回 (合并后原料, 每格记录)。
    """
    materials = merge_materials(sources, material_snapshots)

    failures: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    for cell in cells:
        weights = [float(w) for w in cell["weights"]]
        theoretical: dict[int, float] = {}
        for mat in materials:
            share = sum(f * w for f, w in zip(mat.fractions, weights))
            grams = dry_mass_g * share
            if grams > ZERO_EPS:
                theoretical[mat.material_id] = grams

        units, fail = _round_cell(
            cell["position"], theoretical, division, minimum_weighed
        )
        if fail is not None:
            failures.append(fail)
            continue
        record = _build_cell_record(
            cell["position"], weights, theoretical, units, materials,
            dry_mass_g, division,
        )
        cell_records.append(record)

    if failures:
        from .errors import GlazeError

        raise GlazeError(
            "部分格位按秤的分度舍入后无法凑足批量或达到最小称量，拒绝创建",
            "cell_rounding_infeasible",
            {"cells": failures},
        )
    return materials, cell_records


def _round_cell(
    position: str,
    theoretical: dict[int, float],
    division: float,
    minimum_weighed: float,
) -> tuple[Optional[dict[int, int]], Optional[dict[str, Any]]]:
    """单格最大余数法舍入，返回 (units, failure)。

    先下取整，再把残余分度按余数从大到小补发，使实际总料量恰为目标
    干料量（理论量按比例闭合时总单位必为整数）。任一正用量最终为零
    （不足一个分度）或低于最小称量，记为不可行。
    """
    target_units = round_int(sum(theoretical.values()) / division)
    floors: dict[int, int] = {}
    remainders: dict[int, float] = {}
    for mid, grams in theoretical.items():
        q = grams / division
        floors[mid] = int(q)
        remainders[mid] = q - int(q)

    units: dict[int, int] = {mid: f for mid, f in floors.items() if f >= 1}
    shortfall = target_units - sum(units.values())
    order = sorted(theoretical, key=lambda m: (-remainders[m], -theoretical[m], m))
    for mid in order:
        if shortfall <= 0:
            break
        units[mid] = units.get(mid, 0) + 1
        shortfall -= 1

    issues: list[dict[str, Any]] = []
    if shortfall != 0 or sum(units.values()) != target_units:
        issues.append({
            "material_id": None,
            "reason": "batch_unreachable",
            "target_units": target_units,
            "actual_units": sum(units.values()),
        })
    for mid, grams in theoretical.items():
        weighed = units.get(mid, 0) * division
        if mid not in units:
            issues.append({
                "material_id": mid,
                "theoretical_g": round(grams, 6),
                "weighed_g": 0.0,
                "reason": "rounds_to_zero",
            })
        elif weighed < minimum_weighed - GRID_EPS * division:
            issues.append({
                "material_id": mid,
                "theoretical_g": round(grams, 6),
                "weighed_g": round(weighed, 6),
                "reason": "below_minimum_weighed",
            })

    if issues:
        return None, {"position": position, "issues": issues}
    return units, None


def _build_cell_record(
    position: str,
    weights: list[float],
    theoretical: dict[int, float],
    units: dict[int, int],
    materials: Sequence[SourceMaterial],
    dry_mass_g: float,
    division: float,
) -> dict[str, Any]:
    """组装单格结果：投料表、舍入误差、釉式、烧后质量、成本。"""
    mat_by_id = {m.material_id: m for m in materials}
    doses: list[dict[str, Any]] = []
    actual_tuples = []
    theory_tuples = []
    total_weighed = 0.0
    for mid in sorted(units):
        mat = mat_by_id[mid]
        u = units[mid]
        weighed_g = u * division
        theory_g = theoretical.get(mid, 0.0)
        total_weighed += weighed_g
        actual_tuples.append(
            (mid, mat.name, mat.oxides, mat.loi, mat.price,
             weighed_g * KG_PER_G)
        )
        doses.append({
            "material_id": mid,
            "name": mat.name,
            "theoretical_g": round(theory_g, 6),
            "weighed_g": round(weighed_g, 6),
            "units": u,
            "rounding_error_g": round(weighed_g - theory_g, 6),
        })
    for mid, grams in theoretical.items():
        mat = mat_by_id[mid]
        theory_tuples.append(
            (mid, mat.name, mat.oxides, mat.loi, mat.price,
             grams * KG_PER_G)
        )

    actual = calc_batch(actual_tuples, targets={})
    theory = calc_batch(theory_tuples, targets={})
    seger_shift = {
        o: round(actual["seger"].get(o, 0.0) - theory["seger"].get(o, 0.0), 9)
        for o in sorted(set(actual["seger"]) | set(theory["seger"]))
    }

    return {
        "position": position,
        "weights": weights,
        "dry_mass_g": dry_mass_g,
        "total_weighed_g": round(total_weighed, 6),
        "total_rounding_error_g": round(total_weighed - dry_mass_g, 6),
        "doses": doses,
        "fired_mass_g": round(actual["fired_mass"] / KG_PER_G, 6),
        "loss_on_ignition_g": round(actual["loss_on_ignition"] / KG_PER_G, 6),
        "cost": round(actual["cost"], 9),
        "seger": actual["seger"],
        "seger_shift_from_theoretical": seger_shift,
        "max_abs_seger_shift": round(
            max((abs(v) for v in seger_shift.values()), default=0.0), 9
        ),
        "oxide_moles": actual["oxide_moles"],
        "flux_moles": actual["flux_moles"],
    }


# ---------------------------------------------------------------------------
# 母料（共有料）+ 逐格补料 方案搜索
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MasterProblem:
    """母料搜索的冻结试验上下文（单位均为 g）。"""

    positions: tuple[str, ...]
    material_ids: tuple[int, ...]
    names: dict[int, str]
    oxides: dict[int, dict[str, float]]
    loi: dict[int, float]
    price: dict[int, float]
    # 逐格实际称量单位（分度倍数）：u[(mid, pos)]，缺格视为 0
    units: dict[tuple[int, str], int]
    division: float
    minimum_weighed: float
    dry_mass_g: float
    baseline_seger: dict[str, dict[str, float]]  # pos -> 逐格实际釉式


def build_master_problem(experiment: dict[str, Any]) -> MasterProblem:
    """从冻结试验记录还原母料搜索问题。"""
    setup = experiment["setup"]
    division = float(setup["scale_division_g"])
    minimum_weighed = float(setup["minimum_weighed_g"])
    dry_mass_g = float(setup["dry_mass_per_tile_g"])
    positions: list[str] = []
    units: dict[tuple[int, str], int] = {}
    mids: set[int] = set()
    names: dict[int, str] = {}
    oxides: dict[int, dict[str, float]] = {}
    loi: dict[int, float] = {}
    price: dict[int, float] = {}
    baseline: dict[str, dict[str, float]] = {}
    for cell in experiment["result"]["cells"]:
        pos = cell["position"]
        positions.append(pos)
        baseline[pos] = cell["seger"]
        for d in cell["doses"]:
            mid = int(d["material_id"])
            mids.add(mid)
            units[(mid, pos)] = int(d["units"])
    for mid in sorted(mids):
        snap = experiment["material_snapshot"][str(mid)]
        names[mid] = snap["name"]
        oxides[mid] = snap["oxides"]
        loi[mid] = snap["loi"]
        price[mid] = snap["price"]
    return MasterProblem(
        positions=tuple(positions),
        material_ids=tuple(sorted(mids)),
        names=names,
        oxides=oxides,
        loi=loi,
        price=price,
        units=units,
        division=division,
        minimum_weighed=minimum_weighed,
        dry_mass_g=dry_mass_g,
        baseline_seger=baseline,
    )


def _skeleton_lambdas(n_materials: int) -> list[dict[int, float]]:
    """生成候选骨架（统一覆盖比例 λ、留一与单料），覆盖不同拆分粒度。"""
    indices = list(range(n_materials))
    skeletons: list[dict[int, float]] = []
    # 1) 统一覆盖比例 0.05..1.0
    for k in range(1, 21):
        lam = round(0.05 * k, 2)
        skeletons.append({i: lam for i in indices})
    # 2) λ=1 时逐料留一
    for skip in indices:
        skeletons.append({i: 1.0 for i in indices if i != skip})
    # 3) 单料入母料
    for i in indices:
        skeletons.append({i: 1.0})
    seen: set[tuple] = set()
    uniq: list[dict[int, float]] = []
    for sk in skeletons:
        key = tuple(sorted(sk.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(sk)
    return uniq


def search_master_plans(
    prob: MasterProblem,
    *,
    master_batch_g: Optional[float] = None,
    master_minimum_weighed: Optional[float] = None,
    allowed_leftover_g: Optional[float] = None,
    max_candidates: int = 10,
) -> dict[str, Any]:
    """搜索"母料 + 逐格补料"方案并排序。

    固定批量（``master_batch_g``）时，母料配料总量恰为 T，等分需求
    D=n·d·Σj 之外多配的部分留作剩余料，要求剩余 ≤ allowed_leftover_g；
    不给定批量时母料按等分需求精确配制（剩余 0）。
    """
    from .errors import GlazeError

    division = prob.division
    min_w = (
        float(master_minimum_weighed)
        if master_minimum_weighed is not None
        else prob.minimum_weighed
    )
    if min_w <= 0.0 or division <= 0.0:
        raise GlazeError("母料称量参数必须为正", "invalid_master_parameter")
    if min_w < division - GRID_EPS * division:
        raise GlazeError(
            "母料最小称量不得小于电子秤分度",
            "minimum_below_division",
            {"minimum_weighed_g": min_w, "division_g": division},
        )

    fixed = master_batch_g is not None
    if fixed:
        target_g = float(master_batch_g)
        if target_g <= 0.0:
            raise GlazeError("母料批量必须为正", "invalid_master_parameter")
        if not is_on_grid(target_g, division):
            raise GlazeError(
                f"母料批量 {target_g} g 不是分度 {division} g 的整数倍",
                "master_batch_not_on_grid",
                {"master_batch_g": target_g, "division_g": division},
            )
        leftover_cap = (
            float(allowed_leftover_g) if allowed_leftover_g is not None else 0.0
        )
        if leftover_cap < 0.0:
            raise GlazeError("允许剩余量不得为负", "invalid_master_parameter")
    else:
        target_g = None
        leftover_cap = None

    n_cells = len(prob.positions)
    # 母料只收所有格位共用的原料：某格缺料（端点格常见）则不具共有性。
    u_min: dict[int, int] = {
        mid: min(prob.units.get((mid, pos), 0) for pos in prob.positions)
        for mid in prob.material_ids
    }
    common = [mid for mid in prob.material_ids if u_min[mid] > 0]

    plans: list[dict[str, Any]] = []
    seen_sig: set[tuple] = set()

    def consider(j: dict[int, int]) -> None:
        j_common = {mid: j.get(mid, 0) for mid in common if j.get(mid, 0) > 0}
        sig = tuple(sorted(j_common.items()))
        if sig in seen_sig:
            return
        seen_sig.add(sig)
        plan = _evaluate_plan(
            prob, j_common, u_min, min_w,
            fixed=fixed, target_g=target_g, leftover_cap=leftover_cap,
        )
        if plan is not None:
            plans.append(plan)

    if not fixed:
        consider({})  # 无母料基线

    # 骨架只在共有料上展开（非共有料永远逐格单独称量）
    for skel in _skeleton_lambdas(len(common)):
        # 主选 j：可行（j ≤ u_min）范围内最接近 λ·u_min 的非零整数
        j_round: dict[int, int] = {}
        j_floor: dict[int, int] = {}
        active = False
        for i, mid in enumerate(common):
            lam = skel.get(i, 0.0)
            cap = u_min[mid]
            if lam <= 0.0:
                j_round[mid] = 0
                j_floor[mid] = 0
                continue
            target = lam * cap
            jr = min(max(round_int(target), 1), cap)
            j_round[mid] = jr
            j_floor[mid] = min(max(int(target), 1 if target >= 1.0 else 0), cap)
            active = True
        if active:
            consider(j_round)
            if fixed and j_floor != j_round:
                consider(j_floor)

    if not plans:
        raise GlazeError(
            "给定母料批量与剩余量限制下找不到可行拆分方案",
            "no_master_plan",
            {
                "master_batch_g": target_g,
                "allowed_leftover_g": leftover_cap,
                "n_cells": n_cells,
            },
        )

    plans.sort(key=_plan_sort_key)
    ranked = plans[:max_candidates]
    return {
        "parameters": {
            "master_batch_g": target_g,
            "master_minimum_weighed_g": min_w,
            "allowed_leftover_g": leftover_cap,
            "scale_division_g": division,
            "fixed_batch": fixed,
            "n_cells": n_cells,
        },
        "n_feasible": len(plans),
        "plans": ranked,
        "best": ranked[0],
    }


def _plan_sort_key(p: dict[str, Any]):
    # 最大成分偏差、称量次数、（负）最小称量余量、剩余料，签名兜底稳定
    return (
        round(p["max_abs_seger_shift"], 9),
        p["total_weighings"],
        -round(p["min_weighing_margin_g"], 9),
        round(p["leftover_g"], 9),
        p["signature"],
    )


def _evaluate_plan(
    prob: MasterProblem,
    j: dict[int, int],
    u_min: dict[int, int],
    master_minimum_weighed: float,
    *,
    fixed: bool,
    target_g: Optional[float],
    leftover_cap: Optional[float],
) -> Optional[dict[str, Any]]:
    """评估单个等分向量 j；违反约束返回 None。"""
    division = prob.division
    n_cells = len(prob.positions)
    included = {mid: jj for mid, jj in j.items() if jj > 0}
    if not included:
        return None if fixed else _baseline_plan(prob)

    # 母料配料单位 = j × 格数；配料最小称量用母料秤参数
    bulk_units = {mid: jj * n_cells for mid, jj in included.items()}
    for mid, b in bulk_units.items():
        if b * division < master_minimum_weighed - GRID_EPS * division:
            return None

    # 固定批量：需求总量 D 与剩余 L = T - D
    demanded_units = sum(bulk_units.values())
    demanded_g = demanded_units * division
    if fixed:
        target_units = round_int(target_g / division)
        leftover_units = target_units - demanded_units
        if leftover_units < 0:
            return None
        if leftover_units * division - leftover_cap > GRID_EPS * division:
            return None
        leftover_g = leftover_units * division
        # 母料各原料配料量：按需求比例把 T 单位以最大余数法分配；
        # 等分只取 j·n 单位，多分的部分留在盆中作剩余料（不入格）。
        prepared_units: dict[int, int] = {}
        fracs: list[tuple[float, int]] = []
        for mid, b in bulk_units.items():
            x = b * target_units / demanded_units
            prepared_units[mid] = int(x)
            fracs.append((x - int(x), mid))
        extra_total = target_units - sum(prepared_units.values())
        for _, mid in sorted(fracs, key=lambda t: (-t[0], t[1]))[:extra_total]:
            prepared_units[mid] += 1
    else:
        leftover_g = 0.0
        prepared_units = dict(bulk_units)

    cell_plans: list[dict[str, Any]] = []
    # 称量次数：母料配料每种原料 1 次 + 每格取 1 次混合料等分 + 逐格补料
    n_bulk = len(bulk_units)
    aliquot_total_units = sum(included.values())  # 每格等分总单位（各料相同）
    aliquot_total_g = aliquot_total_units * division
    n_cells_with_aliquot = sum(
        1 for pos in prob.positions
        if any(prob.units.get((mid, pos), 0) > 0 for mid in included)
    )
    n_aliquot = n_cells_with_aliquot
    n_topup = 0
    margins: list[float] = []
    for mid, b in prepared_units.items():
        margins.append(b * division - master_minimum_weighed)

    for pos in prob.positions:
        master_doses: list[dict[str, Any]] = []
        topup_doses: list[dict[str, Any]] = []
        tuples = []
        cell_aliquot_units = 0
        for mid in prob.material_ids:
            u = prob.units.get((mid, pos), 0)
            if mid in included:
                a = included[mid]
                # 共有性：入母料的原料必须每格都用到，且等分不超过该格用量
                if u <= 0 or a > u:
                    return None
                master_doses.append(_dose_entry(prob, mid, a))
                cell_aliquot_units += a
                top = u - a
                if top > 0:
                    if top * division < prob.minimum_weighed - GRID_EPS * division:
                        return None  # 补料低于最小称量
                    topup_doses.append(_dose_entry(prob, mid, top))
                    margins.append(top * division - prob.minimum_weighed)
                    n_topup += 1
                tuples.append(_tuple(prob, mid, u))
            elif u > 0:
                if u * division < prob.minimum_weighed - GRID_EPS * division:
                    return None
                topup_doses.append(_dose_entry(prob, mid, u))
                margins.append(u * division - prob.minimum_weighed)
                n_topup += 1
                tuples.append(_tuple(prob, mid, u))

        if cell_aliquot_units > 0:
            # 混合料等分作为一次称量，总量不得低于最小称量
            aliquot_g = cell_aliquot_units * division
            if aliquot_g < prob.minimum_weighed - GRID_EPS * division:
                return None
            margins.append(aliquot_g - prob.minimum_weighed)

        actual = calc_batch(tuples, targets={})
        base = prob.baseline_seger[pos]
        shift = {
            o: round(actual["seger"].get(o, 0.0) - base.get(o, 0.0), 9)
            for o in sorted(set(actual["seger"]) | set(base))
        }
        cell_max = max((abs(v) for v in shift.values()), default=0.0)
        cell_plans.append({
            "position": pos,
            "master_aliquot": {
                "total_units": cell_aliquot_units,
                "total_g": round(cell_aliquot_units * division, 6),
                "components": master_doses,
            } if cell_aliquot_units else None,
            "topup_doses": topup_doses,
            "seger": actual["seger"],
            "seger_shift": shift,
            "max_abs_seger_shift": round(cell_max, 9),
        })

    max_dev = max(
        (cp["max_abs_seger_shift"] for cp in cell_plans), default=0.0
    )
    signature = ",".join(
        f"{mid}:{included[mid]}" for mid in sorted(included)
    )
    prepared_g = sum(prepared_units.values()) * division
    return {
        "signature": signature,
        "included_material_ids": sorted(included),
        "bulk": [
            {
                "material_id": mid,
                "name": prob.names[mid],
                "prepared_units": prepared_units[mid],
                "prepared_g": round(prepared_units[mid] * division, 6),
                "aliquot_units": included[mid],
                "aliquot_g": round(included[mid] * division, 6),
                "n_aliquots": n_cells,
            }
            for mid in sorted(prepared_units)
        ],
        "aliquot_total_g": round(aliquot_total_g, 6),
        "master_batch_prepared_g": round(prepared_g, 6),
        "master_consumed_g": round(demanded_g, 6),
        "leftover_g": round(leftover_g, 6),
        "total_weighings": n_bulk + n_aliquot + n_topup,
        "n_bulk_weighings": n_bulk,
        "n_aliquot_weighings": n_aliquot,
        "n_topup_weighings": n_topup,
        "min_weighing_margin_g": round(min(margins), 6),
        "max_abs_seger_shift": round(max_dev, 9),
        "cells": cell_plans,
        "weighing_order": _weighing_order(prob, included, prepared_units),
    }


def _baseline_plan(prob: MasterProblem) -> dict[str, Any]:
    """无母料基线：全部逐格按冻结方案称量。"""
    cell_plans: list[dict[str, Any]] = []
    total = 0
    margins: list[float] = []
    for pos in prob.positions:
        topup_doses: list[dict[str, Any]] = []
        tuples = []
        for mid in prob.material_ids:
            u = prob.units.get((mid, pos), 0)
            if u > 0:
                topup_doses.append(_dose_entry(prob, mid, u))
                tuples.append(_tuple(prob, mid, u))
                total += 1
                margins.append(u * prob.division - prob.minimum_weighed)
        actual = calc_batch(tuples, targets={})
        cell_plans.append({
            "position": pos,
            "master_aliquot": None,
            "topup_doses": topup_doses,
            "seger": actual["seger"],
            "seger_shift": {o: 0.0 for o in sorted(actual["seger"])},
            "max_abs_seger_shift": 0.0,
        })
    return {
        "signature": "none",
        "included_material_ids": [],
        "bulk": [],
        "master_batch_prepared_g": 0.0,
        "master_consumed_g": 0.0,
        "leftover_g": 0.0,
        "total_weighings": total,
        "n_bulk_weighings": 0,
        "n_aliquot_weighings": 0,
        "n_topup_weighings": total,
        "min_weighing_margin_g": round(min(margins), 6),
        "max_abs_seger_shift": 0.0,
        "cells": cell_plans,
        "weighing_order": _weighing_order(prob, {}, {}),
    }


def _dose_entry(prob: MasterProblem, mid: int, units: int) -> dict[str, Any]:
    return {
        "material_id": mid,
        "name": prob.names[mid],
        "units": units,
        "weighed_g": round(units * prob.division, 6),
    }


def _tuple(prob: MasterProblem, mid: int, units: int):
    return (
        mid,
        prob.names[mid],
        prob.oxides[mid],
        prob.loi[mid],
        prob.price[mid],
        units * prob.division * KG_PER_G,
    )


def _weighing_order(
    prob: MasterProblem,
    included: dict[int, int],
    prepared_units: dict[int, int],
) -> list[dict[str, Any]]:
    """称量顺序：先逐料配母料，再逐格"取一次混合料等分 → 逐料补料"。"""
    order: list[dict[str, Any]] = []
    step = 0
    for mid in sorted(included):
        step += 1
        order.append({
            "step": step,
            "stage": "bulk",
            "position": None,
            "material_id": mid,
            "name": prob.names[mid],
            "action": "weigh_master_batch",
            "weighed_g": round(prepared_units[mid] * prob.division, 6),
        })
    for pos in prob.positions:
        components: list[dict[str, Any]] = []
        aliquot_units = 0
        for mid in sorted(included):
            u = prob.units.get((mid, pos), 0)
            if u <= 0:
                continue
            components.append({
                "material_id": mid,
                "name": prob.names[mid],
                "units": included[mid],
            })
            aliquot_units += included[mid]
        if components:
            step += 1
            order.append({
                "step": step,
                "stage": "aliquot",
                "position": pos,
                "material_id": None,
                "name": "母料等分",
                "action": "take_master_aliquot",
                "weighed_g": round(aliquot_units * prob.division, 6),
                "components": components,
            })
        for mid in prob.material_ids:
            u = prob.units.get((mid, pos), 0)
            top = u - included.get(mid, 0) if mid in included and u > 0 else u
            if top > 0:
                step += 1
                order.append({
                    "step": step,
                    "stage": "topup",
                    "position": pos,
                    "material_id": mid,
                    "name": prob.names[mid],
                    "action": "weigh_topup",
                    "weighed_g": round(top * prob.division, 6),
                })
    return order
