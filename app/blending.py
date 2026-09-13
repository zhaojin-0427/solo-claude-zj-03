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
母料只提取**所有格位共用**的原料（"各格共有用量"）。

* 未限定母料批量时：每种入料原料按分度整数倍 ``j``（逐格等分单位）
  配料 n·j 单位（n 为格数），逐格取一份均质混合料等分，差额逐料补称。
  等分组成与需求一致，逐格釉式与冻结方案完全相同（偏差 0）。
* 限定母料批量 T（分度整数倍）时：配料按需求比例以最大余数法凑足
  T 单位，组成可能与需求略有出入。每格取固定质量 A（单位）的均质
  混合料，其中各原料质量按**实际配料组成** b_m/T 折算（允许小数），
  再以最大余数法把整数补料分配到各原料并闭合每格总量。补料为正但
  不足最小称量时该方案不可行。剩余料 T - n·A 须在允许上限内。
  这种组成偏差会如实计入最大釉式偏移并参与排序。

方案枚举覆盖共有料的**全部子集**（每种料要么全量入母料、要么不入），
按 (最大成分偏差, 称量次数, 最小称量余量, 剩余料) 字典序排序。

成本口径
--------
相同原料在不同来源配方中的冻结价格可能不同（原料单价可更新，各版本
保留各自快照）。合并后每格该原料的有效单价按各来源在该格的实际份额
加权，因此结果与来源排列顺序无关。
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


def ceil_int(x: float) -> int:
    """向上取整到最近整数（网格容差内的整数视为整数）。"""
    n = int(x)
    if x - n > GRID_EPS:
        return n + 1
    return n


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
    # material_id -> 冻结时该原料在本版本中的单价（元/kg）
    prices: dict[int, float]


@dataclass(frozen=True)
class SourceMaterial:
    """合并后的来源原料（相同原料跨来源按比例叠加）。"""

    material_id: int
    name: str
    oxides: dict[str, float]
    loi: float
    # 各来源配方归一化后该原料的 kg/kg 干料份额，顺序与 sources 一致
    fractions: tuple[float, ...]
    # 各来源该原料的单价（元/kg），顺序与 fractions 一致
    prices: tuple[float, ...]

    def effective_price(self, weights: Sequence[float]) -> float:
        """该格混合后此原料的有效单价：按各来源实际份额加权。

        weights 为该格各来源占比（合计为 1）；该料在格内来自来源 s
        的质量为 fractions[s]·weights[s]，成本按来源单价分别计价。
        """
        mass = sum(f * w for f, w in zip(self.fractions, weights) if w > 0)
        if mass <= 0.0:
            return self.prices[0]
        cost_mass = sum(
            f * w * p
            for f, w, p in zip(self.fractions, weights, self.prices)
            if w > 0
        )
        return cost_mass / mass


def build_sources(versions: Sequence[dict[str, Any]]) -> tuple[FrozenSource, ...]:
    """从冻结版本记录构造归一化来源。

    冻结版本的 ``items`` 若存在同料多条（历史数据），先按原料合计
    再以该版本批量归一化为干料份额。
    """
    sources: list[FrozenSource] = []
    for v in versions:
        totals: dict[int, float] = {}
        prices: dict[int, float] = {}
        for item in v["items"]:
            amt = float(item["amount"])
            if amt > 0.0:
                mid = int(item["material_id"])
                totals[mid] = totals.get(mid, 0.0) + amt
                prices[mid] = float(v["material_snapshot"][str(mid)]["price"])
        batch_mass = sum(totals.values())
        if batch_mass <= 0.0:
            raise ValueError(f"来源配方 {v['id']} 没有正用量原料")
        sources.append(
            FrozenSource(
                version_id=v["id"],
                note=v.get("note"),
                shares={mid: amt / batch_mass for mid, amt in totals.items()},
                original_amounts=dict(sorted(totals.items())),
                prices=prices,
            )
        )
    return tuple(sources)


def merge_materials(
    sources: Sequence[FrozenSource],
    material_snapshots: dict[int, dict[str, Any]],
) -> list[SourceMaterial]:
    """合并相同原料：叠加各来源份额，单价按来源分别保留（加权在格内做）。"""
    merged: dict[int, SourceMaterial] = {}
    n_src = len(sources)
    for s_idx, src in enumerate(sources):
        for mid, share in src.shares.items():
            snap = material_snapshots[mid]
            existing = merged.get(mid)
            if existing is None:
                fractions = [0.0] * n_src
                prices = [0.0] * n_src
                fractions[s_idx] = share
                prices[s_idx] = src.prices[mid]
                merged[mid] = SourceMaterial(
                    material_id=mid,
                    name=snap["name"],
                    oxides=snap["oxides"],
                    loi=snap["loi"],
                    fractions=tuple(fractions),
                    prices=tuple(prices),
                )
            else:
                fractions = list(existing.fractions)
                prices = list(existing.prices)
                fractions[s_idx] = share
                prices[s_idx] = src.prices[mid]
                merged[mid] = SourceMaterial(
                    material_id=mid,
                    name=existing.name,
                    oxides=existing.oxides,
                    loi=existing.loi,
                    fractions=tuple(fractions),
                    prices=tuple(prices),
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
    """组装单格结果：投料表、舍入误差、釉式、烧后质量、成本。

    化学组成与成本分开计算：组成对相同原料与价格无关；成本则按该格
    各来源实际份额对单价加权后的有效单价计算（与来源排列顺序无关）。
    """
    mat_by_id = {m.material_id: m for m in materials}
    doses: list[dict[str, Any]] = []
    actual_tuples = []
    theory_tuples = []
    total_weighed = 0.0
    total_cost = 0.0
    for mid in sorted(units):
        mat = mat_by_id[mid]
        u = units[mid]
        weighed_g = u * division
        theory_g = theoretical.get(mid, 0.0)
        total_weighed += weighed_g
        eff_price = mat.effective_price(weights)
        line_cost = weighed_g * KG_PER_G * eff_price
        total_cost += line_cost
        actual_tuples.append(
            (mid, mat.name, mat.oxides, mat.loi, eff_price,
             weighed_g * KG_PER_G)
        )
        doses.append({
            "material_id": mid,
            "name": mat.name,
            "theoretical_g": round(theory_g, 6),
            "weighed_g": round(weighed_g, 6),
            "units": u,
            "effective_price_per_kg": round(eff_price, 9),
            "cost": round(line_cost, 9),
            "rounding_error_g": round(weighed_g - theory_g, 6),
        })
    for mid, grams in theoretical.items():
        mat = mat_by_id[mid]
        theory_tuples.append(
            (mid, mat.name, mat.oxides, mat.loi,
             mat.effective_price(weights), grams * KG_PER_G)
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
        "cost": round(total_cost, 9),
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
    """母料搜索的冻结试验上下文（质量单位均为 g）。"""

    positions: tuple[str, ...]
    material_ids: tuple[int, ...]
    names: dict[int, str]
    oxides: dict[int, dict[str, float]]
    loi: dict[int, float]
    # 逐格有效单价（元/kg）：price[(mid, pos)]
    price: dict[tuple[int, str], float]
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
    price: dict[tuple[int, str], float] = {}
    mids: set[int] = set()
    names: dict[int, str] = {}
    oxides: dict[int, dict[str, float]] = {}
    loi: dict[int, float] = {}
    baseline: dict[str, dict[str, float]] = {}
    for cell in experiment["result"]["cells"]:
        pos = cell["position"]
        positions.append(pos)
        baseline[pos] = cell["seger"]
        for d in cell["doses"]:
            mid = int(d["material_id"])
            mids.add(mid)
            units[(mid, pos)] = int(d["units"])
            # 冻结记录中已存该格有效单价；旧记录缺字段时退回原料快照价
            price[(mid, pos)] = float(
                d.get("effective_price_per_kg")
                if d.get("effective_price_per_kg") is not None
                else experiment["material_snapshot"][str(mid)]["price"]
            )
    for mid in sorted(mids):
        snap = experiment["material_snapshot"][str(mid)]
        names[mid] = snap["name"]
        oxides[mid] = snap["oxides"]
        loi[mid] = snap["loi"]
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


# 共有料子集全枚举的上限：超出后退回启发式（陶艺配方实际原料数远小于此）
MAX_SUBSET_ENUMERATE = 18


def search_master_plans(
    prob: MasterProblem,
    *,
    master_batch_g: Optional[float] = None,
    master_minimum_weighed: Optional[float] = None,
    allowed_leftover_g: Optional[float] = None,
    max_candidates: int = 10,
) -> dict[str, Any]:
    """搜索"母料 + 逐格补料"方案并排序。

    枚举共有料的全部子集（每种料要么以逐格最小用量全量入母料，要么
    不入；固定批量下额外尝试与需求总量最接近的等分粒度）。非固定批量
    模式下等分组成与需求一致，釉式偏差恒为 0；固定批量模式按实际
    配料组成与实际取用量计算偏差。
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
        target_units = round_int(target_g / division)
    else:
        target_g = None
        leftover_cap = None
        target_units = None

    n_cells = len(prob.positions)
    # 母料只收所有格位共用的原料：某格缺料（端点格常见）则不具共有性。
    u_min: dict[int, int] = {
        mid: min(prob.units.get((mid, pos), 0) for pos in prob.positions)
        for mid in prob.material_ids
    }
    common = [mid for mid in prob.material_ids if u_min[mid] > 0]

    # 剩余量单位上限：严格按填写克数向下取整，不得四舍五入放宽。
    # 例：分度 0.5 g、上限 2.4 g → 4 个单位（2.5 g 剩余即超限）。
    if fixed:
        leftover_cap_units = int(leftover_cap / division + GRID_EPS)
        min_aliquot_units = max(
            1, ceil_int(min_w / division)
        )
    else:
        leftover_cap_units = None
        min_aliquot_units = 1

    plans: list[dict[str, Any]] = []
    seen_sig: set[tuple] = set()

    def consider(included: dict[int, int], aliquot_units: Optional[int]) -> None:
        sig = (aliquot_units, tuple(sorted(included.items())))
        if sig in seen_sig:
            return
        seen_sig.add(sig)
        plan = _evaluate_plan(
            prob, included, u_min, min_w,
            fixed=fixed,
            target_units=target_units,
            leftover_cap_units=leftover_cap_units,
            aliquot_units=aliquot_units,
        )
        if plan is not None:
            plans.append(plan)

    if not fixed:
        consider({}, None)  # 无母料基线

    # ---- 候选：共有料全量子集（全量入母料 = 逐格最小用量）----
    j_full = {mid: u_min[mid] for mid in common}
    subsets = _iter_subsets(common)
    for subset in subsets:
        included = {mid: j_full[mid] for mid in subset}
        if fixed:
            for A in _candidate_aliquot_sizes(
                n_cells,
                target_units,
                leftover_cap_units,
                min_aliquot_units,
            ):
                consider(included, A)
        else:
            consider(included, None)

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


def _iter_subsets(items: list[int]):
    """枚举全部子集（含空集）；超过上限时退化为留一/成对/单料启发式。"""
    m = len(items)
    if m <= MAX_SUBSET_ENUMERATE:
        for mask in range(1 << m):
            yield [items[i] for i in range(m) if mask & (1 << i)]
        return
    # 启发式兜底：空集、单料、成对、留一、全集
    yield []
    for i in items:
        yield [i]
    for i in range(m):
        for k in range(i + 1, m):
            yield [items[i], items[k]]
    for i in range(m):
        yield [items[k] for k in range(m) if k != i]
    yield list(items)


# 固定批量下，每子集枚举的等分粒度上限（超出则等距下采样，端点必含）
MAX_ALIQUOT_GRID_POINTS = 200


def _candidate_aliquot_sizes(
    n_cells: int,
    target_units: int,
    leftover_cap_units: int,
    min_aliquot_units: int,
) -> list[int]:
    """固定批量下候选的每格等分粒度 A（单位：分度）。

    可行区间由剩余量约束严格界定：

        剩余 L = T - n·A ，要求 0 ≤ L ≤ 余量上限

    故 ``A ∈ [ceil((T-余量上限)/n), floor(T/n)]``；再叠加等分称量的
    最小称量下限。零剩余点 ``A = T/n``（整除时）始终在区间内并必被
    枚举——母料正好耗尽的拆分不得遗漏。区间过长时等距下采样，
    但两个端点一定保留。
    """
    lo = max(
        min_aliquot_units,
        ceil_int((target_units - leftover_cap_units) / n_cells),
    )
    hi = target_units // n_cells  # floor，保证 n·A ≤ T
    if lo > hi:
        return []
    span = hi - lo + 1
    if span <= MAX_ALIQUOT_GRID_POINTS:
        return list(range(lo, hi + 1))
    step = span / MAX_ALIQUOT_GRID_POINTS
    picked = {lo, hi}
    k = 1
    while len(picked) < MAX_ALIQUOT_GRID_POINTS:
        picked.add(lo + int(k * step))
        k += 1
    return sorted(picked)


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
    included_in: dict[int, int],
    u_min: dict[int, int],
    master_minimum_weighed: float,
    *,
    fixed: bool,
    target_units: Optional[int],
    leftover_cap_units: Optional[int],
    aliquot_units: Optional[int],
) -> Optional[dict[str, Any]]:
    """评估单个方案；违反约束返回 None。

    included_in: mid -> 每格等分所需的该料单位数 j_m（非固定批量下
    母料配料恰为 n·j_m）。固定批量下 aliquot_units 给出每格实际取料
    单位 A，各料实际取用量按母料**实际配料组成**折算。
    """
    division = prob.division
    n_cells = len(prob.positions)
    included = dict(included_in)
    if not included:
        return None if fixed else _baseline_plan(prob)

    # ---- 母料配料 ----
    if not fixed:
        # 精确模式：每料配 n·j_m，等分组成与需求一致，无剩余
        prepared_units = {mid: j * n_cells for mid, j in included.items()}
        aliquot_A = sum(included.values())
        leftover_units = 0
        consumed_units = sum(prepared_units.values())
    else:
        # 固定批量：需求 D = n·Σj；每格取 A 单位混合料，消耗 n·A
        demanded_per_cell = sum(included.values())
        A = aliquot_units
        if A is None or A <= 0 or A > demanded_per_cell + GRID_EPS:
            return None
        consumed_units = n_cells * A
        leftover_units = target_units - consumed_units
        if leftover_units < 0:
            return None
        if leftover_cap_units is not None and leftover_units > leftover_cap_units:
            return None
        # 配料按需求比例把 T 单位最大余数法分配到各料
        demanded_total = n_cells * demanded_per_cell
        prepared_units = {}
        fracs: list[tuple[float, int]] = []
        for mid, j in included.items():
            x = (j * n_cells) * target_units / demanded_total
            prepared_units[mid] = int(x)
            fracs.append((x - int(x), mid))
        extra_total = target_units - sum(prepared_units.values())
        if extra_total < 0:
            return None
        for _, mid in sorted(fracs, key=lambda t: (-t[0], t[1]))[:extra_total]:
            prepared_units[mid] += 1
        aliquot_A = A

    # 配料最小称量（母料秤）
    for mid, b in prepared_units.items():
        if b * division < master_minimum_weighed - GRID_EPS * division:
            return None

    # ---- 逐格取料/补料 ----
    cell_plans: list[dict[str, Any]] = []
    n_bulk = len(prepared_units)
    n_aliquot = 0
    n_topup = 0
    margins: list[float] = [
        b * division - master_minimum_weighed for b in prepared_units.values()
    ]

    for pos in prob.positions:
        u = {
            mid: prob.units.get((mid, pos), 0)
            for mid in prob.material_ids
        }
        # 入母料的料必须每格都用到且等分不超过用量（共有性）
        if any(u.get(mid, 0) <= 0 for mid in included):
            return None

        aliquot_components: dict[int, float] = {}
        topup_units: dict[int, int] = {}
        if not fixed:
            for mid, j in included.items():
                if j > u[mid]:
                    return None
                aliquot_components[mid] = float(j)
                t = u[mid] - j
                if t > 0:
                    topup_units[mid] = t
            # 未入母料但该格用到的料：逐格单独称量
            for mid in prob.material_ids:
                if mid not in included and u[mid] > 0:
                    topup_units[mid] = u[mid]
        else:
            # 实际取料：A 单位均质混合料按**实际配料组成** b_m/T 折算，
            # 各料分量允许小数（称的是混合料，不是逐料上秤）。
            for mid, b in prepared_units.items():
                aliquot_components[mid] = b * aliquot_A / target_units
            # 理想补料 = 目标 - 等分实取（非负）；补料总量恒为 U_cell - A
            ideal: dict[int, float] = {}
            for mid in prob.material_ids:
                ideal[mid] = max(u[mid] - aliquot_components.get(mid, 0.0), 0.0)
            total_topup = round_int(sum(u.values()) - aliquot_A)
            if total_topup < 0:
                return None
            floors = {mid: int(v) for mid, v in ideal.items()}
            shortfall = total_topup - sum(floors.values())
            order = sorted(
                ideal, key=lambda m: (-(ideal[m] - floors[m]), -ideal[m], m)
            )
            topup_units = dict(floors)
            for mid in order[: max(shortfall, 0)]:
                topup_units[mid] = topup_units.get(mid, 0) + 1
            # 等分实取超过该料目标用量时，理想补料被钳为 0，无法闭合
            if shortfall < 0 or any(
                topup_units.get(mid, 0) > u[mid] for mid in included
            ):
                return None
            topup_units = {m: v for m, v in topup_units.items() if v > 0}

        # 最小称量：正值补料逐料校验
        for mid, t in topup_units.items():
            if t * division < prob.minimum_weighed - GRID_EPS * division:
                return None
            margins.append(t * division - prob.minimum_weighed)
        n_topup += len(topup_units)

        # 等分混合料作为一次称量（各料分量允许小数：称的是均质混合料）
        aliquot_g = aliquot_A * division
        if aliquot_g < prob.minimum_weighed - GRID_EPS * division:
            return None
        margins.append(aliquot_g - prob.minimum_weighed)
        n_aliquot += 1

        actual_units: dict[int, float] = {}
        for mid in prob.material_ids:
            actual_units[mid] = (
                aliquot_components.get(mid, 0.0) + topup_units.get(mid, 0)
            )
        tuples = [
            (
                mid, prob.names[mid], prob.oxides[mid], prob.loi[mid],
                prob.price[(mid, pos)],
                actual_units[mid] * division * KG_PER_G,
            )
            for mid in prob.material_ids
            if actual_units[mid] > ZERO_EPS
        ]
        actual = calc_batch(tuples, targets={})
        base = prob.baseline_seger[pos]
        shift = {
            o: round(actual["seger"].get(o, 0.0) - base.get(o, 0.0), 9)
            for o in sorted(set(actual["seger"]) | set(base))
        }
        cell_max = max((abs(v) for v in shift.values()), default=0.0)

        master_doses = [
            {
                "material_id": mid,
                "name": prob.names[mid],
                "units": (
                    included[mid]
                    if not fixed
                    else round(aliquot_components[mid], 6)
                ),
                "weighed_g": round(aliquot_components[mid] * division, 6),
            }
            for mid in sorted(included)
            if aliquot_components.get(mid, 0.0) > ZERO_EPS
        ]
        cell_plans.append({
            "position": pos,
            "master_aliquot": {
                "total_units": aliquot_A,
                "total_g": round(aliquot_g, 6),
                "components": master_doses,
            },
            "topup_doses": [
                _dose_entry(prob, mid, topup_units[mid])
                for mid in sorted(topup_units)
            ],
            "actual_units": {
                str(mid): round(v, 6) for mid, v in actual_units.items() if v > 0
            },
            "seger": actual["seger"],
            "seger_shift": shift,
            "max_abs_seger_shift": round(cell_max, 9),
        })

    max_dev = max(
        (cp["max_abs_seger_shift"] for cp in cell_plans), default=0.0
    )
    signature = ",".join(
        f"{mid}:{included[mid]}" for mid in sorted(included)
    ) + (f"@A{aliquot_A}" if fixed else "")
    prepared_g = sum(prepared_units.values()) * division
    plan = {
        "signature": signature,
        "included_material_ids": sorted(included),
        "bulk": [
            {
                "material_id": mid,
                "name": prob.names[mid],
                "prepared_units": prepared_units[mid],
                "prepared_g": round(prepared_units[mid] * division, 6),
                "required_units": included[mid] * n_cells,
                "aliquot_units": included[mid],
                "aliquot_g": round(included[mid] * division, 6),
                "n_aliquots": n_cells,
            }
            for mid in sorted(prepared_units)
        ],
        "aliquot_total_g": round(aliquot_A * division, 6),
        "master_batch_prepared_g": round(prepared_g, 6),
        "master_consumed_g": round(consumed_units * division, 6),
        "leftover_g": round(leftover_units * division, 6),
        "total_weighings": n_bulk + n_aliquot + n_topup,
        "n_bulk_weighings": n_bulk,
        "n_aliquot_weighings": n_aliquot,
        "n_topup_weighings": n_topup,
        "min_weighing_margin_g": round(min(margins), 6),
        "max_abs_seger_shift": round(max_dev, 9),
        "cells": cell_plans,
        "weighing_order": None,
    }
    plan["weighing_order"] = _build_weighing_order(
        prob, prepared_units, cell_plans
    )
    return plan


def _build_weighing_order(
    prob: MasterProblem,
    prepared_units: dict[int, int],
    cell_plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从方案对象生成称量顺序：先逐料配母料，再逐格取等分、逐料补料。

    固定批量下每格等分的各料分量可能为小数（均质混合料按组成折算），
    补料则是最大余数法确定的整数，顺序必须与 ``cell_plans`` 完全一致，
    因此在此统一由方案对象构建，避免独立逻辑产生分歧。
    """
    order: list[dict[str, Any]] = []
    step = 0
    for mid in sorted(prepared_units):
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
    for cp in cell_plans:
        pos = cp["position"]
        ma = cp["master_aliquot"]
        if ma is not None:
            step += 1
            order.append({
                "step": step,
                "stage": "aliquot",
                "position": pos,
                "material_id": None,
                "name": "母料等分",
                "action": "take_master_aliquot",
                "weighed_g": ma["total_g"],
                "components": ma["components"],
            })
        for d in cp["topup_doses"]:
            step += 1
            order.append({
                "step": step,
                "stage": "topup",
                "position": pos,
                "material_id": d["material_id"],
                "name": d["name"],
                "action": "weigh_topup",
                "weighed_g": d["weighed_g"],
            })
    return order


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
                tuples.append(_tuple(prob, mid, u, pos))
                total += 1
                margins.append(u * prob.division - prob.minimum_weighed)
        actual = calc_batch(tuples, targets={})
        cell_plans.append({
            "position": pos,
            "master_aliquot": None,
            "topup_doses": topup_doses,
            "actual_units": {
                str(d["material_id"]): d["units"] for d in topup_doses
            },
            "seger": actual["seger"],
            "seger_shift": {o: 0.0 for o in sorted(actual["seger"])},
            "max_abs_seger_shift": 0.0,
        })
    plan = {
        "signature": "none",
        "included_material_ids": [],
        "bulk": [],
        "aliquot_total_g": 0.0,
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
        "weighing_order": None,
    }
    plan["weighing_order"] = _build_weighing_order(prob, {}, cell_plans)
    return plan


def _dose_entry(prob: MasterProblem, mid: int, units: int) -> dict[str, Any]:
    return {
        "material_id": mid,
        "name": prob.names[mid],
        "units": units,
        "weighed_g": round(units * prob.division, 6),
    }


def _tuple(prob: MasterProblem, mid: int, units: float, pos: str):
    return (
        mid,
        prob.names[mid],
        prob.oxides[mid],
        prob.loi[mid],
        prob.price[(mid, pos)],
        units * prob.division * KG_PER_G,
    )
