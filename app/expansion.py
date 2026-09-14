"""釉坯热膨胀适配：重复曲线插值、共同温区膨胀差积分与应力评估。

口径与单位
----------
* 曲线：膨胀仪温度（°C）—相对长度（ΔL/L0，无量纲应变）。登记时可声明
  ``strain_unit = "1" / "ppm" / "%"``，入库一律折算为无量纲应变。
* 方向：``heating`` 升温（温度须严格递增）/ ``cooling`` 降温（严格递减）；
  分析前统一按温度升序排列，升降温分别成对计算。
* 测次配对：每个配方、每个方向，釉条重复编号集合须与同方向坯条完全
  一致，逐对 (方向, 重复编号) 计算；调用方登记的异常测次在配对前剔除。

共同有效温区
------------
全部未排除曲线温度范围的交集 [lo, hi]；须为正区间且覆盖
[室温, 应力释放温度]，否则拒绝分析并指出越界端点。

残余应变与应力
--------------
失配应变 ``m(T) = 釉应变 - 坯应变``（同对曲线各自线性插值后相减）。
在共同温区积分釉坯膨胀差，室温残余应变（釉相对坯）：

    ε = m(T_set) - m(T_room) = ∫[T_room, T_set] (α_釉 - α_坯) dT

正值表示冷却后釉层受拉（开裂方向），负值表示受压（剥釉方向）。
釉层应力按双轴模量 M = E/(1-ν) 与厚度修正：

    σ = M_釉·ε / (1 + M_釉·t_釉 / (M_坯·t_坯))     （MPa，E 取 GPa）

分段线膨胀系数
--------------
共同温区等分为 n_segments 段，段内割线斜率即该段平均线膨胀系数（1/K），
对分段线性曲线为精确均值；逐配方给出釉、坯与差值及测次标准差。

离散度与超限
------------
应力区间取全部测次对的 [min, max]，离散度取测次对标准差（n≥2，ddof=1）。
给定目标应力窗 [low, high]（MPa）：区间上端越出 high 记开裂方向超限，
下端越出 low 记剥釉方向超限。排序键字典序：

  1. 窗外计数（开裂 + 剥釉两个方向）
  2. 最坏区间超限量（MPa）
  3. 测次离散度（应力标准差）
  4. 窗中心偏差（归一化）
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .config import EXPANSION_MAX_ABS_STRAIN
from .errors import GlazeError

DIRECTIONS = ("heating", "cooling")
DIRECTION_LABELS = {"heating": "升温", "cooling": "降温"}

# strain_unit -> 无量纲应变折算系数
STRAIN_UNIT_FACTORS = {"1": 1.0, "ppm": 1e-6, "%": 1e-2}


def _r(x: Optional[float], ndigits: int = 9) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), ndigits)


# ---------------------------------------------------------------------------
# 曲线规范化与校验
# ---------------------------------------------------------------------------

def normalize_points(
    points: Sequence[dict[str, float]], strain_unit: str
) -> list[dict[str, float]]:
    """把登记点折算为无量纲应变，并按温度升序返回 [{"temp_c", "strain"}]。"""
    try:
        factor = STRAIN_UNIT_FACTORS[strain_unit]
    except KeyError:
        raise GlazeError(
            f"未知相对长度单位: {strain_unit}（支持 1 / ppm / %）",
            "unknown_strain_unit",
            {"strain_unit": strain_unit, "known": list(STRAIN_UNIT_FACTORS)},
        )
    out = [
        {"temp_c": float(p["temp_c"]), "strain": float(p["strain"]) * factor}
        for p in points
    ]
    for p in out:
        if abs(p["strain"]) > EXPANSION_MAX_ABS_STRAIN:
            raise GlazeError(
                f"温度 {p['temp_c']} °C 处相对长度 {p['strain']} 超出合理范围 "
                f"±{EXPANSION_MAX_ABS_STRAIN}，请核对 strain_unit 单位",
                "strain_out_of_range",
                {"temp_c": p["temp_c"], "strain": p["strain"],
                 "strain_unit": strain_unit},
            )
    out.sort(key=lambda p: p["temp_c"])
    return out


def check_temperature_order(temps: Sequence[float], direction: str) -> None:
    """升温曲线温度须严格递增，降温曲线须严格递减。"""
    if direction == "heating":
        bad = any(b <= a for a, b in zip(temps, temps[1:]))
    else:
        bad = any(b >= a for a, b in zip(temps, temps[1:]))
    if bad:
        label = DIRECTION_LABELS.get(direction, direction)
        raise GlazeError(
            f"{label}曲线的温度必须严格{'递增' if direction == 'heating' else '递减'}",
            "invalid_temperature_order",
            {"direction": direction, "temps_c": [float(t) for t in temps]},
        )


def curve_arrays(
    points: Sequence[dict[str, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """规范化点列 -> (温度升序数组, 应变数组)；温度不得重复。"""
    temps = np.array([p["temp_c"] for p in points], dtype=float)
    strains = np.array([p["strain"] for p in points], dtype=float)
    if temps.size >= 2 and np.any(np.diff(temps) <= 0.0):
        raise GlazeError(
            "同一曲线内温度不得重复",
            "duplicate_temperature",
            {"temps_c": [float(t) for t in temps]},
        )
    return temps, strains


# ---------------------------------------------------------------------------
# 共同有效温区
# ---------------------------------------------------------------------------

def common_valid_range(
    curves: Sequence[tuple[np.ndarray, np.ndarray]]
) -> tuple[float, float]:
    """全部曲线温度范围的交集 [lo, hi]。"""
    lo = max(float(t[0]) for t, _ in curves)
    hi = min(float(t[-1]) for t, _ in curves)
    return lo, hi


def check_common_range(
    lo: float, hi: float, room_temp_c: float, stress_release_temp_c: float
) -> None:
    """共同温区须为正区间且覆盖 [室温, 应力释放温度]。"""
    if hi - lo <= 0.0:
        raise GlazeError(
            "全部曲线没有共同有效温区（温度范围交集为空），"
            "请检查各曲线的温度覆盖",
            "no_common_temperature_range",
            {"intersection_c": [lo, hi]},
        )
    if room_temp_c >= stress_release_temp_c:
        raise GlazeError(
            f"室温 {room_temp_c} °C 必须低于应力释放温度 "
            f"{stress_release_temp_c} °C",
            "invalid_temperature_span",
            {"room_temp_c": room_temp_c,
             "stress_release_temp_c": stress_release_temp_c},
        )
    missing = []
    if room_temp_c < lo - 1e-9:
        missing.append({"endpoint": "room_temp", "temp_c": room_temp_c})
    if stress_release_temp_c > hi + 1e-9:
        missing.append(
            {"endpoint": "stress_release", "temp_c": stress_release_temp_c}
        )
    if missing:
        raise GlazeError(
            f"共同有效温区 [{lo}, {hi}] °C 未覆盖残余应变积分区间 "
            f"[{room_temp_c}, {stress_release_temp_c}] °C",
            "temperature_range_uncovered",
            {
                "common_range_c": [lo, hi],
                "required_range_c": [room_temp_c, stress_release_temp_c],
                "uncovered": missing,
            },
        )


# ---------------------------------------------------------------------------
# 测次配对
# ---------------------------------------------------------------------------

def pair_replicates(
    body_by_rep: dict[int, Any],
    glaze_by_rep: dict[int, Any],
    *,
    direction: str,
    recipe_label: str,
) -> list[tuple[int, Any, Any]]:
    """同一方向下釉条与坯条按重复编号配对；集合不一致时拒绝。"""
    body_reps = sorted(body_by_rep)
    glaze_reps = sorted(glaze_by_rep)
    if body_reps != glaze_reps:
        raise GlazeError(
            f"配方 {recipe_label} {DIRECTION_LABELS[direction]}方向的釉条测次 "
            f"{glaze_reps} 与坯条测次 {body_reps} 不配对",
            "replicate_mismatch",
            {
                "recipe": recipe_label,
                "direction": direction,
                "glaze_replicates": glaze_reps,
                "body_replicates": body_reps,
                "missing_body": sorted(set(glaze_reps) - set(body_reps)),
                "missing_glaze": sorted(set(body_reps) - set(glaze_reps)),
            },
        )
    return [(rep, glaze_by_rep[rep], body_by_rep[rep]) for rep in body_reps]


# ---------------------------------------------------------------------------
# 单对曲线计算
# ---------------------------------------------------------------------------

def _interp(curve: tuple[np.ndarray, np.ndarray], t: float) -> float:
    temps, strains = curve
    return float(np.interp(t, temps, strains))


def segment_edges(lo: float, hi: float, n_segments: int) -> list[tuple[float, float]]:
    edges = np.linspace(lo, hi, n_segments + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_segments)]


def segment_cte(
    curve: tuple[np.ndarray, np.ndarray], edges: Sequence[tuple[float, float]]
) -> list[float]:
    """各温区段的割线斜率 = 平均线膨胀系数（1/K）。"""
    return [
        (_interp(curve, t1) - _interp(curve, t0)) / (t1 - t0)
        for t0, t1 in edges
    ]


def residual_strain(
    glaze: tuple[np.ndarray, np.ndarray],
    body: tuple[np.ndarray, np.ndarray],
    room_temp_c: float,
    stress_release_temp_c: float,
) -> float:
    """室温残余应变（釉相对坯）：m(T_set) - m(T_room)，正值釉受拉。"""
    m_set = _interp(glaze, stress_release_temp_c) - _interp(
        body, stress_release_temp_c
    )
    m_room = _interp(glaze, room_temp_c) - _interp(body, room_temp_c)
    return m_set - m_room


def glaze_stress_mpa(
    strain: float,
    *,
    glaze_modulus_gpa: float,
    body_modulus_gpa: float,
    glaze_poisson: float,
    body_poisson: float,
    glaze_thickness_mm: float,
    body_thickness_mm: float,
) -> float:
    """釉层应力（MPa，正=拉/开裂方向，负=压/剥釉方向）。"""
    m_g = glaze_modulus_gpa / (1.0 - glaze_poisson)
    m_b = body_modulus_gpa / (1.0 - body_poisson)
    factor = m_g / (1.0 + (m_g * glaze_thickness_mm) / (m_b * body_thickness_mm))
    return factor * strain * 1000.0


# ---------------------------------------------------------------------------
# 研究级分析
# ---------------------------------------------------------------------------

def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    return {
        "mean": _r(arr.mean(), 12),
        "std": _r(arr.std(ddof=1), 12) if arr.size > 1 else 0.0,
        "min": _r(arr.min(), 12),
        "max": _r(arr.max(), 12),
    }


def _window_assessment(
    stress_min: float, stress_max: float, low: float, high: float
) -> dict[str, Any]:
    """按开裂（拉）/剥釉（压）方向评估应力区间对目标窗的越限。"""
    crazing_excess = max(stress_max - high, 0.0)
    shivering_excess = max(low - stress_min, 0.0)
    n_violations = int(crazing_excess > 1e-12) + int(shivering_excess > 1e-12)
    return {
        "low_mpa": low,
        "high_mpa": high,
        "in_window": n_violations == 0,
        "n_violations": n_violations,
        "crazing_excess_mpa": _r(crazing_excess, 6),
        "shivering_excess_mpa": _r(shivering_excess, 6),
        "worst_excess_mpa": _r(max(crazing_excess, shivering_excess), 6),
    }


def analyze_study_data(
    *,
    elastic: dict[str, float],
    body_curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    recipes: Sequence[dict[str, Any]],
    room_temp_c: float,
    n_segments: int,
    stress_window: Optional[tuple[float, float]],
) -> dict[str, Any]:
    """对全部配方执行配对插值、膨胀差积分与应力评估。

    ``body_curves`` / ``recipes[].curves`` 均为 {direction: {replicate_no: curve}}，
    异常测次已在组装阶段剔除。``elastic`` 含应力释放温度与弹性/几何参数。
    """
    t_set = elastic["stress_release_temp_c"]
    stress_kwargs = {
        k: v for k, v in elastic.items() if k != "stress_release_temp_c"
    }
    # 共同有效温区：全部未排除曲线的交集
    all_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for by_rep in body_curves.values():
        all_curves.extend(by_rep.values())
    for recipe in recipes:
        for by_rep in recipe["curves"].values():
            all_curves.extend(by_rep.values())
    if not recipes:
        raise GlazeError("研究内没有任何配方，无法分析", "empty_expansion_study")
    if not all_curves:
        raise GlazeError(
            "研究内没有任何膨胀曲线，无法分析", "empty_expansion_curves"
        )
    lo, hi = common_valid_range(all_curves)
    check_common_range(lo, hi, room_temp_c, t_set)
    edges = segment_edges(lo, hi, n_segments)

    # 坯体分段线膨胀系数（各方向/测次均值）
    body_segment_rows: list[dict[str, Any]] = []
    body_cte_by_pair: dict[tuple[str, int], list[float]] = {}
    for direction in DIRECTIONS:
        for rep, curve in sorted(body_curves.get(direction, {}).items()):
            body_cte_by_pair[(direction, rep)] = segment_cte(curve, edges)
    for seg_idx, (t0, t1) in enumerate(edges):
        vals = [ctes[seg_idx] for ctes in body_cte_by_pair.values()]
        arr = np.array(vals, dtype=float)
        body_segment_rows.append({
            "segment": seg_idx + 1,
            "temp_low_c": _r(t0, 6),
            "temp_high_c": _r(t1, 6),
            "body_cte_per_k": _r(arr.mean(), 12),
            "body_cte_std": _r(arr.std(ddof=1), 12) if arr.size > 1 else 0.0,
        })

    recipe_rows: list[dict[str, Any]] = []
    for recipe in recipes:
        label = f"#{recipe['recipe_index']}（{recipe['version_id']}）"
        pairs: list[dict[str, Any]] = []
        seg_diffs: list[list[float]] = []
        seg_glaze: list[list[float]] = []
        for direction in DIRECTIONS:
            glaze_by_rep = recipe["curves"].get(direction, {})
            if not glaze_by_rep:
                continue
            body_by_rep = body_curves.get(direction, {})
            if not body_by_rep:
                raise GlazeError(
                    f"配方 {label} 有{DIRECTION_LABELS[direction]}釉条曲线，"
                    "但缺少同方向坯条曲线，无法配对",
                    "replicate_mismatch",
                    {"recipe": label, "direction": direction,
                     "glaze_replicates": sorted(glaze_by_rep),
                     "body_replicates": []},
                )
            for rep, glaze, body in pair_replicates(
                body_by_rep, glaze_by_rep, direction=direction, recipe_label=label
            ):
                strain = residual_strain(glaze, body, room_temp_c, t_set)
                stress = glaze_stress_mpa(strain, **stress_kwargs)
                g_cte = segment_cte(glaze, edges)
                b_cte = body_cte_by_pair[(direction, rep)]
                seg_glaze.append(g_cte)
                seg_diffs.append([g - b for g, b in zip(g_cte, b_cte)])
                pairs.append({
                    "direction": direction,
                    "replicate_no": rep,
                    "residual_strain": _r(strain, 12),
                    "stress_mpa": _r(stress, 6),
                })
        if not pairs:
            raise GlazeError(
                f"配方 {label} 没有任何可用测次对（曲线缺失或全部被排除）",
                "no_valid_pairs",
                {"recipe": label},
            )
        strains = [p["residual_strain"] for p in pairs]
        stresses = [p["stress_mpa"] for p in pairs]
        strain_stats = _stats(strains)
        stress_stats = _stats(stresses)
        glaze_arr = np.array(seg_glaze, dtype=float)
        diff_arr = np.array(seg_diffs, dtype=float)
        segments = []
        for seg_idx, (t0, t1) in enumerate(edges):
            g_vals = glaze_arr[:, seg_idx]
            d_vals = diff_arr[:, seg_idx]
            segments.append({
                "segment": seg_idx + 1,
                "temp_low_c": _r(t0, 6),
                "temp_high_c": _r(t1, 6),
                "glaze_cte_per_k": _r(g_vals.mean(), 12),
                "diff_cte_per_k": _r(d_vals.mean(), 12),
                "diff_cte_std": (
                    _r(d_vals.std(ddof=1), 12) if d_vals.size > 1 else 0.0
                ),
            })
        window = None
        if stress_window is not None:
            window = _window_assessment(
                float(stress_stats["min"]), float(stress_stats["max"]),
                stress_window[0], stress_window[1],
            )
        recipe_rows.append({
            "recipe_index": recipe["recipe_index"],
            "version_id": recipe["version_id"],
            "n_pairs": len(pairs),
            "pairs": pairs,
            "residual_strain": strain_stats,
            "stress_mpa": stress_stats,
            "cte_segments": segments,
            "window": window,
        })

    return {
        "common_range_c": [_r(lo, 6), _r(hi, 6)],
        "room_temp_c": room_temp_c,
        "stress_release_temp_c": t_set,
        "n_segments": n_segments,
        "stress_window_mpa": (
            {"low": stress_window[0], "high": stress_window[1]}
            if stress_window is not None
            else None
        ),
        "body": {
            "n_curves": sum(len(v) for v in body_curves.values()),
            "cte_segments": body_segment_rows,
        },
        "recipes": recipe_rows,
        "n_recipes": len(recipe_rows),
    }


# ---------------------------------------------------------------------------
# 配方排列（目标应力窗 / 最坏区间 / 不确定度）
# ---------------------------------------------------------------------------

def rank_recipes(
    analysis: dict[str, Any],
    *,
    stress_low_mpa: float,
    stress_high_mpa: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """按 (窗外计数, 最坏区间超限量, 离散度, 窗中心偏差) 字典序排列配方。"""
    center = (stress_low_mpa + stress_high_mpa) / 2.0
    half = (stress_high_mpa - stress_low_mpa) / 2.0
    scale = half if half > 1e-12 else max(abs(center), 1.0)
    rows: list[dict[str, Any]] = []
    for recipe in analysis["recipes"]:
        stats = recipe["stress_mpa"]
        window = recipe["window"]
        assert window is not None  # 排列前必须带目标应力窗分析
        center_dev = abs(float(stats["mean"]) - center) / scale
        rows.append({
            "recipe_index": recipe["recipe_index"],
            "version_id": recipe["version_id"],
            "n_pairs": recipe["n_pairs"],
            "stress_mean_mpa": _r(stats["mean"], 6),
            "stress_interval_mpa": [_r(stats["min"], 6), _r(stats["max"], 6)],
            "residual_strain_mean": _r(recipe["residual_strain"]["mean"], 12),
            "n_violations": window["n_violations"],
            "in_window": window["in_window"],
            "crazing_excess_mpa": window["crazing_excess_mpa"],
            "shivering_excess_mpa": window["shivering_excess_mpa"],
            "worst_excess_mpa": window["worst_excess_mpa"],
            "uncertainty_mpa": _r(stats["std"], 6),
            "center_deviation": _r(center_dev, 9),
        })
    rows.sort(
        key=lambda r: (
            r["n_violations"],
            r["worst_excess_mpa"],
            r["uncertainty_mpa"],
            r["center_deviation"],
            r["recipe_index"],
        )
    )
    return rows[:max_candidates]


# ---------------------------------------------------------------------------
# 已定稿研究比较
# ---------------------------------------------------------------------------

def compare_results(
    base: dict[str, Any], other: dict[str, Any]
) -> dict[str, Any]:
    """比较两次定稿分析结果：共有配方逐份给应力/残余应变/分段膨胀差变化。"""
    base_recipes = {r["version_id"]: r for r in base["recipes"]}
    other_recipes = {r["version_id"]: r for r in other["recipes"]}
    shared = sorted(set(base_recipes) & set(other_recipes))
    rows = []
    for vid in shared:
        a, b = base_recipes[vid], other_recipes[vid]
        seg_changes = []
        for sa, sb in zip(a["cte_segments"], b["cte_segments"]):
            seg_changes.append({
                "segment": sa["segment"],
                "temp_low_c": sa["temp_low_c"],
                "temp_high_c": sa["temp_high_c"],
                "diff_cte_change_per_k": _r(
                    sb["diff_cte_per_k"] - sa["diff_cte_per_k"], 12
                ),
            })
        rows.append({
            "version_id": vid,
            "recipe_index_base": a["recipe_index"],
            "recipe_index_other": b["recipe_index"],
            "n_pairs_base": a["n_pairs"],
            "n_pairs_other": b["n_pairs"],
            "stress_mean_change_mpa": _r(
                b["stress_mpa"]["mean"] - a["stress_mpa"]["mean"], 6
            ),
            "stress_interval_base_mpa": [
                a["stress_mpa"]["min"], a["stress_mpa"]["max"]
            ],
            "stress_interval_other_mpa": [
                b["stress_mpa"]["min"], b["stress_mpa"]["max"]
            ],
            "residual_strain_change": _r(
                b["residual_strain"]["mean"] - a["residual_strain"]["mean"], 12
            ),
            "in_window_base": a["window"]["in_window"] if a["window"] else None,
            "in_window_other": b["window"]["in_window"] if b["window"] else None,
            "cte_segment_changes": seg_changes,
        })
    return {
        "shared_recipes": rows,
        "only_in_base": sorted(set(base_recipes) - set(other_recipes)),
        "only_in_other": sorted(set(other_recipes) - set(base_recipes)),
    }
