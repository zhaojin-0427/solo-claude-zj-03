"""烧成试片研究：Scheffé 混合响应面拟合、缺陷发生率统计与配比搜索。

口径与单位
----------
* 试片测量：L*a*b*（CIELAB）、60° 光泽度（GU）、烧后厚度（mm），
  针孔 / 缩釉 / 流釉缺陷等级 0~3（0=无，3=严重）。
* 缺陷发生率：格内重复试片中等级 ≥ 1 的比例，Wilson 区间取 95%。
  拟合时按每片 0/1 发生指示作线性概率 Scheffé 面（与定量指标共用
  设计矩阵与自由度），搜索评估时预测值截断到 [0, 1]。

Scheffé 模型
------------
q 个来源（线性 2 份 / 三元 3 份），比例 x_i ≥ 0 且 Σx_i = 1：

* 一次（order=1）：``y = Σ β_i x_i``（无截距项）
* 二次（order=2）：``y = Σ β_i x_i + Σ_{i<j} β_ij x_i x_j``

设计矩阵由"有试片的格位"构成，每片试片一行（重复试片提供纯误差
自由度）。格位数少于项数、试片总数无剩余自由度或设计矩阵欠秩时
拒绝拟合，并指出缺少的比例区域（未使用的来源顶点、缺失的二元
内部混合边、三元共线、比例水平数不足等），不作外推。交叉验证
优先留一格（LOCO：整格重复试片同时留出重拟合）；去掉一格后设计
欠秩时（如格位数恰等于项数的最小设计）退化为留一片 PRESS
（``e_i/(1-h_ii)``），仅 h≈1 的饱和试片被排除。

配比搜索
--------
在试验冻结的比例范围 [ratio_low, ratio_high] 与步长网格上枚举
（各权重为步长整数倍且合计为 1），落在观测格位凸包之外的候选
按外推剔除并计数。排序键字典序：

  1. 约束违规数（各指标预测值超出调用方上限的个数）
  2. 预测不确定度 —— 杠杆值 ``sqrt(x0'(X'X)^{-1} x0)``（与指标无关）
  3. 目标中心偏差 —— 各目标窗口中点的加权归一化距离
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import t as t_dist

from .config import (
    FIRING_MAX_GRID_POINTS,
    FIRING_OUTLIER_THRESHOLD,
    FIRING_WILSON_Z,
)
from .errors import GlazeError

# 指标目录
QUANTITATIVE_METRICS = ("l_star", "a_star", "b_star", "gloss60", "thickness_mm")
DEFECT_TYPES = ("pinhole", "crawling", "running")
DEFECT_RATE_METRICS = tuple(f"{d}_rate" for d in DEFECT_TYPES)
ALL_METRICS = QUANTITATIVE_METRICS + DEFECT_RATE_METRICS

METRIC_LABELS = {
    "l_star": "L*（明度）",
    "a_star": "a*（红绿）",
    "b_star": "b*（黄蓝）",
    "gloss60": "60° 光泽度（GU）",
    "thickness_mm": "烧后厚度（mm）",
    "pinhole_rate": "针孔发生率",
    "crawling_rate": "缩釉发生率",
    "running_rate": "流釉发生率",
}

# 网格（步长整数倍）判定的相对容差
GRID_EPS = 1e-9
# 凸包/共线判定的几何容差
GEOM_EPS = 1e-9


def _r(x: float, ndigits: int = 9) -> float:
    return round(float(x), ndigits)


# ---------------------------------------------------------------------------
# Scheffé 设计矩阵
# ---------------------------------------------------------------------------

def scheffe_terms(q: int, order: int) -> list[tuple[int, ...]]:
    """q 个来源、order 阶 Scheffé 模型的项（以来源下标元组表示）。"""
    terms: list[tuple[int, ...]] = [(i,) for i in range(q)]
    if order >= 2:
        terms += list(combinations(range(q), 2))
    return terms


def term_label(term: tuple[int, ...]) -> str:
    """项标签：x1、x2、x1x2 …（来源编号从 1 起，与试验 sources 顺序一致）。"""
    return "".join(f"x{i + 1}" for i in term)


def design_row(weights: Sequence[float], order: int) -> list[float]:
    q = len(weights)
    row: list[float] = []
    for term in scheffe_terms(q, order):
        v = 1.0
        for i in term:
            v *= weights[i]
        row.append(v)
    return row


def design_problems(
    weights_list: Sequence[Sequence[float]], q: int, order: int
) -> list[dict[str, Any]]:
    """指出设计缺少的比例区域（用于样本不足/欠秩时的诊断，不作外推）。"""
    problems: list[dict[str, Any]] = []
    n_terms = len(scheffe_terms(q, order))
    distinct = {
        tuple(round(float(wi), 12) for wi in w) for w in weights_list
    }
    if weights_list and len(distinct) == 1:
        only = next(iter(distinct))
        problems.append({
            "type": "single_ratio_point",
            "weights": [_r(w, 9) for w in only],
            "message": "全部格位集中在同一比例，缺少不同的比例水平",
        })
    elif len(distinct) < n_terms:
        problems.append({
            "type": "insufficient_distinct_points",
            "distinct_points": len(distinct),
            "needed_points": n_terms,
            "message": (
                f"不同比例水平仅 {len(distinct)} 个，{n_terms} 项模型"
                f"至少需要 {n_terms} 个不同比例；请补充其他比例的格位"
            ),
        })
    for i in range(q):
        if all(w[i] <= 0.0 for w in weights_list):
            problems.append({
                "type": "unused_source",
                "source_index": i,
                "message": f"所有格位中第 {i + 1} 份来源占比均为 0，缺少该端点区域",
            })
    if order >= 2:
        for i, j in combinations(range(q), 2):
            if not any(w[i] > 0.0 and w[j] > 0.0 for w in weights_list):
                problems.append({
                    "type": "missing_edge_interior",
                    "pair": [i, j],
                    "message": (
                        f"缺少来源 {i + 1} 与 {j + 1} 同时为正的内部混合比例区域"
                    ),
                })
    if q == 3 and len(weights_list) >= 3:
        pts = np.array([[w[0], w[1]] for w in weights_list], dtype=float)
        centered = pts - pts.mean(axis=0)
        if int(np.linalg.matrix_rank(centered)) < 2:
            problems.append({
                "type": "collinear_cells",
                "message": "全部格位在三角坐标中共线，缺少三角形内部区域",
            })
    return problems


# ---------------------------------------------------------------------------
# 缺陷发生率（Wilson 区间）
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = FIRING_WILSON_Z) -> tuple[float, float, float]:
    """二项发生率的 Wilson 区间，返回 (rate, low, high)。"""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, max(center - half, 0.0), min(center + half, 1.0)


def defect_rate_stats(tiles: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按重复试片统计各类缺陷：发生片数、发生率、Wilson 区间与平均等级。"""
    n = len(tiles)
    out: dict[str, dict[str, Any]] = {}
    for d in DEFECT_TYPES:
        k = sum(1 for t in tiles if t[d] >= 1)
        rate, lo, hi = wilson_interval(k, n) if n else (0.0, 0.0, 0.0)
        out[d] = {
            "n_tiles": n,
            "n_occurred": k,
            "rate": _r(rate, 6),
            "wilson_low": _r(lo, 6),
            "wilson_high": _r(hi, 6),
            "mean_grade": _r(sum(t[d] for t in tiles) / n, 6) if n else None,
        }
    return out


def summarize_cells(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐格汇总：重复数、各定量指标均值/标准差、各类缺陷发生率。"""
    summary: list[dict[str, Any]] = []
    for c in cells:
        tiles = c["tiles"]
        n = len(tiles)
        entry: dict[str, Any] = {
            "position": c["position"],
            "weights": list(c["weights"]),
            "n_tiles": n,
        }
        if n:
            for m in QUANTITATIVE_METRICS:
                vals = np.array([float(t[m]) for t in tiles], dtype=float)
                entry[f"{m}_mean"] = _r(vals.mean(), 6)
                entry[f"{m}_std"] = (
                    _r(vals.std(ddof=1), 6) if n > 1 else 0.0
                )
            entry["defects"] = defect_rate_stats(tiles)
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Scheffé 响应面拟合
# ---------------------------------------------------------------------------

@dataclass
class SurfaceFit:
    """一次拟合的全部结果（供响应组装与配比搜索复用）。"""

    order: int
    q: int
    terms: list[str]
    design_cells: list[dict[str, Any]]      # 参与设计的格位（position + weights）
    unmeasured_cells: list[str]             # 布局内但无试片的格位
    n_tiles: int
    df: int                                 # 剩余自由度
    t_crit: float
    xtxinv: np.ndarray
    beta: dict[str, np.ndarray]
    sigma: dict[str, float]
    metrics: dict[str, dict[str, Any]]      # 每指标的系数/CI/CV/异常试片
    defect_rates: dict[str, list[dict[str, Any]]]  # 逐格缺陷发生率统计


def _metric_values(tiles: Sequence[dict[str, Any]], metric: str) -> np.ndarray:
    if metric in QUANTITATIVE_METRICS:
        return np.array([float(t[metric]) for t in tiles], dtype=float)
    defect = metric[: -len("_rate")]
    return np.array(
        [1.0 if int(t[defect]) >= 1 else 0.0 for t in tiles], dtype=float
    )


def fit_surfaces(
    cells: Sequence[dict[str, Any]],
    order: int,
    outlier_threshold: float = FIRING_OUTLIER_THRESHOLD,
    unmeasured_cells: Optional[list[str]] = None,
) -> SurfaceFit:
    """对全部指标拟合 Scheffé 响应面。

    ``cells`` 仅含**有试片**的格位：{"position", "weights", "tiles"}。
    样本不足或设计矩阵欠秩时抛出 :class:`GlazeError` 并附缺少的比例区域。
    """
    if order not in (1, 2):
        raise GlazeError("模型阶次仅支持 1（一次）或 2（二次）", "invalid_model_order")
    if not cells:
        raise GlazeError("研究内没有任何试片测量，无法拟合", "empty_firing_study")
    q = len(cells[0]["weights"])
    terms = scheffe_terms(q, order)
    p = len(terms)
    weights_list = [list(c["weights"]) for c in cells]
    problems = design_problems(weights_list, q, order)

    if len(cells) < p:
        raise GlazeError(
            f"有试片的格位仅 {len(cells)} 个，少于 {p} 个模型项，"
            "样本不足，请补充下列比例区域的试片",
            "insufficient_design",
            {
                "n_cells": len(cells),
                "n_terms": p,
                "needed_additional_cells": p - len(cells),
                "missing_regions": problems,
            },
        )
    x_cells = np.array([design_row(w, order) for w in weights_list], dtype=float)
    rank = int(np.linalg.matrix_rank(x_cells))
    if rank < p:
        raise GlazeError(
            f"设计矩阵欠秩（秩 {rank} < 项数 {p}），"
            "现有格位无法区分全部模型项，请补充下列比例区域的试片",
            "rank_deficient_design",
            {"rank": rank, "n_terms": p, "missing_regions": problems},
        )

    # 逐试片展开设计矩阵
    row_cells: list[int] = []
    tiles_flat: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    for c_idx, cell in enumerate(cells):
        row = design_row(cell["weights"], order)
        for tile in cell["tiles"]:
            row_cells.append(c_idx)
            tiles_flat.append(tile)
            x_rows.append(row)
    x = np.array(x_rows, dtype=float)
    n = x.shape[0]
    df = n - p
    if df < 1:
        problems.append({
            "type": "insufficient_replicates",
            "n_tiles": n,
            "n_terms": p,
            "message": (
                f"试片总数 {n} 等于模型项数 {p}，没有剩余自由度；"
                "请为任意格位补充重复试片"
            ),
        })
        raise GlazeError(
            f"试片总数 {n} 等于模型项数 {p}，没有剩余自由度估计误差，"
            "请为至少一个格位补充重复试片",
            "insufficient_design",
            {
                "n_tiles": n,
                "n_terms": p,
                "needed_additional_tiles": p + 1 - n,
                "missing_regions": problems,
            },
        )

    xtxinv = np.linalg.inv(x.T @ x)
    h = np.einsum("ij,jk,ik->i", x, xtxinv, x)  # 帽子矩阵对角元
    t_crit = float(t_dist.ppf(0.975, df))

    # 留一格交叉验证：预先算好"去掉某格后"设计是否仍满秩
    cell_cv_ok: list[bool] = []
    for c_idx in range(len(cells)):
        rest = [w for k, w in enumerate(weights_list) if k != c_idx]
        if len(rest) < p:
            cell_cv_ok.append(False)
            continue
        xr = np.array([design_row(w, order) for w in rest], dtype=float)
        cell_cv_ok.append(int(np.linalg.matrix_rank(xr)) == p)

    metrics: dict[str, dict[str, Any]] = {}
    beta_map: dict[str, np.ndarray] = {}
    sigma_map: dict[str, float] = {}
    for metric in ALL_METRICS:
        y = _metric_values(tiles_flat, metric)
        beta = xtxinv @ (x.T @ y)
        resid = y - x @ beta
        rss = float(resid @ resid)
        sigma2 = rss / df
        sigma = math.sqrt(max(sigma2, 0.0))
        se_beta = np.sqrt(np.maximum(np.diag(xtxinv) * sigma2, 0.0))
        tss = float(((y - y.mean()) ** 2).sum())
        r2 = None if tss <= 0.0 else _r(1.0 - rss / tss, 6)

        coefficients = []
        for t_idx, term in enumerate(terms):
            coefficients.append({
                "term": term_label(term),
                "value": _r(beta[t_idx], 6),
                "se": _r(se_beta[t_idx], 6),
                "ci_low": _r(beta[t_idx] - t_crit * se_beta[t_idx], 6),
                "ci_high": _r(beta[t_idx] + t_crit * se_beta[t_idx], 6),
            })

        # 交叉验证：优先留一格（整格重复试片同时留出重拟合）；
        # 剩余设计欠秩时（如格位数恰等于项数的最小设计）退化为
        # 留一片 PRESS（e_i/(1-h_ii)）；h≈1 的饱和试片才排除。
        sq_err: list[float] = []
        loco_cells: list[str] = []
        loto_cells: list[str] = []
        excluded: list[str] = []
        for c_idx, cell in enumerate(cells):
            if cell_cv_ok[c_idx]:
                mask = np.array([rc != c_idx for rc in row_cells])
                xs, ys = x[mask], y[mask]
                beta_sub = np.linalg.solve(xs.T @ xs, xs.T @ ys)
                pred = float(np.array(design_row(cell["weights"], order)) @ beta_sub)
                for k, rc in enumerate(row_cells):
                    if rc == c_idx:
                        sq_err.append((float(y[k]) - pred) ** 2)
                loco_cells.append(cell["position"])
                continue
            used_press = False
            cell_excluded = False
            for k, rc in enumerate(row_cells):
                if rc != c_idx:
                    continue
                if h[k] >= 1.0 - 1e-10:
                    cell_excluded = True  # 饱和点：留一片后该点不可估
                    continue
                press = float(resid[k]) / (1.0 - h[k])
                sq_err.append(press * press)
                used_press = True
            if used_press:
                loto_cells.append(cell["position"])
            if cell_excluded:
                excluded.append(cell["position"])
        cv_rmse = math.sqrt(sum(sq_err) / len(sq_err)) if sq_err else None

        # 异常试片：学生化残差
        outliers: list[dict[str, Any]] = []
        if sigma > 0.0:
            for k, tile in enumerate(tiles_flat):
                denom = sigma * math.sqrt(max(1.0 - h[k], 1e-12))
                stud = float(resid[k]) / denom
                if abs(stud) >= outlier_threshold:
                    outliers.append({
                        "position": tile["position"],
                        "replicate_no": tile["replicate_no"],
                        "observed": _r(y[k], 6),
                        "fitted": _r(y[k] - resid[k], 6),
                        "studentized_residual": _r(stud, 4),
                    })

        beta_map[metric] = beta
        sigma_map[metric] = sigma
        metrics[metric] = {
            "coefficients": coefficients,
            "sigma": _r(sigma, 6),
            "r2": r2,
            "df": df,
            "cv": {
                "method": "leave_one_cell_out",
                "fallback": "leave_one_tile_out",
                "rmse": _r(cv_rmse, 6) if cv_rmse is not None else None,
                "n_tiles_evaluated": len(sq_err),
                "loco_cells": loco_cells,
                "loto_cells": loto_cells,
                "excluded_cells": excluded,
            },
            "outliers": outliers,
        }

    defect_rates = {
        d: [
            {"position": cell["position"], **defect_rate_stats(cell["tiles"])[d]}
            for cell in cells
        ]
        for d in DEFECT_TYPES
    }

    return SurfaceFit(
        order=order,
        q=q,
        terms=[term_label(t) for t in terms],
        design_cells=[
            {"position": c["position"], "weights": list(c["weights"])}
            for c in cells
        ],
        unmeasured_cells=list(unmeasured_cells or []),
        n_tiles=n,
        df=df,
        t_crit=_r(t_crit, 6),
        xtxinv=xtxinv,
        beta=beta_map,
        sigma=sigma_map,
        metrics=metrics,
        defect_rates=defect_rates,
    )


# ---------------------------------------------------------------------------
# 配比搜索（原比例范围 × 步长网格，凸包内，不外推）
# ---------------------------------------------------------------------------

def mixture_grid(
    q: int, low: float, high: float, step: float
) -> list[tuple[float, ...]]:
    """枚举 simplex 网格点：各权重为 step 整数倍、落在 [low, high]、合计为 1。"""
    n = 1.0 / step
    if abs(n - round(n)) > 1e-9:
        raise GlazeError(
            f"比例步长 {step} 不能整分单位区间（1/step 必须为整数）",
            "step_not_compatible",
            {"step": step},
        )
    steps = int(round(n))
    k_lo = math.ceil(low * steps - GRID_EPS)
    k_hi = math.floor(high * steps + GRID_EPS)
    ks = list(range(max(k_lo, 0), min(k_hi, steps) + 1))
    if not ks:
        raise GlazeError(
            f"比例范围 [{low}, {high}] 内没有步长 {step} 的网格点",
            "empty_search_grid",
            {"low": low, "high": high, "step": step},
        )
    if q == 2:
        points = [
            (k / steps, (steps - k) / steps)
            for k in ks
            if k_lo <= steps - k <= k_hi
        ]
    else:
        if len(ks) ** 2 > 4 * FIRING_MAX_GRID_POINTS:
            raise GlazeError(
                f"步长 {step} 在三元网格上过细（候选超过 "
                f"{FIRING_MAX_GRID_POINTS} 点），请加大步长",
                "grid_too_fine",
                {"step": step, "max_grid_points": FIRING_MAX_GRID_POINTS},
            )
        points = []
        for i in ks:
            for j in ks:
                k = steps - i - j
                if k_lo <= k <= k_hi:
                    points.append((i / steps, j / steps, k / steps))
    if len(points) > FIRING_MAX_GRID_POINTS:
        raise GlazeError(
            f"网格共 {len(points)} 点，超过上限 {FIRING_MAX_GRID_POINTS}，请加大步长",
            "grid_too_fine",
            {"n_points": len(points), "max_grid_points": FIRING_MAX_GRID_POINTS},
        )
    if not points:
        raise GlazeError(
            f"比例范围 [{low}, {high}] 与步长 {step} 下没有满足合计为 1 的配比",
            "empty_search_grid",
            {"low": low, "high": high, "step": step},
        )
    return points


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """单调链凸包，返回逆时针顶点（不重复起点）；共线时退化为两端点。"""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _on_segment(p, a, b, eps: float = GEOM_EPS) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    scale = max(math.hypot(b[0] - a[0], b[1] - a[1]), eps)
    if abs(cross) > eps * scale * 10:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def in_design_hull(weights: Sequence[float], design_weights: Sequence[Sequence[float]]) -> bool:
    """候选配比是否落在观测格位的凸包内（线性 1D / 三元 2D）。"""
    q = len(weights)
    if q == 2:
        ts = [w[1] for w in design_weights]
        return min(ts) - GEOM_EPS <= weights[1] <= max(ts) + GEOM_EPS
    point = (weights[0], weights[1])
    hull = _convex_hull([(w[0], w[1]) for w in design_weights])
    if not hull:
        return False
    if len(hull) == 1:
        return math.dist(point, hull[0]) <= GEOM_EPS * 10
    if len(hull) == 2:
        return _on_segment(point, hull[0], hull[1])
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if cross < -GEOM_EPS * 10:
            return False
    return True


def search_ratios(
    fit: SurfaceFit,
    *,
    ratio_low: float,
    ratio_high: float,
    step: float,
    limits: dict[str, float],
    targets: dict[str, dict[str, float]],
    max_candidates: int,
) -> dict[str, Any]:
    """在比例网格上评估全部指标预测，按 (违规数, 不确定度, 中心偏差) 排序。"""
    unknown = sorted((set(limits) | set(targets)) - set(ALL_METRICS))
    if unknown:
        raise GlazeError(
            f"未知指标: {', '.join(unknown)}",
            "unknown_metric",
            {"unknown": unknown, "known": list(ALL_METRICS)},
        )
    grid = mixture_grid(fit.q, ratio_low, ratio_high, step)
    design_weights = [c["weights"] for c in fit.design_cells]

    candidates: list[dict[str, Any]] = []
    n_extrapolated = 0
    for weights in grid:
        if not in_design_hull(weights, design_weights):
            n_extrapolated += 1
            continue
        x0 = np.array(design_row(weights, fit.order), dtype=float)
        leverage = math.sqrt(max(float(x0 @ fit.xtxinv @ x0), 0.0))
        predictions: dict[str, dict[str, Any]] = {}
        for metric in ALL_METRICS:
            raw = float(x0 @ fit.beta[metric])
            se = fit.sigma[metric] * leverage
            clamped = metric in DEFECT_RATE_METRICS and not (0.0 <= raw <= 1.0)
            value = min(max(raw, 0.0), 1.0) if metric in DEFECT_RATE_METRICS else raw
            predictions[metric] = {
                "value": _r(value, 6),
                "se": _r(se, 6),
                "ci_low": _r(value - fit.t_crit * se, 6),
                "ci_high": _r(value + fit.t_crit * se, 6),
                "clamped": clamped,
            }
        violations = sorted(
            m for m, lim in limits.items()
            if predictions[m]["value"] > lim + 1e-12
        )
        center_dev = 0.0
        for metric, spec in targets.items():
            center = (spec["low"] + spec["high"]) / 2.0
            half = (spec["high"] - spec["low"]) / 2.0
            scale = half if half > 1e-12 else max(abs(center), 1.0)
            center_dev += spec.get("weight", 1.0) * abs(
                predictions[metric]["value"] - center
            ) / scale
        candidates.append({
            "weights": [_r(w, 9) for w in weights],
            "predictions": predictions,
            "n_violations": len(violations),
            "violations": violations,
            "uncertainty": _r(leverage, 9),
            "center_deviation": _r(center_dev, 9),
        })

    candidates.sort(
        key=lambda c: (
            c["n_violations"],
            c["uncertainty"],
            c["center_deviation"],
            c["weights"],
        )
    )
    ranked = candidates[:max_candidates]
    return {
        "domain": {
            "ratio_low": ratio_low,
            "ratio_high": ratio_high,
            "step": step,
            "n_grid_points": len(grid),
            "n_in_hull": len(candidates),
            "n_extrapolated_excluded": n_extrapolated,
        },
        "n_candidates": len(ranked),
        "candidates": ranked,
        "best": ranked[0] if ranked else None,
    }
