"""替代配方搜索：混合整数线性规划（HiGHS milp）。

决策变量（按 step 格数计数）::

    k_i —— 原料 i 的投料格数（locked 时固定为常数/step）
    y_i —— 原料 i 是否使用（0/1）
    z_j —— 目标氧化物 j 是否越界（0/1）
    s_j —— 目标氧化物 j 的越界量（釉式单位，>=0）

釉式 r_j = Q_j / F（氧化物摩尔数 / 助熔摩尔总数）为分式；
第 1 阶段越界计数用 F 的上界做大 M 线性化（精确），加权偏差阶段
对线性分式目标采用 Dinkelbach 迭代求全局最优。

排序优先级（字典序）：
  1. 越界项数
  2. 加权偏差之和
  3. 使用原料种数
  4. 成本（相同成本下批量越接近目标越好）
"""
from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, milp
from scipy.optimize import Bounds, LinearConstraint

from .config import (
    MIN_FLUX_MOLES,
    MIN_FLUX_PER_KG,
    OXIDE_CATALOG,
    SEGER_TOLERANCE,
    settings,
)
from .chemistry import oxides_by_role
from .errors import GlazeError

PCT = 100.0
BIG_M_MARGIN = 1.0 + 1e-6


@dataclass
class MaterialData:
    id: int
    name: str
    oxides_pct: dict[str, float]
    loi: float
    price: float
    available: float
    khi: float
    klo: float
    locked_grid: float | None
    required: bool


@dataclass
class TargetData:
    oxide: str
    low: float | None
    high: float | None
    weight: float


# ---------------------------------------------------------------------------
# 请求预处理与矛盾检查
# ---------------------------------------------------------------------------

def prepare_materials(materials: dict, req) -> list[MaterialData]:
    """生成参与优化的原料列表，并执行结构性矛盾检查。"""
    tol = req.batch_tolerance if req.batch_tolerance is not None else req.step
    batch_hi = max(req.batch_size + tol, 0.0)

    referenced = set(req.required) | set(req.forbidden) | set(req.limits) | set(req.locked)
    missing = sorted(i for i in referenced if i not in materials)
    if missing:
        raise GlazeError(
            f"约束引用了不存在的原料: {missing}",
            "material_not_found",
            {"material_ids": missing},
        )

    # 防御性检查（schema 通常已拦截；直接调用 service 时仍保证一致语义）
    forbidden_locked = sorted(set(req.forbidden) & set(req.locked))
    if forbidden_locked:
        raise GlazeError(
            f"原料同时被禁用与锁定用量: {forbidden_locked}",
            "forbidden_locked_conflict",
            {"material_ids": forbidden_locked},
        )
    for mid in sorted(set(req.forbidden) & set(req.limits)):
        lim = req.limits[mid]
        if (lim.low or 0.0) > 0.0 or (lim.high is not None and lim.high > 0.0):
            raise GlazeError(
                f"禁用原料 {mid} 又被赋予正用量约束",
                "forbidden_with_positive_limit",
                {"material_id": mid},
            )

    locked_total = sum(req.locked.values())
    if locked_total > req.batch_size + tol + 1e-9:
        raise GlazeError(
            f"锁定用量合计 {locked_total:.3f} kg 已超过批量上限 "
            f"{req.batch_size + tol:.3f} kg",
            "locked_exceeds_batch",
            {"locked_total": locked_total, "batch_upper": req.batch_size + tol},
        )

    other_cap = max(batch_hi - locked_total, 0.0)

    result: list[MaterialData] = []
    for mid, mat in materials.items():
        if mid in req.forbidden:
            continue
        lim = req.limits.get(mid)
        high = min(
            mat.available,
            lim.high if (lim and lim.high is not None) else mat.available,
        )
        # 任何非锁定原料的用量不可能超过“批量上限 - 已锁定总量”
        if mid not in req.locked:
            high = min(high, other_cap)
        low = lim.low if (lim and lim.low is not None) else 0.0
        if low > high + 1e-9:
            raise GlazeError(
                f"原料「{mat.name}」下限 {low} kg 高于可用上限 {high} kg"
                "（库存或单项上限）",
                "material_bound_vs_stock",
                {"material_id": mid, "low": low, "high": high},
            )

        if mid in req.locked:
            locked = req.locked[mid]
            if locked > high + 1e-9 or locked < low - 1e-9:
                raise GlazeError(
                    f"原料「{mat.name}」锁定 {locked} kg 超出 [{low}, {high}] kg",
                    "lock_vs_bounds",
                    {"material_id": mid, "locked": locked, "low": low, "high": high},
                )
            locked_grid = float(round(locked / req.step))
            klo = khi = locked_grid
        else:
            klo = float(math.ceil(low / req.step - 1e-9))
            khi = float(math.floor(high / req.step + 1e-9))
            if mid in req.required and klo < 1.0:
                klo = 1.0  # 必用至少一个步进
            if klo > khi:
                raise GlazeError(
                    f"原料「{mat.name}」在步进 {req.step} kg 下无可行格点："
                    f"[{low}, {high}] kg",
                    "no_grid_point",
                    {"material_id": mid, "low": low, "high": high, "step": req.step},
                )
            locked_grid = None
        result.append(
            MaterialData(
                id=mid,
                name=mat.name,
                oxides_pct=mat.oxides,
                loi=mat.loi,
                price=mat.price,
                available=mat.available,
                khi=khi,
                klo=klo,
                locked_grid=locked_grid,
                required=mid in req.required,
            )
        )

    if not result:
        raise GlazeError("没有任何可用原料（全部被禁用或库为空）", "no_materials")
    return result


def build_targets(req) -> list[TargetData]:
    return [
        TargetData(o, t.low, t.high, t.weight) for o, t in req.targets.items()
    ]


# ---------------------------------------------------------------------------
# MILP 构造
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    mats: list[MaterialData]
    targets: list[TargetData]
    step: float
    batch_lo: float
    batch_hi: float

    def __post_init__(self):
        self.n = len(self.mats)
        self.nt = len(self.targets)
        # 变量列布局： k[0:n] | y[n:2n] | z[2n:2n+nt] | s[...]
        self.idx_y = self.n
        self.idx_z = 2 * self.n
        self.idx_s = 2 * self.n + self.nt
        self.nvars = 2 * self.n + 2 * self.nt

        oxide_set: set[str] = set()
        for m in self.mats:
            oxide_set.update(m.oxides_pct)
        self.mole_per_kg: dict[str, np.ndarray] = {}
        for o in oxide_set:
            col = np.zeros(self.n)
            mw = OXIDE_CATALOG[o].molwt
            for i, m in enumerate(self.mats):
                col[i] = m.oxides_pct.get(o, 0.0) / (PCT * mw)
            self.mole_per_kg[o] = col

        flux_cols = [
            self.mole_per_kg.get(o, np.zeros(self.n)) for o in oxides_by_role("flux")
        ]
        self.flux_per_kg = (
            np.sum(flux_cols, axis=0) if flux_cols else np.zeros(self.n)
        )
        self.flux_max_moles = float(
            sum(self.flux_per_kg[i] * min(self.mats[i].khi,
                                          self.batch_hi / self.step) * self.step
                for i in range(self.n))
        )
        # 每个目标氧化物的摩尔量上界（按各料上限线性组合），
        # 供大 M 线性化使用，避免用全局过宽的上界削弱 LP 松弛。
        eff_khi = np.array([
            min(m.khi, self.batch_hi / self.step) for m in self.mats
        ])
        self.oxide_max_moles: dict[str, float] = {}
        for t in self.targets:
            cj = self.mole_per_kg.get(t.oxide, np.zeros(self.n))
            self.oxide_max_moles[t.oxide] = float(
                np.dot(np.maximum(cj, 0.0), eff_khi * self.step)
            )

    # ---- 变量界 / 整数性 ----
    def bounds_integrality(self):
        lb = np.zeros(self.nvars)
        ub = np.full(self.nvars, np.inf)
        integrality = np.zeros(self.nvars)
        for i, m in enumerate(self.mats):
            lb[i] = m.klo
            ub[i] = m.khi
            if m.locked_grid is None:
                integrality[i] = 1.0
            # y_i：锁定时随用量固定；必用时下界为 1
            if m.locked_grid is not None:
                lb[self.idx_y + i] = ub[self.idx_y + i] = (
                    1.0 if m.locked_grid > 0 else 0.0
                )
            else:
                lb[self.idx_y + i] = 1.0 if m.required else 0.0
                ub[self.idx_y + i] = 1.0
            integrality[self.idx_y + i] = 1.0
        for j in range(self.nt):
            ub[self.idx_z + j] = 1.0
            integrality[self.idx_z + j] = 1.0
        return Bounds(lb, ub), integrality

    def _empty_row(self) -> np.ndarray:
        return np.zeros(self.nvars)

    # ---- 静态约束 ----
    def static_constraints(self, extra_cuts: list[np.ndarray] | None = None):
        rows, lbs, ubs = [], [], []

        def add(row, lo, hi):
            rows.append(row)
            lbs.append(lo)
            ubs.append(hi)

        # batch_lo <= step * Σk <= batch_hi
        r = self._empty_row()
        r[0:self.n] = self.step
        add(r, self.batch_lo, self.batch_hi)

        for i, m in enumerate(self.mats):
            if m.locked_grid is not None:
                continue
            # y_i <= k_i <= khi*y_i
            r = self._empty_row()
            r[i] = 1.0
            r[self.idx_y + i] = -1.0
            add(r, 0.0, np.inf)            # k_i - y_i >= 0
            r = self._empty_row()
            r[i] = 1.0
            r[self.idx_y + i] = -m.khi
            add(r, -np.inf, 0.0)           # k_i - khi*y_i <= 0

        # 助熔摩尔数下限（绝对值 + 对批量的相对下限）
        r = self._empty_row()
        r[0:self.n] = self.flux_per_kg * self.step
        add(r, MIN_FLUX_MOLES, np.inf)
        r = self._empty_row()
        r[0:self.n] = (self.flux_per_kg - MIN_FLUX_PER_KG) * self.step
        add(r, 0.0, np.inf)

        # no-good 割：排除给定用料集合组合
        for used_mask in extra_cuts or []:
            r = self._empty_row()
            for i in range(self.n):
                if used_mask[i]:
                    r[self.idx_y + i] = 1.0
            add(r, -np.inf, float(used_mask.sum() - 1))

        return LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))

    # ---- 越界计数（精确大 M；M 按各氧化物实际摩尔范围收紧） ----
    def stage1_constraint(self):
        rows, lbs, ubs = [], [], []
        for j, t in enumerate(self.targets):
            cj = self.mole_per_kg.get(t.oxide, np.zeros(self.n))
            # 表达式 (cj - b*flux) 的上界按正/负系数在原料上限处取值
            def _big_m(coef: np.ndarray) -> float:
                m = sum(
                    max(coef[i], 0.0) * self.mats[i].khi * self.step
                    for i in range(self.n)
                )
                return max(m * BIG_M_MARGIN, MIN_FLUX_MOLES * 10.0)

            if t.high is not None:
                coef = cj - (t.high + SEGER_TOLERANCE) * self.flux_per_kg
                r = self._empty_row()
                r[0:self.n] = coef * self.step
                r[self.idx_z + j] = -_big_m(coef)
                rows.append(r)
                lbs.append(-np.inf)
                ubs.append(0.0)
            if t.low is not None:
                coef = (t.low - SEGER_TOLERANCE) * self.flux_per_kg - cj
                r = self._empty_row()
                r[0:self.n] = coef * self.step
                r[self.idx_z + j] = -_big_m(coef)
                rows.append(r)
                lbs.append(-np.inf)
                ubs.append(0.0)
        if not rows:
            return None
        return LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))

    # ---- 越界量（摩尔尺度，s_j 单位为"摩尔"；s_j/F 即釉式偏差） ----
    def deviation_constraint(self):
        """线性化越界量（含 SEGER_TOLERANCE 容差）。

        高目标: Q_j - high_j*F - TOL·F - s_j <= 0
        低目标: low_j*F - Q_j - TOL·F - s_j <= 0
        即釉式偏差在 TOL 以内时 s_j 可为 0，吸收浮点/MIP 噪声。
        """
        rows, lbs, ubs = [], [], []
        for j, t in enumerate(self.targets):
            cj = self.mole_per_kg.get(t.oxide, np.zeros(self.n))
            if t.high is not None:
                r = self._empty_row()
                r[0:self.n] = (
                    cj - (t.high + SEGER_TOLERANCE) * self.flux_per_kg
                ) * self.step
                r[self.idx_s + j] = -1.0
                rows.append(r)
                lbs.append(-np.inf)
                ubs.append(0.0)
            if t.low is not None:
                r = self._empty_row()
                r[0:self.n] = (
                    (t.low - SEGER_TOLERANCE) * self.flux_per_kg - cj
                ) * self.step
                r[self.idx_s + j] = -1.0
                rows.append(r)
                lbs.append(-np.inf)
                ubs.append(0.0)
        if not rows:
            return None
        return LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))

    def zsum_constraint(self, value: float, slack: float = 1e-6):
        r = self._empty_row()
        r[self.idx_z:self.idx_z + self.nt] = 1.0
        return LinearConstraint(r[None, :], value - slack, value + slack)

    def weighted_dev_moles_constraint(self, value: float, slack: float = 1e-8):
        """Σ w_j·s_j <= value（摩尔尺度加权偏差），下游阶段保真用。"""
        r = self._empty_row()
        for j, t in enumerate(self.targets):
            r[self.idx_s + j] = t.weight
        scale = max(abs(value), 1e-12)
        return LinearConstraint(
            r[None, :], -np.inf, value + slack * scale + 1e-12
        )

    def ysum_constraint(self, value: float):
        r = self._empty_row()
        r[self.idx_y:self.idx_y + self.n] = 1.0
        return LinearConstraint(r[None, :], -np.inf, value + 1e-6)

    def cost_constraint(self, value: float):
        r = self._empty_row()
        for i, m in enumerate(self.mats):
            r[i] = m.price * self.step
        return LinearConstraint(r[None, :], -np.inf, value + 1e-6 * max(value, 1.0))

    def extract(self, x: np.ndarray) -> dict:
        grids = np.rint(x[0:self.n]).astype(int)
        amounts = grids.astype(float) * self.step
        zs = x[self.idx_z:self.idx_z + self.nt]
        ss = x[self.idx_s:self.idx_s + self.nt]
        return {
            "grids": grids,
            "amounts": amounts,
            "used": amounts > 0.0,
            "z": zs,
            "s": ss,
        }


def _merge(static: LinearConstraint, *extra: LinearConstraint | None) -> LinearConstraint:
    cs = [static] + [e for e in extra if e is not None]
    return LinearConstraint(
        np.vstack([c.A for c in cs]),
        np.concatenate([c.lb for c in cs]),
        np.concatenate([c.ub for c in cs]),
    )


def heuristic_start(prob: "Problem") -> np.ndarray | None:
    """构造一个高质量可行整数初始解（热启动）。

    步骤：
      1) Charnes-Cooper 分式 LP 求"加权越界最小"的连续配比；
      2) 就近取整并投影到批量区间，满足锁定/必用/库存格点；
      3) 以实际釉式计算 z（越界指示）与 s（摩尔越界量）。
    返回完整变量向量，不可行时退回简单格点构造，再不行返回 None。
    """
    n = prob.n
    step = prob.step

    grids = _continuous_rounded_grids(prob)
    if grids is None:
        grids = np.zeros(n)
        for i, m in enumerate(prob.mats):
            grids[i] = m.locked_grid if m.locked_grid is not None else (
                max(m.klo, 1.0 if m.required else 0.0)
            )

    if not _project_to_batch(prob, grids):
        return None

    # 单格邻域局部下降：在保持总量的前提下，逐格在原料间转移，
    # 降低 (越界项数, 加权偏差)，以得到质量接近最优的整数起点。
    grids = _local_descent(prob, grids)

    amounts = grids * step
    flux = float(np.dot(prob.flux_per_kg, amounts))
    if flux <= MIN_FLUX_MOLES or flux < MIN_FLUX_PER_KG * float(amounts.sum()):
        return None

    x = np.zeros(prob.nvars)
    x[0:n] = grids
    x[prob.idx_y:prob.idx_y + n] = (grids > 0).astype(float)
    for j, t in enumerate(prob.targets):
        cj = prob.mole_per_kg.get(t.oxide, np.zeros(n))
        r = float(np.dot(cj, amounts)) / flux
        dev = 0.0
        if t.high is not None:
            dev = max(dev, r - t.high)
        if t.low is not None:
            dev = max(dev, t.low - r)
        in_dev = 0.0 if dev <= SEGER_TOLERANCE else dev
        x[prob.idx_z + j] = 1.0 if dev > SEGER_TOLERANCE else 0.0
        x[prob.idx_s + j] = in_dev * flux
    return x


def _continuous_rounded_grids(prob: "Problem") -> np.ndarray | None:
    """Charnes-Cooper 求连续最优配比并就近取整（含多目标权重）。"""
    n, nt = prob.n, prob.nt
    step = prob.step
    rows, lo, hi = [], [], []

    def add(r, l, h):
        rows.append(np.asarray(r, float))
        lo.append(l)
        hi.append(h)

    add(np.append(np.full(n, step), -prob.batch_lo), 0.0, np.inf)
    add(np.append(np.full(n, step), -prob.batch_hi), -np.inf, 0.0)
    add(np.append(prob.flux_per_kg * step, 0.0), 1.0, 1.0)
    add(np.append((prob.flux_per_kg - MIN_FLUX_PER_KG) * step, 0.0), 0.0, np.inf)
    for i, m in enumerate(prob.mats):
        if m.locked_grid is not None:
            r = np.zeros(n + 1)
            r[i] = 1.0
            r[n] = -m.locked_grid
            add(r, 0.0, 0.0)
        else:
            r = np.zeros(n + 1)
            r[i] = 1.0
            r[n] = -m.klo
            add(r, 0.0, np.inf)
            r = np.zeros(n + 1)
            r[i] = 1.0
            r[n] = -m.khi
            add(r, -np.inf, 0.0)
            if m.required:
                r = np.zeros(n + 1)
                r[i] = 1.0
                r[n] = -1.0
                add(r, 0.0, np.inf)
    bounds_cc = [(0.0, np.inf)] * n + [(1e-12, 1.0 / MIN_FLUX_MOLES)]

    # 引入 s'(=s/F) 变量最小化加权偏差：需要扩展到 n+1+nt
    # 直接对每个目标加权越界构造目标行（在 CC 空间 s' 为釉式偏差）
    extra_rows = list(rows)
    extra_lo, extra_hi = list(lo), list(hi)
    nv = n + 1 + nt
    for j, t in enumerate(prob.targets):
        cj = prob.mole_per_kg.get(t.oxide, np.zeros(n))
        if t.high is not None:
            r = np.zeros(nv)
            r[0:n] = (cj - t.high * prob.flux_per_kg) * step
            r[n + 1 + j] = -1.0
            extra_rows.append(r)
            extra_lo.append(-np.inf)
            extra_hi.append(0.0)
        if t.low is not None:
            r = np.zeros(nv)
            r[0:n] = (t.low * prob.flux_per_kg - cj) * step
            r[n + 1 + j] = -1.0
            extra_rows.append(r)
            extra_lo.append(-np.inf)
            extra_hi.append(0.0)
    # 原 CC 约束补齐到 nv 列
    A_full = np.zeros((len(extra_rows), nv))
    for idx, r in enumerate(extra_rows):
        A_full[idx, :len(r)] = r
    b_full = (np.array(extra_lo), np.array(extra_hi))
    bounds_full = bounds_cc + [(0.0, np.inf)] * nt
    cobj = np.zeros(nv)
    for j, t in enumerate(prob.targets):
        cobj[n + 1 + j] = t.weight
    sol = _range_lp(cobj, A_full, b_full[0], b_full[1], bounds_full)
    if sol is None:
        return None
    # k' = k/F；t=1/F ⇒ k = k'/t
    t_val = max(sol.x[n], 1e-12)
    cont_grids = sol.x[0:n] / t_val
    locked = {i for i, m in enumerate(prob.mats) if m.locked_grid is not None}
    grids = np.zeros(n)
    for i in range(n):
        g = prob.mats[i].locked_grid if i in locked else int(round(cont_grids[i]))
        g = min(max(g, prob.mats[i].klo), prob.mats[i].khi)
        grids[i] = g
    return grids


def _grids_score(prob: "Problem", grids: np.ndarray) -> tuple[int, float, float]:
    """格点配方的排序键：(越界项数, 加权偏差, 成本)。"""
    amounts = grids * prob.step
    flux = float(np.dot(prob.flux_per_kg, amounts))
    if flux <= MIN_FLUX_MOLES:
        return (10**9, 1e18, 1e18)
    nv = 0
    wdev = 0.0
    for t in prob.targets:
        cj = prob.mole_per_kg.get(t.oxide, np.zeros(prob.n))
        r = float(np.dot(cj, amounts)) / flux
        d = 0.0
        if t.high is not None:
            d = max(d, r - t.high)
        if t.low is not None:
            d = max(d, t.low - r)
        if d > SEGER_TOLERANCE:
            nv += 1
            wdev += t.weight * d
    cost = float(np.dot([m.price for m in prob.mats], amounts))
    return (nv, wdev, cost)


def zset_constraint(prob: "Problem", zset: np.ndarray) -> LinearConstraint:
    """把每个 z_j 固定为给定 0/1（具体越界集合），取代计数约束。"""
    rows = []
    for j in range(prob.nt):
        r = prob._empty_row()
        r[prob.idx_z + j] = 1.0
        rows.append(r)
    A = np.vstack(rows)
    # 用紧容差把整数变量钉在 0/1，避免 MIP 解出现 0.999 取整翻转
    vals = zset.astype(float)
    return LinearConstraint(A, vals - 1e-8, vals + 1e-8)


def _local_descent(prob: "Problem", grids: np.ndarray,
                   max_passes: int = 4) -> np.ndarray:
    """保持总量不变的单格/两格转移局部搜索，最小化 (越界, 加权偏差, 成本)。"""
    grids = grids.astype(float).copy()
    cur = _grids_score(prob, grids)
    locked = {i for i, m in enumerate(prob.mats) if m.locked_grid is not None}
    free = [i for i in range(prob.n) if i not in locked]
    for _ in range(max_passes):
        improved = False
        # 单格转移
        for src in free:
            if grids[src] <= prob.mats[src].klo + 1e-9:
                continue
            for dst in free:
                if dst == src or grids[dst] >= prob.mats[dst].khi - 1e-9:
                    continue
                trial = grids.copy()
                trial[src] -= 1.0
                trial[dst] += 1.0
                sc = _grids_score(prob, trial)
                if sc < cur:
                    grids, cur, improved = trial, sc, True
        # 两格交换（src->dst1, src->dst2 一次）帮助跳出单格平台
        for src in free:
            if grids[src] <= prob.mats[src].klo + 1e-9:
                continue
            for d1 in free:
                if grids[d1] >= prob.mats[d1].khi - 1e-9:
                    continue
                for d2 in free:
                    if d2 == src or d2 == d1 or grids[d2] >= prob.mats[d2].khi - 1e-9:
                        continue
                    trial = grids.copy()
                    trial[src] -= 2.0
                    trial[d1] += 1.0
                    trial[d2] += 1.0
                    if (trial >= np.array([prob.mats[i].klo for i in range(prob.n)]) - 1e-9
                            ).all():
                        sc = _grids_score(prob, trial)
                        if sc < cur:
                            grids, cur, improved = trial, sc, True
        if not improved:
            break
    return grids


def _project_to_batch(prob: "Problem", grids: np.ndarray) -> bool:
    """把格点总量投影进批量格点区间，尊重锁定/必用/单项上下界。"""
    lo_g = int(math.ceil(prob.batch_lo / prob.step - 1e-9))
    hi_g = int(math.floor(prob.batch_hi / prob.step + 1e-9))
    total = int(round(grids.sum()))
    if total > hi_g:
        need = total - hi_g
        order = sorted(
            range(prob.n),
            key=lambda i: (
                prob.mats[i].locked_grid is not None,
                prob.mats[i].required,
            ),
        )
        for i in order:
            reducible = max(grids[i] - prob.mats[i].klo, 0.0)
            cut = min(reducible, float(need))
            grids[i] -= cut
            need -= int(cut)
            if need <= 0:
                break
        if need > 0:
            return False
    elif total < lo_g:
        need = lo_g - total
        room = np.array([
            0.0 if prob.mats[i].locked_grid is not None
            else prob.mats[i].khi - grids[i]
            for i in range(prob.n)
        ])
        for i in np.argsort(-room):
            add = min(room[i], float(need))
            grids[i] += add
            need -= int(add)
            if need <= 0:
                break
        if need > 0:
            return False
    return True


@contextlib.contextmanager
def _silence_native_stdout():
    """静音 HiGHS C++ 层直接写入 std::cout 的调试输出。

    scipy 打包的 HiGHS 留有遗留的无条件 ``std::cout`` 打印
    （``HighsMipSolverData::transform...``），``disp=False`` 与
    ``output_flag`` 均无法关闭，且其内容在进程退出随 C++ 流缓冲
    刷新——调用结束就恢复 fd 反而会让残留缓冲刷回真实 stdout。
    因此整个搜索/诊断生命周期都需保持 fd1 指向 /dev/null；
    Python 侧的 ``sys.stdout`` 先 flush，避免吞掉自己的输出。
    """
    import sys

    sys.stdout.flush()
    saved = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
        # 恢复前把 C 层 stdio（含 HiGHS 遗留的 std::cout 文本，
        # 经 C 缓冲兼容写出）排空到 /dev/null，防止退出时刷回终端
        try:
            import ctypes

            ctypes.CDLL(None).fflush(None)
        except Exception:  # pragma: no cover - 非 glibc 平台退化
            pass
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)


def _milp(c, constraints, integrality, bounds, start=None, time_limit=None):
    # 注：scipy 的 milp 不透传 HiGHS MIP 热启动（mip_start 仅 verbatim
    # 且被忽略），故 start 不送入求解器；它用于调用方比较/兜底。
    options = {
        "time_limit": settings.milp_time_limit if time_limit is None else time_limit,
        "mip_rel_gap": settings.mip_rel_gap,
        "disp": False,
    }
    try:
        return milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options=options,
        )
    except Exception as exc:  # pragma: no cover
        raise GlazeError(f"优化求解失败: {exc}", "solver_error")


def _full_start(prob: "Problem", sol: dict) -> np.ndarray:
    """把 extract 的解展开为完整变量向量，作为下阶段热启动。"""
    x = np.zeros(prob.nvars)
    x[0:prob.n] = sol["grids"]
    x[prob.idx_y:prob.idx_y + prob.n] = (sol["amounts"] > 0).astype(float)
    x[prob.idx_z:prob.idx_z + prob.nt] = np.clip(sol["z"], 0.0, 1.0)
    x[prob.idx_s:prob.idx_s + prob.nt] = np.maximum(sol["s"], 0.0)
    return x


def lexicographic_search(
    prob: Problem, cuts: list[np.ndarray] | None = None
) -> dict | None:
    """四阶段字典序优化，返回 extract 形式的解；无解返回 None。"""
    bounds, integrality = prob.bounds_integrality()
    static = prob.static_constraints(cuts)

    # ---- 阶段 1：确定最小越界项数 ----
    # 先用高质量启发式（连续分式 LP 取整）拿到可行 z 计数兜底；
    # 再以短时限让 MILP 尝试证明/改进，超时则采用启发式值——
    # 后续 Dinkelbach 阶段在固定越界集合下优化加权偏差。
    c1 = prob._empty_row()
    c1[prob.idx_z:prob.idx_z + prob.nt] = 1.0
    hs = heuristic_start(prob)
    n_violations_heur = None
    if hs is not None:
        n_violations_heur = int(round(float(
            hs[prob.idx_z:prob.idx_z + prob.nt].sum()
        )))
    res1 = _milp(
        c1,
        _merge(static, prob.stage1_constraint()),
        integrality,
        bounds,
        start=hs,
        time_limit=settings.stage1_time_limit if n_violations_heur is not None
        else settings.milp_time_limit,
    )
    if res1.x is not None:
        sol = prob.extract(res1.x)
        zset = np.rint(sol["z"]).astype(int)
    elif hs is not None:
        sol = prob.extract(hs)
        zset = np.rint(sol["z"]).astype(int)
    else:
        return None
    # 固定“具体越界集合”。后续阶段 z 成为常数，可移除昂贵的大 M
    # 计数约束，子问题变为普通 MILP（实测从数十秒降到亚秒级）。
    zfix = zset_constraint(prob, zset)

    # ---- 阶段 2：最小化加权偏差（Dinkelbach 分式规划） ----
    # 目标  min Σ w_j·s_j / F ，其中 s_j/F 是釉式偏差，F 为助熔摩尔数。
    # 对线性分式（整数可行域）用 Dinkelbach 迭代：
    #   min_k [Σ w_j·s_j - λ·F]，再以所得解的真实比值更新 λ，
    # 残差归零即全局最优；避免固定分母线性化选错解。
    dev_con = prob.deviation_constraint()
    s_moles_best = None
    if prob.nt:
        lam = 0.0
        seen_keys: set[tuple] = set()
        for _ in range(max(settings.dinkelbach_iters, 1)):
            c2 = prob._empty_row()
            for j, t in enumerate(prob.targets):
                c2[prob.idx_s + j] = t.weight
            c2[0:prob.n] -= lam * prob.flux_per_kg * prob.step
            res2 = _milp(
                c2,
                _merge(static, zfix, dev_con),
                integrality,
                bounds,
                start=_full_start(prob, sol),
            )
            if res2.x is None:
                break
            sol = prob.extract(res2.x)
            key = tuple(int(g) for g in sol["grids"])
            if key in seen_keys:
                break
            seen_keys.add(key)
            s_moles = float(
                sum(prob.targets[j].weight * max(sol["s"][j], 0.0)
                    for j in range(prob.nt))
            )
            flux = float(np.dot(prob.flux_per_kg, sol["amounts"]))
            if flux <= MIN_FLUX_MOLES:
                break
            s_moles_best = s_moles
            residual = res2.fun  # min 目标值 = s_moles - lam*flux
            if abs(residual) <= 1e-8 * max(s_moles, flux, 1.0):
                break
            lam = s_moles / flux
        wfix = prob.weighted_dev_moles_constraint(
            s_moles_best if s_moles_best is not None else 0.0
        )
    else:
        wfix = None

    # ---- 阶段 3：最小化原料种数 ----
    c3 = prob._empty_row()
    c3[prob.idx_y:prob.idx_y + prob.n] = 1.0
    res3 = _milp(
        c3,
        _merge(static, zfix, dev_con, wfix),
        integrality,
        bounds,
        start=_full_start(prob, sol),
    )
    if res3.x is not None:
        sol = prob.extract(res3.x)
    n_materials = int(sol["used"].sum())
    yfix = prob.ysum_constraint(float(n_materials))

    # ---- 阶段 4：最小化成本 ----
    c4 = prob._empty_row()
    for i, m in enumerate(prob.mats):
        c4[i] = m.price * prob.step
    res4 = _milp(
        c4,
        _merge(static, zfix, dev_con, wfix, yfix),
        integrality,
        bounds,
        start=_full_start(prob, sol),
    )
    if res4.x is not None:
        sol = prob.extract(res4.x)
    cost_val = float(np.dot([m.price for m in prob.mats], sol["amounts"]))
    cost_fix = prob.cost_constraint(cost_val)

    # 同成本下的批量贴合 tie-break：分别极小/极大批量，取最接近目标中心者
    b0 = 0.5 * (prob.batch_lo + prob.batch_hi)
    cons = _merge(static, zfix, dev_con, wfix, yfix, cost_fix)

    def _extreme(sign: float):
        c = prob._empty_row()
        c[0:prob.n] = sign * prob.step
        res = _milp(c, cons, integrality, bounds, start=_full_start(prob, sol))
        return prob.extract(res.x) if res.x is not None else None

    best = sol
    best_dist = abs(float(best["amounts"].sum()) * prob.step - b0)
    for cand in (_extreme(+1.0), _extreme(-1.0)):
        if cand is None:
            continue
        cand_cost = float(np.dot([m.price for m in prob.mats], cand["amounts"]))
        dist = abs(float(cand["amounts"].sum()) * prob.step - b0)
        if cand_cost <= cost_val + 1e-6 * max(cost_val, 1.0) and dist < best_dist - 1e-12:
            best, best_dist = cand, dist
    return best


def _seger_of(prob: Problem, amounts: np.ndarray) -> dict[str, float]:
    moles = {}
    for o, col in prob.mole_per_kg.items():
        val = float(np.dot(col, amounts))
        if val > 0:
            moles[o] = val
    flux = sum(moles.get(o, 0.0) for o in oxides_by_role("flux"))
    if flux <= MIN_FLUX_MOLES:
        return {}
    return {o: v / flux for o, v in moles.items()}


def _weighted_deviation(targets: list[TargetData], seger: dict[str, float]) -> float:
    total = 0.0
    for t in targets:
        v = seger.get(t.oxide, 0.0)
        d = 0.0
        if t.low is not None:
            d = max(d, t.low - v)
        if t.high is not None:
            d = max(d, v - t.high)
        total += t.weight * d
    return total


# ---------------------------------------------------------------------------
# 无解诊断：连续松弛 + Charnes-Cooper 分式 LP
# ---------------------------------------------------------------------------

def diagnose(prob: Problem) -> dict:
    n = prob.n
    diag: dict = {
        "feasible_relaxed": False,
        "limiting_oxides": [],
        "tight_materials": [],
        "structural_issues": [],
        "note": "",
    }

    # 1) 去掉釉式目标的连续松弛可行性
    A_rows, A_lb, A_ub = [], [], []
    r = np.zeros(n)
    r[:] = prob.step
    A_rows.append(r)
    A_lb.append(prob.batch_lo)
    A_ub.append(prob.batch_hi)
    r = np.zeros(n)
    r[:] = prob.flux_per_kg * prob.step
    A_rows.append(r)
    A_lb.append(MIN_FLUX_MOLES)
    A_ub.append(np.inf)
    # 相对助熔下限：(flux_per_kg - MIN_FLUX_PER_KG)*step*Σk >= 0
    r = np.zeros(n)
    r[:] = (prob.flux_per_kg - MIN_FLUX_PER_KG) * prob.step
    A_rows.append(r)
    A_lb.append(0.0)
    A_ub.append(np.inf)
    A = np.vstack(A_rows)
    bounds_k = [(m.klo, m.khi) for m in prob.mats]
    base = _range_lp(np.zeros(n), A, np.array(A_lb), np.array(A_ub), bounds_k)
    if base is None:
        diag["structural_issues"].append(
            "即使完全放开釉式目标，批量/库存/锁定约束仍无解"
        )
        diag["note"] = "限制来自批量与库存（及锁定用量），与釉式目标无关"
        _structural_details(prob, diag)
        return diag
    diag["feasible_relaxed"] = True

    if not prob.targets:
        diag["note"] = "连续松弛可行但整数格点不可行；可减小 step 或放宽批量容差。"
        return diag

    # 2) Charnes-Cooper：变量 (k'_1..k'_n, t)，k'=k/F, t=1/F
    cc_A, cc_lo, cc_hi = [], [], []

    def add_cc(row, lo, hi):
        cc_A.append(np.asarray(row, dtype=float))
        cc_lo.append(lo)
        cc_hi.append(hi)

    add_cc(np.append(np.full(n, prob.step), -prob.batch_lo), 0.0, np.inf)
    add_cc(np.append(np.full(n, prob.step), -prob.batch_hi), -np.inf, 0.0)
    add_cc(np.append(prob.flux_per_kg * prob.step, 0.0), 1.0, 1.0)
    # 相对助熔下限：(flux - floor)*step*Σk' >= floor  （因 step*Σk = step*Σk'/t）
    add_cc(
        np.append((prob.flux_per_kg - MIN_FLUX_PER_KG) * prob.step, 0.0),
        0.0,
        np.inf,
    )

    for i, m in enumerate(prob.mats):
        r1 = np.zeros(n + 1)
        r1[i] = 1.0
        if m.locked_grid is not None:
            r1[n] = -m.locked_grid
            add_cc(r1, 0.0, 0.0)                 # k'_i = locked*t
        else:
            r1[n] = -m.klo
            add_cc(r1, 0.0, np.inf)              # k'_i >= klo*t
            r2 = np.zeros(n + 1)
            r2[i] = 1.0
            r2[n] = -m.khi
            add_cc(r2, -np.inf, 0.0)             # k'_i <= khi*t
            if m.required:
                r3 = np.zeros(n + 1)
                r3[i] = 1.0
                r3[n] = -1.0
                add_cc(r3, 0.0, np.inf)          # k'_i >= t（至少 1 格）

    A_cc = np.vstack(cc_A)
    bounds_cc = [(0.0, np.inf)] * n + [(1e-12, 1.0 / MIN_FLUX_MOLES)]

    probe = _range_lp(np.zeros(n + 1), A_cc, np.array(cc_lo), np.array(cc_hi),
                      bounds_cc)
    if probe is None:
        diag["note"] = "釉式可行域 LP 失败，通常是助熔含量过低或锁定/必用组合矛盾。"
        return diag

    limiting = []
    for tgt in prob.targets:
        cj = prob.mole_per_kg.get(tgt.oxide, np.zeros(n))
        cvec = np.append(cj * prob.step, 0.0)
        lo_sol = _range_lp(cvec, A_cc, np.array(cc_lo), np.array(cc_hi), bounds_cc)
        hi_sol = _range_lp(-cvec, A_cc, np.array(cc_lo), np.array(cc_hi), bounds_cc)
        if lo_sol is None or hi_sol is None:
            continue
        ach_lo = float(lo_sol.fun)
        ach_hi = float(-hi_sol.fun)
        reasons = []
        if tgt.low is not None and ach_hi < tgt.low - 1e-7:
            reasons.append(f"可达釉式最高 {ach_hi:.4f} 仍低于下限 {tgt.low}")
        if tgt.high is not None and ach_lo > tgt.high + 1e-7:
            reasons.append(f"可达釉式最低 {ach_lo:.4f} 仍高于上限 {tgt.high}")
        if not reasons:
            continue
        ranked = sorted(
            ((prob.mats[i].name, float(cj[i])) for i in range(n)),
            key=lambda x: -x[1],
        )
        limiting.append(
            {
                "oxide": tgt.oxide,
                "low": tgt.low,
                "high": tgt.high,
                "achievable_low": round(ach_lo, 6),
                "achievable_high": round(ach_hi, 6),
                "reason": "；".join(reasons),
                "highest_contribution_materials": [
                    {"material": nm, "mole_per_kg": round(v, 7)}
                    for nm, v in ranked[:3] if v > 0
                ],
                "lowest_contribution_materials": [
                    {"material": nm, "mole_per_kg": round(v, 7)}
                    for nm, v in sorted(
                        ((prob.mats[i].name, float(cj[i])) for i in range(n)),
                        key=lambda x: x[1],
                    )[:3]
                ],
            }
        )
    diag["limiting_oxides"] = limiting

    # 3) 紧库存 / 锁定信息
    total_cap = sum(m.khi * prob.step for m in prob.mats)
    if total_cap < prob.batch_lo - 1e-9:
        diag["tight_materials"].append(
            {
                "type": "total_stock",
                "message": f"可用库存合计 {total_cap:.3f} kg 小于批量下限 "
                           f"{prob.batch_lo:.3f} kg",
            }
        )
    for m in prob.mats:
        if m.locked_grid is not None and m.locked_grid > 0:
            diag["tight_materials"].append(
                {
                    "type": "locked",
                    "material_id": m.id,
                    "name": m.name,
                    "amount": round(m.locked_grid * prob.step, 6),
                }
            )

    if limiting:
        names = "、".join(x["oxide"] for x in limiting)
        diag["note"] = (
            f"限制目标的氧化物：{names}。可放宽其目标区间、补充高含量原料，"
            "或解除相关原料的禁用/库存限制；受限原料见 tight_materials。"
        )
    elif not diag["tight_materials"]:
        diag["note"] = (
            "连续松弛下目标可达，但步进整数格点上不可行；"
            "可减小 step 或放宽批量容差。"
        )
    return diag


def _range_lp(c, A, lb, ub, bounds):
    """区间线性约束 LP：lb <= Ax <= ub。返回 linprog 结果或 None。"""
    rows, rhs = [], []
    finite_lo = np.where(np.isfinite(lb))[0]
    finite_hi = np.where(np.isfinite(ub))[0]
    if len(finite_lo):
        rows.append(-A[finite_lo])
        rhs.append(-lb[finite_lo])
    if len(finite_hi):
        rows.append(A[finite_hi])
        rhs.append(ub[finite_hi])
    if rows:
        A_ub = np.vstack(rows)
        b_ub = np.concatenate(rhs)
    else:
        A_ub, b_ub = None, None
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res if res.success else None


def _structural_details(prob: Problem, diag: dict) -> None:
    locked_sum = sum(
        m.locked_grid * prob.step for m in prob.mats if m.locked_grid is not None
    )
    if locked_sum > prob.batch_hi + 1e-9:
        diag["structural_issues"].append(
            f"锁定用量合计 {locked_sum:.3f} kg 超过批量上限 {prob.batch_hi:.3f} kg"
        )
    cap = sum(m.khi * prob.step for m in prob.mats)
    if cap < prob.batch_lo - 1e-9:
        diag["structural_issues"].append(
            f"库存合计上限 {cap:.3f} kg 低于批量下限 {prob.batch_lo:.3f} kg"
        )
