"""应用配置与计算常量。

计算常量（氧化物分子量 / 助熔分类 / 容差）全部冻结在
``CONSTANTS_VERSION`` 中：配方版本快照会记录该版本号，
保证日后修改常量表后，旧版本的重算结果仍可追溯。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 氧化物目录：摩尔质量 (g/mol) 与釉式分类
#   flux       —— 助熔氧化物（RO / R2O），釉式中其摩尔数之和归一化为 1
#   amphoteric —— 中性氧化物（R2O3 等）
#   acid       —— 酸性氧化物（RO2）
# 数值采用常用原子量（C=12.011, O=15.999, Na=22.990, K=39.098,
# Ca=40.078, Mg=24.305, Ba=137.327, Zn=65.38, Al=26.982, Si=28.085,
# Fe=55.845, Ti=47.867, B=10.81, P=30.974, Zr=91.224, Li=6.94,
# Sr=87.62, Pb=207.2, Sn=118.71, Cr=51.9961, Sb=121.76, Mn=54.938）。
# ---------------------------------------------------------------------------


class OxideInfo(BaseModel):
    molwt: float
    role: str  # flux / amphoteric / acid


OXIDE_CATALOG: dict[str, OxideInfo] = {
    # 助熔氧化物
    "K2O":   OxideInfo(molwt=94.196,  role="flux"),
    "Na2O":  OxideInfo(molwt=61.979,  role="flux"),
    "Li2O":  OxideInfo(molwt=29.881,  role="flux"),
    "CaO":   OxideInfo(molwt=56.077,  role="flux"),
    "MgO":   OxideInfo(molwt=40.304,  role="flux"),
    "BaO":   OxideInfo(molwt=153.326, role="flux"),
    "SrO":   OxideInfo(molwt=103.619, role="flux"),
    "ZnO":   OxideInfo(molwt=81.379,  role="flux"),
    "PbO":   OxideInfo(molwt=223.199, role="flux"),
    "MnO":   OxideInfo(molwt=70.937,  role="flux"),
    # 中性氧化物
    "Al2O3": OxideInfo(molwt=101.961, role="amphoteric"),
    "Fe2O3": OxideInfo(molwt=159.687, role="amphoteric"),
    "B2O3":  OxideInfo(molwt=69.620,  role="amphoteric"),
    "Cr2O3": OxideInfo(molwt=151.991, role="amphoteric"),
    "Sb2O3": OxideInfo(molwt=291.518, role="amphoteric"),
    # 酸性氧化物
    "SiO2":  OxideInfo(molwt=60.084,  role="acid"),
    "TiO2":  OxideInfo(molwt=79.866,  role="acid"),
    "ZrO2":  OxideInfo(molwt=123.222, role="acid"),
    "SnO2":  OxideInfo(molwt=150.709, role="acid"),
    "P2O5":  OxideInfo(molwt=141.945, role="acid"),
}

# 常量集版本：分子量 / 分类 / 默认容差任何变化都应递增
CONSTANTS_VERSION = "seg-2026-09-v1"

# 原料分析（氧化物质量分数 + LOI）合计允许的绝对误差
ANALYSIS_TOLERANCE = 0.02  # 100 ± 2 g（以百分数质量计）

# 助熔总摩尔数的正下限，防止零助熔配方做釉式归一化时除零
MIN_FLUX_MOLES = 1e-7
# 助熔摩尔数对生料质量(kg)的相对下限：强制配方含实质助熔，
# 避免优化器用极微量助熔制造数值上非零但化学上无意义的釉式。
# 1e-3 mol/kg 约为纯钾长石助熔摩尔浓度（~1.8e-3）的一半。
MIN_FLUX_PER_KG = 1e-3

# 釉式偏差判定容差：小于该值的偏差视为达标（吸收浮点/MIP 间隙噪声）。
# 取 1e-4：远小于陶艺配方关心的釉式差异（通常 1e-2 量级）。
SEGER_TOLERANCE = 1e-4

# 优化器参数
MILP_TIME_LIMIT = 20.0      # 常规 milp 求解秒数
STAGE1_TIME_LIMIT = 3.0     # 越界计数阶段秒数（启发式兜底，不必证明到最优）
MIP_REL_GAP = 1e-6          # 相对间隙（釉式尺度 1e-6 足够区分排序）
DINKELBACH_ITERS = 8        # 加权偏差分式规划的最大迭代次数
MAX_ALTERNATIVES = 5        # 替代配方最多返回条数

# 批次波动研究参数
DEFAULT_N_RESAMPLES = 400      # 默认重采样次数
MIN_RESAMPLES = 20             # 重采样次数下限（再低统计意义不足）
MAX_RESAMPLES = 10000          # 重采样次数上限
DEFAULT_STUDY_SEED = 20260913  # 缺省随机种子（固定种子保证可复现）
ROBUST_MAX_ITERS = 40          # 稳健配方搜索的贪心迭代上限
TOP_VIOLATION_COMBOS = 5       # 结果中返回的常见越界组合条数

# 釉浆调制参数
DENSITY_CLOSURE_TOLERANCE_G_ML = 0.02  # 比重杯实测与理论比重的闭合容差（g/mL）
CORRECTION_MAX_GRID_COMBOS = 200_000   # 纠偏网格枚举组合数上限

# 烧成试片研究参数
FIRING_MAX_DEFECT_GRADE = 3        # 缺陷等级上限（0=无，3=严重）
FIRING_OUTLIER_THRESHOLD = 2.5     # 异常试片判定的学生化残差阈值
FIRING_WILSON_Z = 1.96             # 缺陷发生率 Wilson 区间 z 值（95%）
FIRING_MAX_GRID_POINTS = 100_000   # 配比搜索网格点数上限


class Settings(BaseModel):
    db_path: str = str(Path(__file__).resolve().parent.parent / "glaze.db")
    analysis_tolerance: float = ANALYSIS_TOLERANCE
    min_flux_moles: float = MIN_FLUX_MOLES
    milp_time_limit: float = MILP_TIME_LIMIT
    stage1_time_limit: float = STAGE1_TIME_LIMIT
    mip_rel_gap: float = MIP_REL_GAP
    dinkelbach_iters: int = DINKELBACH_ITERS
    max_alternatives: int = MAX_ALTERNATIVES


settings = Settings()


def constants_snapshot() -> dict:
    """完整常量快照：仅凭它即可还原分子量与氧化物角色。

    配方版本冻结时保存此结构（而不仅是版本号），这样即使日后
    目录扩充或数值修订，旧版本仍能凭自身重算 Seger 釉式。
    """
    return {
        "constants_version": CONSTANTS_VERSION,
        "analysis_tolerance": ANALYSIS_TOLERANCE,
        "min_flux_moles": MIN_FLUX_MOLES,
        "min_flux_per_kg": MIN_FLUX_PER_KG,
        "oxides": {
            name: {"molwt": info.molwt, "role": info.role}
            for name, info in OXIDE_CATALOG.items()
        },
    }
