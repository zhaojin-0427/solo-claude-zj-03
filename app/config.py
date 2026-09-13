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

# 优化器参数
MILP_TIME_LIMIT = 30.0      # 单次 milp 求解秒数
RESCALE_PASSES = 2          # 釉式分数约束定点重标定轮数
MAX_ALTERNATIVES = 5        # 替代配方最多返回条数


class Settings(BaseModel):
    db_path: str = str(Path(__file__).resolve().parent.parent / "glaze.db")
    analysis_tolerance: float = ANALYSIS_TOLERANCE
    min_flux_moles: float = MIN_FLUX_MOLES
    milp_time_limit: float = MILP_TIME_LIMIT
    rescale_passes: int = RESCALE_PASSES
    max_alternatives: int = MAX_ALTERNATIVES


settings = Settings()
