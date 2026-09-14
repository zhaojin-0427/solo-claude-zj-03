"""SQLite 持久层：原料库、不可变配方版本、批次波动研究与釉浆调制批次。

只有原料允许更新/删除；``recipe_versions``、``studies``、
``robust_versions`` 仅 INSERT 与 SELECT，通过对输入哈希建唯一索引
实现"同一版本重复读取保持一致"。釉浆批次在 planned/mixing 期间
可追加台账，定稿（``slurry_freezes``）后全部记录冻结且幂等。
烧成试片研究与釉坯热膨胀适配研究按 草稿 -> 定稿只读 -> 复制新版
流转：草稿期可增删试片/曲线/配方/排除，定稿（``firing_freezes`` /
``expansion_freezes``）后全部记录冻结，后续测量只能写入复制出的
新版本草稿，不得改写已定稿结果。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .config import CONSTANTS_VERSION, constants_snapshot, settings
from .errors import NotFoundError
from .schemas import Material, MaterialCreate, MaterialUpdate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    oxides              TEXT NOT NULL,  -- JSON: 归一化后的氧化物百分数
    loi                 REAL NOT NULL,
    price               REAL NOT NULL,
    available           REAL NOT NULL,
    analysis_tolerance  REAL NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS moisture_measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id     INTEGER NOT NULL,
    lot             TEXT NOT NULL,          -- 适用批号
    sample_mass_g   REAL NOT NULL,          -- 取样（湿料）质量
    dried_mass_g    REAL NOT NULL,          -- 烘干后质量
    moisture_pct    REAL NOT NULL,          -- 湿基含水率（百分数）
    valid_from      TEXT NOT NULL,          -- 生效起始日（含，YYYY-MM-DD）
    valid_to        TEXT NOT NULL,          -- 生效截止日（含）
    note            TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (material_id) REFERENCES materials(id),
    UNIQUE (material_id, lot, sample_mass_g, dried_mass_g, valid_from, valid_to)
);

CREATE TABLE IF NOT EXISTS recipe_versions (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    items              TEXT NOT NULL,             -- 冻结时的投料 [{material_id, amount}]
    note               TEXT,
    material_snapshot  TEXT NOT NULL,             -- 冻结时原料分析全量快照
    constraints        TEXT NOT NULL,             -- 搜索/计算请求中的约束
    constants_version  TEXT NOT NULL,             -- 计算常量版本号
    constants_snapshot TEXT,                      -- 分子量/角色/容差完整快照（JSON）
    moisture_plan      TEXT,                      -- 冻结时的批号湿料称量方案（JSON）
    result             TEXT NOT NULL,             -- 完整计算结果 JSON
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS studies (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    version_id         TEXT NOT NULL,             -- 来源冻结配方版本
    source_items       TEXT NOT NULL,             -- 来源配方投料 [{material_id, amount}]
    batches            TEXT NOT NULL,             -- 归一化后的化验批次快照
    linked_groups      TEXT NOT NULL,             -- 同批联动抽样组
    targets            TEXT NOT NULL,             -- 釉式目标区间
    n_resamples        INTEGER NOT NULL,          -- 重采样次数
    seed               INTEGER NOT NULL,          -- 随机种子
    note               TEXT,
    constants_version  TEXT NOT NULL,
    constants_snapshot TEXT NOT NULL,             -- 分子量/角色完整快照（JSON）
    result             TEXT NOT NULL,             -- 重采样统计结果 JSON
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS robust_versions (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    study_id           TEXT NOT NULL,             -- 来源研究
    items              TEXT NOT NULL,             -- 选定投料 [{material_id, amount}]
    note               TEXT,
    search_constraints TEXT,                      -- 稳健搜索约束（原样存档）
    source_snapshot    TEXT NOT NULL,             -- 来源配方+化验数据+抽样规则+种子
    result             TEXT NOT NULL,             -- 选定方案的重采样统计 JSON
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS blend_experiments (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    mode               TEXT NOT NULL,             -- linear / ternary
    sources            TEXT NOT NULL,             -- 来源冻结版本 id 与份额快照
    layout             TEXT NOT NULL,             -- 试片布局 [{position, weights}]
    setup              TEXT NOT NULL,             -- 干料量/分度/最小称量/比例范围等常量
    material_snapshot  TEXT NOT NULL,             -- 合并原料的冻结分析快照
    constants_version  TEXT NOT NULL,
    constants_snapshot TEXT NOT NULL,             -- 分子量/角色完整快照（JSON）
    note               TEXT,
    result             TEXT NOT NULL,             -- 逐格投料/釉式/烧后质量/成本 JSON
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS blend_versions (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    experiment_id      TEXT NOT NULL,             -- 来源混合试验
    note               TEXT,
    search_constraints TEXT NOT NULL,             -- 母料搜索参数
    plan               TEXT NOT NULL,             -- 选定方案（母料拆分+称量顺序）
    source_snapshot    TEXT NOT NULL,             -- 来源配方+布局+常量完整快照
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS slurry_batches (
    id                        TEXT PRIMARY KEY,    -- slb_<uuid>
    version_id                TEXT NOT NULL,       -- 来源冻结配方版本
    status                    TEXT NOT NULL,       -- planned / mixing / finalized
    target_dry_mass_kg        REAL NOT NULL,
    solids_low                REAL NOT NULL,
    solids_high               REAL NOT NULL,
    density_low               REAL NOT NULL,
    density_high              REAL NOT NULL,
    powder_true_density_g_ml  REAL NOT NULL,       -- kg/L 与 g/mL 数值相同
    water_temp_c              REAL NOT NULL,
    water_density_g_ml        REAL NOT NULL,       -- 创建时按水温插值
    container_capacity_ml     REAL NOT NULL,
    additive_ratio            REAL NOT NULL,
    note                      TEXT,
    initial_plan              TEXT NOT NULL,       -- 初始称量（干料/水/添加剂）
    recipe_snapshot           TEXT NOT NULL,       -- 来源版本 id/投料/份额/原料名
    freeze_id                 TEXT,                -- 定稿后指向 slurry_freezes.id
    created_at                TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                TEXT NOT NULL DEFAULT (datetime('now')),
    finalized_at              TEXT
);

CREATE TABLE IF NOT EXISTS slurry_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT NOT NULL,          -- 来源调制批次
    seq         INTEGER NOT NULL,       -- 批次内严格递增的台账序号
    entry_type  TEXT NOT NULL,          -- addition/recycle/premix/adjustment/reading
    payload     TEXT NOT NULL,          -- 逐笔登记的原始内容
    state_after TEXT,                   -- 台账动作后的守恒状态；读数为闭合评估
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (batch_id, seq)
);

CREATE TABLE IF NOT EXISTS slurry_freezes (
    id          TEXT PRIMARY KEY,       -- slf_<hash>
    batch_id    TEXT NOT NULL UNIQUE,   -- 一批次只能定稿一次，重复定稿取旧记录
    note        TEXT,
    final_state TEXT NOT NULL,          -- 定稿时刻质量守恒状态与告警
    snapshot    TEXT NOT NULL,          -- 投料/读数/调整/计算常量完整快照
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS firing_studies (
    id              TEXT PRIMARY KEY,   -- fst_<uuid>
    experiment_id   TEXT NOT NULL,      -- 来源冻结混合试验
    kiln_run        TEXT NOT NULL,      -- 窑次标签
    status          TEXT NOT NULL,      -- draft / finalized
    version_no      INTEGER NOT NULL,   -- 谱系内版本号（复制补测递增）
    parent_id       TEXT,               -- 复制来源研究 id
    root_id         TEXT NOT NULL,      -- 谱系根研究 id
    body            TEXT NOT NULL,      -- 坯体
    firing_curve    TEXT NOT NULL,      -- JSON: [{time_min, temp_c}]
    atmosphere      TEXT,
    kiln_position   TEXT NOT NULL,      -- 窑位
    fired_on        TEXT,               -- 烧成日期
    note            TEXT,
    freeze_id       TEXT,               -- 定稿后指向 firing_freezes.id
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finalized_at    TEXT
);

CREATE TABLE IF NOT EXISTS firing_tiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        TEXT NOT NULL,      -- 所属研究
    position        TEXT NOT NULL,      -- 格位（来源试验布局）
    replicate_no    INTEGER NOT NULL,   -- 格内重复编号（从 1 起）
    l_star          REAL NOT NULL,
    a_star          REAL NOT NULL,
    b_star          REAL NOT NULL,
    gloss60         REAL NOT NULL,      -- 60° 光泽度（GU）
    thickness_mm    REAL NOT NULL,      -- 烧后厚度（mm）
    pinhole         INTEGER NOT NULL,   -- 针孔等级 0~3
    crawling        INTEGER NOT NULL,   -- 缩釉等级 0~3
    running         INTEGER NOT NULL,   -- 流釉等级 0~3
    note            TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (study_id, position, replicate_no)
);

CREATE TABLE IF NOT EXISTS firing_freezes (
    id          TEXT PRIMARY KEY,       -- fsf_<hash>
    study_id    TEXT NOT NULL UNIQUE,   -- 一研究只能定稿一次，重复定稿取旧记录
    note        TEXT,
    final_state TEXT NOT NULL,          -- 定稿时逐格统计汇总
    snapshot    TEXT NOT NULL,          -- 试验/窑次/试片/常量完整快照
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS firing_result_freezes (
    id                 TEXT PRIMARY KEY,  -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    study_id           TEXT NOT NULL,     -- 来源研究（须已定稿）
    note               TEXT,
    fit_options        TEXT NOT NULL,     -- 拟合选项（模型阶次/异常阈值）
    search_constraints TEXT NOT NULL,     -- 上限/目标窗口/步长/候选序号
    selected           TEXT NOT NULL,     -- 选定配比与全部指标预测
    source_snapshot    TEXT NOT NULL,     -- 来源试验+试片数据+拟合结果快照
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS expansion_studies (
    id                          TEXT PRIMARY KEY,  -- tex_<uuid>
    body_name                   TEXT NOT NULL,     -- 坯体型号
    status                      TEXT NOT NULL,     -- draft / finalized
    version_no                  INTEGER NOT NULL,  -- 谱系内版本号（复制补测递增）
    parent_id                   TEXT,              -- 复制来源研究 id
    root_id                     TEXT NOT NULL,     -- 谱系根研究 id
    glaze_thickness_mm          REAL NOT NULL,     -- 釉层厚度
    body_thickness_mm           REAL NOT NULL,     -- 坯体厚度
    glaze_elastic_modulus_gpa   REAL NOT NULL,     -- 釉弹性模量
    body_elastic_modulus_gpa    REAL NOT NULL,     -- 坯体弹性模量
    glaze_poisson_ratio         REAL NOT NULL,
    body_poisson_ratio          REAL NOT NULL,
    stress_release_temp_c       REAL NOT NULL,     -- 应力释放温度
    note                        TEXT,
    freeze_id                   TEXT,              -- 定稿后指向 expansion_freezes.id
    created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    finalized_at                TEXT
);

CREATE TABLE IF NOT EXISTS expansion_body_curves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        TEXT NOT NULL,      -- 所属研究
    replicate_no    INTEGER NOT NULL,   -- 重复测次编号（从 1 起）
    direction       TEXT NOT NULL,      -- heating / cooling
    points          TEXT NOT NULL,      -- JSON: [{temp_c, strain}]（无量纲应变，温度升序）
    n_points        INTEGER NOT NULL,
    note            TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (study_id, replicate_no, direction)
);

CREATE TABLE IF NOT EXISTS expansion_recipes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id         TEXT NOT NULL,     -- 所属研究
    recipe_index     INTEGER NOT NULL,  -- 研究内配方序号（1~20，删除后不重排）
    version_id       TEXT NOT NULL,     -- 来源冻结配方版本
    recipe_snapshot  TEXT NOT NULL,     -- 来源配方快照（投料/釉式）
    note             TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (study_id, recipe_index),
    UNIQUE (study_id, version_id)
);

CREATE TABLE IF NOT EXISTS expansion_glaze_curves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id       INTEGER NOT NULL,   -- 所属配方（expansion_recipes.id）
    replicate_no    INTEGER NOT NULL,   -- 重复测次编号（从 1 起）
    direction       TEXT NOT NULL,      -- heating / cooling
    points          TEXT NOT NULL,      -- JSON: [{temp_c, strain}]（无量纲应变，温度升序）
    n_points        INTEGER NOT NULL,
    note            TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (recipe_id, replicate_no, direction),
    FOREIGN KEY (recipe_id) REFERENCES expansion_recipes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS expansion_exclusions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        TEXT NOT NULL,      -- 所属研究
    scope           TEXT NOT NULL,      -- body / glaze
    recipe_index    INTEGER NOT NULL,   -- 釉条为配方序号；坯条固定 0
    replicate_no    INTEGER NOT NULL,
    direction       TEXT NOT NULL,
    reason          TEXT NOT NULL,      -- 排除原因（必填）
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (study_id, scope, recipe_index, replicate_no, direction)
);

CREATE TABLE IF NOT EXISTS expansion_freezes (
    id          TEXT PRIMARY KEY,       -- texf_<hash>
    study_id    TEXT NOT NULL UNIQUE,   -- 一研究只能定稿一次，重复定稿取旧记录
    note        TEXT,
    final_state TEXT NOT NULL,          -- 定稿时的全部分析结果（含计算参数）
    snapshot    TEXT NOT NULL,          -- 原始曲线/来源配方/排除/参数完整快照
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """为旧库补列并回填常量快照，使既有版本也可凭自身还原分子量。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(recipe_versions)")}
    if cols and "constants_snapshot" not in cols:
        conn.execute(
            "ALTER TABLE recipe_versions ADD COLUMN constants_snapshot TEXT"
        )
        snapshot = json.dumps(constants_snapshot(), sort_keys=True)
        conn.execute(
            "UPDATE recipe_versions SET constants_snapshot = ? "
            "WHERE constants_snapshot IS NULL",
            (snapshot,),
        )
    # 既有版本冻结时没有含水称量方案：回退为按干料称量，保持原结果
    if cols and "moisture_plan" not in cols:
        conn.execute(
            "ALTER TABLE recipe_versions ADD COLUMN moisture_plan TEXT"
        )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


# ---------------------------------------------------------------------------
# 原料
# ---------------------------------------------------------------------------

def _row_to_material(row: sqlite3.Row) -> Material:
    data = dict(row)
    data["oxides"] = json.loads(data["oxides"])
    return Material(**data)


def list_materials() -> list[Material]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM materials ORDER BY id").fetchall()
    return [_row_to_material(r) for r in rows]


def get_material(material_id: int) -> Material:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM materials WHERE id = ?", (material_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(f"原料不存在: id={material_id}", "material_not_found")
    return _row_to_material(row)


def get_materials_map(ids: list[int]) -> dict[int, Material]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM materials WHERE id IN ({placeholders})", ids
        ).fetchall()
    return {r["id"]: _row_to_material(r) for r in rows}


def create_material(
    payload: MaterialCreate,
    normalized_oxides: dict[str, float],
    normalized_loi: float,
) -> Material:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO materials (name, oxides, loi, price, available, analysis_tolerance)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                payload.name,
                json.dumps(normalized_oxides, sort_keys=True),
                normalized_loi,
                payload.price,
                payload.available,
                payload.analysis_tolerance,
            ),
        )
        material_id = cur.lastrowid
    return get_material(material_id)


def update_material(material_id: int, payload: MaterialUpdate) -> Material:
    existing = get_material(material_id)
    merged = existing.model_dump()
    for field, value in payload.model_dump(exclude_unset=True).items():
        merged[field] = value
    # 触发完整校验（Pydantic + 化学合计检查）
    from .chemistry import validate_analysis

    oxides, loi = validate_analysis(
        merged["oxides"], merged["loi"], merged["analysis_tolerance"]
    )
    with get_conn() as conn:
        conn.execute(
            """UPDATE materials
               SET name=?, oxides=?, loi=?, price=?, available=?,
                   analysis_tolerance=?, updated_at=datetime('now')
               WHERE id=?""",
            (
                merged["name"],
                json.dumps(oxides, sort_keys=True),
                loi,
                merged["price"],
                merged["available"],
                merged["analysis_tolerance"],
                material_id,
            ),
        )
    return get_material(material_id)


def delete_material(material_id: int) -> None:
    get_material(material_id)
    with get_conn() as conn:
        conn.execute("DELETE FROM materials WHERE id = ?", (material_id,))


# ---------------------------------------------------------------------------
# 原料含水测定（只追加、不可变）
# ---------------------------------------------------------------------------

def _moisture_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "material_id": row["material_id"],
        "lot": row["lot"],
        "sample_mass_g": row["sample_mass_g"],
        "dried_mass_g": row["dried_mass_g"],
        "moisture_fraction": row["moisture_pct"] / 100.0,
        "moisture_pct": row["moisture_pct"],
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
        "note": row["note"],
        "created_at": row["created_at"],
    }


def list_moisture_measurements(
    material_id: Optional[int] = None,
    lot: Optional[str] = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM moisture_measurements"
    clauses: list[str] = []
    params: list[Any] = []
    if material_id is not None:
        clauses.append("material_id = ?")
        params.append(material_id)
    if lot is not None:
        clauses.append("lot = ?")
        params.append(lot)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY material_id, valid_from, id"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_moisture_row_to_dict(r) for r in rows]


def find_overlapping_moisture(
    material_id: int, lot: str, valid_from: str, valid_to: str
) -> Optional[dict[str, Any]]:
    """返回同原料同批号生效区间重叠（闭区间相交）的既有测定。"""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM moisture_measurements
               WHERE material_id = ? AND lot = ?
                 AND valid_from <= ? AND valid_to >= ?
               ORDER BY id LIMIT 1""",
            (material_id, lot, valid_to, valid_from),
        ).fetchone()
    return _moisture_row_to_dict(row) if row is not None else None


def create_moisture_measurement(
    *,
    material_id: int,
    lot: str,
    sample_mass_g: float,
    dried_mass_g: float,
    moisture_pct: float,
    valid_from: str,
    valid_to: str,
    note: Optional[str],
) -> dict[str, Any]:
    """追加一条不可变含水测定；完全重复的记录幂等返回旧记录。"""
    with get_conn() as conn:
        existing = conn.execute(
            """SELECT * FROM moisture_measurements
               WHERE material_id = ? AND lot = ? AND sample_mass_g = ?
                 AND dried_mass_g = ? AND valid_from = ? AND valid_to = ?""",
            (material_id, lot, sample_mass_g, dried_mass_g, valid_from, valid_to),
        ).fetchone()
        if existing is not None:
            return _moisture_row_to_dict(existing)
        cur = conn.execute(
            """INSERT INTO moisture_measurements
                   (material_id, lot, sample_mass_g, dried_mass_g, moisture_pct,
                    valid_from, valid_to, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                material_id, lot, sample_mass_g, dried_mass_g, moisture_pct,
                valid_from, valid_to, note,
            ),
        )
        new_id = cur.lastrowid
        row = conn.execute(
            "SELECT * FROM moisture_measurements WHERE id = ?", (new_id,)
        ).fetchone()
    return _moisture_row_to_dict(row)


def get_effective_moisture(
    material_id: int, lot: str, as_of: str
) -> Optional[dict[str, Any]]:
    """取某原料某批号在基准日生效（闭区间包含该日）的测定；无则返回 None。"""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM moisture_measurements
               WHERE material_id = ? AND lot = ?
                 AND valid_from <= ? AND valid_to >= ?
               ORDER BY id DESC LIMIT 1""",
            (material_id, lot, as_of, as_of),
        ).fetchone()
    return _moisture_row_to_dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# 配方版本（不可变）
# ---------------------------------------------------------------------------

def save_version(
    input_hash: str,
    items: list[dict[str, Any]],
    note: Optional[str],
    material_snapshot: dict[str, Any],
    constraints: dict[str, Any],
    result: dict[str, Any],
    constants: Optional[dict[str, Any]] = None,
    moisture_plan: Optional[dict[str, Any]] = None,
) -> tuple[Any, bool]:
    """写入不可变版本；同 hash 已存在则直接返回旧记录（幂等）。

    返回 ``(record, created)``。
    """
    constants = constants or constants_snapshot()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM recipe_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        if existing is not None:
            return _version_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO recipe_versions
                   (id, input_hash, items, note, material_snapshot,
                    constraints, constants_version, constants_snapshot,
                    moisture_plan, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                json.dumps(items, sort_keys=True),
                note,
                json.dumps(material_snapshot, sort_keys=True),
                json.dumps(constraints, sort_keys=True),
                CONSTANTS_VERSION,
                json.dumps(constants, sort_keys=True),
                json.dumps(moisture_plan, sort_keys=True, ensure_ascii=False)
                if moisture_plan is not None
                else None,
                json.dumps(result, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT * FROM recipe_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
    return _version_row_to_dict(row), True


def get_version(version_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM recipe_versions WHERE id = ?", (version_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(f"配方版本不存在: {version_id}", "version_not_found")
    return _version_row_to_dict(row)


def list_versions(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM recipe_versions ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [_version_row_to_dict(r) for r in rows]


def _version_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("items", "material_snapshot", "constraints", "result"):
        data[key] = json.loads(data[key])
    raw = data.get("constants_snapshot")
    if raw:
        data["constants_snapshot"] = json.loads(raw)
    else:
        # 旧库（迁移前创建的版本）：用当前常量补齐，保证读取结构一致
        data["constants_snapshot"] = constants_snapshot()
    # 旧版本（含水功能上线前冻结）无称量方案：显式按干料，绝不暗用现值
    plan_raw = data.get("moisture_plan")
    data["moisture_plan"] = json.loads(plan_raw) if plan_raw else None
    return data


# ---------------------------------------------------------------------------
# 批次波动研究（不可变）
# ---------------------------------------------------------------------------

def save_study(
    *,
    input_hash: str,
    version_id: str,
    source_items: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    linked_groups: list[list[int]],
    targets: dict[str, Any],
    n_resamples: int,
    seed: int,
    note: Optional[str],
    constants: dict[str, Any],
    result: dict[str, Any],
) -> tuple[Any, bool]:
    """写入研究快照；同 hash 已存在则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM studies WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        if existing is not None:
            return _study_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO studies
                   (id, input_hash, version_id, source_items, batches,
                    linked_groups, targets, n_resamples, seed, note,
                    constants_version, constants_snapshot, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                version_id,
                json.dumps(source_items, sort_keys=True),
                json.dumps(batches, sort_keys=True),
                json.dumps(linked_groups, sort_keys=True),
                json.dumps(targets, sort_keys=True),
                n_resamples,
                seed,
                note,
                CONSTANTS_VERSION,
                json.dumps(constants, sort_keys=True),
                json.dumps(result, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT * FROM studies WHERE input_hash = ?", (input_hash,)
        ).fetchone()
    return _study_row_to_dict(row), True


def get_study(study_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM studies WHERE id = ?", (study_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(f"批次波动研究不存在: {study_id}", "study_not_found")
    return _study_row_to_dict(row)


def list_studies(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM studies ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [_study_row_to_dict(r) for r in rows]


def _study_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in (
        "source_items",
        "batches",
        "linked_groups",
        "targets",
        "constants_snapshot",
        "result",
    ):
        data[key] = json.loads(data[key])
    return data


# ---------------------------------------------------------------------------
# 稳健配方冻结（不可变）
# ---------------------------------------------------------------------------

def save_robust_version(
    *,
    input_hash: str,
    study_id: str,
    items: list[dict[str, Any]],
    note: Optional[str],
    search_constraints: Optional[dict[str, Any]],
    source_snapshot: dict[str, Any],
    result: dict[str, Any],
) -> tuple[Any, bool]:
    """写入稳健配方冻结记录；同 hash 已存在则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM robust_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        if existing is not None:
            return _robust_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO robust_versions
                   (id, input_hash, study_id, items, note,
                    search_constraints, source_snapshot, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                study_id,
                json.dumps(items, sort_keys=True),
                note,
                json.dumps(search_constraints, sort_keys=True)
                if search_constraints is not None
                else None,
                json.dumps(source_snapshot, sort_keys=True),
                json.dumps(result, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT * FROM robust_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
    return _robust_row_to_dict(row), True


def get_robust_version(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM robust_versions WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"稳健配方冻结不存在: {freeze_id}", "robust_version_not_found"
        )
    return _robust_row_to_dict(row)


def list_robust_versions(study_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM robust_versions WHERE study_id = ? "
            "ORDER BY created_at DESC, id LIMIT ?",
            (study_id, limit),
        ).fetchall()
    return [_robust_row_to_dict(r) for r in rows]


def _robust_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("items", "source_snapshot", "result"):
        data[key] = json.loads(data[key])
    if data.get("search_constraints"):
        data["search_constraints"] = json.loads(data["search_constraints"])
    return data


# ---------------------------------------------------------------------------
# 配方混合试验（不可变）
# ---------------------------------------------------------------------------

def save_blend_experiment(
    *,
    input_hash: str,
    mode: str,
    sources: list[dict[str, Any]],
    layout: list[dict[str, Any]],
    setup: dict[str, Any],
    material_snapshot: dict[str, Any],
    constants: dict[str, Any],
    note: Optional[str],
    result: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """写入混合试验快照；同 hash 已存在则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM blend_experiments WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        if existing is not None:
            return _blend_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO blend_experiments
                   (id, input_hash, mode, sources, layout, setup,
                    material_snapshot, constants_version, constants_snapshot,
                    note, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                mode,
                json.dumps(sources, sort_keys=True, ensure_ascii=False),
                json.dumps(layout, sort_keys=True, ensure_ascii=False),
                json.dumps(setup, sort_keys=True),
                json.dumps(material_snapshot, sort_keys=True, ensure_ascii=False),
                CONSTANTS_VERSION,
                json.dumps(constants, sort_keys=True),
                note,
                json.dumps(result, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM blend_experiments WHERE input_hash = ?", (input_hash,)
        ).fetchone()
    return _blend_row_to_dict(row), True


def get_blend_experiment(experiment_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM blend_experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"配方混合试验不存在: {experiment_id}", "blend_experiment_not_found"
        )
    return _blend_row_to_dict(row)


def list_blend_experiments(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM blend_experiments ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [_blend_row_to_dict(r) for r in rows]


def _blend_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in (
        "sources",
        "layout",
        "setup",
        "material_snapshot",
        "constants_snapshot",
        "result",
    ):
        data[key] = json.loads(data[key])
    return data


# ---------------------------------------------------------------------------
# 母料拆分方案冻结（不可变）
# ---------------------------------------------------------------------------

def save_blend_version(
    *,
    input_hash: str,
    experiment_id: str,
    note: Optional[str],
    search_constraints: dict[str, Any],
    plan: dict[str, Any],
    source_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """写入母料方案冻结记录；同 hash 已存在则返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM blend_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        if existing is not None:
            return _blend_version_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO blend_versions
                   (id, input_hash, experiment_id, note,
                    search_constraints, plan, source_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                experiment_id,
                note,
                json.dumps(search_constraints, sort_keys=True, ensure_ascii=False),
                json.dumps(plan, sort_keys=True, ensure_ascii=False),
                json.dumps(source_snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM blend_versions WHERE input_hash = ?", (input_hash,)
        ).fetchone()
    return _blend_version_row_to_dict(row), True


def get_blend_version(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM blend_versions WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"母料拆分冻结不存在: {freeze_id}", "blend_version_not_found"
        )
    return _blend_version_row_to_dict(row)


def list_blend_versions(experiment_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM blend_versions WHERE experiment_id = ? "
            "ORDER BY created_at DESC, id LIMIT ?",
            (experiment_id, limit),
        ).fetchall()
    return [_blend_version_row_to_dict(r) for r in rows]


def _blend_version_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("search_constraints", "plan", "source_snapshot"):
        data[key] = json.loads(data[key])
    return data


# ---------------------------------------------------------------------------
# 釉浆调制批次（计划/调制中可变；定稿后冻结）
# ---------------------------------------------------------------------------

def create_slurry_batch(
    *,
    batch_id: str,
    version_id: str,
    params: dict[str, Any],
    initial_plan: dict[str, Any],
    recipe_snapshot: dict[str, Any],
) -> dict[str, Any]:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO slurry_batches
                   (id, version_id, status, target_dry_mass_kg,
                    solids_low, solids_high, density_low, density_high,
                    powder_true_density_g_ml, water_temp_c, water_density_g_ml,
                    container_capacity_ml, additive_ratio, note,
                    initial_plan, recipe_snapshot)
               VALUES (?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                version_id,
                params["target_dry_mass_kg"],
                params["solids_low"],
                params["solids_high"],
                params["density_low"],
                params["density_high"],
                params["powder_true_density_kg_l"],
                params["water_temp_c"],
                params["water_density_g_ml"],
                params["container_capacity_ml"],
                params["additive_ratio"],
                params.get("note"),
                json.dumps(initial_plan, sort_keys=True, ensure_ascii=False),
                json.dumps(recipe_snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
    return get_slurry_batch(batch_id)


def get_slurry_batch(batch_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM slurry_batches WHERE id = ?", (batch_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"釉浆调制批次不存在: {batch_id}", "slurry_batch_not_found"
        )
    return _slurry_row_to_dict(row)


def list_slurry_batches(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM slurry_batches ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [_slurry_row_to_dict(r) for r in rows]


def update_slurry_status(
    batch_id: str, status: str, freeze_id: Optional[str] = None
) -> None:
    with get_conn() as conn:
        if status == "finalized":
            conn.execute(
                """UPDATE slurry_batches
                   SET status=?, freeze_id=?,
                       finalized_at=datetime('now'), updated_at=datetime('now')
                   WHERE id=?""",
                (status, freeze_id, batch_id),
            )
        else:
            conn.execute(
                "UPDATE slurry_batches SET status=?, updated_at=datetime('now') "
                "WHERE id=?",
                (status, batch_id),
            )


def append_slurry_entry(
    batch_id: str,
    entry_type: str,
    payload: dict[str, Any],
    state_after: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """追加一条台账（投料/回收浆/预混粉/调整/读数），序号严格递增。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM slurry_entries "
            "WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        seq = int(row["max_seq"]) + 1
        conn.execute(
            """INSERT INTO slurry_entries
                   (batch_id, seq, entry_type, payload, state_after)
               VALUES (?, ?, ?, ?, ?)""",
            (
                batch_id,
                seq,
                entry_type,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                json.dumps(state_after, sort_keys=True, ensure_ascii=False)
                if state_after is not None
                else None,
            ),
        )
        saved = conn.execute(
            "SELECT * FROM slurry_entries WHERE batch_id = ? AND seq = ?",
            (batch_id, seq),
        ).fetchone()
    return _slurry_entry_row_to_dict(saved)


def list_slurry_entries(batch_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM slurry_entries WHERE batch_id = ? ORDER BY seq",
            (batch_id,),
        ).fetchall()
    return [_slurry_entry_row_to_dict(r) for r in rows]


def save_slurry_freeze(
    *,
    freeze_id: str,
    batch_id: str,
    note: Optional[str],
    final_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """定稿冻结；同批次已有冻结记录则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM slurry_freezes WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if existing is not None:
            return _slurry_freeze_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO slurry_freezes (id, batch_id, note, final_state, snapshot)
               VALUES (?, ?, ?, ?, ?)""",
            (
                freeze_id,
                batch_id,
                note,
                json.dumps(final_state, sort_keys=True, ensure_ascii=False),
                json.dumps(snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM slurry_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    return _slurry_freeze_row_to_dict(row), True


def get_slurry_freeze(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM slurry_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"釉浆定稿冻结不存在: {freeze_id}", "slurry_freeze_not_found"
        )
    return _slurry_freeze_row_to_dict(row)


def find_slurry_freeze(batch_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM slurry_freezes WHERE batch_id = ?", (batch_id,)
        ).fetchone()
    return _slurry_freeze_row_to_dict(row) if row is not None else None


def _slurry_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["initial_plan"] = json.loads(data["initial_plan"])
    data["recipe_snapshot"] = json.loads(data["recipe_snapshot"])
    return data


def _slurry_entry_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["payload"] = json.loads(data["payload"])
    if data.get("state_after"):
        data["state_after"] = json.loads(data["state_after"])
    return data


def _slurry_freeze_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["final_state"] = json.loads(data["final_state"])
    data["snapshot"] = json.loads(data["snapshot"])
    return data


# ---------------------------------------------------------------------------
# 烧成试片研究（草稿可变；定稿后只读，补测须复制新版）
# ---------------------------------------------------------------------------

def create_firing_study(
    *,
    study_id: str,
    experiment_id: str,
    kiln_run: str,
    version_no: int,
    parent_id: Optional[str],
    root_id: str,
    body: str,
    firing_curve: list[dict[str, Any]],
    atmosphere: Optional[str],
    kiln_position: str,
    fired_on: Optional[str],
    note: Optional[str],
) -> dict[str, Any]:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO firing_studies
                   (id, experiment_id, kiln_run, status, version_no,
                    parent_id, root_id, body, firing_curve, atmosphere,
                    kiln_position, fired_on, note)
               VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                study_id,
                experiment_id,
                kiln_run,
                version_no,
                parent_id,
                root_id,
                body,
                json.dumps(firing_curve, sort_keys=True, ensure_ascii=False),
                atmosphere,
                kiln_position,
                fired_on,
                note,
            ),
        )
    return get_firing_study(study_id)


def get_firing_study(study_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM firing_studies WHERE id = ?", (study_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"烧成试片研究不存在: {study_id}", "firing_study_not_found"
        )
    return _firing_study_row_to_dict(row)


def list_firing_studies(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM firing_studies ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [_firing_study_row_to_dict(r) for r in rows]


def firing_lineage_max_version(root_id: str) -> int:
    """谱系（同一根研究及其全部复制版本）内的最大版本号。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) AS max_v FROM firing_studies "
            "WHERE root_id = ?",
            (root_id,),
        ).fetchone()
    return int(row["max_v"])


def update_firing_study_status(
    study_id: str, status: str, freeze_id: Optional[str] = None
) -> None:
    with get_conn() as conn:
        if status == "finalized":
            conn.execute(
                """UPDATE firing_studies
                   SET status=?, freeze_id=?,
                       finalized_at=datetime('now'), updated_at=datetime('now')
                   WHERE id=?""",
                (status, freeze_id, study_id),
            )
        else:
            conn.execute(
                "UPDATE firing_studies SET status=?, updated_at=datetime('now') "
                "WHERE id=?",
                (status, study_id),
            )


def _firing_study_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["firing_curve"] = json.loads(data["firing_curve"])
    return data


# ---------------------------------------------------------------------------
# 烧成试片（草稿期可增删；定稿后随研究只读）
# ---------------------------------------------------------------------------

def _firing_tile_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def add_firing_tile(
    study_id: str, tile: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """登记一片重复试片；同格位同重复编号已存在则幂等返回旧记录。

    重复编号相同但测量值不同视为冲突，抛出 :class:`GlazeError`。
    """
    from .errors import GlazeError

    with get_conn() as conn:
        existing = conn.execute(
            """SELECT * FROM firing_tiles
               WHERE study_id = ? AND position = ? AND replicate_no = ?""",
            (study_id, tile["position"], tile["replicate_no"]),
        ).fetchone()
        if existing is not None:
            old = _firing_tile_row_to_dict(existing)
            same = all(
                old[k] == tile[k]
                for k in (
                    "l_star", "a_star", "b_star", "gloss60", "thickness_mm",
                    "pinhole", "crawling", "running",
                )
            )
            if same:
                return old, False
            raise GlazeError(
                f"格位 {tile['position']} 的重复编号 {tile['replicate_no']} "
                "已存在且测量值不同，请改用新的重复编号",
                "duplicate_replicate",
                {
                    "position": tile["position"],
                    "replicate_no": tile["replicate_no"],
                    "existing_tile_id": old["id"],
                },
            )
        cur = conn.execute(
            """INSERT INTO firing_tiles
                   (study_id, position, replicate_no, l_star, a_star, b_star,
                    gloss60, thickness_mm, pinhole, crawling, running, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                study_id,
                tile["position"],
                tile["replicate_no"],
                tile["l_star"],
                tile["a_star"],
                tile["b_star"],
                tile["gloss60"],
                tile["thickness_mm"],
                tile["pinhole"],
                tile["crawling"],
                tile["running"],
                tile.get("note"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM firing_tiles WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return _firing_tile_row_to_dict(row), True


def list_firing_tiles(study_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM firing_tiles WHERE study_id = ? "
            "ORDER BY position, replicate_no, id",
            (study_id,),
        ).fetchall()
    return [_firing_tile_row_to_dict(r) for r in rows]


def delete_firing_tile(study_id: str, tile_id: int) -> None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM firing_tiles WHERE study_id = ? AND id = ?",
            (study_id, tile_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(
                f"试片不存在: study={study_id} tile={tile_id}",
                "firing_tile_not_found",
            )
        conn.execute(
            "DELETE FROM firing_tiles WHERE study_id = ? AND id = ?",
            (study_id, tile_id),
        )


# ---------------------------------------------------------------------------
# 烧成研究定稿快照（不可变）
# ---------------------------------------------------------------------------

def save_firing_freeze(
    *,
    freeze_id: str,
    study_id: str,
    note: Optional[str],
    final_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """定稿冻结；同研究已有冻结记录则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM firing_freezes WHERE study_id = ?", (study_id,)
        ).fetchone()
        if existing is not None:
            return _firing_freeze_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO firing_freezes (id, study_id, note, final_state, snapshot)
               VALUES (?, ?, ?, ?, ?)""",
            (
                freeze_id,
                study_id,
                note,
                json.dumps(final_state, sort_keys=True, ensure_ascii=False),
                json.dumps(snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM firing_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    return _firing_freeze_row_to_dict(row), True


def get_firing_freeze(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM firing_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"烧成研究定稿冻结不存在: {freeze_id}", "firing_freeze_not_found"
        )
    return _firing_freeze_row_to_dict(row)


def find_firing_freeze(study_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM firing_freezes WHERE study_id = ?", (study_id,)
        ).fetchone()
    return _firing_freeze_row_to_dict(row) if row is not None else None


def _firing_freeze_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["final_state"] = json.loads(data["final_state"])
    data["snapshot"] = json.loads(data["snapshot"])
    return data


# ---------------------------------------------------------------------------
# 烧成配比结果冻结（不可变）
# ---------------------------------------------------------------------------

def save_firing_result_freeze(
    *,
    input_hash: str,
    study_id: str,
    note: Optional[str],
    fit_options: dict[str, Any],
    search_constraints: dict[str, Any],
    selected: dict[str, Any],
    source_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """写入配比结果冻结记录；同 hash 已存在则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM firing_result_freezes WHERE input_hash = ?",
            (input_hash,),
        ).fetchone()
        if existing is not None:
            return _firing_result_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO firing_result_freezes
                   (id, input_hash, study_id, note, fit_options,
                    search_constraints, selected, source_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                study_id,
                note,
                json.dumps(fit_options, sort_keys=True, ensure_ascii=False),
                json.dumps(search_constraints, sort_keys=True, ensure_ascii=False),
                json.dumps(selected, sort_keys=True, ensure_ascii=False),
                json.dumps(source_snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM firing_result_freezes WHERE input_hash = ?",
            (input_hash,),
        ).fetchone()
    return _firing_result_row_to_dict(row), True


def get_firing_result_freeze(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM firing_result_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"烧成配比结果冻结不存在: {freeze_id}",
            "firing_result_freeze_not_found",
        )
    return _firing_result_row_to_dict(row)


def list_firing_result_freezes(
    study_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM firing_result_freezes WHERE study_id = ? "
            "ORDER BY created_at DESC, id LIMIT ?",
            (study_id, limit),
        ).fetchall()
    return [_firing_result_row_to_dict(r) for r in rows]


def _firing_result_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("fit_options", "search_constraints", "selected", "source_snapshot"):
        data[key] = json.loads(data[key])
    return data


# ---------------------------------------------------------------------------
# 釉坯热膨胀适配研究（草稿 -> 定稿只读 -> 复制新版补测）
# ---------------------------------------------------------------------------

def create_expansion_study(
    *,
    study_id: str,
    body_name: str,
    version_no: int,
    parent_id: Optional[str],
    root_id: str,
    params: dict[str, Any],
    note: Optional[str],
) -> dict[str, Any]:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO expansion_studies
                   (id, body_name, status, version_no, parent_id, root_id,
                    glaze_thickness_mm, body_thickness_mm,
                    glaze_elastic_modulus_gpa, body_elastic_modulus_gpa,
                    glaze_poisson_ratio, body_poisson_ratio,
                    stress_release_temp_c, note)
               VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                study_id,
                body_name,
                version_no,
                parent_id,
                root_id,
                params["glaze_thickness_mm"],
                params["body_thickness_mm"],
                params["glaze_elastic_modulus_gpa"],
                params["body_elastic_modulus_gpa"],
                params["glaze_poisson_ratio"],
                params["body_poisson_ratio"],
                params["stress_release_temp_c"],
                note,
            ),
        )
    return get_expansion_study(study_id)


def get_expansion_study(study_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expansion_studies WHERE id = ?", (study_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"釉坯热膨胀研究不存在: {study_id}", "expansion_study_not_found"
        )
    return dict(row)


def list_expansion_studies(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM expansion_studies ORDER BY created_at DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def expansion_lineage_max_version(root_id: str) -> int:
    """谱系（同一根研究及其全部复制版本）内的最大版本号。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) AS max_v FROM expansion_studies "
            "WHERE root_id = ?",
            (root_id,),
        ).fetchone()
    return int(row["max_v"])


def update_expansion_study_status(
    study_id: str, status: str, freeze_id: Optional[str] = None
) -> None:
    with get_conn() as conn:
        if status == "finalized":
            conn.execute(
                """UPDATE expansion_studies
                   SET status=?, freeze_id=?,
                       finalized_at=datetime('now'), updated_at=datetime('now')
                   WHERE id=?""",
                (status, freeze_id, study_id),
            )
        else:
            conn.execute(
                "UPDATE expansion_studies SET status=?, updated_at=datetime('now') "
                "WHERE id=?",
                (status, study_id),
            )


# ---------------------------------------------------------------------------
# 坯条膨胀曲线（草稿期可增删；定稿后随研究只读）
# ---------------------------------------------------------------------------

def _curve_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["points"] = json.loads(data["points"])
    return data


def add_expansion_body_curve(
    study_id: str, curve: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """登记一条坯条膨胀曲线；同测次同方向已存在则幂等返回旧记录。

    测次/方向相同但曲线点不同视为冲突，抛出 :class:`GlazeError`。
    """
    from .errors import GlazeError

    with get_conn() as conn:
        existing = conn.execute(
            """SELECT * FROM expansion_body_curves
               WHERE study_id = ? AND replicate_no = ? AND direction = ?""",
            (study_id, curve["replicate_no"], curve["direction"]),
        ).fetchone()
        if existing is not None:
            old = _curve_row_to_dict(existing)
            if old["points"] == curve["points"]:
                return old, False
            raise GlazeError(
                f"坯条测次 {curve['replicate_no']}（{curve['direction']}）已存在"
                "且曲线点不同，请改用新的测次编号",
                "duplicate_replicate",
                {
                    "replicate_no": curve["replicate_no"],
                    "direction": curve["direction"],
                    "existing_curve_id": old["id"],
                },
            )
        cur = conn.execute(
            """INSERT INTO expansion_body_curves
                   (study_id, replicate_no, direction, points, n_points, note)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                study_id,
                curve["replicate_no"],
                curve["direction"],
                json.dumps(curve["points"], sort_keys=True),
                len(curve["points"]),
                curve.get("note"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM expansion_body_curves WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return _curve_row_to_dict(row), True


def list_expansion_body_curves(study_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM expansion_body_curves WHERE study_id = ? "
            "ORDER BY direction, replicate_no, id",
            (study_id,),
        ).fetchall()
    return [_curve_row_to_dict(r) for r in rows]


def delete_expansion_body_curve(study_id: str, curve_id: int) -> None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM expansion_body_curves WHERE study_id = ? AND id = ?",
            (study_id, curve_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(
                f"坯条曲线不存在: study={study_id} curve={curve_id}",
                "expansion_curve_not_found",
            )
        conn.execute(
            "DELETE FROM expansion_body_curves WHERE study_id = ? AND id = ?",
            (study_id, curve_id),
        )


# ---------------------------------------------------------------------------
# 研究配方与釉条曲线（草稿期可增删；定稿后随研究只读）
# ---------------------------------------------------------------------------

def _expansion_recipe_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["recipe_snapshot"] = json.loads(data["recipe_snapshot"])
    return data


def next_expansion_recipe_index(study_id: str) -> int:
    """研究内最小的空闲配方序号（1 起；删除后的空位可复用）。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT recipe_index FROM expansion_recipes WHERE study_id = ?",
            (study_id,),
        ).fetchall()
    used = {int(r["recipe_index"]) for r in rows}
    idx = 1
    while idx in used:
        idx += 1
    return idx


def count_expansion_recipes(study_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM expansion_recipes WHERE study_id = ?",
            (study_id,),
        ).fetchone()
    return int(row["n"])


def add_expansion_recipe(
    study_id: str,
    *,
    recipe_index: int,
    version_id: str,
    recipe_snapshot: dict[str, Any],
    curves: list[dict[str, Any]],
    note: Optional[str],
) -> dict[str, Any]:
    """登记一份配方及其同批釉条曲线；同版本重复登记抛出 :class:`GlazeError`。"""
    from .errors import GlazeError

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id, recipe_index FROM expansion_recipes "
            "WHERE study_id = ? AND version_id = ?",
            (study_id, version_id),
        ).fetchone()
        if existing is not None:
            raise GlazeError(
                f"配方版本 {version_id} 已登记为第 {existing['recipe_index']} 份配方",
                "duplicate_recipe",
                {"version_id": version_id,
                 "recipe_index": existing["recipe_index"]},
            )
        cur = conn.execute(
            """INSERT INTO expansion_recipes
                   (study_id, recipe_index, version_id, recipe_snapshot, note)
               VALUES (?, ?, ?, ?, ?)""",
            (
                study_id,
                recipe_index,
                version_id,
                json.dumps(recipe_snapshot, sort_keys=True, ensure_ascii=False),
                note,
            ),
        )
        recipe_id = cur.lastrowid
        for curve in curves:
            conn.execute(
                """INSERT INTO expansion_glaze_curves
                       (recipe_id, replicate_no, direction, points, n_points, note)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    recipe_id,
                    curve["replicate_no"],
                    curve["direction"],
                    json.dumps(curve["points"], sort_keys=True),
                    len(curve["points"]),
                    curve.get("note"),
                ),
            )
        row = conn.execute(
            "SELECT * FROM expansion_recipes WHERE id = ?", (recipe_id,)
        ).fetchone()
    return _expansion_recipe_row_to_dict(row)


def list_expansion_recipes(study_id: str) -> list[dict[str, Any]]:
    """研究内全部配方（按序号），每份附带釉条曲线。"""
    with get_conn() as conn:
        recipes = conn.execute(
            "SELECT * FROM expansion_recipes WHERE study_id = ? "
            "ORDER BY recipe_index",
            (study_id,),
        ).fetchall()
        out = []
        for r in recipes:
            data = _expansion_recipe_row_to_dict(r)
            curves = conn.execute(
                "SELECT * FROM expansion_glaze_curves WHERE recipe_id = ? "
                "ORDER BY direction, replicate_no, id",
                (data["id"],),
            ).fetchall()
            data["curves"] = [_curve_row_to_dict(c) for c in curves]
            out.append(data)
    return out


def add_expansion_glaze_curve(
    recipe_id: int, curve: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """向既有配方补登一条釉条曲线；同测次同方向已存在则幂等返回旧记录。

    测次/方向相同但曲线点不同视为冲突，抛出 :class:`GlazeError`。
    """
    from .errors import GlazeError

    with get_conn() as conn:
        existing = conn.execute(
            """SELECT * FROM expansion_glaze_curves
               WHERE recipe_id = ? AND replicate_no = ? AND direction = ?""",
            (recipe_id, curve["replicate_no"], curve["direction"]),
        ).fetchone()
        if existing is not None:
            old = _curve_row_to_dict(existing)
            if old["points"] == curve["points"]:
                return old, False
            raise GlazeError(
                f"釉条测次 {curve['replicate_no']}（{curve['direction']}）已存在"
                "且曲线点不同，请改用新的测次编号",
                "duplicate_replicate",
                {
                    "replicate_no": curve["replicate_no"],
                    "direction": curve["direction"],
                    "existing_curve_id": old["id"],
                },
            )
        cur = conn.execute(
            """INSERT INTO expansion_glaze_curves
                   (recipe_id, replicate_no, direction, points, n_points, note)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                recipe_id,
                curve["replicate_no"],
                curve["direction"],
                json.dumps(curve["points"], sort_keys=True),
                len(curve["points"]),
                curve.get("note"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM expansion_glaze_curves WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return _curve_row_to_dict(row), True


def delete_expansion_glaze_curve(recipe_id: int, curve_id: int) -> None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM expansion_glaze_curves WHERE recipe_id = ? AND id = ?",
            (recipe_id, curve_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(
                f"釉条曲线不存在: recipe={recipe_id} curve={curve_id}",
                "expansion_curve_not_found",
            )
        conn.execute(
            "DELETE FROM expansion_glaze_curves WHERE recipe_id = ? AND id = ?",
            (recipe_id, curve_id),
        )


def get_expansion_recipe(study_id: str, recipe_index: int) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expansion_recipes WHERE study_id = ? AND recipe_index = ?",
            (study_id, recipe_index),
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"配方不存在: study={study_id} recipe_index={recipe_index}",
            "expansion_recipe_not_found",
        )
    return _expansion_recipe_row_to_dict(row)


def delete_expansion_recipe(study_id: str, recipe_index: int) -> None:
    recipe = get_expansion_recipe(study_id, recipe_index)  # 404
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM expansion_glaze_curves WHERE recipe_id = ?",
            (recipe["id"],),
        )
        conn.execute(
            "DELETE FROM expansion_recipes WHERE study_id = ? AND recipe_index = ?",
            (study_id, recipe_index),
        )
        # 指向该配方的排除记录一并清理
        conn.execute(
            "DELETE FROM expansion_exclusions "
            "WHERE study_id = ? AND scope = 'glaze' AND recipe_index = ?",
            (study_id, recipe_index),
        )


# ---------------------------------------------------------------------------
# 异常测次排除（草稿期可增删；定稿后随研究只读）
# ---------------------------------------------------------------------------

def add_expansion_exclusion(
    study_id: str, exclusion: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """登记一条测次排除；完全相同的排除幂等返回旧记录，同键不同原因视为冲突。"""
    from .errors import GlazeError

    with get_conn() as conn:
        existing = conn.execute(
            """SELECT * FROM expansion_exclusions
               WHERE study_id = ? AND scope = ? AND recipe_index = ?
                 AND replicate_no = ? AND direction = ?""",
            (
                study_id,
                exclusion["scope"],
                exclusion["recipe_index"],
                exclusion["replicate_no"],
                exclusion["direction"],
            ),
        ).fetchone()
        if existing is not None:
            old = dict(existing)
            if old["reason"] == exclusion["reason"]:
                return old, False
            raise GlazeError(
                "同一测次已登记排除且原因不同，请先删除原排除再重新登记",
                "duplicate_exclusion",
                {"existing_exclusion_id": old["id"]},
            )
        cur = conn.execute(
            """INSERT INTO expansion_exclusions
                   (study_id, scope, recipe_index, replicate_no, direction, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                study_id,
                exclusion["scope"],
                exclusion["recipe_index"],
                exclusion["replicate_no"],
                exclusion["direction"],
                exclusion["reason"],
            ),
        )
        row = conn.execute(
            "SELECT * FROM expansion_exclusions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def list_expansion_exclusions(study_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM expansion_exclusions WHERE study_id = ? "
            "ORDER BY scope, recipe_index, direction, replicate_no, id",
            (study_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_expansion_exclusion(study_id: str, exclusion_id: int) -> None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM expansion_exclusions WHERE study_id = ? AND id = ?",
            (study_id, exclusion_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(
                f"排除记录不存在: study={study_id} exclusion={exclusion_id}",
                "expansion_exclusion_not_found",
            )
        conn.execute(
            "DELETE FROM expansion_exclusions WHERE study_id = ? AND id = ?",
            (study_id, exclusion_id),
        )


# ---------------------------------------------------------------------------
# 热膨胀研究定稿快照（不可变）
# ---------------------------------------------------------------------------

def save_expansion_freeze(
    *,
    freeze_id: str,
    study_id: str,
    note: Optional[str],
    final_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """定稿冻结；同研究已有冻结记录则直接返回旧记录（幂等）。"""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM expansion_freezes WHERE study_id = ?", (study_id,)
        ).fetchone()
        if existing is not None:
            return _expansion_freeze_row_to_dict(existing), False
        conn.execute(
            """INSERT INTO expansion_freezes (id, study_id, note, final_state, snapshot)
               VALUES (?, ?, ?, ?, ?)""",
            (
                freeze_id,
                study_id,
                note,
                json.dumps(final_state, sort_keys=True, ensure_ascii=False),
                json.dumps(snapshot, sort_keys=True, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM expansion_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    return _expansion_freeze_row_to_dict(row), True


def get_expansion_freeze(freeze_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expansion_freezes WHERE id = ?", (freeze_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(
            f"热膨胀研究定稿冻结不存在: {freeze_id}", "expansion_freeze_not_found"
        )
    return _expansion_freeze_row_to_dict(row)


def find_expansion_freeze(study_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expansion_freezes WHERE study_id = ?", (study_id,)
        ).fetchone()
    return _expansion_freeze_row_to_dict(row) if row is not None else None


def _expansion_freeze_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["final_state"] = json.loads(data["final_state"])
    data["snapshot"] = json.loads(data["snapshot"])
    return data
