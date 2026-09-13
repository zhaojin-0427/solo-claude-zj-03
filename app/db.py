"""SQLite 持久层：原料库、不可变配方版本与批次波动研究。

只有原料允许更新/删除；``recipe_versions``、``studies``、
``robust_versions`` 仅 INSERT 与 SELECT，通过对输入哈希建唯一索引
实现"同一版本重复读取保持一致"。
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

CREATE TABLE IF NOT EXISTS recipe_versions (
    id                 TEXT PRIMARY KEY,          -- input_hash
    input_hash         TEXT NOT NULL UNIQUE,
    items              TEXT NOT NULL,             -- 冻结时的投料 [{material_id, amount}]
    note               TEXT,
    material_snapshot  TEXT NOT NULL,             -- 冻结时原料分析全量快照
    constraints        TEXT NOT NULL,             -- 搜索/计算请求中的约束
    constants_version  TEXT NOT NULL,             -- 计算常量版本号
    constants_snapshot TEXT,                      -- 分子量/角色/容差完整快照（JSON）
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
                    constraints, constants_version, constants_snapshot, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                input_hash,
                json.dumps(items, sort_keys=True),
                note,
                json.dumps(material_snapshot, sort_keys=True),
                json.dumps(constraints, sort_keys=True),
                CONSTANTS_VERSION,
                json.dumps(constants, sort_keys=True),
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
