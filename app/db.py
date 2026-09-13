"""SQLite 持久层：原料库与不可变配方版本。

只有原料允许更新/删除；``recipe_versions`` 仅 INSERT 与 SELECT，
通过对输入哈希建唯一索引实现"同一版本重复读取保持一致"。
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
