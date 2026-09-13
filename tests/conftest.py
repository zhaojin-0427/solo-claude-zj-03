"""pytest 夹具：临时 SQLite + TestClient + 种子数据。"""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("GLAZE_DB", tmp.name)

    # 在导入 app.config 前注入临时库路径
    from app import config
    monkeypatch.setattr(config.settings, "db_path", tmp.name)

    from app import db
    db.init_db()

    from app.seed import SEED_MATERIALS
    from app.chemistry import validate_analysis
    from app.schemas import MaterialCreate
    for data in SEED_MATERIALS:
        payload = MaterialCreate(**data)
        normalized_oxides, normalized_loi = validate_analysis(
            payload.oxides, payload.loi, payload.analysis_tolerance
        )
        db.create_material(payload, normalized_oxides, normalized_loi)

    from app.main import create_app
    app = create_app()
    with TestClient(app) as c:
        c.base_targets = None  # type: ignore[attr-defined]
        yield c

    os.unlink(tmp.name)
