"""直接计算接口测试：釉式、烧失、成本与错误处理。"""
from __future__ import annotations

import pytest

from app.config import OXIDE_CATALOG


def _id_by_name(client, name: str) -> int:
    for m in client.get("/materials").json():
        if m["name"] == name:
            return m["id"]
    raise AssertionError(name)


def test_compute_seger_and_loi(client):
    kf = _id_by_name(client, "钾长石")   # K2O 16.92 Al2O3 18.32 SiO2 64.76
    cc = _id_by_name(client, "方解石")   # CaO 56.03, LOI 43.97
    qz = _id_by_name(client, "石英")
    resp = client.post("/compute", json={"items": [
        {"material_id": kf, "amount": 50.0},
        {"material_id": cc, "amount": 20.0},
        {"material_id": qz, "amount": 30.0},
    ]})
    assert resp.status_code == 200, resp.text
    r = resp.json()

    # 烧后质量：50 + 20*(1-0.4397) + 30
    assert r["fired_mass"] == pytest.approx(50 + 20 * 0.5603 + 30, abs=1e-9)
    assert r["batch_mass"] == pytest.approx(100.0)
    # 烧失
    assert r["loss_on_ignition"] == pytest.approx(20 * 0.4397, abs=1e-9)
    assert r["loss_on_ignition_pct"] == pytest.approx(20 * 0.4397, abs=1e-6)

    # 手工复算摩尔数（kg / (g/mol) 单位下统一按百分数质量）
    nK = 50 * 0.1692 / OXIDE_CATALOG["K2O"].molwt
    nCa = 20 * 0.5603 / OXIDE_CATALOG["CaO"].molwt
    flux = nK + nCa
    nAl = 50 * 0.1832 / OXIDE_CATALOG["Al2O3"].molwt
    nSi = (50 * 0.6476 + 30 * 1.0) / OXIDE_CATALOG["SiO2"].molwt
    assert r["seger"]["K2O"] == pytest.approx(nK / flux, abs=1e-9)
    assert r["seger"]["CaO"] == pytest.approx(nCa / flux, abs=1e-9)
    assert r["seger"]["Al2O3"] == pytest.approx(nAl / flux, abs=1e-9)
    assert r["seger"]["SiO2"] == pytest.approx(nSi / flux, abs=1e-9)
    # 助熔归一化：K2O + CaO = 1
    assert r["seger"]["K2O"] + r["seger"]["CaO"] == pytest.approx(1.0, abs=1e-9)

    # 烧后氧化物质量比合计为 1
    assert sum(r["fired_oxide_mass_ratio"].values()) == pytest.approx(1.0, abs=1e-9)


def test_compute_cost_and_breakdown(client):
    kf = _id_by_name(client, "钾长石")
    qz = _id_by_name(client, "石英")
    mats = {m["id"]: m for m in client.get("/materials").json()}
    resp = client.post("/compute", json={"items": [
        {"material_id": kf, "amount": 10.0},
        {"material_id": qz, "amount": 5.0},
    ]})
    r = resp.json()
    expected_cost = 10 * mats[kf]["price"] + 5 * mats[qz]["price"]
    assert r["cost"] == pytest.approx(expected_cost, abs=1e-9)
    assert len(r["breakdown"]) == 2


def test_compute_negative_amount(client):
    resp = client.post("/compute", json={"items": [
        {"material_id": 1, "amount": -3.0},
    ]})
    assert resp.status_code == 422


def test_compute_unknown_material(client):
    resp = client.post("/compute", json={"items": [
        {"material_id": 9999, "amount": 10.0},
    ]})
    assert resp.status_code == 404


def test_compute_no_flux_rejected(client):
    # 纯石英+高岭土（高岭土无碱/碱土助熔）
    qz = _id_by_name(client, "石英")
    ka = _id_by_name(client, "高岭土")
    resp = client.post("/compute", json={"items": [
        {"material_id": qz, "amount": 50.0},
        {"material_id": ka, "amount": 50.0},
    ]})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_flux"


def test_compute_empty_batch_rejected(client):
    resp = client.post("/compute", json={"items": [
        {"material_id": 1, "amount": 0.0},
    ]})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_batch"
