"""原料库 CRUD 与输入校验测试。"""
from __future__ import annotations


def test_list_and_get_seed(client):
    resp = client.get("/materials")
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()]
    assert "钾长石" in names and "石英" in names
    mid = resp.json()[0]["id"]
    one = client.get(f"/materials/{mid}")
    assert one.status_code == 200
    assert one.json()["id"] == mid


def test_get_missing_404(client):
    assert client.get("/materials/9999").status_code == 404


def test_create_and_delete(client):
    resp = client.post("/materials", json={
        "name": "试制药料",
        "oxides": {"CaO": 55.0, "MgO": 1.0},
        "loi": 43.5,
        "price": 3.2,
        "available": 42.0,
    })
    assert resp.status_code == 201, resp.text
    mid = resp.json()["id"]
    # 小幅分析误差被接受并归一化到 100
    assert abs(sum(resp.json()["oxides"].values()) + resp.json()["loi"] - 100.0) < 1e-6

    patch = client.patch(f"/materials/{mid}", json={"price": 4.0})
    assert patch.status_code == 200
    assert patch.json()["price"] == 4.0

    assert client.delete(f"/materials/{mid}").status_code == 204
    assert client.get(f"/materials/{mid}").status_code == 404


def test_negative_component_rejected(client):
    resp = client.post("/materials", json={
        "name": "坏料",
        "oxides": {"CaO": -5.0, "SiO2": 100.0},
        "loi": 0.0,
        "price": 1.0,
        "available": 10.0,
    })
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "negative_component"


def test_unknown_oxide_rejected(client):
    resp = client.post("/materials", json={
        "name": "外星料",
        "oxides": {"UnOb": 100.0},
        "loi": 0.0,
        "price": 1.0,
        "available": 10.0,
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_oxide"


def test_analysis_sum_out_of_tolerance(client):
    resp = client.post("/materials", json={
        "name": "合计错",
        "oxides": {"SiO2": 80.0},
        "loi": 10.0,  # 合计 90，超出默认 ±2
        "price": 1.0,
        "available": 10.0,
    })
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "analysis_sum_error"
    assert body["details"]["total"] == 90.0


def test_analysis_sum_custom_tolerance(client):
    resp = client.post("/materials", json={
        "name": "宽容差料",
        "oxides": {"SiO2": 80.0},
        "loi": 10.0,
        "price": 1.0,
        "available": 10.0,
        "analysis_tolerance": 11.0,
    })
    assert resp.status_code == 201


def test_negative_price_rejected(client):
    resp = client.post("/materials", json={
        "name": "负价料",
        "oxides": {"SiO2": 100.0},
        "loi": 0.0,
        "price": -1.0,
        "available": 10.0,
    })
    assert resp.status_code == 422
