"""Flask 路由层测试（含本地样例加载）。"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from app.web import app


def _commit_bytes(actions):
    return ("\n".join(json.dumps(a) for a in actions)).encode("utf-8")


def test_index_renders():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Delta Lake 版本可恢复性分析".encode() in resp.data


def test_upload_analyze_flow():
    client = app.test_client()
    data = {
        "from_version": "0",
        "to_version": "1",
        "log_files": [
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/a.parquet"}}])),
             "00000000000000000000.json"),
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/b.parquet"}}])),
             "00000000000000000001.json"),
        ],
        "manifest_files": [
            (io.BytesIO(b"data/a.parquet\ndata/b.parquet\n"), "existing.txt"),
        ],
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "可重建 · 文件齐全".encode() in resp.data


def test_upload_two_batches_with_conflict():
    """批次 A + 批次 B 一并提交：同名不同内容的提交判冲突，与顺序无关。"""
    client = app.test_client()
    v1_a = _commit_bytes([{"add": {"path": "data/b.parquet"}}])
    v1_b = _commit_bytes([{"add": {"path": "data/clash.parquet"}}])

    def post(batch_a_first: bool):
        batches = {
            "log_files": [
                (io.BytesIO(_commit_bytes([{"add": {"path": "data/a.parquet"}}])),
                 "00000000000000000000.json"),
                (io.BytesIO(v1_a), "00000000000000000001.json"),
            ],
            "log_files_b": [
                (io.BytesIO(v1_b), "00000000000000000001.json"),
            ],
        }
        if not batch_a_first:
            batches["log_files"], batches["log_files_b"] = (
                batches["log_files_b"], batches["log_files"])
        data = {
            "from_version": "0", "to_version": "1",
            "manifest_files": [
                (io.BytesIO(b"data/a.parquet\ndata/b.parquet\n"), "existing.txt")],
            **batches,
        }
        return client.post("/analyze", data=data,
                           content_type="multipart/form-data")

    for order in (True, False):
        resp = post(order)
        assert resp.status_code == 200
        # 冲突文件的文件名出现在报告里，v1 结论为冲突且不核对文件
        assert b'"conflict_files"' in resp.data
        assert b"00000000000000000001.json" in resp.data
        assert b'"conflict_commits": [1]' in resp.data
        assert b'"file_check_hidden": true' in resp.data


def test_upload_identical_copies_across_batches_deduped():
    client = app.test_client()
    v0 = _commit_bytes([{"add": {"path": "data/a.parquet"}}])
    data = {
        "from_version": "0", "to_version": "0",
        "log_files": [(io.BytesIO(v0), "00000000000000000000.json")],
        "log_files_b": [(io.BytesIO(v0), "00000000000000000000.json")],
        "manifest_files": [(io.BytesIO(b"data/a.parquet\n"), "existing.txt")],
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b'"conflict_files": []' in resp.data
    assert b"00000000000000000000.json" in resp.data  # 出现在去重清单中


def test_upload_missing_file_reports_400():
    client = app.test_client()
    resp = client.post("/analyze", data={
        "from_version": "0", "to_version": "1",
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "至少上传".encode() in resp.data


def test_sample_api_healthy():
    client = app.test_client()
    resp = client.post("/api/analyze-sample/healthy_table")
    assert resp.status_code == 200, resp.data
    report = resp.get_json()
    assert report["range"] == {"from": 0, "to": 4}
    assert report["summary"]["recoverable"] == 5
    assert report["conflict_files"] == []


def test_sample_api_incident_two_batches():
    """事故样例：两批备份一并加载，冲突 / 去重 / 缺提交 / 缺文件都出现。"""
    client = app.test_client()
    resp = client.post("/api/analyze-sample/incident_table")
    assert resp.status_code == 200, resp.data
    report = resp.get_json()
    assert report["range"] == {"from": 0, "to": 6}

    # 同名不同内容：v3 提交与 v2 检查点整组弃用
    conflicts = {c["name"]: c for c in report["conflict_files"]}
    assert set(conflicts) == {"00000000000000000003.json",
                              "00000000000000000002.checkpoint.parquet"}
    assert conflicts["00000000000000000003.json"]["kind"] == "commit"
    assert conflicts["00000000000000000002.checkpoint.parquet"]["kind"] == "checkpoint"
    # 字节一致的重叠文件：v2 / v4 提交与 v4 检查点已去重
    assert set(report["duplicate_files"]) == {
        "00000000000000000002.json",
        "00000000000000000004.json",
        "00000000000000000004.checkpoint.parquet",
    }
    assert report["available_commits"] == [0, 1, 2, 4, 5]
    assert report["available_checkpoints"] == [4]

    by_ver = {v["version"]: v for v in report["versions"]}
    # v0–v2：v2 检查点冲突弃用后从 0 版提交重放，文件齐全
    for ver in (0, 1, 2):
        assert by_ver[ver]["status"] == "recoverable"
        assert by_ver[ver]["base"] == "commit-0"
    # v3：提交冲突弃用且无可用检查点 -> 无法重建，不核对文件
    assert by_ver[3]["status"] == "conflict"
    assert by_ver[3]["conflict_commits"] == [3]
    assert by_ver[3]["missing_commits"] == []
    assert by_ver[3]["file_check_hidden"] is True
    assert by_ver[3]["referenced_files"] == []
    # v4–v5：以无冲突的 checkpoint-4 为基线，缺数据文件 part-00004
    for ver in (4, 5):
        assert by_ver[ver]["status"] == "missing_files"
        assert by_ver[ver]["base"] == "checkpoint-4"
        assert by_ver[ver]["missing_files"] == ["data/part-00004-incident.parquet"]
    # v6：提交两批都未上传 -> 与冲突分开列为“未上传”，仍照常核对文件
    assert by_ver[6]["status"] == "missing_commit"
    assert by_ver[6]["missing_commits"] == [6]
    assert by_ver[6]["conflict_commits"] == []
    assert by_ver[6]["file_check_hidden"] is False

    assert report["summary"] == {"recoverable": 3, "missing_commit": 1,
                                 "missing_files": 2, "conflict": 1}


def test_unknown_sample_404():
    resp = app.test_client().post("/api/analyze-sample/nope")
    assert resp.status_code == 404


def test_sample_api_list():
    resp = app.test_client().get("/api/samples")
    assert resp.status_code == 200
    names = resp.get_json()
    assert "healthy_table" in names and "incident_table" in names
