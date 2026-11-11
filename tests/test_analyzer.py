"""analyzer 核心逻辑测试。

运行（仓库根目录）：
    /tmp/dvenv/bin/python -I -m pytest tests -q
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.analyzer import (
    AnalyzerError,
    InputFile,
    STATUS_CONFLICT,
    STATUS_MISSING_COMMIT,
    STATUS_MISSING_FILES,
    STATUS_RECOVERABLE,
    analyze,
    normalize_data_path,
    parse_checkpoint,
    parse_commit,
)


# ---------- 构造工具 ----------

ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("size", pa.int64()),
    ("modificationTime", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
ADD_DV_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionVector", pa.struct([
        ("storageType", pa.string()),
        ("pathOrInlineDv", pa.string()),
    ])),
])


def commit_file(version: int, actions: list[dict]) -> InputFile:
    payload = "\n".join(json.dumps(a) for a in actions)
    return InputFile(filename=f"{version:020d}.json",
                     data=payload.encode("utf-8"))


def add(path: str, dv: dict | None = None) -> dict:
    action = {"add": {"path": path, "size": 1, "dataChange": True}}
    if dv is not None:
        action["add"]["deletionVector"] = dv
    return action


def remove(path: str, dv: dict | None = None) -> dict:
    action = {"remove": {"path": path, "dataChange": True}}
    if dv is not None:
        action["remove"]["deletionVector"] = dv
    return action


def checkpoint_file(version: int, active: list[str],
                    tombstones: list[str] | None = None) -> InputFile:
    n = max(len(active), len(tombstones or []), 1)
    add_rows = [{"path": p, "size": 1, "modificationTime": None,
                 "dataChange": False} for p in active] + [None] * (n - len(active))
    rm_rows = [{"path": p, "deletionTimestamp": None,
                "dataChange": False} for p in (tombstones or [])]
    rm_rows += [None] * (n - len(rm_rows))
    table = pa.table({
        "add": pa.array(add_rows, type=ADD_TYPE),
        "remove": pa.array(rm_rows, type=pa.struct([
            ("path", pa.string()),
            ("deletionTimestamp", pa.timestamp("us")),
            ("dataChange", pa.bool_()),
        ])),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return InputFile(filename=f"{version:020d}.checkpoint.parquet",
                     data=buf.getvalue())


def manifest(paths: list[str]) -> InputFile:
    return InputFile(filename="existing_files.txt",
                     data=("\n".join(paths) + "\n").encode("utf-8"))


def f(n: int) -> str:
    return f"data/part-{n:05d}.parquet"


def basic_commits() -> list[InputFile]:
    return [
        commit_file(0, [add(f(0)), add(f(1))]),
        commit_file(1, [add(f(2))]),
        commit_file(2, [add(f(3)), remove(f(1))]),
        commit_file(3, [add(f(4)), remove(f(0))]),
    ]


# ---------- 归一化与解析 ----------

def test_normalize_data_path():
    assert normalize_data_path("./data/a.parquet") == "data/a.parquet"
    assert normalize_data_path("/data/a.parquet") == "data/a.parquet"
    assert normalize_data_path("data/space%20file.parquet") == "data/space file.parquet"
    with pytest.raises(AnalyzerError):
        normalize_data_path(123)


def test_parse_commit_handles_ndjson_and_array():
    nd = parse_commit(0, b'{"add": {"path": "a.parquet"}}\n{"remove": {"path": "b.parquet"}}',
                      "x.json")
    assert nd.adds == [("a.parquet", {"path": "a.parquet"})]
    assert nd.removes == ["b.parquet"]

    arr = parse_commit(1, b'[{"add": {"path": "c.parquet"}}]', "y.json")
    assert arr.adds[0][0] == "c.parquet"


def test_parse_checkpoint_basic():
    ck = checkpoint_file(2, [f(0), f(2)], tombstones=[f(1)])
    parsed = parse_checkpoint(2, ck.data, ck.filename)
    assert parsed.adds == [f(0), f(2)]
    assert parsed.tombstones == [f(1)]
    assert parsed.dv_detected is False


def test_bad_parquet_rejected():
    with pytest.raises(AnalyzerError):
        parse_checkpoint(1, b"not parquet", "00000000000000000001.checkpoint.parquet")


# ---------- 完整分析：仅提交 ----------

def test_replay_commits_only_all_recoverable():
    report = analyze(basic_commits(), [manifest([f(i) for i in range(5)])], 0, 3)
    statuses = {v["version"]: v["status"] for v in report["versions"]}
    assert statuses == {0: STATUS_RECOVERABLE, 1: STATUS_RECOVERABLE,
                        2: STATUS_RECOVERABLE, 3: STATUS_RECOVERABLE}
    v3 = report["versions"][3]
    assert v3["referenced_files"] == sorted([f(2), f(3), f(4)])
    assert v3["base"] == "commit-0"
    assert v3["missing_files"] == []
    assert report["summary"][STATUS_RECOVERABLE] == 4


def test_remove_then_readd_same_path():
    commits = [
        commit_file(0, [add(f(0))]),
        commit_file(1, [remove(f(0))]),
        commit_file(2, [add(f(0))]),
    ]
    report = analyze(commits, [manifest([f(0)])], 2, 2)
    v2 = report["versions"][0]
    assert v2["status"] == STATUS_RECOVERABLE
    assert v2["referenced_files"] == [f(0)]


def test_missing_data_file_flagged_with_path():
    report = analyze(basic_commits(), [manifest([f(0), f(1), f(2), f(3)])], 3, 3)
    v3 = report["versions"][0]
    assert v3["status"] == STATUS_MISSING_FILES
    assert v3["missing_files"] == [f(4)]
    assert v3["missing_commits"] == []


def test_missing_commit_breaks_replay_and_lists_version():
    commits = [c for c in basic_commits() if not c.filename.startswith(f"{2:020d}")]
    report = analyze(commits, [manifest([f(i) for i in range(5)])], 0, 3)
    by_ver = {v["version"]: v for v in report["versions"]}
    assert by_ver[0]["status"] == STATUS_RECOVERABLE
    assert by_ver[1]["status"] == STATUS_RECOVERABLE
    assert by_ver[2]["missing_commits"] == [2]
    assert by_ver[2]["status"] == STATUS_MISSING_COMMIT
    # v3 因 v2 断档同样不可重建，但仍把缺失提交号明确列出
    assert by_ver[3]["status"] == STATUS_MISSING_COMMIT
    assert by_ver[3]["missing_commits"] == [2]


def test_no_base_without_commit_zero_or_checkpoint():
    commits = [commit_file(1, [add(f(0))])]
    report = analyze(commits, [manifest([f(0)])], 0, 1)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_MISSING_COMMIT
    assert v0["missing_commits"] == [0]
    assert v0["base"] is None


# ---------- 完整分析：检查点基线 ----------

def test_checkpoint_base_short_circuits_missing_early_commits():
    # 只有 v2 检查点 + v3 提交；v0/v1 提交缺失，但检查点不晚于目标版本
    logs = [checkpoint_file(2, [f(0), f(2), f(3)]),
            commit_file(3, [add(f(4)), remove(f(0))])]
    report = analyze(logs, [manifest([f(0), f(2), f(3), f(4)])], 2, 3)
    v2, v3 = report["versions"]
    assert v2["status"] == STATUS_RECOVERABLE
    assert v2["base"] == "checkpoint-2"
    assert v3["base"] == "checkpoint-2"
    assert v3["referenced_files"] == sorted([f(2), f(3), f(4)])


def test_checkpoint_tombstones_not_part_of_snapshot():
    ck = checkpoint_file(1, [f(0)], tombstones=[f(9)])
    report = analyze([ck], [manifest([f(0)])], 1, 1)
    v1 = report["versions"][0]
    assert v1["referenced_files"] == [f(0)]
    assert f(9) not in v1["referenced_files"]


def test_checkpoint_older_than_target_requires_continuous_commits():
    logs = [checkpoint_file(1, [f(0)]),
            commit_file(2, [add(f(1))])]
    # 检查点之后的 v2 提交存在，连续重放成功
    ok = analyze(logs, [manifest([f(0), f(1)])], 2, 2)
    assert ok["versions"][0]["status"] == STATUS_RECOVERABLE

    # v2 提交缺失 -> 检查点基线之后断档
    bad = analyze([checkpoint_file(1, [f(0)])], [manifest([f(0)])], 2, 2)
    v = bad["versions"][0]
    assert v["status"] == STATUS_MISSING_COMMIT
    assert v["missing_commits"] == [2]


# ---------- 删除向量拒绝 ----------

def test_deletion_vector_in_commit_rejected():
    dv = {"storageType": "u", "pathOrInlineDv": "abc"}
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([commit_file(0, [add(f(0), dv)])], [manifest([])], 0, 0)
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([commit_file(0, [remove(f(0), dv)])], [manifest([])], 0, 0)


def test_deletion_vector_protocol_feature_rejected():
    commits = [commit_file(0, [
        {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                      "readerFeatures": ["deletionVectors"],
                      "writerFeatures": ["deletionVectors"]}},
        add(f(0)),
    ])]
    with pytest.raises(AnalyzerError, match="deletionVectors"):
        analyze(commits, [manifest([f(0)])], 0, 0)


def test_deletion_vector_in_checkpoint_rejected():
    table = pa.table({"add": pa.array([
        {"path": f(0), "deletionVector": {"storageType": "u",
                                          "pathOrInlineDv": "x"}},
    ], type=ADD_DV_TYPE)})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    ck = InputFile(filename=f"{1:020d}.checkpoint.parquet", data=buf.getvalue())
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([ck], [manifest([f(0)])], 1, 1)


# ---------- 输入校验与杂项 ----------

def test_range_validation():
    with pytest.raises(AnalyzerError, match="起点不能大于终点"):
        analyze(basic_commits(), [], 3, 1)
    with pytest.raises(AnalyzerError, match="负数"):
        analyze(basic_commits(), [], -1, 1)


def test_empty_and_unrecognized_inputs():
    with pytest.raises(AnalyzerError, match="至少上传"):
        analyze([], [], 0, 1)
    weird = InputFile(filename="README.md", data=b"hi")
    with pytest.raises(AnalyzerError, match="没有可识别"):
        analyze([weird], [], 0, 0)


def test_ignored_multi_part_checkpoint_reported():
    logs = basic_commits() + [InputFile(
        filename=f"{1:020d}.checkpoint.00000000000000000001.00000000000000000002.parquet",
        data=b"x")]
    report = analyze(logs, [manifest([f(i) for i in range(5)])], 0, 0)
    ignored_names = [i["name"] for i in report["ignored_files"]]
    assert any("checkpoint" in n and n.endswith(".parquet") for n in ignored_names)


def test_manifest_comments_blank_lines_and_duplicates():
    m = InputFile(filename="m.txt",
                  data=b"# comment\n\n./data/part-00000.parquet\n/data/part-00000.parquet\n")
    report = analyze([commit_file(0, [add("data/part-00000.parquet")])], [m], 0, 0)
    assert report["existing_file_count"] == 1
    assert report["versions"][0]["status"] == STATUS_RECOVERABLE


def test_duplicate_upload_deduped():
    c = commit_file(0, [add(f(0))])
    report = analyze([c, InputFile(filename=c.filename, data=c.data)],
                     [manifest([f(0)])], 0, 0)
    assert c.filename in report["duplicate_files"]
    assert report["conflict_files"] == []
    assert report["versions"][0]["status"] == STATUS_RECOVERABLE


# ---------- 两批备份：同名副本去重与冲突 ----------

def clashing_commit(version: int, path: str = "data/clash.parquet") -> InputFile:
    """构造与 commit_file 同版本但内容不同的提交文件。"""
    payload = json.dumps({"add": {"path": path, "size": 999, "dataChange": True}})
    return InputFile(filename=f"{version:020d}.json", data=payload.encode("utf-8"))


def test_identical_copies_deduped_regardless_of_order():
    c0 = commit_file(0, [add(f(0))])
    c1 = commit_file(1, [add(f(1))])
    dup1 = InputFile(filename=c1.filename, data=c1.data)
    for logs in ([c0, c1, dup1], [c0, dup1, c1]):
        report = analyze(logs, [manifest([f(0), f(1)])], 1, 1)
        assert report["versions"][0]["status"] == STATUS_RECOVERABLE
        assert report["duplicate_files"] == [c1.filename]
        assert report["conflict_files"] == []
        assert report["available_commits"] == [0, 1]


def test_conflicting_commit_excluded_and_blocks_replay():
    logs = [commit_file(0, [add(f(0))]), commit_file(1, [add(f(1))]),
            clashing_commit(1)]
    report = analyze(logs, [manifest([f(0), f(1)])], 0, 1)
    # 冲突文件列出文件名与副本数，整组不参与重放
    assert report["conflict_files"] == [
        {"name": f"{1:020d}.json", "kind": "commit", "version": 1, "copies": 2}]
    assert report["available_commits"] == [0]
    v0, v1 = report["versions"]
    assert v0["status"] == STATUS_RECOVERABLE
    assert v1["status"] == STATUS_CONFLICT
    assert v1["conflict_commits"] == [1]
    assert v1["missing_commits"] == []
    # 因冲突不能重建的版本不展示文件核对结果
    assert v1["file_check_hidden"] is True
    assert v1["referenced_files"] == []
    assert v1["missing_files"] == []
    assert report["summary"][STATUS_CONFLICT] == 1


def test_conflict_detection_independent_of_upload_order():
    c0 = commit_file(0, [add(f(0))])
    c1 = commit_file(1, [add(f(1))])
    clash = clashing_commit(1)
    first = analyze([c0, c1, clash], [manifest([f(0), f(1)])], 1, 1)
    swapped = analyze([c0, clash, c1], [manifest([f(0), f(1)])], 1, 1)
    for report in (first, swapped):
        assert report["versions"][0]["status"] == STATUS_CONFLICT
        assert report["versions"][0]["conflict_commits"] == [1]


def test_three_copies_one_diverges_is_conflict():
    c1 = commit_file(1, [add(f(1))])
    same = InputFile(filename=c1.filename, data=c1.data)
    diff = InputFile(filename=c1.filename, data=c1.data + b"\n")
    logs = [commit_file(0, [add(f(0))]), c1, same, diff]
    report = analyze(logs, [manifest([f(0), f(1)])], 1, 1)
    assert report["versions"][0]["status"] == STATUS_CONFLICT
    assert report["conflict_files"][0]["copies"] == 3


def test_conflicting_checkpoint_excluded_falls_back_to_commit_zero():
    ck_a = checkpoint_file(1, [f(0)])
    ck_b = InputFile(filename=ck_a.filename,
                     data=checkpoint_file(1, [f(0), f(9)]).data)
    logs = [commit_file(0, [add(f(0))]), commit_file(1, [add(f(1))]), ck_a, ck_b]
    report = analyze(logs, [manifest([f(0), f(1)])], 1, 1)
    assert report["available_checkpoints"] == []
    assert [c["name"] for c in report["conflict_files"]] == [ck_a.filename]
    assert report["conflict_files"][0]["kind"] == "checkpoint"
    v1 = report["versions"][0]
    # 无冲突检查点不可用 -> 回退到 0 版提交起算，结论不受影响
    assert v1["base"] == "commit-0"
    assert v1["status"] == STATUS_RECOVERABLE
    assert v1["conflict_commits"] == []


def test_conflicted_commit_covered_by_later_checkpoint_does_not_block():
    # v3 提交冲突弃用，但 v4 检查点无冲突且已包含 v3 的改动
    logs = [checkpoint_file(4, [f(0), f(3), f(4)]),
            commit_file(3, [add(f(3))]), clashing_commit(3)]
    report = analyze(logs, [manifest([f(0), f(3), f(4)])], 4, 4)
    v4 = report["versions"][0]
    assert v4["base"] == "checkpoint-4"
    assert v4["status"] == STATUS_RECOVERABLE
    assert v4["conflict_commits"] == []
    assert v4["file_check_hidden"] is False


def test_missing_and_conflicting_commits_listed_separately():
    logs = [commit_file(0, [add(f(0))]),
            commit_file(1, [add(f(1))]), clashing_commit(1)]
    # v2 提交两批都未上传；v1 提交冲突弃用
    report = analyze(logs, [manifest([f(0), f(1)])], 2, 2)
    v2 = report["versions"][0]
    assert v2["status"] == STATUS_CONFLICT
    assert v2["conflict_commits"] == [1]
    assert v2["missing_commits"] == [2]
    assert v2["file_check_hidden"] is True


def test_conflicted_commit_zero_means_no_base():
    logs = [commit_file(0, [add(f(0))]), clashing_commit(0, "data/other.parquet"),
            commit_file(1, [add(f(1))])]
    report = analyze(logs, [manifest([f(0), f(1)])], 1, 1)
    v1 = report["versions"][0]
    assert v1["base"] is None
    assert v1["status"] == STATUS_CONFLICT
    assert v1["conflict_commits"] == [0]
    assert v1["missing_commits"] == []
    assert v1["file_check_hidden"] is True


def test_conflicting_corrupt_copy_never_parsed():
    # 冲突副本在解析前整组弃用：其中一份是坏字节也不影响结论
    c1 = commit_file(1, [add(f(1))])
    garbage = InputFile(filename=c1.filename, data=b"\x00\xff not json")
    report = analyze([commit_file(0, [add(f(0))]), c1, garbage],
                     [manifest([f(0), f(1)])], 1, 1)
    assert report["versions"][0]["status"] == STATUS_CONFLICT
    assert report["versions"][0]["conflict_commits"] == [1]


def test_no_manifest_means_every_referenced_file_is_missing():
    report = analyze(basic_commits(), [], 2, 2)
    v2 = report["versions"][0]
    assert v2["status"] == STATUS_MISSING_FILES
    assert v2["missing_files"] == sorted([f(0), f(2), f(3)])
    assert report["existing_file_count"] == 0
