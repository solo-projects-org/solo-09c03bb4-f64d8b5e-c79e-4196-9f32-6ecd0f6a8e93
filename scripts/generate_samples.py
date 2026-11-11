"""生成两套本地样例 Delta 表，供网页一键加载或手动上传。

样例只构造 _delta_log 中的 JSON 提交、经典单文件 Parquet 检查点
和现存数据文件路径清单，不依赖真实 Spark / Delta Lake 运行时。

- healthy_table：单批留存的健康表；
- incident_table：对象存储故障后的两批日志备份（batch_a / batch_b），
  内含字节一致的重叠文件（应去重）与内容不一致的同名文件（应判冲突、
  整组弃用），可在网页上分两批一并上传。

运行：python -I scripts/generate_samples.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# ---- Delta 检查点列结构（仅保留重放所需字段，无 deletionVector 列） ----
ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("size", pa.int64()),
    ("modificationTime", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
REMOVE_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionTimestamp", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
PROTOCOL_TYPE = pa.struct([
    ("minReaderVersion", pa.int32()),
    ("minWriterVersion", pa.int32()),
    ("readerFeatures", pa.list_(pa.string())),
    ("writerFeatures", pa.list_(pa.string())),
])


def commit_name(version: int) -> str:
    return f"{version:020d}.json"


def checkpoint_name(version: int) -> str:
    return f"{version:020d}.checkpoint.parquet"


def write_commit(log_dir: Path, version: int, actions: list[dict]) -> None:
    lines = [json.dumps(action, ensure_ascii=False) for action in actions]
    (log_dir / commit_name(version)).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def add_action(path: str) -> dict:
    return {"add": {"path": path, "partitionValues": {}, "size": 1024,
                    "modificationTime": 1700000000000, "dataChange": True}}


def remove_action(path: str) -> dict:
    return {"remove": {"path": path, "deletionTimestamp": 1700000100000,
                       "dataChange": True}}


PROTOCOL_ACTION = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}


def write_checkpoint(log_dir: Path, version: int,
                     active_paths: list[str], tombstone_paths: list[str]) -> None:
    """写经典单文件 Parquet 检查点。"""
    n_add, n_rm = len(active_paths), len(tombstone_paths)
    n_rows = max(n_add, n_rm, 1)  # protocol 至少占一行

    add_rows = [
        {"path": p, "size": 1024,
         "modificationTime": 1700000000000, "dataChange": False}
        for p in active_paths
    ] + [None] * (n_rows - n_add)
    remove_rows = [
        {"path": p, "deletionTimestamp": 1700000100000, "dataChange": False}
        for p in tombstone_paths
    ] + [None] * (n_rows - n_rm)
    protocol_rows = [
        {"minReaderVersion": 1, "minWriterVersion": 2,
         "readerFeatures": [], "writerFeatures": []}
    ] + [None] * (n_rows - 1)

    table = pa.table({
        "add": pa.array(add_rows, type=ADD_TYPE),
        "remove": pa.array(remove_rows, type=REMOVE_TYPE),
        "protocol": pa.array(protocol_rows, type=PROTOCOL_TYPE),
    })
    pq.write_table(table, log_dir / checkpoint_name(version))


def write_manifest(table_dir: Path, existing: list[str]) -> None:
    lines = ["# 现存数据文件表内相对路径清单（# 开头为注释，空行忽略）"]
    for i, p in enumerate(existing):
        # 故意混入 ./ 前缀与空行，验证服务端归一化逻辑
        lines.append(("./" + p) if i % 2 == 0 else p)
        if i == 1:
            lines.append("")
    (table_dir / "existing_files.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    # 同步建占位文件，让 samples 目录看起来像真实表布局
    for p in existing:
        target = table_dir / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder parquet bytes for sample")


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def build_healthy() -> None:
    """健康表：v0-v4 全部可重建，含 v2 检查点，所有数据文件均在。"""
    table_dir = SAMPLES / "healthy_table"
    reset_dir(table_dir)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True)

    f = lambda n: f"data/part-{n:05d}-healthy.parquet"
    write_commit(log_dir, 0, [PROTOCOL_ACTION, add_action(f(0)), add_action(f(1))])
    write_commit(log_dir, 1, [add_action(f(2))])
    write_commit(log_dir, 2, [add_action(f(3)), remove_action(f(1))])
    # v2 检查点：当前活动文件 f0,f2,f3；f1 是墓碑
    write_checkpoint(log_dir, 2, [f(0), f(2), f(3)], [f(1)])
    write_commit(log_dir, 3, [add_action(f(4))])
    write_commit(log_dir, 4, [add_action(f(5)), remove_action(f(2))])

    (log_dir / "_last_checkpoint").write_text(json.dumps({
        "version": 2, "size": 0}), encoding="utf-8")
    write_manifest(table_dir, [f(i) for i in range(6)])


def build_incident() -> None:
    """误删事故表：对象存储故障后抢救出的两批日志备份。

    真实历史 v0–v6：v0 建表（a0,a1）→ v1 加 a2 → v2 加 a3 删 a1（写检查点）
    → v3 加 a4 → v4 加 a5（写检查点）→ v5 加 a6 删 a2 → v6 加 a7。

    - 批次 A（主存储事故前的备份，覆盖 v0–v4）；
    - 批次 B（副本存储的备份，覆盖 v2–v5），与 A 有重叠：
      v2 / v4 提交与 v4 检查点字节一致（应去重）；
      v3 提交与 v2 检查点内容不一致（应判冲突、整组弃用）；
    - v6 提交两批都没有（未上传）；
    - 现存清单缺 a4（事故中被删），a1 逻辑删除但物理仍在。
    """
    table_dir = SAMPLES / "incident_table"
    reset_dir(table_dir)
    log_a = table_dir / "batch_a" / "_delta_log"
    log_b = table_dir / "batch_b" / "_delta_log"
    log_a.mkdir(parents=True)
    log_b.mkdir(parents=True)

    a = lambda n: f"data/part-{n:05d}-incident.parquet"

    # ---- 批次 A：主存储事故前的备份（v0–v4）----
    write_commit(log_a, 0, [PROTOCOL_ACTION, add_action(a(0)), add_action(a(1))])
    write_commit(log_a, 1, [add_action(a(2))])
    write_commit(log_a, 2, [add_action(a(3)), remove_action(a(1))])
    # v2 检查点：当前活动文件 a0,a2,a3；a1 是墓碑
    write_checkpoint(log_a, 2, [a(0), a(2), a(3)], [a(1)])
    write_commit(log_a, 3, [add_action(a(4))])
    write_commit(log_a, 4, [add_action(a(5))])
    # v4 检查点在 v3 之后生成，保留了 v3 引入的 a4
    write_checkpoint(log_a, 4, [a(0), a(2), a(3), a(4), a(5)], [a(1)])
    (log_a / "_last_checkpoint").write_text(json.dumps({
        "version": 4, "size": 0}), encoding="utf-8")

    # ---- 批次 B：副本存储的备份（v2–v5），部分重叠文件与 A 不一致 ----
    # 与 A 字节一致的重叠文件 → 应自动去重
    shutil.copyfile(log_a / commit_name(2), log_b / commit_name(2))
    shutil.copyfile(log_a / commit_name(4), log_b / commit_name(4))
    shutil.copyfile(log_a / checkpoint_name(4), log_b / checkpoint_name(4))
    # 与 A 内容不一致的同名文件 → 应判冲突、整组弃用：
    # v3 提交在副本侧是一次未完成的重写（同路径、不同元数据）
    write_commit(log_b, 3, [{"add": {
        "path": a(4), "partitionValues": {}, "size": 4096,
        "modificationTime": 1700000099999, "dataChange": True}}])
    # v2 检查点在副本侧是残缺版本（漏了 v2 才加入的 a3）
    write_checkpoint(log_b, 2, [a(0), a(2)], [a(1)])
    # 仅批次 B 有的新提交
    write_commit(log_b, 5, [add_action(a(6)), remove_action(a(2))])
    # 一个分片检查点残片：本工具明确不支持，应在页面中列为“已忽略”
    (log_b / f"{1:020d}.checkpoint.00000000000000000001.00000000000000000002.parquet")\
        .write_bytes(b"not-a-real-multi-part-checkpoint")
    (log_b / "_last_checkpoint").write_text(json.dumps({
        "version": 4, "size": 0}), encoding="utf-8")

    # v6 提交（add a7）两批备份都没有 → 逐版报告列为“未上传”
    # 现存清单：a4 已随误删消失；a1 虽已逻辑删除但物理文件仍在
    write_manifest(table_dir, [a(0), a(1), a(2), a(3), a(5), a(6)])


def main() -> None:
    build_healthy()
    build_incident()
    print(f"样例已生成于 {SAMPLES}")
    for table in sorted(SAMPLES.iterdir()):
        if not table.is_dir():
            continue
        print(f"\n[{table.name}]")
        for log_dir in sorted(table.rglob("_delta_log")):
            print(f"  {log_dir.relative_to(table)}/")
            for path in sorted(log_dir.iterdir()):
                print(f"    - {path.name}")


if __name__ == "__main__":
    main()
