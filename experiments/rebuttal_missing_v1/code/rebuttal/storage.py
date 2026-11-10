"""Move managed data within one filesystem, retaining aliases for old paths.

The caller must hold resources/.resource_preparation.lock. Preflight all conflicts
before moving anything; individual renames and the journal allow safe resumption.
"""
import os
from pathlib import Path

from .common import atomic_json, file_hash, now, read_json


def exists(path):
    return path.exists() or path.is_symlink()


def same_location(one, two):
    return one.resolve() == two.resolve()


def _preflight(source, destination):
    if not exists(source) or same_location(source, destination):
        return
    if source.resolve() in destination.resolve().parents or destination.resolve() in source.resolve().parents:
        raise ValueError(f"迁移源/目标不能相互包含：{source} -> {destination}")
    parent = destination
    while not parent.exists():
        parent = parent.parent
    if source.lstat().st_dev != parent.stat().st_dev:
        raise ValueError(f"迁移需在同一文件系统内：{source} -> {destination}")
    if not exists(destination):
        return
    if source.is_symlink() or destination.is_symlink():
        if source.is_symlink() and destination.is_symlink() and os.readlink(source) == os.readlink(destination):
            return
        raise ValueError(f"迁移路径含不一致的软链接：{source} -> {destination}")
    if source.is_dir() and destination.is_dir():
        for child in source.iterdir():
            _preflight(child, destination / child.name)
        return
    if source.is_file() and destination.is_file() and file_hash(source) == file_hash(destination):
        return
    raise ValueError(f"目标存在不同内容，未覆盖：{destination}")


def _move(source, destination):
    if not exists(source) or same_location(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not exists(destination):
        source.rename(destination)
    elif source.is_dir() and not source.is_symlink():
        # Move provenance first so an interrupted merge remains identifiable.
        for child in sorted(source.iterdir(), key=lambda p: (p.name != "download_source.json", p.name)):
            _move(child, destination / child.name)
        source.rmdir()
    else:
        # Duplicate equality was established in preflight, while holding the lock.
        source.unlink()


def migrate_legacy_layout(resources, target, cache, *, repo, revision):
    resources, target, cache = Path(resources), Path(target), Path(cache)
    old_data = resources / "datasets/imagenet100"
    namespace = "datasets--" + repo.replace("/", "--")
    pairs = [(old_data, target), (resources / "hf_cache" / namespace, cache / namespace),
             (resources / "core_transfer_events.jsonl", cache.parent / "core_transfer_events.jsonl")]
    active = [(src, dst) for src, dst in pairs if exists(src) and not same_location(src, dst)]
    journal_path = resources / "imagenet100_storage_migration.json"
    prior = read_json(journal_path) if journal_path.exists() else {}
    owned = {(p["from"], p["to"]) for p in prior.get("paths", [])}
    if prior.get("repo") == repo and prior.get("revision") == revision:
        # Recover a crash after rename/rmdir but before creation of the old alias.
        active += [(src, dst) for src, dst in pairs if not exists(src) and exists(dst)
                   and (str(src), str(dst)) in owned]
    if not active:
        return
    # Only the fixed, managed ImageNet100 directory may be merged automatically.
    if any(src == old_data for src, _ in active):
        for root in (old_data, target):
            source_file = root / "download_source.json"
            if source_file.exists():
                source = read_json(source_file)
                if source.get("repo") != repo or source.get("revision") != revision:
                    raise ValueError(f"迁移目录来自其他数据版本：{root}")
            elif root.exists() and any(root.iterdir()):
                raise ValueError(f"迁移目录含未登记的数据，未修改：{root}")
    for source, destination in active:
        if source.is_symlink():
            raise ValueError(f"旧路径已链接到其他迁移位置，请核对数据/缓存配置：{source}")
        _preflight(source, destination)
    record = {"repo": repo, "revision": revision, "updated": now(), "state": "in_progress",
              "paths": [{"from": str(src), "to": str(dst)} for src, dst in pairs
                        if (src, dst) in active or (str(src), str(dst)) in owned]}
    atomic_json(journal_path, record)
    for source, destination in active:
        print(f"迁移并保留旧路径链接：{source} -> {destination}", flush=True)
        _move(source, destination)
        source.symlink_to(destination.resolve(), target_is_directory=destination.is_dir())
    record.update(state="complete", updated=now())
    atomic_json(journal_path, record)
