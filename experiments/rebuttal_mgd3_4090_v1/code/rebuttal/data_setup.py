"""Explicit preparation of the pinned ImageNet100 core experiment data.

No GPU imports, model downloads, environment installs, or scheduler calls.
"""
import argparse
import ast
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from .common import ROOT, atomic_json, digest, file_hash, now, read_json
from . import network
from .resources import (configured, has_vae, imagenet_aliases, metadata_names,
                        name_to_wnid, normalize_name, resolve_row_wnid)
from .transfers import DownloadError
from .image_formats import EXTENSIONS, inspect_image
from .storage import migrate_legacy_layout

REPO = "clane9/imagenet-100"
REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"
SHARDS = ["data/validation-00000-of-00001.parquet"] + [f"data/train-{i:05d}-of-00017.parquet" for i in range(17)]
TOTAL_BYTES = 8406986315
EXPECTED_ROWS = {"train": 126689, "val": 5000}
REQUIRED_FREE = 30 * 1024 ** 3


def split_of(filename):
    return "val" if "validation-" in filename else "train"


def manifest_path(resources):
    return Path(resources) / "imagenet100_download_manifest.json"


def dataset_cache(target):
    return Path(configured("REBUTTAL_DATA_CACHE") or Path(target) / ".hf_cache")


def require_space(cache, target):
    if min(shutil.disk_usage(cache).free, shutil.disk_usage(target).free) < REQUIRED_FREE:
        raise ValueError("数据准备需至少 30 GiB 可用空间（不含后续实验输出）；请核对空间和用户配额")


def validate_manifest(value, classes):
    if value.get("repo") != REPO or value.get("revision") != REVISION:
        raise ValueError("数据清单不是固定的 clane9/imagenet-100 版本")
    if value.get("manifest_hash") != digest({k: v for k, v in value.items() if k != "manifest_hash"}):
        raise ValueError("下载清单内容/hash 不一致")
    files = value["files"]
    if [f["filename"] for f in files] != SHARDS or sum(f["size"] for f in files) != TOTAL_BYTES:
        raise ValueError("下载清单的 18 个分片或总大小不符")
    for item in files:
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or not isinstance(item["size"], int) or item["size"] <= 0:
            raise ValueError("缺少有效分片大小/SHA-256")
    mapping = value["label_to_wnid"]
    if set(mapping) != {str(i) for i in range(100)} or set(mapping.values()) != set(classes) or len(set(classes)) != 100:
        raise ValueError("数据源标签与 class100.txt 的 100 个 WNID 不符")
    return value


def label_mapping(names, classes, source_classes=None):
    if isinstance(names, list):
        names = {str(i): name for i, name in enumerate(names)}
    if not isinstance(names, dict) or set(map(str, names)) != {str(i) for i in range(100)}:
        raise ValueError("源数据必须包含 100 个明确的 ClassLabel 标签")
    aliases = imagenet_aliases()
    if source_classes is not None:
        if set(source_classes) != set(classes):
            raise ValueError("源 classes.py 与 class100.txt 的 WNID 集合不符")
        for wnid, name in source_classes.items():
            aliases[normalize_name(name)].add(wnid)
    mapping = {str(i): name_to_wnid(name, aliases) for i, name in names.items()}
    if set(mapping.values()) != set(classes) or len(set(classes)) != 100:
        raise ValueError("无法从 ClassLabel 名称唯一验证全部 100 个 WNID；拒绝按 class100.txt 位置猜标签")
    return mapping


def source_class_dict(raw):
    # Read a literal mapping as data. Never execute repository Python code.
    tree = ast.parse(raw)
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if len(value) == 100 and all(isinstance(k, str) and re.fullmatch(r"n\d{8}", k)
                                         and isinstance(v, str) for k, v in value.items()):
                candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("源 classes.py 没有唯一的 100 类字典")
    return candidates[0]


def metadata_manifest(info, classes, names, source_classes=None):
    if info.get("sha") != REVISION:
        raise ValueError("仓库 API 返回了其他版本，拒绝下载")
    entries = {p["rfilename"]: p for p in info.get("siblings", [])}
    files = []
    for filename in SHARDS:
        lfs = entries.get(filename, {}).get("lfs", {})
        files.append({"filename": filename, "size": lfs.get("size"), "sha256": lfs.get("sha256")})
    value = {"repo": REPO, "revision": REVISION, "files": files,
             "label_to_wnid": label_mapping(names, classes, source_classes), "label_names": names,
             "expected_rows": EXPECTED_ROWS, "created": now(),
             "image_policy": "Preserve source bytes; publisher's images have short side 160 pixels (bicubic)."}
    value["manifest_hash"] = digest(value)
    return validate_manifest(value, classes)


def successful(result):
    return result["curl_exit"] == 0 and 200 <= result["http_status"] < 300 and not result.get("limit_exceeded")


def diagnose(resources, classes, cached=None):
    report = {"created": now(), "repo": REPO, "revision": REVISION, "proxy": network.proxy_probe(),
              "checks": [], "routes": []}
    manifest = cached
    for endpoint in network.endpoints():
        tls_modes = ["default"]
        for tls in tls_modes:
            print(f"检查元数据：{endpoint}，TLS={tls}", flush=True)
            api_url = f"{endpoint}/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
            body, result = network.fetch_small(api_url, tls=tls)
            result.update({"stage": "metadata", "endpoint": endpoint})
            report["checks"].append(result)
            if result["curl_exit"] == 35 and tls == "default":
                tls_modes.append("1.2")
                continue
            api_ok = False
            if successful(result):
                try:
                    info = json.loads(body)
                    if info.get("sha") != REVISION:
                        raise ValueError("API revision mismatch")
                    api_ok = True
                    if manifest is None:
                        card = info.get("cardData") or {}
                        candidates = card.get("dataset_info", {})
                        if not isinstance(candidates, list):
                            candidates = [candidates]
                        names = next((metadata_names(v) for v in candidates if metadata_names(v)), None)
                        if not names:
                            import yaml
                            raw, readme_result = network.fetch_small(f"{endpoint}/datasets/{REPO}/raw/{REVISION}/README.md", tls=tls)
                            readme_result.update({"stage": "class_metadata", "endpoint": endpoint})
                            report["checks"].append(readme_result)
                            if not successful(readme_result):
                                raise ValueError("不能读取固定版本的 ClassLabel 元数据")
                            front = yaml.safe_load(raw.decode().split("---", 2)[1])
                            candidates = front.get("dataset_info", {})
                            if not isinstance(candidates, list):
                                candidates = [candidates]
                            names = next((metadata_names(v) for v in candidates if metadata_names(v)), None)
                        try:
                            manifest = metadata_manifest(info, classes, names)
                        except ValueError:
                            raw, mapping_result = network.fetch_small(
                                f"{endpoint}/datasets/{REPO}/raw/{REVISION}/scripts/classes.py", tls=tls)
                            mapping_result.update({"stage": "wnid_metadata", "endpoint": endpoint})
                            report["checks"].append(mapping_result)
                            if not successful(mapping_result):
                                raise ValueError("不能读取固定版本的 WNID 元数据")
                            manifest = metadata_manifest(info, classes, names, source_class_dict(raw))
                        atomic_json(manifest_path(resources), manifest)
                except (ValueError, TypeError, KeyError, IndexError, SyntaxError) as error:
                    api_ok = False
                    result["validation_error"] = network.sanitize(error)
            # A cached manifest permits resumption even when the API is down.
            # Always probe the actual artifact/CDN; a working website is insufficient.
            print(f"检查真实分片和 CDN：{endpoint}，最多读取 64 KiB，TLS={tls}", flush=True)
            body, result = network.fetch_small(network.resolver(endpoint, REPO, REVISION, SHARDS[0]),
                                               tls=tls, byte_range="0-65535", limit=65536)
            result.update({"stage": "shard_range", "endpoint": endpoint})
            report["checks"].append(result)
            expected_range = f"bytes 0-65535/{manifest['files'][0]['size']}" if manifest else None
            range_ok = (successful(result) and result["http_status"] == 206 and len(body) == 65536
                        and body.startswith(b"PAR1") and (result["content_range"] == expected_range if manifest else
                                                         bool(re.fullmatch(r"bytes 0-65535/\d+", result["content_range"] or ""))))
            result["range_verified"] = range_ok
            if result["curl_exit"] == 35 and tls == "default":
                tls_modes.append("1.2")
            if range_ok and manifest is not None and (api_ok or cached is not None):
                report["routes"].append({"endpoint": endpoint, "tls": tls})
                break
    report["ready_to_download"] = bool(report["routes"] and manifest)
    if not report["ready_to_download"]:
        proxy = report["proxy"]
        if proxy.get("reachable") is False and proxy.get("error_type") == "PermissionError":
            report["next_action"] = "本进程连接代理被系统权限拒绝；请在有正常网络权限的集群终端执行同一准备命令。"
        elif proxy.get("reachable") is False:
            report["next_action"] = "配置的代理端口不可达；检查该代理服务/SSH 转发是否仍运行，并核对代理地址。脚本未修改代理。"
        elif manifest:
            report["next_action"] = "元数据已缓存，但真实分片/CDN 未通过；按 shard_range 的 redirect_hosts、effective_host 和 error 检查出口。"
        else:
            report["next_action"] = "元数据尚未齐备；检查 metadata/class_metadata/wnid_metadata 阶段的 error 或 validation_error。"
    atomic_json(Path(resources) / "core_network_diagnostics.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["ready_to_download"]:
        raise DownloadError("没有通过真实分片 Range 读取的下载路线；详见 resources/core_network_diagnostics.json。"
                            + report["next_action"] + " 未开始大文件下载。")
    return manifest, report["routes"]


def marker_paths(target, item):
    base = Path(target) / "download_records" / Path(item["filename"]).stem
    return base.with_suffix(".done.json"), base.with_suffix(".jsonl")


def check_records(target, item, manifest):
    """Full local image hash + decode check; never accepts mere marker existence."""
    marker, record = marker_paths(target, item)
    saved = read_json(marker)
    if (saved.get("revision") != REVISION or saved.get("shard_sha256") != item["sha256"]
            or saved.get("manifest_hash") != manifest["manifest_hash"] or file_hash(record) != saved["record_sha256"]):
        raise ValueError(f"分片提取记录不匹配：{item['filename']}")
    counts, outputs = Counter(), set()
    with record.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            wnid = manifest["label_to_wnid"].get(str(row["label"]))
            rel = Path(row["output"])
            if (row["row"] != index or wnid != row["wnid"] or rel.is_absolute() or ".." in rel.parts
                    or len(rel.parts) != 3 or rel.parts[:2] != (split_of(item["filename"]), wnid)
                    or not rel.name.startswith(Path(item["filename"]).stem + f"_{index:08d}.")
                    or rel.suffix.lower() not in EXTENSIONS or row["output"] in outputs):
                raise ValueError("提取记录含重复图片、错误路径或标签")
            path = Path(target) / rel
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"图片 SHA-256 不符：{rel}")
            decoded = inspect_image(path, context=f"shard={item['filename']}, row={index}, wnid={wnid}")
            if any(key in row and row[key] != decoded[key] for key in ("format", "mode", "size")):
                raise ValueError(f"图片格式记录不一致：{rel}")
            outputs.add(row["output"])
            counts[wnid] += 1
    if sum(counts.values()) != saved["rows"] or saved["written"] != saved["rows"]:
        raise ValueError("提取记录行数不一致")
    return counts, outputs


def extract(target, blob, item, manifest):
    import pyarrow.parquet as pq
    marker, record = marker_paths(target, item)
    if marker.exists():
        try:
            check_records(target, item, manifest)
            print(f"复用已验证图片：{item['filename']}", flush=True)
            return
        except (OSError, ValueError, KeyError):
            network.quarantine(marker, "invalid_record")
    print(f"提取并验证图片：{item['filename']}", flush=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(blob)
    metadata = parquet.schema_arrow.metadata or {}
    names = metadata_names(json.loads(metadata[b"huggingface"])) if b"huggingface" in metadata else None
    aliases, inferred = imagenet_aliases(), dict(manifest["label_to_wnid"])
    temporary = record.with_suffix(".jsonl.partial")
    count, formats = 0, Counter()
    with temporary.open("w") as stream:
        for batch in parquet.iter_batches(batch_size=64):
            columns = batch.to_pydict()
            for offset in range(batch.num_rows):
                row = {key: values[offset] for key, values in columns.items()}
                wnid = resolve_row_wnid(row, names, inferred, aliases)
                if wnid != manifest["label_to_wnid"].get(str(row.get("label"))):
                    raise ValueError("Parquet 标签与固定版本标签清单冲突")
                raw = row["image"].get("bytes") if isinstance(row["image"], dict) else row["image"]
                if not isinstance(raw, bytes):
                    raise ValueError("Parquet 未内嵌图片字节")
                decoded = inspect_image(io.BytesIO(raw),
                                        context=f"shard={item['filename']}, row={count}, wnid={wnid}")
                extension = decoded["extension"]
                rel = Path(split_of(item["filename"])) / wnid / f"{Path(item['filename']).stem}_{count:08d}{extension}"
                path = Path(target) / rel
                checksum = hashlib.sha256(raw).hexdigest()
                if not path.exists() or file_hash(path) != checksum:
                    if path.exists():
                        # Keep corrupt originals outside ImageFolder's class directories.
                        quarantine = Path(target) / "corrupt_images" / rel
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        network.quarantine(quarantine, "previous")
                        path.replace(quarantine)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    part = path.with_suffix(path.suffix + ".partial")
                    part.write_bytes(raw)
                    part.replace(path)
                stream.write(json.dumps({"row": count, "label": row["label"], "wnid": wnid,
                                         "output": str(rel), "sha256": checksum,
                                         **{key: decoded[key] for key in ("format", "mode", "size")}}) + "\n")
                formats[decoded["format"]] += 1
                count += 1
                if count % 1000 == 0:
                    print(f"  已提取 {count:,}/{parquet.metadata.num_rows:,} 张", flush=True)
    if count != parquet.metadata.num_rows:
        raise ValueError("Parquet 行数不一致")
    temporary.replace(record)
    atomic_json(marker, {"repo": REPO, "revision": REVISION, "shard": item["filename"],
                        "shard_sha256": item["sha256"], "manifest_hash": manifest["manifest_hash"],
                        "record_sha256": file_hash(record), "rows": count, "written": count,
                        "formats": dict(formats), "completed": now()})
    print(f"分片提取完成：{count:,} 张，格式 {dict(formats)}", flush=True)


def verify_dataset(resources, target, manifest, classes):
    counts = {"train": Counter(), "val": Counter()}
    outputs = set()
    for item in manifest["files"]:
        blob, _ = network.blob_paths(dataset_cache(target), REPO, item["sha256"])
        if not network.valid_blob(blob, item):
            raise ValueError(f"缺失/损坏的分片：{item['filename']}")
        print(f"本地校验分片及图片：{item['filename']}", flush=True)
        partial_counts, partial_outputs = check_records(target, item, manifest)
        if outputs.intersection(partial_outputs):
            raise ValueError("多个分片含重复输出路径")
        outputs.update(partial_outputs)
        counts[split_of(item["filename"])].update(partial_counts)
    for split, expected in EXPECTED_ROWS.items():
        directories = {p.name for p in (Path(target) / split).iterdir() if p.is_dir()}
        if directories != set(classes):
            raise ValueError(f"{split} 类别目录与 class100.txt 不符")
        if sum(counts[split].values()) != expected or set(counts[split]) != set(classes):
            raise ValueError(f"{split} 数量/类别错误：{sum(counts[split].values())}/{expected}")
        if split == "val" and set(counts[split].values()) != {50}:
            raise ValueError("验证集必须每类 50 张")
    # ImageFolder scans recursively: reject extra files/classes that would change the experiment.
    actual = {str(p.relative_to(target)) for split in ("train", "val")
              for p in (Path(target) / split).rglob("*") if p.is_file()}
    if actual != outputs:
        raise ValueError(f"图片目录与提取记录不同：额外 {len(actual - outputs)}，缺少 {len(outputs - actual)}")
    return {wnid: {split: counts[split][wnid] for split in counts} for wnid in classes}


def check_models():
    dit, vae = configured("DIT_CKPT"), configured("VAE_PATH")
    if not dit or not Path(dit).is_file() or Path(dit).stat().st_size < 1_000_000_000:
        raise ValueError("本地 DiT 不可用，请核对 rebuttal.core.env 的 DIT_CKPT；本入口不自动下载模型")
    if not vae or not has_vae(vae):
        raise ValueError("本地 VAE 不可用，请核对 rebuttal.core.env 的 VAE_PATH；本入口不自动下载模型")
    print(f"复用 DiT：{Path(dit).resolve()}\n复用 VAE：{Path(vae).resolve()}", flush=True)


def run(mode, resources, target, classes):
    resources, target = Path(resources), Path(target)
    resources.mkdir(parents=True, exist_ok=True)
    # Share the old preparation lock so old and new downloaders cannot race.
    with (resources / ".resource_preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("已有资源准备/验证进程运行，请等待其完成")
        path = manifest_path(resources)
        manifest = validate_manifest(read_json(path), classes) if path.exists() else None
        if mode == "diagnose":
            diagnose(resources, classes, manifest)
            return
        check_models()
        if configured("REPORT_DIR") and Path(configured("REPORT_DIR")).resolve() != (target / "val").resolve():
            raise ValueError("核心批次 REPORT_DIR 必须是 DATA_ROOT/val")
        if mode == "verify":
            if manifest is None:
                raise ValueError("缺少固定版本下载清单，ImageNet100 尚未准备完成")
            ready = read_json(target / "resource_manifest.json")
            if (ready.get("download_manifest_hash") != manifest["manifest_hash"] or not ready.get("valid")
                    or ready.get("repo") != REPO or ready.get("revision") != REVISION):
                raise ValueError("缺少有效的完整资源就绪标记")
            counts = verify_dataset(resources, target, manifest, classes)
            if ready.get("counts") != counts:
                raise ValueError("完整资源就绪标记与实际图片数量不符")
            print("本地校验通过：18 个分片、126689 张训练图、5000 张验证图；无需联网。", flush=True)
            return
        cache = dataset_cache(target)
        migrate_legacy_layout(resources, target, cache, repo=REPO, revision=REVISION)
        # Check compatibility before any downloads or extracting into an existing directory.
        target.mkdir(parents=True, exist_ok=True)
        source_path = target / "download_source.json"
        if source_path.exists():
            source = read_json(source_path)
            if source.get("repo") != REPO or source.get("revision") != REVISION:
                raise ValueError("现有数据目录来自其他版本；拒绝混合数据")
        else:
            if any((target / split).exists() for split in ("train", "val")):
                raise ValueError("DATA_ROOT 已有未登记的数据目录，拒绝混入下载数据")
            atomic_json(source_path, {"repo": REPO, "revision": REVISION, "created": now()})
        routes = []
        cache.mkdir(parents=True, exist_ok=True)
        print(f"数据目录：{target}\n数据缓存：{cache}\n固定版本共 18 个分片；先处理本地缓存，缺失时续传。", flush=True)
        if manifest is None:
            require_space(cache, target)
            manifest, routes = diagnose(resources, classes, manifest)
        # Import without installing packages or changing the research environment.
        import pyarrow.parquet  # noqa: F401
        ready_path = target / "resource_manifest.json"
        if ready_path.exists():
            network.quarantine(ready_path, "previous")
        for item in manifest["files"]:
            cached = any(network.valid_blob(p, item) for p in network.blob_paths(cache, REPO, item["sha256"]))
            if not cached and not routes:
                require_space(cache, target)
                _, routes = diagnose(resources, classes, manifest)
            blob = network.download(item, repo=REPO, revision=REVISION, cache=cache, routes=routes)
            extract(target, blob, item, manifest)
            if split_of(item["filename"]) == "val":
                counts, _ = check_records(target, item, manifest)
                if set(counts) != set(classes) or set(counts.values()) != {50}:
                    raise ValueError("验证集标签/数量未通过，不继续下载训练集")
        counts = verify_dataset(resources, target, manifest, classes)
        atomic_json(ready_path, {"repo": REPO, "revision": REVISION, "valid": True,
                    "root": str(target.resolve()), "report": str((target / "val").resolve()),
                    "counts": counts, "label_to_wnid": manifest["label_to_wnid"],
                    "download_manifest_hash": manifest["manifest_hash"], "image_policy": manifest["image_policy"],
                    "created": now(), "verification": "all shard hashes, all image hashes and full image decode"})
        print("数据准备完成且全部校验通过。现在可以运行 scripts/submit_rebuttal_core.sh。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--diagnose", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--resource-root", default=str(ROOT / "resources"))
    parser.add_argument("--class-file", default=os.environ.get("CLASS_FILE", str(ROOT / "misc/class100.txt")))
    args = parser.parse_args()
    mode = "diagnose" if args.diagnose else "verify" if args.verify else "prepare"
    try:
        classes = Path(args.class_file).read_text().splitlines()
        if len(classes) != 100 or len(set(classes)) != 100 or not all(re.fullmatch(r"n\d{8}", c) for c in classes):
            raise ValueError("核心批次需要 100 个不同的合法 WNID")
        target = configured("DATA_ROOT") or str(Path(args.resource_root) / "datasets/imagenet100")
        run(mode, args.resource_root, target, classes)
        atomic_json(Path(args.resource_root) / "core_prepare_status.json",
                    {"created": now(), "mode": mode, "exit_code": 0})
    except (OSError, ValueError, KeyError, TypeError, ImportError, DownloadError) as error:
        code = 75 if isinstance(error, DownloadError) else 1
        record = {"created": now(), "mode": mode, "error_type": type(error).__name__,
                  "message": network.sanitize(error), "exit_code": code}
        atomic_json(Path(args.resource_root) / "core_prepare_error.json", record)
        atomic_json(Path(args.resource_root) / "core_prepare_status.json", record)
        print(f"资源未就绪：{record['message']}", file=sys.stderr)
        if mode == "verify":
            print("请先执行：bash scripts/prepare_rebuttal_core.sh；提交入口不会自动下载。", file=sys.stderr)
        sys.exit(code)


if __name__ == "__main__":
    main()
