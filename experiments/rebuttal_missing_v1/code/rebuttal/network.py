"""Bounded curl diagnostics and one-layer, SHA-256 checked resumable transfers.

Use the caller's proxy and CA settings unchanged. Never log proxy credentials or
signed CDN query strings, and never disable certificate verification.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from .common import file_hash, now
from .transfers import DownloadError


def host(url):
    try:
        return urlsplit(url).hostname or "unknown"
    except ValueError:
        return "invalid-url"


def sanitize(value):
    # Keep useful TLS/connection errors, but strip credentials and signed URLs.
    text = re.sub(r"(?:https?|socks5h?)://[^\s\"'<>]+",
                  lambda m: "https://" + host(m[0]) + "/[redacted]", str(value))
    text = re.sub(r"(?i)(authorization|token|password|signature)[=: ]+\S+", r"\1=[redacted]", text)
    return text[-3000:]


def endpoints():
    values = [os.environ.get("HF_ENDPOINT"), "https://huggingface.co", "https://hf-mirror.com"]
    result = list(dict.fromkeys(v.rstrip("/") for v in values if v))
    for value in result:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("HF_ENDPOINT must be an HTTPS origin without credentials/query strings")
    return result


def proxy_probe():
    # curl prefers the lowercase variable for HTTPS, then HTTPS_PROXY, then ALL_PROXY.
    value = next((os.environ[k] for k in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
                  if os.environ.get(k)), None)
    if not value:
        return {"configured": False}
    parsed = urlsplit(value if "://" in value else "http://" + value)
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 1080 if parsed.scheme.startswith("socks") else 80)
        with socket.create_connection((parsed.hostname, port), timeout=3):
            pass
        return {"configured": True, "host": parsed.hostname, "port": port, "reachable": True,
                "no_proxy_set": bool(os.environ.get("no_proxy") or os.environ.get("NO_PROXY"))}
    except (OSError, ValueError) as error:
        return {"configured": True, "host": parsed.hostname, "port": locals().get("port"), "reachable": False,
                "error": sanitize(error), "error_type": type(error).__name__}


def curl_command(tls="default"):
    if not shutil.which("curl"):
        raise DownloadError("curl is not installed")
    # -q must be first: avoid a .curlrc silently adding retries or insecure TLS.
    command = ["curl", "-q", "--fail", "--location", "--max-redirs", "8", "--http1.1",
               "--silent", "--show-error", "--connect-timeout", "10",
               "--proto", "=https", "--proto-redir", "=https"]
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if ca:
        command += ["--cacert", ca]
    if tls == "1.2":
        command += ["--tlsv1.2", "--tls-max", "1.2"]
    elif tls != "default":
        raise ValueError("Unknown TLS mode")
    return command


def curl_request(url, output, *, tls="default", byte_range=None, limit=2 * 1024 * 1024, transfer=False):
    """Bound time, response size for probes, and retain only sanitized diagnostics."""
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix="cags-curl-") as tmp:
        headers, stdout, stderr = [Path(tmp) / p for p in ("headers", "stdout", "stderr")]
        command = curl_command(tls) + ["--dump-header", str(headers), "--write-out", "%{json}",
                                      "--output", str(output)]
        if transfer:
            command += ["--continue-at", "-", "--speed-limit", "1024", "--speed-time", "60", "--max-time", "1800"]
        else:
            command += ["--max-time", "25", "--max-filesize", str(limit)]
        if byte_range:
            command += ["--range", byte_range]
        command += ["--url", url]
        started, last_print, bounded = time.monotonic(), time.monotonic(), False
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err)
            try:
                while process.poll() is None:
                    size = output.stat().st_size if output.exists() else 0
                    # Also enforce limits for responses without Content-Length.
                    if (not transfer and size > limit) or time.monotonic() - started > (1830 if transfer else 30):
                        bounded = True
                        process.terminate()
                        break
                    if transfer and time.monotonic() - last_print >= 10:
                        print(f"  下载进度：{size:,} bytes；{host(url)}", flush=True)
                        last_print = time.monotonic()
                    time.sleep(0.1 if not transfer else 1)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        try:
            info = json.loads(stdout.read_text())
        except ValueError:
            info = {}
        header_text = headers.read_text(errors="replace") if headers.exists() else ""
        locations = re.findall(r"(?im)^location:\s*(\S+)", header_text)
        ranges = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", header_text)
        return {"curl_exit": process.returncode, "http_status": info.get("http_code", 0), "tls": tls,
                "requested_host": host(url), "effective_host": host(info.get("url_effective", url)),
                "redirect_hosts": [host(v) for v in locations], "content_range": ranges[-1] if ranges else None,
                "bytes": output.stat().st_size if output.exists() else 0,
                "error": sanitize(stderr.read_text()), "limit_exceeded": bounded}


def fetch_small(url, *, tls="default", byte_range=None, limit=2 * 1024 * 1024):
    with tempfile.TemporaryDirectory(prefix="cags-probe-") as tmp:
        path = Path(tmp) / "response"
        result = curl_request(url, path, tls=tls, byte_range=byte_range, limit=limit)
        body = path.read_bytes()[:limit] if path.exists() else b""
        return body, result


def resolver(endpoint, repo, revision, filename):
    # New resolver request per attempt avoids reusing expired signed CDN URLs.
    return f"{endpoint}/datasets/{repo}/resolve/{revision}/{filename}?download=true&rebuttal={time.time_ns()}"


def blob_paths(cache, repo, checksum):
    target = Path(cache) / ("datasets--" + repo.replace("/", "--")) / "blobs" / checksum
    return target, target.with_name(checksum + ".curl.incomplete")


def valid_blob(path, item):
    return path.is_file() and path.stat().st_size == item["size"] and file_hash(path) == item["sha256"]


def quarantine(path, reason):
    if path.exists():
        saved = path.with_name(path.name + f".{reason}_{time.time_ns()}")
        path.replace(saved)
        print(f"保留异常文件：{saved}", flush=True)


def download(item, *, repo, revision, cache, routes):
    """Use a frozen manifest; local completed files require no metadata requests."""
    target, partial = blob_paths(cache, repo, item["sha256"])
    if valid_blob(target, item):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    quarantine(target, "bad_hash")
    hub_partial = target.with_name(target.name + ".incomplete")
    if not partial.exists() and hub_partial.is_file() and hub_partial.stat().st_size:
        shutil.copyfile(hub_partial, partial)
    if valid_blob(partial, item):
        partial.replace(target)
        return target
    if not routes:
        raise DownloadError("No verified download route; run prepare_rebuttal_core.sh --diagnose")
    attempts = int(os.environ.get("REBUTTAL_DOWNLOAD_RETRIES", "3"))
    if attempts < 1:
        raise ValueError("REBUTTAL_DOWNLOAD_RETRIES must be positive")
    errors, stalled = [], 0
    for attempt in range(attempts):
        if partial.exists() and partial.stat().st_size >= item["size"]:
            quarantine(partial, "bad_hash")
        before = partial.stat().st_size if partial.exists() else 0
        route = routes[attempt % len(routes)]
        print(f"{item['filename']}：续传 {before:,}/{item['size']:,} bytes，尝试 {attempt + 1}/{attempts}", flush=True)
        result = curl_request(resolver(route["endpoint"], repo, revision, item["filename"]),
                              partial, tls=route["tls"], transfer=True)
        verified = valid_blob(partial, item)
        after = partial.stat().st_size if partial.exists() else 0
        result.update({"before": before, "after": after, "expected": item["size"]})
        log = Path(cache).parent / "core_transfer_events.jsonl"
        with log.open("a") as stream:
            stream.write(json.dumps({"created": now(), "filename": item["filename"],
                                     "sha256_verified": verified, **result}) + "\n")
        if verified:
            partial.replace(target)
            print(f"SHA-256 通过：{item['filename']}", flush=True)
            return target
        errors.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        stalled = stalled + 1 if after <= before else 0
        if after >= item["size"]:
            quarantine(partial, "bad_hash")
        elif result["curl_exit"] == 33:
            quarantine(partial, "range_rejected")
        # Do not burn nine nested attempts when neither endpoint makes progress.
        if stalled >= 2:
            break
    raise DownloadError(f"{item['filename']}: 下载未完成；断点已保留。" + json.dumps(errors, ensure_ascii=False))
