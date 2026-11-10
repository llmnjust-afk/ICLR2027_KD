"""Verified, resumable public Hub downloads, with a separate curl TLS transport."""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from requests.exceptions import RequestException

from .common import file_hash


class DownloadError(RequestException):
    """All configured transports/endpoints failed; partial files are retained."""


def endpoints():
    return list(dict.fromkeys(x.rstrip('/') for x in [
        os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'),
        'https://hf-mirror.com', 'https://huggingface.co']))


def artifact_metadata(repo, filename, *, repo_type, revision, endpoint):
    """Resolve LFS identity even when a mirror strips resolver HEAD headers."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    import requests

    def checked(checksum, size, commit):
        checksum = re.sub(r'^W/', '', str(checksum or '')).strip('"').lower()
        if re.fullmatch(r'[0-9a-f]{40}', revision) and commit != revision:
            raise ValueError('metadata does not match pinned revision')
        if not re.fullmatch(r'[0-9a-f]{64}', checksum) or not isinstance(size, int) or size <= 0:
            raise ValueError('missing LFS SHA-256 or file size')
        return checksum, size

    errors = []
    url = hf_hub_url(repo, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)
    try:
        # Metadata uses the canonical resolver URL: some mirrors mishandle query
        # strings on HEAD. Only the actual transfer needs a fresh CDN signature.
        meta = get_hf_file_metadata(url, token=False, timeout=20)
        return checked(meta.etag, meta.size, meta.commit_hash)
    except (RequestException, ValueError) as error:
        errors.append(f'HEAD: {type(error).__name__}')

    try:
        info = HfApi(endpoint=endpoint, token=False).repo_info(
            repo, repo_type=repo_type, revision=revision, files_metadata=True, timeout=20)
        sibling = next((f for f in (info.siblings or []) if f.rfilename == filename), None)
        lfs = getattr(sibling, 'lfs', None) or {}
        checksum = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        size = lfs.get('size') if isinstance(lfs, dict) else getattr(lfs, 'size', None)
        result = checked(checksum, size, info.sha)
        print(f'LFS metadata recovered from repository API: {filename}', flush=True)
        return result
    except (RequestException, ValueError) as error:
        errors.append(f'repository API: {type(error).__name__}')

    # The raw URL returns the small Git LFS pointer, not the binary shard. Bound
    # the read in case a mirror redirects /raw to the binary instead.
    raw_url = url.replace(f'/resolve/{revision}/', f'/raw/{revision}/', 1)
    try:
        with requests.get(raw_url, stream=True, timeout=(20, 20)) as response:
            response.raise_for_status()
            buffer = bytearray()
            for chunk in response.iter_content(4096):
                buffer.extend(chunk)
                if len(buffer) >= 4096:
                    break
            pointer = bytes(buffer[:4096]).decode('ascii', errors='replace')
        match = re.fullmatch(r'version https://git-lfs.github.com/spec/v1\r?\n'
                             r'oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\s*', pointer)
        if not match:
            raise ValueError('response is not a Git LFS pointer')
        result = checked(match[1], int(match[2]), revision)
        print(f'LFS metadata recovered from pinned raw pointer: {filename}', flush=True)
        return result
    except (RequestException, ValueError) as error:
        errors.append(f'raw pointer: {type(error).__name__}')
    raise DownloadError(f'{endpoint}: no verifiable LFS metadata for {filename} ({"; ".join(errors)})')


def curl_download(repo, filename, *, repo_type, revision, cache_dir):
    from huggingface_hub import hf_hub_url
    from .network import sanitize

    if not shutil.which('curl'):
        raise DownloadError('curl is not installed; install curl or use REBUTTAL_DOWNLOAD_TRANSPORT=hub')
    attempts = int(os.environ.get('REBUTTAL_DOWNLOAD_RETRIES', '3'))
    if attempts < 1:
        raise ValueError('REBUTTAL_DOWNLOAD_RETRIES must be positive')
    errors = []
    for endpoint in endpoints():
        url = hf_hub_url(repo, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)
        try:
            checksum, expected_size = artifact_metadata(
                repo, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)
        except RequestException as error:
            errors.append(str(error))
            print(f'Metadata unavailable at {endpoint}; trying the next endpoint.', flush=True)
            continue
        # Use the standard Hub blob location so both transports share completed files.
        blobs = Path(cache_dir) / ((repo_type + 's--') + repo.replace('/', '--')) / 'blobs'
        blobs.mkdir(parents=True, exist_ok=True)
        target = blobs / checksum
        partial = blobs / (checksum + '.curl.incomplete')

        def verified(path):
            return path.is_file() and path.stat().st_size == expected_size and file_hash(path) == checksum

        if verified(target):
            return str(target)
        # Import a Hub partial once; leave its original available to the Hub client.
        hub_partial = blobs / (checksum + '.incomplete')
        if not partial.exists() and hub_partial.is_file() and hub_partial.stat().st_size:
            shutil.copyfile(hub_partial, partial)
        for attempt in range(attempts):
            if verified(partial):
                partial.replace(target)
                return str(target)
            if partial.exists() and partial.stat().st_size >= expected_size:
                partial.rename(partial.with_name(partial.name + f'.invalid_{time.time_ns()}'))
            offset = partial.stat().st_size if partial.exists() else 0
            print(f'curl {endpoint}: {filename}, resume={offset}/{expected_size} bytes, '
                  f'attempt={attempt + 1}/{attempts}', flush=True)
            command = ['curl', '--fail', '--location', '--http1.1', '--show-error', '--silent',
                       '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '120',
                       '--max-time', '600', '--continue-at', '-', '--output', str(partial),
                       '--proto', '=https', '--proto-redir', '=https']
            # Honor the same custom CA bundle as requests, without disabling TLS checks.
            ca = os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('CURL_CA_BUNDLE')
            if ca:
                command += ['--cacert', ca]
            command += ['--url', url + '?download=true&rebuttal=' + str(time.time_ns())]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if verified(partial):
                partial.replace(target)
                print(f'SHA-256 verified: {filename}', flush=True)
                return str(target)
            errors.append(f'{endpoint}: curl exit={result.returncode}, '
                          f'bytes={partial.stat().st_size if partial.exists() else 0}/{expected_size}; '
                          f'detail={sanitize(result.stderr)}')
            # Restart servers that refuse Range requests, keeping the old partial for inspection.
            if result.returncode == 33 and partial.exists():
                partial.rename(partial.with_name(partial.name + f'.range_rejected_{time.time_ns()}'))
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise DownloadError(f'Cannot download {repo}/{filename}. ' + '; '.join(errors) +
                        '. Partial downloads retained. Check proxy/CDN access or run --prepare-only on a connected node.')


def hub_download(repo, filename, *, repo_type='model', revision, cache_dir, token=False):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    mode = os.environ.get('REBUTTAL_DOWNLOAD_TRANSPORT', 'auto')
    if mode not in {'auto', 'hub', 'curl'}:
        raise ValueError('REBUTTAL_DOWNLOAD_TRANSPORT must be auto, hub, or curl')
    kwargs = dict(repo_type=repo_type, revision=revision, cache_dir=str(cache_dir), token=False)
    try:
        return hf_hub_download(repo, filename, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        pass
    if mode != 'curl':
        try:
            return hf_hub_download(repo, filename, **kwargs)
        except RequestException as error:
            if mode == 'hub':
                raise
            print(f'Hub download failed ({type(error).__name__}); retrying with resumable curl.', flush=True)
    return curl_download(repo, filename, repo_type=repo_type, revision=revision, cache_dir=cache_dir)
