"""Serial, resumable Hub downloads with verified local reuse and rate backoff."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import time


class DownloadIntegrityError(ValueError):
    pass


def local_target(source, name):
    relative = PurePosixPath(name)
    root = Path(source).resolve()
    path = root.joinpath(*relative.parts)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or not relative.parts or not path.resolve().is_relative_to(root)):
        raise ValueError(f'Invalid download path: {name!r}')
    return path


def verified_local(source, name, revision):
    # The Hub version is pinned in pyproject.toml. Its local-dir receipts bind
    # each completed download to both the commit and LFS SHA256/git blob SHA1.
    from huggingface_hub._local_folder import read_download_metadata
    path = local_target(source, name)
    if not path.is_file():
        return False
    metadata = read_download_metadata(Path(source), name)
    if metadata is None or metadata.commit_hash != revision:
        return False
    if re.fullmatch(r'[0-9a-f]{64}', metadata.etag):
        digest = hashlib.sha256()
    elif re.fullmatch(r'[0-9a-f]{40}', metadata.etag):
        digest = hashlib.sha1()
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    else:
        return False
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest() == metadata.etag


def retry_delay(error, attempt, now=None):
    """Return a bounded-backoff delay, honoring longer server Retry-After values.

    None means permanent failure. In particular, 401/403/404 are never skipped
    or reclassified as rate limits. A 429 pauses the only download worker.
    """
    import httpx
    response, status, network = None, None, False
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, 'response', None)
        status = getattr(response, 'status_code', None)
        network |= isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError))
        if status is None and isinstance(error, ConnectionError):
            # hf-xet may wrap HTTP errors as a built-in ConnectionError.
            match = re.search(r'HTTP status (?:client|server) error \((\d{3})\b', str(error))
            status = int(match[1]) if match else None
        if status is not None:
            break
        # Hub HEAD failures can become LocalEntryNotFoundError or ValueError
        # with an explicit HTTP/transport cause. Preserve its retry headers.
        error = error.__cause__
    if status is not None:
        if status not in {408, 429, 500, 502, 503, 504}:
            return None
    elif not network:
        return None
    delay = min(300., (60. if status == 429 else 10.) * 2**min(attempt, 8))
    header = response.headers.get('Retry-After') if response is not None else None
    if header:
        try:
            value = float(header)
        except ValueError:
            try:
                deadline = parsedate_to_datetime(header)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                value = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
            except (TypeError, ValueError, OverflowError):
                value = 0.
        if math.isfinite(value):
            delay = max(delay, value)
    return delay


def wait_seconds(seconds):
    while seconds > 0:
        chunk = min(seconds, 30.)
        time.sleep(chunk)
        seconds -= chunk


def download_files(source, names, repo, revision, *, interval=2., max_retries=8,
                   log=print, progress=None, authenticated=False, transport='http'):
    """Download only missing/invalid files, one file at a time.

    Completed files are hashed locally before any Hub call. The normal Hub
    downloader retains its .incomplete files and resumes interrupted transfers.
    HTTP is the default for many small files. Large authenticated Parquet jobs
    can explicitly use Xet. Both modes honor rate backoff. A stored token is
    requested with token=True and never included in reports or log messages.
    """
    from huggingface_hub import constants, hf_hub_download
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Downloads require an immutable commit SHA')
    if not math.isfinite(interval) or interval < 0 or type(max_retries) is not int or not 0 <= max_retries <= 20:
        raise ValueError('Invalid download interval or retry budget')
    if transport not in {'http', 'xet'} or type(authenticated) is not bool:
        raise ValueError('Invalid Hub transport/authentication setting')
    source = Path(source).resolve()
    names = list(dict.fromkeys(names))
    for name in names:
        local_target(source, name)
    state = dict(status='running', complete=False, revision=revision, total=len(names),
                 processed=0, reused=0, downloaded=0, retries=0, current_file=None)
    def emit():
        if progress:
            progress(dict(state))
    last_request = None
    old_xet = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = transport == 'http'
    try:
        log(f'[download] {len(names)} files; transport={transport}; interval={interval:g}s; verifying local cache')
        emit()
        for name in names:
            state.update(status='running', current_file=name, retry_wait_seconds=0.)
            emit()
            if verified_local(source, name, revision):
                state['reused'] += 1
                fetched = False
            else:
                integrity_retried = False
                for attempt in range(max_retries+1):
                    if last_request is not None:
                        wait_seconds(max(0., interval - (time.monotonic()-last_request)))
                    last_request = time.monotonic()
                    try:
                        # Force replacement only for a present but unverified
                        # file; a missing final file keeps its partial download.
                        hf_hub_download(repo, name, repo_type='dataset', revision=revision,
                                        local_dir=source, force_download=local_target(source, name).is_file(),
                                        **({'token': True} if authenticated else {}))
                        if not verified_local(source, name, revision):
                            raise DownloadIntegrityError(f'Download failed commit/content verification: {name}')
                        break
                    except Exception as error:
                        # An old Xet .incomplete may contain reconstructed ranges
                        # rather than an HTTP prefix. If resumed bytes fail the
                        # hash, replace the unverified final file once from zero.
                        integrity_retry = isinstance(error, DownloadIntegrityError) and not integrity_retried
                        delay = interval if integrity_retry else retry_delay(error, attempt)
                        if delay is None or attempt == max_retries:
                            raise
                        integrity_retried |= integrity_retry
                        state.update(status='waiting', retry_wait_seconds=delay)
                        state['retries'] += 1
                        emit()
                        reason = 'content hash mismatch; retry from zero' if integrity_retry else 'temporary network/rate error'
                        log(f'[download] {name}: {reason}; wait {delay:g}s; retry {attempt+1}/{max_retries}')
                        wait_seconds(delay)
                        state.update(status='running', retry_wait_seconds=0.)
                        emit()
                state['downloaded'] += 1
                fetched = True
            state['processed'] += 1
            emit()
            if fetched or state['processed'] % 50 == 0 or state['processed'] == len(names):
                log(f"[download] {state['processed']}/{len(names)}; reused={state['reused']}; downloaded={state['downloaded']}")
        state.update(status='completed', complete=True, current_file=None)
        emit()
        log('[download] Complete: all planned files verified locally.')
        return state
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error_type=type(error).__name__)
        emit()
        raise
    finally:
        constants.HF_HUB_DISABLE_XET = old_xet
