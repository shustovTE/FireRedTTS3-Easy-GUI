"""Download the pinned Russian spontaneous corpus; never persist API credentials."""
from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request

from filelock import FileLock

DATASET_ID = "cmu5mg3pr00simh07epeylc55"
CORPUS = "sps-corpus-5.0-2026-09-11-ru"
FILENAME = "common-voice-spontaneous-speech-5-0-russ-883d3f86.tar.gz"
ARCHIVE_SHA256 = "aa0910f5fef7f24afccf5bca075995f0b7a970d96da72626c80507d629d05a6f"
ARCHIVE_BYTES = 93031233
DEFAULT_ARCHIVE = f"training/raw/common-voice-sps-5-ru/{FILENAME}"
USER_AGENT = "FireRedTTS3-Easy-GUI/0.1.1"


def safe_failure(phase, error):
    """Diagnose transport/status failures without reflecting tokens or signed URLs."""
    if isinstance(error, urllib.error.HTTPError):
        headers = {name.lower(): str(value).strip() for name, value in error.headers.items()} if error.headers else {}
        content_type = headers.get("content-type", "").lower()
        challenge = headers.get("cf-mitigated", "").lower() == "challenge"
        kind = "JSON" if "json" in content_type else "HTML" if "html" in content_type else "non-JSON"
        cloudflare_code = None
        try:
            payload = json.loads(error.read(4096))
            if isinstance(payload, dict) and payload.get("cloudflare_error") is True:
                code = payload.get("error_code")
                if type(code) is int and 1000 <= code <= 9999:
                    cloudflare_code = code
        except Exception:
            pass
        finally:
            error.close()
        if challenge:
            hint = ("Cloudflare challenge (cf-mitigated: challenge) replaced the API/storage response. "
                    "This is not an API-key validation result. Use the dataset's normal browser download "
                    "or contact Mozilla Data Collective about API access from this host; changing the key "
                    "alone does not address this challenge.")
        elif cloudflare_code == 1010:
            hint = "Cloudflare 1010 rejected the client signature before API authentication; update the downloader/User-Agent."
        else:
            hint = {
                401: "Authentication was rejected; verify the Mozilla API key (not an HF token).",
                403: "Access was denied; this can be a request filter or account/dataset access restriction, not necessarily an invalid key.",
                429: "Rate limited; wait before retrying.",
            }.get(error.code, "The service returned an HTTP error; retry later for 5xx responses.")
        return RuntimeError(f"{phase}: HTTP {error.code}, response={kind}. {hint} Response bodies, credentials and signed URLs are omitted.")
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        detail = "TLS certificate verification failed; check system time and CA certificates"
    elif isinstance(reason, ssl.SSLError):
        detail = "TLS connection failed"
    elif isinstance(reason, socket.gaierror):
        detail = "DNS lookup failed"
    elif isinstance(reason, TimeoutError):
        detail = "request timed out"
    elif isinstance(error, json.JSONDecodeError):
        detail = "expected JSON but received an invalid/non-JSON response"
    else:
        detail = f"request failed ({type(error).__name__})"
    return RuntimeError(f"{phase}: {detail}. Credentials and signed URLs are omitted.")


def verify_archive(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Common Voice archive differs from the pinned Russian 5.0 release")
    return path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API bearer token to a redirected host.
        return None


def download(archive=DEFAULT_ARCHIVE, log=print):
    archive = Path(archive).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(archive) + ".lock", timeout=0):
        if archive.exists():
            verify_archive(archive)
            log(f"[common-voice] Verified existing archive: {archive}")
            return archive
        key = os.environ.get("MOZILLA_DATA_COLLECTIVE_API_KEY") or getpass.getpass("Mozilla Data Collective API key: ")
        if not key.strip():
            raise ValueError("Mozilla API key is required")
        request = urllib.request.Request(
            f"https://mozilladatacollective.com/api/datasets/{DATASET_ID}/download",
            data=b"{}", method="POST",
            headers={"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json",
                     "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
                url = json.load(response).get("downloadUrl")
        except Exception as exc:
            raise safe_failure("Mozilla download API", exc) from None
        finally:
            del request, key
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Mozilla API did not return an HTTPS download URL")
        with tempfile.TemporaryDirectory(prefix=".cv-download-", dir=archive.parent) as temporary:
            candidate = Path(temporary) / FILENAME
            try:
                # Separate request: the bearer token is NOT sent to storage.
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=90) as response, candidate.open("wb") as out:
                    received = 0
                    while block := response.read(4 * 1024 * 1024):
                        received += len(block)
                        if received > ARCHIVE_BYTES:
                            raise ValueError("Oversized archive")
                        out.write(block)
                        log(f"[common-voice] Downloaded {received / 1024**2:.1f} MiB")
            except Exception as exc:
                raise safe_failure("Common Voice storage download (no partial archive installed)", exc) from None
            verify_archive(candidate)
            candidate.replace(archive)
    return archive


def extract_verified(archive, destination):
    verify_archive(archive)
    destination = Path(destination).resolve()
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > 1024**3:
            raise ValueError("Unexpected archive expansion size")
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not path.parts or path.parts[0] != CORPUS
                    or not (member.isdir() or member.isfile()) or member.name in seen):
                raise ValueError("Unsafe or unexpected archive member")
            seen.add(member.name)
        source.extractall(destination, members=members, filter="data")
    return destination / CORPUS
