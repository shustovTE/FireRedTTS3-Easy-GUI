"""Download the pinned Russian spontaneous corpus; never persist API credentials."""
from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
            headers={"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json"})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
                url = json.load(response).get("downloadUrl")
        except Exception:
            raise RuntimeError("Mozilla download API failed; check account access/API key and retry. Credentials and response URLs were not logged.") from None
        finally:
            del request, key
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Mozilla API did not return an HTTPS download URL")
        with tempfile.TemporaryDirectory(prefix=".cv-download-", dir=archive.parent) as temporary:
            candidate = Path(temporary) / FILENAME
            try:
                # Separate request: the bearer token is NOT sent to storage.
                with urllib.request.urlopen(url, timeout=90) as response, candidate.open("wb") as out:
                    received = 0
                    while block := response.read(4 * 1024 * 1024):
                        received += len(block)
                        if received > ARCHIVE_BYTES:
                            raise ValueError("Oversized archive")
                        out.write(block)
                        log(f"[common-voice] Downloaded {received / 1024**2:.1f} MiB")
            except Exception:
                raise RuntimeError("Common Voice storage download failed; rerun download. No partial archive was installed.") from None
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
