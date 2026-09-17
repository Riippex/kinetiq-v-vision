"""Download, verify, and cache pinned model artifacts for real ONNX inference.

Weights are never committed to the repository: verified downloads land under
the git-ignored `weights/` directory (see `.gitignore`). Every artifact is
checked against the manifest's declared size and SHA-256 before it is ever
handed to `cv2.dnn.readNetFromONNX`; a corrupted or wrong download is deleted
and never silently substituted for the real model.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kinetiq_v_vision.infrastructure.inference.manifest import (
    calculate_file_sha256,
    verify_artifact_sha256,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WEIGHTS_DIR = _REPO_ROOT / "weights"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ModelArtifactUnavailableError(RuntimeError):
    """Raised when a pinned model artifact cannot be obtained and verified.

    Carries the exact per-URL blocker rather than ever falling back to a
    mock or previously-known-good result.
    """


def resolve_model_artifact(
    manifest: dict[str, Any],
    weights_dir: Path | str = DEFAULT_WEIGHTS_DIR,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Ensure the artifact declared by `manifest` exists locally and is verified.

    Reuses an existing hash-verified copy under `weights_dir` when present.
    Otherwise tries each of the manifest's `download_urls` in order, verifying
    both size_bytes and sha256 before accepting the download. Raises
    `ModelArtifactUnavailableError` naming every URL's exact failure reason
    when no declared source yields a verified artifact.
    """
    artifact = manifest["artifact"]
    filename = artifact["filename"]
    expected_sha256 = artifact["sha256"]
    expected_size = artifact["size_bytes"]
    model_id = manifest.get("model_id", filename)

    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    destination = weights_path / filename

    if destination.is_file():
        if destination.stat().st_size == expected_size and verify_artifact_sha256(
            destination, expected_sha256
        ):
            return destination
        # Stale or corrupted local copy: remove so a fresh download is attempted.
        destination.unlink()

    download_urls: list[str] = artifact.get("download_urls", [])
    if not download_urls:
        raise ModelArtifactUnavailableError(
            f"Manifest for '{model_id}' declares no download_urls; cannot obtain '{filename}'."
        )

    failures: list[str] = []
    for url in download_urls:
        try:
            _download(url, destination, timeout_seconds)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            failures.append(f"{url}: request failed ({exc})")
            destination.unlink(missing_ok=True)
            continue

        actual_size = destination.stat().st_size
        if actual_size != expected_size:
            failures.append(
                f"{url}: downloaded {actual_size} bytes, manifest declares {expected_size}"
            )
            destination.unlink(missing_ok=True)
            continue

        if not verify_artifact_sha256(destination, expected_sha256):
            actual_hash = calculate_file_sha256(destination)
            failures.append(
                f"{url}: SHA-256 mismatch (expected {expected_sha256}, got {actual_hash})"
            )
            destination.unlink(missing_ok=True)
            continue

        return destination

    raise ModelArtifactUnavailableError(
        f"Could not obtain a size- and hash-verified copy of '{filename}' for "
        f"'{model_id}' from any declared download URL:\n" + "\n".join(f"  - {f}" for f in failures)
    )


def _download(url: str, destination: Path, timeout_seconds: float) -> None:
    """Stream `url` to `destination`. Raises on any transport-level failure."""
    request = urllib.request.Request(url, headers={"User-Agent": "kinetiq-v-vision-benchmark/1.0"})
    with (
        urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        open(destination, "wb") as handle,
    ):
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            handle.write(chunk)
