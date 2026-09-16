"""Model and runtime environment manifest loading, validation, and checksum verification."""

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_SCHEMA_PATH = (
    _REPO_ROOT / "contracts" / "models" / "schema" / "model-manifest.schema.json"
)
_RUNTIME_SCHEMA_PATH = (
    _REPO_ROOT / "contracts" / "models" / "schema" / "runtime-manifest.schema.json"
)


def get_model_manifest_schema_path() -> Path:
    return _MODEL_SCHEMA_PATH


def get_runtime_manifest_schema_path() -> Path:
    return _RUNTIME_SCHEMA_PATH


def _load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Manifest schema not found at {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_model_manifest(manifest: dict[str, Any]) -> None:
    """Validate model manifest against Draft 2020-12 schema.

    Raises:
        ValueError: If validation fails.
    """
    schema = _load_schema(_MODEL_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(manifest))
    if errors:
        error_msgs = [f"{e.json_path or '<root>'}: {e.message}" for e in errors]
        raise ValueError("Model manifest validation failed:\n" + "\n".join(error_msgs))


def load_model_manifest(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a model manifest from a file path or dict."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Model manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    elif isinstance(source, dict):
        manifest = source
    else:
        raise TypeError(f"Expected Path, str, or dict, got {type(source)}")

    validate_model_manifest(manifest)
    return manifest


def validate_runtime_manifest(manifest: dict[str, Any]) -> None:
    """Validate runtime environment manifest against Draft 2020-12 schema.

    Raises:
        ValueError: If validation fails.
    """
    schema = _load_schema(_RUNTIME_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(manifest))
    if errors:
        error_msgs = [f"{e.json_path or '<root>'}: {e.message}" for e in errors]
        raise ValueError(
            "Runtime manifest validation failed:\n" + "\n".join(error_msgs)
        )


def load_runtime_manifest(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a runtime manifest from a file path or dict."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Runtime manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    elif isinstance(source, dict):
        manifest = source
    else:
        raise TypeError(f"Expected Path, str, or dict, got {type(source)}")

    validate_runtime_manifest(manifest)
    return manifest


def calculate_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for checksum: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifact_sha256(file_path: Path | str, expected_sha256: str) -> bool:
    """Verify if a file matches the expected SHA-256 checksum."""
    actual_hash = calculate_file_sha256(file_path)
    return actual_hash.lower() == expected_sha256.lower()


def _parse_version(version_str: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version_str.split(".") if p.isdigit())


def get_current_platform_identifier() -> str:
    """Return normalized platform string (e.g. linux-x86_64, windows-x86_64, macos-arm64)."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    if sys_name == "linux":
        os_id = "linux"
    elif sys_name == "windows":
        os_id = "windows"
    elif sys_name == "darwin":
        os_id = "macos"
    else:
        os_id = sys_name

    if machine in ("x86_64", "amd64"):
        arch_id = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch_id = "arm64" if os_id == "macos" else "aarch64"
    else:
        arch_id = machine

    return f"{os_id}-{arch_id}"


def check_runtime_compatibility(runtime_manifest: dict[str, Any]) -> list[str]:
    """Check current Python interpreter and host platform against the runtime manifest.

    Returns:
        List of compatibility error messages. Empty list indicates full compatibility.
    """
    errors: list[str] = []

    # 1. Python version check
    current_py = (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    py_spec = runtime_manifest.get("python_spec", {})
    min_py_str = py_spec.get("min_version")
    max_py_str = py_spec.get("max_exclusive_version")

    if min_py_str:
        min_py = _parse_version(min_py_str)
        if current_py < min_py:
            errors.append(
                f"Python version {sys.version.split()[0]} is below minimum required {min_py_str}"
            )

    if max_py_str:
        max_py = _parse_version(max_py_str)
        if current_py >= max_py:
            errors.append(
                f"Python version {sys.version.split()[0]} meets or exceeds maximum exclusive version {max_py_str}"
            )

    # 2. Supported platform check
    current_platform = get_current_platform_identifier()
    supported_platforms = runtime_manifest.get("supported_platforms", [])
    if current_platform not in supported_platforms:
        errors.append(
            f"Current platform '{current_platform}' is not in supported platforms: {supported_platforms}"
        )

    # 3. NumPy version check (if installed)
    try:
        import numpy as np

        numpy_spec = runtime_manifest.get("numpy_spec", {})
        min_np_str = numpy_spec.get("min_version")
        if min_np_str:
            np_ver = _parse_version(np.__version__)
            min_np = _parse_version(min_np_str)
            if np_ver < min_np:
                errors.append(
                    f"NumPy version {np.__version__} is below minimum required {min_np_str}"
                )
    except ImportError:
        errors.append("NumPy is not installed in the current environment")

    return errors
