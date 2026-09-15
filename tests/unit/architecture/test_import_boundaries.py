import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "kinetiq_v_vision"


def get_all_python_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.py"))


def extract_imported_modules(file_path: Path) -> list[str]:
    """Parse a python file AST and return a list of top-level and relative module imports."""
    with open(file_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_domain_has_zero_outer_or_framework_dependencies() -> None:
    domain_dir = SRC_ROOT / "domain"
    assert domain_dir.exists()

    forbidden_patterns = [
        "fastapi",
        "starlette",
        "cv2",
        "boto3",
        "mlflow",
        "pydantic",
        "kinetiq_v_vision.application",
        "kinetiq_v_vision.infrastructure",
        "kinetiq_v_vision.interfaces",
        "kinetiq_v_vision.bootstrap",
    ]

    for py_file in get_all_python_files(domain_dir):
        imports = extract_imported_modules(py_file)
        for imp in imports:
            for forbidden in forbidden_patterns:
                assert not (imp == forbidden or imp.startswith(f"{forbidden}.")), (
                    f"Domain file {py_file.name} violates Clean Architecture by importing '{imp}'"
                )


def test_application_has_zero_outer_or_framework_dependencies() -> None:
    app_dir = SRC_ROOT / "application"
    assert app_dir.exists()

    forbidden_patterns = [
        "fastapi",
        "starlette",
        "cv2",
        "boto3",
        "mlflow",
        "kinetiq_v_vision.infrastructure",
        "kinetiq_v_vision.interfaces",
        "kinetiq_v_vision.bootstrap",
    ]

    for py_file in get_all_python_files(app_dir):
        imports = extract_imported_modules(py_file)
        for imp in imports:
            for forbidden in forbidden_patterns:
                assert not (imp == forbidden or imp.startswith(f"{forbidden}.")), (
                    f"Application file {py_file.name} violates Clean Architecture by importing '{imp}'"
                )


def test_infrastructure_does_not_import_interfaces_or_bootstrap() -> None:
    infra_dir = SRC_ROOT / "infrastructure"
    assert infra_dir.exists()

    forbidden_patterns = [
        "kinetiq_v_vision.interfaces",
        "kinetiq_v_vision.bootstrap",
    ]

    for py_file in get_all_python_files(infra_dir):
        imports = extract_imported_modules(py_file)
        for imp in imports:
            for forbidden in forbidden_patterns:
                assert not (imp == forbidden or imp.startswith(f"{forbidden}.")), (
                    f"Infrastructure file {py_file.name} violates Clean Architecture by importing '{imp}'"
                )
