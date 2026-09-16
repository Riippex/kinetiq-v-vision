import argparse
import sys


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kinetiq-v-vision",
        description="Kinetiq V Vision engine CLI interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start the FastAPI REST service")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    run_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Check readiness command
    subparsers.add_parser(
        "check", help="Verify engine runtime and configuration readiness"
    )

    # Baseline benchmark command
    baseline_parser = subparsers.add_parser(
        "baseline", help="Run pose candidate baselines on a fixed split"
    )
    baseline_parser.add_argument(
        "--manifest",
        default=None,
        help="Path to dataset manifest JSON (defaults to fixtures/data/synthetic_manifest.json)",
    )
    baseline_parser.add_argument(
        "--split",
        default="development",
        choices=["development", "heldout"],
        help="Dataset split to evaluate (default: development)",
    )
    baseline_parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Specific candidate IDs to benchmark (default: all)",
    )
    baseline_parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum frames per clip to evaluate (default: 10)",
    )
    baseline_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write benchmark JSON report",
    )

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        import uvicorn

        from kinetiq_v_vision.bootstrap.container import Container

        container = Container()
        app = container.create_configured_app()
        uvicorn.run(app, host=parsed.host, port=parsed.port)
        return 0

    if parsed.command == "check":
        import json
        from pathlib import Path

        from kinetiq_v_vision.infrastructure.inference.manifest import (
            check_runtime_compatibility,
            load_model_manifest,
            load_runtime_manifest,
        )

        repo_root = Path(__file__).resolve().parents[4]
        manifests_dir = repo_root / "model-manifests"
        runtime_manifest_path = manifests_dir / "runtime-environment.json"

        # Check runtime manifest
        if not runtime_manifest_path.exists():
            print(f"Error: Runtime manifest not found at {runtime_manifest_path}")
            return 1

        try:
            runtime_manifest = load_runtime_manifest(runtime_manifest_path)
            compat_errors = check_runtime_compatibility(runtime_manifest)
            if compat_errors:
                print("Runtime environment compatibility errors:")
                for err in compat_errors:
                    print(f"  - {err}")
                return 1
            print("Runtime environment: compatible.")
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error validating runtime manifest: {e}")
            return 1

        # Check model manifests
        model_manifest_files = [
            manifests_dir / "person_detection_mediapipe_v1.json",
            manifests_dir / "pose_estimation_mediapipe_v1.json",
        ]
        for mf in model_manifest_files:
            if not mf.exists():
                print(f"Error: Model manifest not found: {mf}")
                return 1
            try:
                load_model_manifest(mf)
                print(f"Model manifest {mf.name}: valid.")
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error validating {mf.name}: {e}")
                return 1

        print("Kinetiq V Vision engine: ready.")
        return 0

    if parsed.command == "baseline":
        from pathlib import Path

        from kinetiq_v_vision.evaluation.baselines import (
            run_pose_baseline_benchmark,
        )

        repo_root = Path(__file__).resolve().parents[4]
        default_manifest = (
            repo_root / "fixtures" / "data" / "synthetic_manifest.json"
        )
        manifest_path = parsed.manifest or str(default_manifest)

        try:
            report = run_pose_baseline_benchmark(
                manifest_path=manifest_path,
                split=parsed.split,
                selected_candidate_ids=parsed.candidates,
                max_frames_per_clip=parsed.max_frames,
            )
            print(report.summary_markdown())

            if parsed.output:
                out_file = Path(parsed.output).resolve()
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(report.to_json(), encoding="utf-8")
                print(f"\nReport saved to: {out_file}")

            return 0
        except Exception as e:
            print(f"Error executing baseline benchmark: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0



if __name__ == "__main__":
    sys.exit(main())
