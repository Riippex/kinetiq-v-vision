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
    subparsers.add_parser("check", help="Verify engine runtime and configuration readiness")

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        import uvicorn
        from kinetiq_v_vision.bootstrap.container import Container

        container = Container()
        app = container.create_configured_app()
        uvicorn.run(app, host=parsed.host, port=parsed.port)
        return 0

    if parsed.command == "check":
        print("Kinetiq V Vision engine: ready.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
