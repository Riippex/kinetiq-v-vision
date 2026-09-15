"""Main module entrypoint allowing `python -m kinetiq_v_vision`."""

import sys
from kinetiq_v_vision.interfaces.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
