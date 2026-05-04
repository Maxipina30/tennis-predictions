from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_LOG = ROOT / "streamlit.out.log"
ERR_LOG = ROOT / "streamlit.err.log"


def main() -> None:
    flags = 0
    if sys.platform.startswith("win"):
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )

    out = OUT_LOG.open("a", encoding="utf-8")
    err = ERR_LOG.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        creationflags=flags,
        close_fds=False,
    )
    time.sleep(2)
    if process.poll() is not None:
        print(f"Streamlit exited early with code {process.returncode}. Check {ERR_LOG}.")
        return
    print(f"Started Streamlit pid={process.pid} url=http://127.0.0.1:8501")


if __name__ == "__main__":
    main()
