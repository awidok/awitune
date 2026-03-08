"""Proxy process lifecycle helpers for dashboard runtime."""

import os
import subprocess
import time
from pathlib import Path


def start_proxy(rt, awitune_dir: Path, proxy_host: str):
    if rt.proxy_proc and rt.proxy_proc.poll() is None:
        return
    venv_py = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "python3"
    py = str(venv_py) if venv_py.exists() else "python3"
    env = os.environ.copy()
    env["PROXY_HOST"] = proxy_host
    env["PROXY_PORT"] = str(rt.cfg.proxy_port)
    rt.proxy_proc = subprocess.Popen(
        [py, str(awitune_dir / "lib" / "proxy.py")],
        cwd=str(awitune_dir),
        env=env,
        stdout=open("/tmp/awitune_proxy.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(2)


def stop_proxy(rt):
    if rt.proxy_proc and rt.proxy_proc.poll() is None:
        rt.proxy_proc.terminate()
        try:
            rt.proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rt.proxy_proc.kill()
    rt.proxy_proc = None
