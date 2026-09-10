"""Run and archive one Colab clip with bounded notebook output and live logs."""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--drive", type=Path, default=Path("/content/drive/MyDrive/ezstats"))
    args = parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. In Colab select Runtime > Change runtime type > T4 GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; starting {args.video}", flush=True)
    repo = Path(__file__).resolve().parents[1]
    source = repo / "data/raw" / args.video
    if not source.is_file():
        raise FileNotFoundError(source)
    # Repeated video seeks against mounted Drive can stall even with a fast GPU.
    inputs = repo / "outputs/inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    video = inputs / source.name
    if not video.exists() or video.stat().st_size != source.stat().st_size:
        print(f"Copying {source.name} to local runtime storage...", flush=True)
        shutil.copy2(source, video)
    logs = repo / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{datetime.now():%Y%m%d_%H%M%S}_{source.stem}.log"
    messages = queue.Queue()
    proc = subprocess.Popen([sys.executable, "-u", str(repo / "run_pipeline_v3.py"),
                             "--video", str(video), "--device", "0"], cwd=repo, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt")

    def read_output():
        try:
            for line in proc.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    run_dir = None
    last_print = started = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as stream:
            while True:
                try:
                    line = messages.get(timeout=20)
                except queue.Empty:
                    print(f"Still running {source.name}: {(time.monotonic() - started) / 60:.1f} min; log: {log.name}", flush=True)
                    continue
                if line is None:
                    break
                stream.write(line)
                stream.flush()
                if line.startswith("Run dir:"):
                    run_dir = Path(line.split(":", 1)[1].strip())
                progress = "%|" in line or "it/s" in line or "s/it" in line
                if not progress or time.monotonic() - last_print >= 20:
                    print(line, end="", flush=True)
                    last_print = time.monotonic()
        proc.wait()
    finally:
        if proc.poll() is None:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()
        reader.join(timeout=5)
        destination = args.drive / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, destination / log.name)
        if run_dir is not None and run_dir.is_dir():
            print(f"Saving {run_dir.name} to Drive...", flush=True)
            shutil.copytree(run_dir, destination / run_dir.name, dirs_exist_ok=True)
            print(f"Saved: {destination / run_dir.name}", flush=True)
    if proc.returncode:
        raise SystemExit(f"Run failed (exit {proc.returncode}); full log: {log}")
    print(f"Finished {source.name}", flush=True)


if __name__ == "__main__":
    main()
