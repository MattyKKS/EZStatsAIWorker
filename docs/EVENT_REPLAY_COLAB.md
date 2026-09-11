# Event replay, without another detection run

This is an experimental ground-pass revision, not a complete football event
model. Goals, shots, crosses, assists and aerial contacts are unsupported. The
old pipeline remains the default; compare outputs before adopting this engine.
Neither detection nor player identities is changed by replay.

Use the existing v3 Colab setup and mounted Drive. Let any current analysis
finish before updating the repository. Run this setup cell once:

```python
%cd /content/EZStatsAIWorker
import subprocess, runpy
subprocess.run(["git", "pull", "--ff-only", "origin", "test"], check=True)
replay = runpy.run_path("scripts/replay_events.py")["main"]
drive = "/content/drive/MyDrive/ezstats/outputs"
```

## Messi, first separate cell

```python
messi_result = replay([
    "--run-dir", f"{drive}/20260910_170402",
    "--copy-to", drive,
])
```

## 08, second separate cell

```python
benchmark_result = replay([
    "--run-dir", f"{drive}/20260910_173839",
    "--copy-to", drive,
])
```

## Brighton, third separate cell

```python
brighton_result = replay([
    "--run-dir", f"{drive}/20260910_174331",
    "--copy-to", drive,
])
```

Each cell computes events locally, renders `stats_video.mp4`, then copies a new
`*_contacts_*` folder to Drive. Original runs are not overwritten. JSON replay
does not need a GPU; rendering is CPU work. A T4 remains useful for full detection.

If the runtime has restarted and the original `/content/...` video is gone,
add `"--video", "/actual/path/to/the/source.mp4"` to that cell's argument list.
Use the identical source clip, not a differently trimmed version. Add
`"--skip-video"` to inspect JSON without loading any source video.

Read `event_evidence.json` for accepted/rejected contacts and transfer reasons.
`ball_transfer` deliberately does NOT mean confirmed interception. Missing goals
in this mode mean **unsupported**, not evidence that the clip has no goal.

Full new inference with the experimental engine, only when needed:

```python
!python -u run_pipeline_v3.py --video data/raw/leo_messi_30pass.mp4 --device 0 --event-engine contacts
```

Do not retrain detectors merely because event counts are wrong. Evaluate missed
ball/player boxes separately from ownership, identity and event errors.
