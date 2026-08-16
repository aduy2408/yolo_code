import marimo._code_mode as cm

async def add_launch_cell():
    async with cm.get_context() as ctx:
        cell_code = """
import marimo as mo

def _launch_verifiers():
    import sys as _sys
    import subprocess
    import os
    import re
    from huggingface_hub import HfApi as _HfApi
    from pathlib import Path

    # Parse HF_TOKEN dynamically from notebook.py to avoid hardcoding credentials
    token_re = re.compile(r'HF_TOKEN\s*=\s*"(hf_[A-Za-z0-9]+)"')
    try:
        with open("/marimo/notebook.py", "r", encoding="utf-8") as f:
            content = f.read()
        match = token_re.search(content)
        if not match:
            raise ValueError("Could not find HF_TOKEN pattern in notebook.py")
        HF_TOKEN = match.group(1)
    except Exception as e:
        # Fallback to env or print error
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        if not HF_TOKEN:
            raise RuntimeError(f"Failed to read HF_TOKEN from notebook.py: {e}")

    ROOT = Path("/marimo/yolo_code")
    _run_root = ROOT / "runs/levir_yolov8n_p2_gap_ftal_verifiers"
    _run_root.mkdir(parents=True, exist_ok=True)
    _pid_path = _run_root / "train.pid"
    _log_path = _run_root / "train.log"
    if _pid_path.is_file():
        _old_pid = int(_pid_path.read_text().strip())
        if __import__("pathlib").Path(f"/proc/{_old_pid}").exists():
            return {"status": "already running", "pid": _old_pid, "log": str(_log_path)}
            
    _command = [
        _sys.executable, "-u", "train_levir_scripts/train_all_levir_verifier.py",
        "--data-root", "/marimo/LevirShipData",
        "--dataset-root", str(ROOT / "datasets"),
        "--project", str(_run_root),
        "--pretrained", "yolov8n.pt",
        "--epochs", "100",
        "--imgsz", "512",
        "--batch-size", "8",
        "--device", "cuda",
        "--workers", "4",
        "--patience", "20",
        "--hf-repo-id", "duyle2408/levir-yolov8n-p2-gap-ftal-verifiers-seed42",
        "--verifier-alpha", "0.0",
        "--verifier-loss-gain", "0.5",
    ]
    _env = os.environ.copy()
    _env["HF_TOKEN"] = HF_TOKEN
    with _log_path.open("a", encoding="utf-8") as _log:
        _process = subprocess.Popen(
            _command, cwd=ROOT, env=_env, stdout=_log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    _pid_path.write_text(str(_process.process.pid if hasattr(_process, "process") else _process.pid))
    return {"status": "launched", "pid": _process.pid, "log": str(_log_path)}

verifiers_launch = _launch_verifiers()
mo.status.toast(f"🧪 Verifiers {verifiers_launch['status']} — PID {verifiers_launch['pid']}")
mo.md(f'''### Box-Conditioned Verifiers (A1, A3, A4) Matrix

- Status: **{verifiers_launch['status']}**
- PID: `{verifiers_launch['pid']}`
- Log: `{verifiers_launch['log']}`
- Order: `a1_box_fovea`, `a3_semantic_structural`, `a4_raw_adapted`
- Epochs: `100`; seed: `42`
''')
"""
        # Delete old cell first to prevent redefinition errors
        if 'launch_verifiers' in ctx.cells:
            ctx.delete_cell(ctx.cells['launch_verifiers'].id)
        cid = ctx.create_cell(cell_code, name="launch_verifiers", hide_code=False)
        ctx.run_cell(cid)
        print("Cell created and run with ID:", cid)

await add_launch_cell()
