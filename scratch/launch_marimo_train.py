import os
import re
import subprocess
import sys
from pathlib import Path
import marimo._code_mode as cm

async def main():
    # 1. Extract HF_TOKEN from cells
    token = None
    async with cm.get_context() as ctx:
        for cell in ctx.cells:
            match = re.search(r"HF_TOKEN\s*=\s*[\"']([^\"']+)[\"']", cell.code)
            if match:
                token = match.group(1)
                break
                
    if not token:
        print("ERROR: Could not find HF_TOKEN in notebook cells.")
        sys.exit(1)
        
    yolo_code_root = Path("/marimo/yolo_code")
    run_dir = yolo_code_root / "runs/levir_yolov8n_p2_interaction"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    pid_path = run_dir / "train.pid"
    log_path = run_dir / "train.log"
    
    # 2. Check duplicate process
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
            # Check if running
            os.kill(old_pid, 0)
            print(f"ERROR: A training run is already running under PID {old_pid}")
            sys.exit(1)
        except (ValueError, OSError):
            # Process is not running, clean up old PID file
            pid_path.unlink(missing_ok=True)
            
    # 3. Launch training
    cmd = [
        "python3",
        "train_all_levir_yolov8n_p2_interaction.py",
        "--data-root", "/marimo/LevirShipData",
        "--dataset-root", "/marimo/datasets",
        "--project", str(run_dir),
        "--epochs", "100",
        "--device", "0",
        "--hf-repo-id", "duyle2408/levir-yolov8n-p2-interaction-seed42"
    ]
    
    print("Launching detached training command:")
    print(" ".join(cmd))
    
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(yolo_code_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp # detach process
    )
    
    # Write PID
    pid_path.write_text(f"{proc.pid}\n")
    print(f"Training started successfully in background with PID: {proc.pid}")
    print(f"Logs are being written to: {log_path}")

await main()
