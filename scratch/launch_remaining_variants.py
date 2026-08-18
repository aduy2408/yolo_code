import os
import re
import subprocess
import sys
from pathlib import Path
import marimo._code_mode as cm

async def main():
    # 1. Extract HF_TOKEN
    token = None
    async with cm.get_context() as ctx:
        for cell in ctx.cells:
            match = re.search(r"HF_TOKEN\s*=\s*[\"']([^\"']+)[\"']", cell.code)
            if match:
                token = match.group(1)
                break
                
    if not token:
        print("ERROR: Could not find HF_TOKEN.")
        sys.exit(1)
        
    yolo_code_root = Path("/marimo/yolo_code")
    run_dir = yolo_code_root / "runs/levir_yolov8n_p2_interaction"
    
    # 2. Stop existing training process
    pid_path = run_dir / "train.pid"
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
            print(f"Stopping existing training process group for PID: {old_pid}")
            os.killpg(os.getpgid(old_pid), 15)
            print("Process terminated.")
        except Exception as e:
            print(f"Process already stopped or error: {e}")
        finally:
            pid_path.unlink(missing_ok=True)
            
    # Clean up runs directory to start fresh
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Launch training for C2, C3, C4
    cmd = [
        "python3",
        "train_all_levir_yolov8n_p2_interaction.py",
        "--data-root", "/marimo/LevirShipData",
        "--dataset-root", "/marimo/datasets",
        "--project", str(run_dir),
        "--epochs", "100",
        "--device", "0",
        "--variants", "c2_agreement", "c3_polarity", "c4_rank4",
        "--hf-repo-id", "duyle2408/levir-yolov8n-p2-interaction-seed42"
    ]
    
    print("Launching remaining variants sequentially (C2, C3, C4):")
    print(" ".join(cmd))
    
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    
    log_path = run_dir / "train.log"
    log_file = open(log_path, "w", encoding="utf-8")
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(yolo_code_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp
    )
    
    pid_path.write_text(f"{proc.pid}\n")
    print(f"Remaining variants training started in background with PID: {proc.pid}")

await main()
