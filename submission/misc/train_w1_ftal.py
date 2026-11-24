#!/usr/bin/env python3
"""Train, evaluate, and upload W1 with canonical FTAL."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys
from pathlib import Path
from misc.prepare_levir_ship import prepare
from utils.marimo_ops import require_training_context
ROOT=Path(__file__).resolve().parents[1]; ULTRALYTICS=ROOT/"models_related/ultralytics"
CONFIG=ROOT/"models_related/models_config/yolov8/levir/yolov8n_p2_levir_oaief_w1.yaml"
TRAIN_REQUIRED=("weights/best.pt","weights/last.pt","results.csv")
COMPLETE_REQUIRED=(*TRAIN_REQUIRED,"evaluation_metrics.json","manifest.json")
FTAL={"factorized_tal_target":True,"factorized_tal_mode":"legacy","factorized_tal_tau":0.75,"factorized_tal_kappa":1.5,"factorized_tal_lambda":0.5,"factorized_tal_s_max":32.0,"factorized_tal_warmup_start":5,"factorized_tal_warmup_end":15,"factorized_tal_p2_only":True}
def local(): sys.path.insert(0,str(ULTRALYTICS))
def has(p,files): return all((p/f).is_file() for f in files)
def seed(s):
 import numpy as np, torch
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def evaluate(run_dir,data,a):
 local(); from ultralytics import YOLO
 model=YOLO(run_dir/"weights/best.pt"); metrics={}
 for split in ("val","test"):
  r=model.val(data=str(data),split=split,imgsz=a.imgsz,batch=a.batch_size,device=a.device,workers=a.workers,plots=False,iou=0.5,project=str(run_dir/"evaluation"),name=split,exist_ok=True)
  metrics.update({f"{split}/{k}":float(v) for k,v in (r.results_dict or {}).items()}); metrics[f"{split}/AP75"]=float(r.box.map75)
 (run_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
def upload(run_dir,repo_id):
 from huggingface_hub import HfApi
 api=HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(repo_id=repo_id,repo_type="dataset",private=False,exist_ok=True); prefix="runs/W1_FTAL"
 api.upload_folder(folder_path=str(run_dir),path_in_repo=prefix,repo_id=repo_id,repo_type="dataset")
 remote={x.rfilename for x in api.list_repo_tree(repo_id=repo_id,repo_type="dataset",path_in_repo=prefix,recursive=True) if hasattr(x,"rfilename")}
 missing={f"{prefix}/{x}" for x in COMPLETE_REQUIRED}-remote
 if missing: raise RuntimeError(f"Remote upload verification failed: {sorted(missing)}")
 (run_dir/"upload_complete.json").write_text(json.dumps({"repo_id":repo_id,"remote_prefix":prefix},indent=2)+"\n")
def run(a):
 require_training_context(hf_repo_id=a.hf_repo_id); data=prepare(a.data_root,a.dataset_root/f"levir_ship_yolo_seed{a.seed}",a.seed); run_dir=a.project/"W1_FTAL"; run_dir.mkdir(parents=True,exist_ok=True); local(); from ultralytics import YOLO
 sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); (run_dir/"manifest.json").write_text(json.dumps({"experiment":"W1_FTAL","config":str(CONFIG),"commit_sha":sha,"seed":a.seed,"split":["val","test"],"nms_iou":0.5,"epochs":a.epochs,"patience":a.patience,"ftal":FTAL,"hf_repo_id":a.hf_repo_id},indent=2)+"\n")
 if not has(run_dir,TRAIN_REQUIRED):
  seed(a.seed); model=YOLO(str(CONFIG)); model.load("yolov8n.pt",smart_transfer=True); model.train(data=str(data),epochs=a.epochs,patience=a.patience,imgsz=a.imgsz,batch=a.batch_size,device=a.device,workers=a.workers,amp=a.amp,seed=a.seed,deterministic=True,project=str(a.project),name="W1_FTAL",exist_ok=True,**FTAL)
 if not has(run_dir,TRAIN_REQUIRED): raise FileNotFoundError(run_dir)
 if not (run_dir/"evaluation_metrics.json").is_file(): evaluate(run_dir,data,a)
 if not has(run_dir,COMPLETE_REQUIRED): raise FileNotFoundError(run_dir)
 upload(run_dir,a.hf_repo_id); print("COMPLETE W1_FTAL",flush=True)
def args():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--project",type=Path,required=True); p.add_argument("--hf-repo-id",required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=100); p.add_argument("--patience",type=int,default=0); p.add_argument("--imgsz",type=int,default=512); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--device",default="0"); p.add_argument("--workers",type=int,default=4); p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True); return p.parse_args()
if __name__=="__main__": run(args())
