#!/usr/bin/env python3
"""Train, evaluate, and upload the global-plus-object OAIEF correction."""
from __future__ import annotations
import argparse,json,os,random,subprocess,sys
from pathlib import Path
from misc.prepare_levir_ship import prepare
from utils.marimo_ops import require_training_context
ROOT=Path(__file__).resolve().parents[1]; ULTRALYTICS=ROOT/"models_related/ultralytics"; CONFIG=ROOT/"models_related/models_config/yolov8/levir/yolov8n_p2_levir_oaief_global_object.yaml"
TRAIN_REQUIRED=("weights/best.pt","weights/last.pt","results.csv"); COMPLETE_REQUIRED=(*TRAIN_REQUIRED,"evaluation_metrics.json","manifest.json")
def local(): sys.path.insert(0,str(ULTRALYTICS))
def has(p,files): return all((p/f).is_file() for f in files)
def seed(s):
 import numpy as np,torch
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def evaluate(p,data,a):
 local(); from ultralytics import YOLO
 model=YOLO(p/"weights/best.pt"); out={}
 for split in ("val","test"):
  r=model.val(data=str(data),split=split,imgsz=a.imgsz,batch=a.batch_size,device=a.device,workers=a.workers,plots=False,iou=0.5,project=str(p/"evaluation"),name=split,exist_ok=True)
  out.update({f"{split}/{k}":float(v) for k,v in (r.results_dict or {}).items()}); out[f"{split}/AP75"]=float(r.box.map75)
 (p/"evaluation_metrics.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def upload(p,repo):
 from huggingface_hub import HfApi
 api=HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(repo_id=repo,repo_type="dataset",private=False,exist_ok=True); prefix="runs/M4_GLOBAL_OBJECT"
 api.upload_folder(folder_path=str(p),path_in_repo=prefix,repo_id=repo,repo_type="dataset")
 remote={x.rfilename for x in api.list_repo_tree(repo_id=repo,repo_type="dataset",path_in_repo=prefix,recursive=True) if hasattr(x,"rfilename")}; missing={f"{prefix}/{x}" for x in COMPLETE_REQUIRED}-remote
 if missing: raise RuntimeError(f"Remote upload verification failed: {sorted(missing)}")
 (p/"upload_complete.json").write_text(json.dumps({"repo_id":repo,"remote_prefix":prefix},indent=2)+"\n")
def run(a):
 require_training_context(hf_repo_id=a.hf_repo_id); data=prepare(a.data_root,a.dataset_root/f"levir_ship_yolo_seed{a.seed}",a.seed); p=a.project/"M4_GLOBAL_OBJECT"; p.mkdir(parents=True,exist_ok=True); local(); from ultralytics import YOLO
 sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); (p/"manifest.json").write_text(json.dumps({"experiment":"M4_GLOBAL_OBJECT","config":str(CONFIG),"commit_sha":sha,"seed":a.seed,"split":["val","test"],"nms_iou":0.5,"epochs":a.epochs,"patience":a.patience,"modulation_alpha":0.5,"router":"detach(source_prob)","hf_repo_id":a.hf_repo_id},indent=2)+"\n")
 if not has(p,TRAIN_REQUIRED):
  seed(a.seed); model=YOLO(str(CONFIG)); model.load("yolov8n.pt",smart_transfer=True); model.train(data=str(data),epochs=a.epochs,patience=a.patience,imgsz=a.imgsz,batch=a.batch_size,device=a.device,workers=a.workers,amp=a.amp,seed=a.seed,deterministic=True,project=str(a.project),name="M4_GLOBAL_OBJECT",exist_ok=True)
 if not has(p,TRAIN_REQUIRED): raise FileNotFoundError(p)
 if not (p/"evaluation_metrics.json").is_file(): evaluate(p,data,a)
 if not has(p,COMPLETE_REQUIRED): raise FileNotFoundError(p)
 upload(p,a.hf_repo_id); print("COMPLETE M4_GLOBAL_OBJECT",flush=True)
def args():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--project",type=Path,required=True); p.add_argument("--hf-repo-id",required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=100); p.add_argument("--patience",type=int,default=0); p.add_argument("--imgsz",type=int,default=512); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--device",default="0"); p.add_argument("--workers",type=int,default=4); p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True); return p.parse_args()
if __name__=="__main__": run(args())
