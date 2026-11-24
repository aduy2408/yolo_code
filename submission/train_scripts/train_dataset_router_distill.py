#!/usr/bin/env python3
"""Distill the full router into a MobileNetV3-Small width-0.5 student."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from train_dataset_router_mobilenet import CLASSES, MEAN, STD, Letterbox, ImageSet, MobileNet_V3_Small_Weights, files, metric, mobilenet_v3_small, records
import torchvision.transforms as T

def student_model(width_mult):
    config, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", width_mult=width_mult)
    model = MobileNetV3(config, last_channel, num_classes=1000)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    return model

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--teacher", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--max-per-class", type=int, default=1600); ap.add_argument("--temperature", type=float, default=4.0); ap.add_argument("--width-mult", type=float, default=0.5); ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    root = args.data_root; rec = records(root, args.seed); n = min(args.max_per_class, min(len(rec[c]["train"]) for c in CLASSES))
    for c in CLASSES: rec[c]["train"] = rec[c]["train"][:n]; rec[c]["val"] = rec[c]["val"][:500]; rec[c]["test"] = rec[c]["test"][:500]
    train_tf = T.Compose([Letterbox(224), T.RandomHorizontalFlip(), T.ColorJitter(.20,.20,.20,.05), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([Letterbox(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    def make(split, tf): return ImageSet([(p, i) for i,c in enumerate(CLASSES) for p in rec[c][split]], tf)
    loader = DataLoader(make("train", train_tf), 64, shuffle=True, num_workers=0, pin_memory=True); val_loader = DataLoader(make("val", eval_tf), 128, shuffle=False, num_workers=0); test_loader = DataLoader(make("test", eval_tf), 128, shuffle=False, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = mobilenet_v3_small(weights=None); teacher.classifier[3] = nn.Linear(teacher.classifier[3].in_features, len(CLASSES)); teacher.load_state_dict(torch.load(args.teacher, map_location="cpu")["model"]); teacher.to(dev).eval()
    student = student_model(args.width_mult).to(dev); optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4); ce = nn.CrossEntropyLoss(label_smoothing=.05); t = args.temperature; history=[]
    for epoch in range(1, args.epochs + 1):
        student.train(); total=0.; count=0
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): teacher_logits=teacher(x)
            student_logits=student(x); hard=ce(student_logits,y); soft=F.kl_div(F.log_softmax(student_logits/t,1),F.softmax(teacher_logits/t,1),reduction="batchmean")*(t*t); loss=.5*hard+.5*soft; loss.backward(); optimizer.step(); total += loss.detach().item()*len(y); count += len(y)
        val=metric(student,val_loader,dev); row={"epoch":epoch,"train_loss":total/count,"val":val}; history.append(row); print(row)
    test=metric(student,test_loader,dev); meta={"classes":list(CLASSES),"architecture":"MobileNetV3-Small-distilled","width_mult":args.width_mult,"input_size":224,"resize":"letterbox","mean":list(MEAN),"std":list(STD),"seed":args.seed,"temperature":t,"train_per_class":n,"history":history,"test":test,"nms_iou":0.5}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":student.state_dict(),"metadata":meta},args.output); args.output.with_suffix(".json").write_text(json.dumps(meta,indent=2)); print(json.dumps({"artifact":str(args.output),"parameters":sum(p.numel() for p in student.parameters()),"test":test},indent=2))
if __name__ == "__main__": main()
