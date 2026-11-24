#!/usr/bin/env python3
"""Improved lightweight three-dataset router using pretrained MobileNetV3-Small."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf

CLASSES = ("varroa", "levirship", "tinyperson")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def files(root): return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
def bucket(p): return int(hashlib.sha1(str(p).encode()).hexdigest()[:8], 16) / 2**32

def records(root, seed):
    r = {c: {s: [] for s in ("train", "val", "test")} for c in CLASSES}
    for s in r["varroa"]: r["varroa"][s] = files(root / "Varroa" / s)
    for p in files(root / "LevirShip"):
        b = bucket(p); r["levirship"]["train" if b < .70 else ("val" if b < .85 else "test")].append(p)
    for p in files(root / "TinyPerson"):
        if "test" in p.parts: r["tinyperson"]["test"].append(p)
        elif "train" in p.parts: r["tinyperson"]["val" if bucket(p) >= .85 else "train"].append(p)
    for c in CLASSES:
        for s in r[c]: random.Random(seed + sum(map(ord, c + s))).shuffle(r[c][s])
    return r

class Letterbox:
    def __init__(self, size, fill=(114, 114, 114)): self.size, self.fill = size, fill
    def __call__(self, image):
        scale = min(self.size / image.width, self.size / image.height)
        nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

class ImageSet(Dataset):
    def __init__(self, items, transform): self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        try: im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        except Exception: im = Image.new("RGB", (224, 224))
        return self.transform(im), y

def metric(model, loader, device):
    model.eval(); ys=[]; ps=[]; cs=[]
    with torch.inference_mode():
        for x,y in loader:
            q=model(x.to(device)).softmax(1); ys += y.tolist(); ps += q.argmax(1).cpu().tolist(); cs += q.max(1).values.cpu().tolist()
    cm=[[sum(a==i and b==j for a,b in zip(ys,ps)) for j in range(3)] for i in range(3)]
    rec=[cm[i][i]/max(1,sum(cm[i])) for i in range(3)]
    return {"accuracy":sum(a==b for a,b in zip(ys,ps))/len(ys),"per_class_recall":dict(zip(CLASSES,rec)),"confusion_matrix":cm,"mean_confidence":sum(cs)/len(cs),"n":len(ys)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--max-per-class",type=int,default=1600); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--width-mult",type=float,default=1.0); args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    r=records(args.data_root,args.seed); n=min(args.max_per_class,min(len(r[c]["train"]) for c in CLASSES))
    for c in CLASSES:
        r[c]["train"]=r[c]["train"][:n]; r[c]["val"]=r[c]["val"][:500]; r[c]["test"]=r[c]["test"][:500]
        print(c,{s:len(r[c][s]) for s in r[c]})
    size=224
    train_tf=T.Compose([Letterbox(size),T.RandomHorizontalFlip(),T.ColorJitter(.20,.20,.20,.05),T.ToTensor(),T.Normalize(MEAN,STD)])
    eval_tf=T.Compose([Letterbox(size),T.ToTensor(),T.Normalize(MEAN,STD)])
    def make(s,t): return ImageSet([(p,i) for i,c in enumerate(CLASSES) for p in r[c][s]],t)
    tr=make("train",train_tf); va=make("val",eval_tf); te=make("test",eval_tf)
    loader=DataLoader(tr,64,shuffle=True,num_workers=0,pin_memory=True); vl=DataLoader(va,128,shuffle=False,num_workers=0); tl=DataLoader(te,128,shuffle=False,num_workers=0)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.width_mult == 1.0:
        weights=MobileNet_V3_Small_Weights.DEFAULT
        model=mobilenet_v3_small(weights=weights)
    else:
        config,last_channel=_mobilenet_v3_conf("mobilenet_v3_small",width_mult=args.width_mult)
        model=MobileNetV3(config,last_channel,num_classes=1000)
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,3); model.to(dev)
    if args.width_mult == 1.0:
        for p in model.features.parameters(): p.requires_grad=False
        opt=torch.optim.AdamW(model.classifier.parameters(),lr=2e-3,weight_decay=1e-4)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss=nn.CrossEntropyLoss(label_smoothing=.05); hist=[]
    for epoch in range(1,args.epochs+1):
        if args.width_mult == 1.0 and epoch==3:
            for p in model.features.parameters(): p.requires_grad=True
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        model.train(); total=0.; count=0
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); z=loss(model(x),y); z.backward(); opt.step(); total+=z.detach().item()*len(y); count+=len(y)
        m=metric(model,vl,dev); row={"epoch":epoch,"train_loss":total/count,"val":m}; hist.append(row); print(row)
    test=metric(model,tl,dev); meta={"classes":list(CLASSES),"architecture":"MobileNetV3-Small","width_mult":args.width_mult,"input_size":224,"resize":"letterbox","mean":list(MEAN),"std":list(STD),"seed":args.seed,"train_per_class":n,"history":hist,"test":test,"nms_iou":0.5,"split_note":"LEVIR uses deterministic hash split; production should use scene-disjoint groups."}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"metadata":meta},args.output); args.output.with_suffix(".json").write_text(json.dumps(meta,indent=2)); print(json.dumps({"artifact":str(args.output),"parameters":sum(p.numel() for p in model.parameters()),"test":test},indent=2))
if __name__=="__main__": main()
