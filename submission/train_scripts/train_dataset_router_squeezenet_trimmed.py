#!/usr/bin/env python3
"""Trim the final SqueezeNet Fire blocks and fine-tune a 3-class router."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from train_dataset_router_mobilenet import CLASSES, MEAN, STD, Letterbox, ImageSet, metric, records

class TrimmedSqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        base=squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        self.features=nn.Sequential(*list(base.features.children())[:11])
        self.classifier=nn.Sequential(nn.Dropout(.5),nn.Conv2d(384,len(CLASSES),1),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d((1,1)))
    def forward(self,x): return torch.flatten(self.classifier(self.features(x)),1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=6); ap.add_argument('--max-per-class',type=int,default=1000); ap.add_argument('--input-size',type=int,default=128); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); rec=records(args.data_root,args.seed); n=min(args.max_per_class,min(len(rec[c]['train']) for c in CLASSES))
    for c in CLASSES: rec[c]['train']=rec[c]['train'][:n]; rec[c]['val']=rec[c]['val'][:500]; rec[c]['test']=rec[c]['test'][:500]
    train_tf=T.Compose([Letterbox(args.input_size),T.RandomHorizontalFlip(),T.ColorJitter(.20,.20,.20,.05),T.ToTensor(),T.Normalize(MEAN,STD)]); eval_tf=T.Compose([Letterbox(args.input_size),T.ToTensor(),T.Normalize(MEAN,STD)])
    def make(s,t): return ImageSet([(p,i) for i,c in enumerate(CLASSES) for p in rec[c][s]],t)
    loader=DataLoader(make('train',train_tf),64,shuffle=True,num_workers=0,pin_memory=True); vl=DataLoader(make('val',eval_tf),128,shuffle=False,num_workers=0); tl=DataLoader(make('test',eval_tf),128,shuffle=False,num_workers=0); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=TrimmedSqueezeNet().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4); loss=nn.CrossEntropyLoss(label_smoothing=.05); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; count=0
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); z=loss(model(x),y); z.backward(); opt.step(); total+=z.detach().item()*len(y); count+=len(y)
        val=metric(model,vl,dev); row={'epoch':epoch,'train_loss':total/count,'val':val}; history.append(row); print(row)
    test=metric(model,tl,dev); meta={'classes':list(CLASSES),'architecture':'Trimmed-SqueezeNet1.1','removed_blocks':['fire11','fire12'],'input_size':args.input_size,'resize':'letterbox','mean':list(MEAN),'std':list(STD),'seed':args.seed,'train_per_class':n,'history':history,'test':test,'nms_iou':0.5}; args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'metadata':meta},args.output); args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({'artifact':str(args.output),'parameters':sum(p.numel() for p in model.parameters()),'test':test},indent=2))
if __name__=='__main__': main()
