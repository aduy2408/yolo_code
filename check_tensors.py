from ultralytics import YOLO
import torch

model = YOLO("yolov8x.pt")
csd = model.model.state_dict()
layers_in_csd = {}
for k in csd.keys():
    layer_idx = int(k.split('.')[1])
    layers_in_csd[layer_idx] = layers_in_csd.get(layer_idx, 0) + 1

for l in sorted(layers_in_csd.keys()):
    print(f"Layer {l}: {layers_in_csd[l]} tensors")
print(f"Total tensors: {sum(layers_in_csd.values())}")
