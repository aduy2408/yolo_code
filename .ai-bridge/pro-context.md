# YOLOv8 local block inspection

Generated: 2026-07-06T08:57:34.014Z
Workspace: /mnt/data/varroa/yolo_related
Workspace ID: ws_9d5e8cd2d1d155051e793f22
Write mode: workspace
Bash mode: safe
Tool mode: standard

Purpose: paste this bundle into a high-context ChatGPT model when that model cannot call the CodexPro MCP tools directly.
Instruction for ChatGPT: use this as repository context, produce a narrow Codex execution plan, and avoid inventing files or runtime facts not shown here.

## Repository Tree

.
├── __pycache__/
├── baseline_reuslts/
├── data/
├── datasets/
├── misc/
├── models_related/
├── runs/
├── SKILLS/
├── SR-TOD/
├── test_results/
├── Understand-Anything/
├── __init__.py
├── =
├── bee_gt_grid_3x4_close.png
├── best_api_pooledge_p2p3_vlf.pt
├── best_api_pooledge_p2p3.pt
├── best_p2_poolingEdge.pt
├── BOUNDARY_API_COLLAPSE_NOTES.md
├── cascade_rcnn_varroa.py
├── DATASET_MODEL_NOTES.md
├── KVCA_SWEEP_NOTES.md
├── quality_debug_predictions-1.csv
├── quality_debug_predictions.csv
├── quality_debug_summary.json
├── quality_debug_summary(1).json
├── README.md
├── skills-lock.json
├── VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md
├── VARROA_POOLING_BOX_ERROR_DIAGNOSTIC.md
├── yolo26n.pt
├── yolov8n.pt
└── yolov8x.pt

## Git Status

```text
## main...origin/main
?? models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2p3_edge_pooling_srdgfe_noapi.yaml
?? models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_srdgfe_noapi.yaml
```

## Recent Commits

```text
7f3dc63 (HEAD -> main, origin/main) .
ee31d46 .
dcb18c2 .
4295925 .
b747cae .
182a53d .
addde0f .
845d8a7 .
```

## Selected Files

Changed files detected: models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2p3_edge_pooling_srdgfe_noapi.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_srdgfe_noapi.yaml
Auto-include important root files: no
Auto-include changed files: no
Explicit selected paths: models_related/ultralytics/ultralytics/nn/modules/block.py, models_related/ultralytics/ultralytics/nn/modules/__init__.py, models_related/ultralytics/ultralytics/nn/tasks.py
Extra globs: none
Files included below: models_related/ultralytics/ultralytics/nn/modules/__init__.py, models_related/ultralytics/ultralytics/nn/modules/block.py, models_related/ultralytics/ultralytics/nn/tasks.py

## File Contents

### models_related/ultralytics/ultralytics/nn/modules/__init__.py

Bytes: 4580
SHA-256: 207cd5d826a223ea1cb82a3c46da94f21ea54d7c6d836f1792b45ad49c7fb189
Lines: 1-245 of 245

```python
  1 | # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
  2 | """
  3 | Ultralytics neural network modules.
  4 | 
  5 | This module provides access to various neural network components used in Ultralytics models, including convolution
  6 | blocks, attention mechanisms, transformer components, and detection/segmentation heads.
  7 | 
  8 | Examples:
  9 |     Visualize a module with Netron
 10 |     >>> from ultralytics.nn.modules import Conv
 11 |     >>> import torch
 12 |     >>> import subprocess
 13 |     >>> x = torch.ones(1, 128, 40, 40)
 14 |     >>> m = Conv(128, 128)
 15 |     >>> f = f"{m._get_name()}.onnx"
 16 |     >>> torch.onnx.export(m, x, f)
 17 |     >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
 18 | """
 19 | 
 20 | from .block import (
 21 |     C1,
 22 |     C2,
 23 |     C2PSA,
 24 |     C3,
 25 |     C3TR,
 26 |     CIB,
 27 |     DFL,
 28 |     ELAN1,
 29 |     PSA,
 30 |     SPP,
 31 |     SPPELAN,
 32 |     SPPF,
 33 |     A2C2f,
 34 |     AConv,
 35 |     ADown,
 36 |     AdversarialPerturbationInjection,
 37 |     ASFAttention,
 38 |     Attention,
 39 |     BiLevelRoutingAttention,
 40 |     BNContrastiveHead,
 41 |     BoundaryFeatureBlock,
 42 |     Bottleneck,
 43 |     BottleneckCSP,
 44 |     C2f,
 45 |     C2fCBAM,
 46 |     C2fAttn,
 47 |     C2fCIB,
 48 |     C2fKV,
 49 |     C2fNAT,
 50 |     C2fPSA,
 51 |     EnSimAM,
 52 |     EnSimAMEdgeRepC2f,
 53 |     FeatureDGFE,
 54 |     C3CBAM,
 55 |     C3Ghost,
 56 |     C3k2,
 57 |     C3x,
 58 |     CBFuse,
 59 |     CBLinear,
 60 |     ContrastiveHead,
 61 |     GhostBottleneck,
 62 |     HGBlock,
 63 |     HGStem,
 64 |     ImagePoolingAttn,
 65 |     KVCompressedAttention,
 66 |     KVCompressedAttentionPartial,
 67 |     KVCompressedTransformerEncoder,
 68 |     M3NATFuse,
 69 |     NATBlock,
 70 |     MaxSigmoidAttnBlock,
 71 |     Proto,
 72 |     RegionRoutingAttentionLite,
 73 |     RepC3,
 74 |     RepC2f,
 75 |     PoolingEdgeRepC2f,
 76 |     RepNCSPELAN4,
 77 |     RepVGGDW,
 78 |     ResNetLayer,
 79 |     SCDown,
 80 |     ScalSeq,
 81 |     TorchVision,
 82 |     TopKAdaptiveGroupKVAttention,
 83 |     TopKGlobalGroupKVAttention,
 84 |     WeightedAdd,
 85 |     clear_boundary_context,
 86 |     set_boundary_context,
 87 |     set_boundary_enabled,
 88 | )
 89 | from .conv import (
 90 |     CBAM,
 91 |     ChannelAttention,
 92 |     Concat,
 93 |     Conv,
 94 |     Conv2,
 95 |     ConvTranspose,
 96 |     DWConv,
 97 |     DWConvTranspose2d,
 98 |     Focus,
 99 |     GhostConv,
100 |     Index,
101 |     LightConv,
102 |     RepConv,
103 |     SpatialAttention,
104 | )
105 | from .head import (
106 |     OBB,
107 |     OBB26,
108 |     Classify,
109 |     Detect,
110 |     LRPCHead,
111 |     Pose,
112 |     Pose26,
113 |     RTDETRDecoder,
114 |     Segment,
115 |     Segment26,
116 |     SemanticSegment,
117 |     WorldDetect,
118 |     YOLOEDetect,
119 |     YOLOESegment,
120 |     YOLOESegment26,
121 |     v10Detect,
122 | )
123 | from .transformer import (
124 |     AIFI,
125 |     MLP,
126 |     DeformableTransformerDecoder,
127 |     DeformableTransformerDecoderLayer,
128 |     LayerNorm2d,
129 |     MLPBlock,
130 |     MSDeformAttn,
131 |     TransformerBlock,
132 |     TransformerEncoderLayer,
133 |     TransformerLayer,
134 | )
135 | 
136 | __all__ = (
137 |     "AIFI",
138 |     "C1",
139 |     "C2",
140 |     "C2PSA",
141 |     "C3",
142 |     "C3TR",
143 |     "CBAM",
144 |     "CIB",
145 |     "DFL",
146 |     "ELAN1",
147 |     "MLP",
148 |     "OBB",
149 |     "OBB26",
150 |     "PSA",
151 |     "SPP",
152 |     "SPPELAN",
153 |     "SPPF",
154 |     "A2C2f",
155 |     "AConv",
156 |     "ADown",
157 |     "AdversarialPerturbationInjection",
158 |     "ASFAttention",
159 |     "Attention",
160 |     "BiLevelRoutingAttention",
161 |     "BNContrastiveHead",
162 |     "BoundaryFeatureBlock",
163 |     "Bottleneck",
164 |     "BottleneckCSP",
165 |     "C2f",
166 |     "C2fCBAM",
167 |     "C2fAttn",
168 |     "C2fCIB",
169 |     "C2fKV",
170 |     "C2fNAT",
171 |     "C2fPSA",
172 |     "EnSimAM",
173 |     "EnSimAMEdgeRepC2f",
174 |     "FeatureDGFE",
175 |     "C3CBAM",
176 |     "C3Ghost",
177 |     "C3k2",
178 |     "C3x",
179 |     "CBFuse",
180 |     "CBLinear",
181 |     "ChannelAttention",
182 |     "Classify",
183 |     "Concat",
184 |     "ContrastiveHead",
185 |     "Conv",
186 |     "Conv2",
187 |     "ConvTranspose",
188 |     "DWConv",
189 |     "DWConvTranspose2d",
190 |     "DeformableTransformerDecoder",
191 |     "DeformableTransformerDecoderLayer",
192 |     "Detect",
193 |     "Focus",
194 |     "GhostBottleneck",
195 |     "GhostConv",
196 |     "HGBlock",
197 |     "HGStem",
198 |     "ImagePoolingAttn",
199 |     "Index",
200 |     "KVCompressedAttention",
201 |     "KVCompressedAttentionPartial",
202 |     "KVCompressedTransformerEncoder",
203 |     "LRPCHead",
204 |     "LayerNorm2d",
205 |     "LightConv",
206 |     "M3NATFuse",
207 |     "NATBlock",
208 |     "MLPBlock",
209 |     "MSDeformAttn",
210 |     "MaxSigmoidAttnBlock",
211 |     "Pose",
212 |     "Pose26",
213 |     "Proto",
214 |     "RegionRoutingAttentionLite",
215 |     "RTDETRDecoder",
216 |     "RepC3",
217 |     "RepC2f",
218 |     "PoolingEdgeRepC2f",
219 |     "RepConv",
220 |     "RepNCSPELAN4",
221 |     "RepVGGDW",
222 |     "ResNetLayer",
223 |     "SCDown",
224 |     "ScalSeq",
225 |     "Segment",
226 |     "Segment26",
227 |     "SemanticSegment",
228 |     "SpatialAttention",
229 |     "TorchVision",
230 |     "TopKAdaptiveGroupKVAttention",
231 |     "TopKGlobalGroupKVAttention",
232 |     "TransformerBlock",
233 |     "TransformerEncoderLayer",
234 |     "TransformerLayer",
235 |     "WorldDetect",
236 |     "YOLOEDetect",
237 |     "YOLOESegment",
238 |     "YOLOESegment26",
239 |     "WeightedAdd",
240 |     "clear_boundary_context",
241 |     "set_boundary_context",
242 |     "set_boundary_enabled",
243 |     "v10Detect",
244 | )
245 | 
```

### models_related/ultralytics/ultralytics/nn/modules/block.py

Bytes: 144643
SHA-256: 81a2e9f5ffb25e5e689fcb7301c824d84246f73ce97dc331e4671d7e0f5ad0c7
Lines: 1-3710 of 3710

```python
   1 | # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
   2 | """Block modules."""
   3 | 
   4 | from __future__ import annotations
   5 | 
   6 | from dataclasses import dataclass
   7 | 
   8 | import torch
   9 | import torch.nn as nn
  10 | import torch.nn.functional as F
  11 | 
  12 | from ultralytics.utils.torch_utils import fuse_conv_and_bn
  13 | 
  14 | from .conv import CBAM, Conv, DWConv, GhostConv, LightConv, RepConv, autopad
  15 | from .transformer import LayerNorm2d, TransformerBlock
  16 | 
  17 | __all__ = (
  18 |     "C1",
  19 |     "C2",
  20 |     "C2PSA",
  21 |     "C3",
  22 |     "C3TR",
  23 |     "CIB",
  24 |     "DFL",
  25 |     "ELAN1",
  26 |     "PSA",
  27 |     "SPP",
  28 |     "SPPELAN",
  29 |     "SPPF",
  30 |     "AConv",
  31 |     "ADown",
  32 |     "AdversarialPerturbationInjection",
  33 |     "ASFAttention",
  34 |     "Attention",
  35 |     "BiLevelRoutingAttention",
  36 |     "BNContrastiveHead",
  37 |     "BoundaryFeatureBlock",
  38 |     "Bottleneck",
  39 |     "BottleneckCSP",
  40 |     "C2f",
  41 |     "C2fCBAM",
  42 |     "C2fAttn",
  43 |     "C2fCIB",
  44 |     "C2fKV",
  45 |     "C2fNAT",
  46 |     "C2fPSA",
  47 |     "EnSimAM",
  48 |     "EnSimAMEdgeRepC2f",
  49 |     "FeatureDGFE",
  50 |     "C3CBAM",
  51 |     "C3Ghost",
  52 |     "C3k2",
  53 |     "C3x",
  54 |     "CBFuse",
  55 |     "CBLinear",
  56 |     "ContrastiveHead",
  57 |     "GhostBottleneck",
  58 |     "HGBlock",
  59 |     "HGStem",
  60 |     "ImagePoolingAttn",
  61 |     "KVCompressedAttention",
  62 |     "KVCompressedAttentionPartial",
  63 |     "KVCompressedTransformerEncoder",
  64 |     "M3NATFuse",
  65 |     "Proto",
  66 |     "RegionRoutingAttentionLite",
  67 |     "TopKAdaptiveGroupKVAttention",
  68 |     "TopKGlobalGroupKVAttention",
  69 |     "RepC3",
  70 |     "RepC2f",
  71 |     "PoolingEdgeRepC2f",
  72 |     "RepNCSPELAN4",
  73 |     "RepVGGDW",
  74 |     "ResNetLayer",
  75 |     "ScalSeq",
  76 |     "SCDown",
  77 |     "TorchVision",
  78 |     "WeightedAdd",
  79 |     "clear_boundary_context",
  80 |     "set_boundary_context",
  81 |     "set_boundary_enabled",
  82 | )
  83 | 
  84 | 
  85 | @dataclass
  86 | class BoundaryContext:
  87 |     """Train-time labels used by BoundaryFeatureBlock."""
  88 | 
  89 |     batch_idx: torch.Tensor
  90 |     bboxes: torch.Tensor
  91 |     image_shape: tuple[int, int, int, int]
  92 | 
  93 | 
  94 | _BOUNDARY_CONTEXT: BoundaryContext | None = None
  95 | _BOUNDARY_ENABLED = True
  96 | 
  97 | 
  98 | def set_boundary_enabled(enabled: bool) -> None:
  99 |     """Enable or disable train-time boundary feature refinement."""
 100 | 
 101 |     global _BOUNDARY_ENABLED
 102 |     _BOUNDARY_ENABLED = enabled
 103 | 
 104 | 
 105 | def set_boundary_context(
 106 |     batch_idx: torch.Tensor | None,
 107 |     bboxes: torch.Tensor | None,
 108 |     image_shape: tuple[int, int, int, int],
 109 | ) -> None:
 110 |     """Store normalized YOLO xywh labels for train-time boundary masks."""
 111 | 
 112 |     global _BOUNDARY_CONTEXT
 113 |     if batch_idx is None or bboxes is None:
 114 |         _BOUNDARY_CONTEXT = None
 115 |         return
 116 | 
 117 |     _BOUNDARY_CONTEXT = BoundaryContext(
 118 |         batch_idx=batch_idx.detach(),
 119 |         bboxes=bboxes.detach(),
 120 |         image_shape=image_shape,
 121 |     )
 122 | 
 123 | 
 124 | def clear_boundary_context() -> None:
 125 |     """Clear train-time boundary labels after a forward pass."""
 126 | 
 127 |     global _BOUNDARY_CONTEXT
 128 |     _BOUNDARY_CONTEXT = None
 129 | 
 130 | 
 131 | class BoundaryFeatureBlock(nn.Module):
 132 |     """Train-only GT-guided boundary/background feature refinement for one FPN level."""
 133 | 
 134 |     def __init__(
 135 |         self,
 136 |         c1: int,
 137 |         ring: float = 1.0,
 138 |         shrinkage: float = 0.25,
 139 |         reduction: int = 4,
 140 |         alpha_init: float = 0.1,
 141 |         alpha_max: float = 0.1,
 142 |     ) -> None:
 143 |         """Initialize a residual transform applied only to boundary and near-background cells."""
 144 | 
 145 |         super().__init__()
 146 |         hidden_channels = max(c1 // max(int(reduction), 1), 8)
 147 |         self.ring = max(float(ring), 0.0)
 148 |         self.shrinkage = max(float(shrinkage), 0.0)
 149 |         self.alpha_max = max(float(alpha_max), 0.0)
 150 |         limit = self.alpha_max * 0.999
 151 |         init = max(min(float(alpha_init), limit), -limit)
 152 |         init = 0.0 if self.alpha_max == 0 else torch.atanh(torch.tensor(init / self.alpha_max)).item()
 153 |         self.alpha = nn.Parameter(torch.tensor(init))
 154 |         self.transform = nn.Sequential(
 155 |             nn.Conv2d(c1, hidden_channels, kernel_size=3, padding=1, bias=False),
 156 |             nn.SiLU(inplace=True),
 157 |             nn.Conv2d(hidden_channels, c1, kernel_size=3, padding=1, bias=False),
 158 |         )
 159 | 
 160 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 161 |         """Refine boundary/background cells during training and stay identity otherwise."""
 162 | 
 163 |         if not _BOUNDARY_ENABLED or not self.training or _BOUNDARY_CONTEXT is None:
 164 |             return x
 165 | 
 166 |         mask = _build_boundary_background_mask(
 167 |             batch_idx=_BOUNDARY_CONTEXT.batch_idx,
 168 |             bboxes=_BOUNDARY_CONTEXT.bboxes,
 169 |             batch_size=x.shape[0],
 170 |             feature_height=x.shape[2],
 171 |             feature_width=x.shape[3],
 172 |             device=x.device,
 173 |             dtype=x.dtype,
 174 |             ring=self.ring,
 175 |             shrinkage=self.shrinkage,
 176 |         )
 177 |         alpha = torch.tanh(self.alpha).to(dtype=x.dtype) * self.alpha_max
 178 |         return x + alpha * self.transform(x) * mask
 179 | 
 180 | 
 181 | def _clamp_interval(start: float, end: float, limit: int) -> tuple[int, int]:
 182 |     """Convert a continuous feature interval to a non-empty clamped integer slice."""
 183 | 
 184 |     if limit <= 0:
 185 |         return 0, 0
 186 |     start_i = max(0, min(limit - 1, int(start)))
 187 |     end_i = max(start_i + 1, min(limit, int(end) + 1))
 188 |     return start_i, end_i
 189 | 
 190 | 
 191 | def _build_boundary_background_mask(
 192 |     batch_idx: torch.Tensor,
 193 |     bboxes: torch.Tensor,
 194 |     batch_size: int,
 195 |     feature_height: int,
 196 |     feature_width: int,
 197 |     device: torch.device,
 198 |     dtype: torch.dtype,
 199 |     ring: float,
 200 |     shrinkage: float,
 201 | ) -> torch.Tensor:
 202 |     """Build a mask for dilated-boundary and near-background cells from normalized xywh GT boxes."""
 203 | 
 204 |     mask = torch.zeros((batch_size, 1, feature_height, feature_width), device=device, dtype=dtype)
 205 |     if bboxes.numel() == 0:
 206 |         return mask
 207 | 
 208 |     batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)
 209 |     bboxes = bboxes.to(device=device, dtype=dtype)
 210 |     ring = max(float(ring), 0.0)
 211 |     near_ring = ring * 3.0
 212 | 
 213 |     for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
 214 |         if idx < 0 or idx >= batch_size:
 215 |             continue
 216 | 
 217 |         x_center, y_center, width, height = [float(v) for v in box]
 218 |         x1 = (x_center - width / 2.0) * feature_width
 219 |         y1 = (y_center - height / 2.0) * feature_height
 220 |         x2 = (x_center + width / 2.0) * feature_width
 221 |         y2 = (y_center + height / 2.0) * feature_height
 222 | 
 223 |         dx1, dx2 = _clamp_interval(x1 - ring, x2 + ring, feature_width)
 224 |         dy1, dy2 = _clamp_interval(y1 - ring, y2 + ring, feature_height)
 225 |         nx1, nx2 = _clamp_interval(x1 - near_ring, x2 + near_ring, feature_width)
 226 |         ny1, ny2 = _clamp_interval(y1 - near_ring, y2 + near_ring, feature_height)
 227 |         pad_x = min(max(x2 - x1, 0.0) * shrinkage, 0.5)
 228 |         pad_y = min(max(y2 - y1, 0.0) * shrinkage, 0.5)
 229 |         ix1, ix2 = _clamp_interval(x1 + pad_x, x2 - pad_x, feature_width)
 230 |         iy1, iy2 = _clamp_interval(y1 + pad_y, y2 - pad_y, feature_height)
 231 | 
 232 |         mask[idx, :, ny1:ny2, nx1:nx2] = 1.0
 233 |         mask[idx, :, dy1:dy2, dx1:dx2] = 1.0
 234 |         mask[idx, :, iy1:iy2, ix1:ix2] = 0.0
 235 | 
 236 |     return mask
 237 | 
 238 | 
 239 | class UpBlock(nn.Module):
 240 |     """SR-TOD reconstruction upsample block."""
 241 | 
 242 |     def __init__(self, c1: int, c2: int) -> None:
 243 |         super().__init__()
 244 |         self.block = nn.Sequential(
 245 |             nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
 246 |             nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
 247 |             nn.ReLU(inplace=True),
 248 |             nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
 249 |             nn.ReLU(inplace=True),
 250 |         )
 251 | 
 252 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 253 |         return self.block(x)
 254 | 
 255 | 
 256 | class FeatureDGFE(nn.Module):
 257 |     """Image-space SR-DGFE for P2 feature enhancement."""
 258 | 
 259 |     def __init__(
 260 |         self,
 261 |         c1: int,
 262 |         reduction: int = 8,
 263 |         threshold_init: float = 0.0156862,
 264 |         sharpness: float = 10.0,
 265 |         alpha_init: float = 1e-3,
 266 |         alpha_max: float = 1.0,
 267 |         recon_ratio: float = 0.5,
 268 |         upsample_steps: int = 2,
 269 |     ) -> None:
 270 |         super().__init__()
 271 |         upsample_steps = max(int(upsample_steps), 1)
 272 |         channel_hidden = max(c1 // max(int(reduction), 1), 8)
 273 | 
 274 |         up_blocks = []
 275 |         in_channels = c1
 276 |         out_channels = max(int(c1 * float(recon_ratio)), 8)
 277 |         for _ in range(upsample_steps):
 278 |             up_blocks.append(UpBlock(in_channels, out_channels))
 279 |             in_channels = out_channels
 280 |             out_channels = max(out_channels // 2, 8)
 281 |         self.upsample = nn.Sequential(*up_blocks)
 282 |         self.reconstruct = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1), nn.Sigmoid())
 283 |         self.channel_mlp = nn.Sequential(
 284 |             nn.Conv2d(c1, channel_hidden, kernel_size=1),
 285 |             nn.ReLU(inplace=True),
 286 |             nn.Conv2d(channel_hidden, c1, kernel_size=1),
 287 |         )
 288 |         self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
 289 |         self.sharpness = float(sharpness)
 290 |         self.alpha_max = max(float(alpha_max), 0.0)
 291 |         p = max(min(float(alpha_init) / max(self.alpha_max, 1e-12), 1.0 - 1e-6), 1e-6)
 292 |         self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(p)))
 293 |         self.last_aux: dict[str, torch.Tensor] | None = None
 294 | 
 295 |     @property
 296 |     def alpha(self) -> torch.Tensor:
 297 |         return torch.sigmoid(self.alpha_logit) * self.alpha_max
 298 | 
 299 |     def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
 300 |         recon = self.reconstruct(self.upsample(x))
 301 |         if recon.shape[-2:] != img.shape[-2:]:
 302 |             recon = F.interpolate(recon, size=img.shape[-2:], mode="bilinear", align_corners=False)
 303 | 
 304 |         diff = (recon - img).abs().mean(dim=1, keepdim=True)
 305 |         spatial_logits_img = self.sharpness * (diff - self.threshold.to(dtype=diff.dtype, device=diff.device))
 306 |         spatial_logits = F.interpolate(spatial_logits_img, size=x.shape[-2:], mode="bilinear", align_corners=False)
 307 |         spatial_gate = 1.0 + torch.sigmoid(spatial_logits)
 308 | 
 309 |         avg_gate = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
 310 |         max_gate = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
 311 |         channel_gate = torch.sigmoid(avg_gate + max_gate)
 312 | 
 313 |         attention = channel_gate * spatial_gate
 314 |         alpha = self.alpha.to(dtype=x.dtype, device=x.device)
 315 |         out = x * (1.0 + alpha * (attention - 1.0))
 316 |         self.last_aux = (
 317 |             {
 318 |                 "recon": recon,
 319 |                 "spatial_logits": spatial_logits,
 320 |                 "spatial_gate": spatial_gate,
 321 |                 "alpha": alpha.reshape(1),
 322 |             }
 323 |             if self.training
 324 |             else None
 325 |         )
 326 |         return out
 327 | 
 328 | 
 329 | class MS_Scharr_EnSimAM(nn.Module):
 330 |     """Multi-scale local-variance EnSimAM with Scharr edge attention."""
 331 | 
 332 |     def __init__(self, lambd: float = 1e-4, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
 333 |         """Initialize learnable local and branch fusion weights with fixed Scharr kernels."""
 334 |         super().__init__()
 335 |         self.lambd = lambd
 336 |         self.alpha = alpha
 337 |         self.beta = beta
 338 |         self.eps = eps
 339 |         self.local_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
 340 |         self.branch_weights = nn.Parameter(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32))
 341 |         scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]).view(1, 1, 3, 3)
 342 |         scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]).view(1, 1, 3, 3)
 343 |         self.register_buffer("scharr_x", scharr_x, persistent=False)
 344 |         self.register_buffer("scharr_y", scharr_y, persistent=False)
 345 | 
 346 |     @staticmethod
 347 |     def local_variance(x: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor:
 348 |         """Compute local variance while preserving [B, C, H, W]."""
 349 |         local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
 350 |         return F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=padding)
 351 | 
 352 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 353 |         """Apply global, multi-kernel local, and Scharr edge attention."""
 354 |         mean = x.mean(dim=(2, 3), keepdim=True)
 355 |         var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
 356 |         energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
 357 |         a_global = torch.sigmoid(1.0 / energy)
 358 | 
 359 |         local_var_3 = self.local_variance(x, kernel_size=3, padding=1)
 360 |         local_var_5 = self.local_variance(x, kernel_size=5, padding=2)
 361 |         local_var_7 = self.local_variance(x, kernel_size=7, padding=3)
 362 |         local_weights = torch.softmax(self.local_weights, dim=0)
 363 |         local_var = local_weights[0] * local_var_3 + local_weights[1] * local_var_5 + local_weights[2] * local_var_7
 364 |         a_local = torch.sigmoid(self.alpha * local_var)
 365 | 
 366 |         c = x.shape[1]
 367 |         scharr_x = self.scharr_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
 368 |         scharr_y = self.scharr_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
 369 |         gx = F.conv2d(x, scharr_x, padding=1, groups=c)
 370 |         gy = F.conv2d(x, scharr_y, padding=1, groups=c)
 371 |         edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
 372 |         a_edge = torch.sigmoid(self.beta * edge)
 373 | 
 374 |         branch_weights = torch.softmax(self.branch_weights, dim=0)
 375 |         attention = branch_weights[0] * a_global + branch_weights[1] * a_local + branch_weights[2] * a_edge
 376 |         return x * attention
 377 | 
 378 | 
 379 | class AdversarialPerturbationInjection(nn.Module):
 380 |     """Train-only gradient-based feature perturbation injection for one FPN level."""
 381 | 
 382 |     def __init__(
 383 |         self,
 384 |         c1: int,
 385 |         rho: float = 0.02,
 386 |         api_weight: float = 0.25,
 387 |         target_mode: str = "foreground",
 388 |         eps: float = 1e-6,
 389 |         use_partial_forward: bool = False,
 390 |         use_rho_warmup: bool = False,
 391 |         warmup_epochs: int = 10,
 392 |         use_per_box_norm: bool = False,
 393 |         use_fgsm_dropout: bool = False,
 394 |         fgsm_drop_rate: float = 0.1,
 395 |     ) -> None:
 396 |         """Initialize SET-style API parameters.
 397 | 
 398 |         The perturbation is supplied by the training loss path from the
 399 |         auxiliary-loss gradient w.r.t. the captured feature. A small auxiliary
 400 |         head scores the perturbed P2 feature without a second full detector forward.
 401 | 
 402 |         Flags (all default False = original behavior):
 403 |             use_partial_forward: Re-run only layers from this layer onward for perturbed
 404 |                 forward, skipping the backbone. Requires tasks.py to cache clean y-list
 405 |                 and set ``_cached_clean_y`` before the perturbed pass.
 406 |             use_rho_warmup: Linearly warm-up ``rho`` and ``api_weight`` from 10 % of their
 407 |                 target values over ``warmup_epochs`` epochs. Set ``_epoch`` each epoch from
 408 |                 the trainer.
 409 |             warmup_epochs: Number of epochs for the rho/weight warm-up schedule.
 410 |             use_per_box_norm: Normalize the adversarial gradient per GT bounding-box region
 411 |                 rather than globally, amplifying the perturbation for small objects.
 412 |             use_fgsm_dropout: In perturb mode, zero out the ``fgsm_drop_rate`` fraction of
 413 |                 channels with the highest perturbation magnitude, forcing the model to learn
 414 |                 redundant representations for robust detection.
 415 |             fgsm_drop_rate: Fraction of channels to drop (0.0-1.0). Active only when
 416 |                 ``use_fgsm_dropout=True`` and in training perturb mode.
 417 |         """
 418 | 
 419 |         super().__init__()
 420 |         self.rho = max(float(rho), 0.0)
 421 |         self.api_weight = max(float(api_weight), 0.0)
 422 |         self.target_mode = str(target_mode)
 423 |         self.eps = max(float(eps), 1e-12)
 424 | 
 425 |         # --- Feature flags (all off = original behavior) ---
 426 |         self.use_partial_forward = bool(use_partial_forward)
 427 |         self.use_rho_warmup = bool(use_rho_warmup)
 428 |         self.warmup_epochs = max(1, int(warmup_epochs))
 429 |         self.use_per_box_norm = bool(use_per_box_norm)
 430 |         self.use_fgsm_dropout = bool(use_fgsm_dropout)
 431 |         self.fgsm_drop_rate = float(max(0.0, min(1.0, fgsm_drop_rate)))
 432 | 
 433 |         # --- Aux head (used only in foreground/boundary target_mode) ---
 434 |         self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)
 435 | 
 436 |         # --- Runtime state ---
 437 |         self.mode = "off"
 438 |         self.captured: torch.Tensor | None = None
 439 |         self.perturbation: torch.Tensor | None = None
 440 |         self.last_perturbation_norm: torch.Tensor | None = None
 441 | 
 442 |         # Partial forward state (populated from tasks.py)
 443 |         self.layer_idx: int | None = None          # position of this module in self.model
 444 |         self._clean_input: torch.Tensor | None = None     # input tensor to this layer in clean pass
 445 |         self._cached_clean_y: list | None = None   # full y-list from clean _predict_once
 446 | 
 447 |         # Rho warmup state (epoch synced from trainer)
 448 |         self._epoch: int = 0
 449 | 
 450 |     # ------------------------------------------------------------------
 451 |     # Scheduled hyper-parameters
 452 |     # ------------------------------------------------------------------
 453 | 
 454 |     @property
 455 |     def current_rho(self) -> float:
 456 |         """Return rho after applying linear warm-up if enabled."""
 457 |         if not self.use_rho_warmup or self._epoch >= self.warmup_epochs:
 458 |             return self.rho
 459 |         t = self._epoch / self.warmup_epochs  # 0 -> 1 over warmup_epochs
 460 |         return self.rho * (0.1 + 0.9 * t)
 461 | 
 462 |     @property
 463 |     def current_api_weight(self) -> float:
 464 |         """Return api_weight after applying linear warm-up if enabled."""
 465 |         if not self.use_rho_warmup or self._epoch >= self.warmup_epochs:
 466 |             return self.api_weight
 467 |         t = self._epoch / self.warmup_epochs
 468 |         return self.api_weight * (0.1 + 0.9 * t)
 469 | 
 470 |     # ------------------------------------------------------------------
 471 |     # State control
 472 |     # ------------------------------------------------------------------
 473 | 
 474 |     def clear_state(self) -> None:
 475 |         """Clear captured feature, perturbation, and partial-forward cache."""
 476 | 
 477 |         self.mode = "off"
 478 |         self.captured = None
 479 |         self.perturbation = None
 480 |         self._clean_input = None
 481 |         self._cached_clean_y = None
 482 | 
 483 |     def capture(self) -> None:
 484 |         """Capture the next forward feature without perturbing it."""
 485 | 
 486 |         self.clear_state()
 487 |         self.mode = "capture"
 488 | 
 489 |     def perturb(self) -> None:
 490 |         """Apply the stored perturbation on the next forward."""
 491 | 
 492 |         self.mode = "perturb"
 493 | 
 494 |     # ------------------------------------------------------------------
 495 |     # Perturbation construction
 496 |     # ------------------------------------------------------------------
 497 | 
 498 |     def set_perturbation_from_grad(
 499 |         self,
 500 |         grad: torch.Tensor | None,
 501 |         bboxes: torch.Tensor | None = None,
 502 |         batch_idx: torch.Tensor | None = None,
 503 |     ) -> bool:
 504 |         """Create an adversarial perturbation from a feature gradient.
 505 | 
 506 |         When ``use_per_box_norm`` is enabled and valid bboxes are provided, the
 507 |         gradient is weighted per GT box region before L2 normalisation.  Small
 508 |         boxes receive a higher weight (proportional to 1/sqrt(area)) so that the
 509 |         perturbation is not diluted by large background regions.
 510 | 
 511 |         Args:
 512 |             grad: Gradient tensor [B, C, H, W] w.r.t. the captured feature.
 513 |             bboxes: Normalised xywh GT boxes [N, 4] (optional, for per-box norm).
 514 |             batch_idx: Integer sample indices [N] matching each box to a batch item.
 515 | 
 516 |         Returns:
 517 |             True if the perturbation is finite and was stored; False otherwise.
 518 |         """
 519 | 
 520 |         if grad is None or self.current_rho == 0 or self.current_api_weight == 0:
 521 |             self.perturbation = None
 522 |             return False
 523 | 
 524 |         if self.use_per_box_norm and bboxes is not None and bboxes.numel() > 0 and batch_idx is not None:
 525 |             B, C, H, W = grad.shape
 526 |             weight_map = torch.ones(B, 1, H, W, device=grad.device, dtype=torch.float32)
 527 |             _batch_idx = batch_idx.to(device=grad.device, dtype=torch.long).view(-1)
 528 |             _bboxes = bboxes.to(device=grad.device, dtype=torch.float32)
 529 |             for b_idx, box in zip(_batch_idx.tolist(), _bboxes.tolist()):
 530 |                 if b_idx < 0 or b_idx >= B:
 531 |                     continue
 532 |                 xc, yc, bw, bh = box
 533 |                 x1 = int((xc - bw / 2) * W)
 534 |                 x2 = max(x1 + 1, int((xc + bw / 2) * W))
 535 |                 y1 = int((yc - bh / 2) * H)
 536 |                 y2 = max(y1 + 1, int((yc + bh / 2) * H))
 537 |                 x1, x2 = max(0, x1), min(W, x2)
 538 |                 y1, y2 = max(0, y1), min(H, y2)
 539 |                 box_area = max(1, (x2 - x1) * (y2 - y1))
 540 |                 # Small objects get amplified (inversely proportional to sqrt(area))
 541 |                 scale = float(H * W / box_area) ** 0.5
 542 |                 weight_map[b_idx, :, y1:y2, x1:x2] *= scale
 543 |             weighted_grad = grad.detach().float() * weight_map
 544 |             flat = weighted_grad.flatten(1)
 545 |             norm = flat.norm(p=2, dim=1).clamp(min=self.eps).view(-1, 1, 1, 1)
 546 |             perturbation = weighted_grad / norm * self.current_rho
 547 |         else:
 548 |             # Default: global L2 norm (original behavior)
 549 |             flat = grad.detach().float().flatten(1)
 550 |             norm = flat.norm(p=2, dim=1).clamp(min=self.eps).view(-1, 1, 1, 1)
 551 |             perturbation = grad.detach().float() / norm * self.current_rho
 552 | 
 553 |         self.perturbation = perturbation.to(dtype=grad.dtype, device=grad.device)
 554 |         self.last_perturbation_norm = self.perturbation.detach().float().flatten(1).norm(p=2, dim=1)
 555 |         return bool(torch.isfinite(self.perturbation).all())
 556 | 
 557 |     # ------------------------------------------------------------------
 558 |     # Forward
 559 |     # ------------------------------------------------------------------
 560 | 
 561 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 562 |         """Capture or perturb a feature during training; stay identity otherwise."""
 563 | 
 564 |         if not self.training:
 565 |             return x
 566 |         if self.mode == "capture":
 567 |             self._clean_input = x   # cache for partial forward
 568 |             self.captured = x
 569 |             if x.requires_grad:
 570 |                 x.retain_grad()
 571 |             return x
 572 |         if self.mode == "perturb" and self.perturbation is not None:
 573 |             out = x + self.perturbation.to(device=x.device, dtype=x.dtype)
 574 |             if self.use_fgsm_dropout:
 575 |                 # Channel-wise structured dropout: suppress top-fgsm_drop_rate channels
 576 |                 # that have the highest perturbation magnitude.  Forces the model to
 577 |                 # learn redundant, attack-resistant feature representations.
 578 |                 ch_mag = self.perturbation.abs().mean(dim=(2, 3))  # [B, C]
 579 |                 k = max(1, int(ch_mag.shape[1] * self.fgsm_drop_rate))
 580 |                 # Per-sample threshold: top-k channel magnitude
 581 |                 thresh = ch_mag.topk(k, dim=1).values[:, -1].view(-1, 1, 1, 1)  # [B,1,1,1]
 582 |                 keep_mask = (ch_mag.unsqueeze(-1).unsqueeze(-1) < thresh).to(dtype=out.dtype)  # [B,C,1,1]
 583 |                 keep_frac = max(1.0 - self.fgsm_drop_rate, 1e-3)
 584 |                 out = out * keep_mask / keep_frac
 585 |             return out
 586 |         return x
 587 | 
 588 |     # ------------------------------------------------------------------
 589 |     # Auxiliary head (used only in foreground / boundary target_mode)
 590 |     # ------------------------------------------------------------------
 591 | 
 592 |     def auxiliary_logits(self, feature: torch.Tensor | None = None) -> torch.Tensor:
 593 |         """Return auxiliary objectness logits for a captured or supplied feature."""
 594 | 
 595 |         if feature is None:
 596 |             if self.captured is None:
 597 |                 raise RuntimeError("API auxiliary_logits requires captured feature.")
 598 |             feature = self.captured
 599 |         return self.aux_head(feature)
 600 | 
 601 |     def auxiliary_loss(self, target: torch.Tensor, feature: torch.Tensor | None = None) -> torch.Tensor:
 602 |         """Compute BCE objectness loss for the auxiliary P2 API branch."""
 603 | 
 604 |         logits = self.auxiliary_logits(feature)
 605 |         target = target.to(device=logits.device, dtype=logits.dtype)
 606 |         if target.shape[-2:] != logits.shape[-2:]:
 607 |             target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
 608 |         return F.binary_cross_entropy_with_logits(logits, target)
 609 | 
 610 |     def adversarial_auxiliary_loss(self, target: torch.Tensor) -> torch.Tensor:
 611 |         """Compute auxiliary loss on the captured feature plus stored perturbation."""
 612 | 
 613 |         if self.captured is None or self.perturbation is None:
 614 |             raise RuntimeError("API adversarial_auxiliary_loss requires captured feature and perturbation.")
 615 |         feature = self.captured + self.perturbation.to(device=self.captured.device, dtype=self.captured.dtype)
 616 |         return self.auxiliary_loss(target, feature=feature)
 617 | 
 618 | 
 619 | class DFL(nn.Module):
 620 |     """Integral module of Distribution Focal Loss (DFL).
 621 | 
 622 |     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
 623 |     """
 624 | 
 625 |     def __init__(self, c1: int = 16):
 626 |         """Initialize a convolutional layer with a given number of input channels.
 627 | 
 628 |         Args:
 629 |             c1 (int): Number of input channels.
 630 |         """
 631 |         super().__init__()
 632 |         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
 633 |         x = torch.arange(c1, dtype=torch.float)
 634 |         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
 635 |         self.c1 = c1
 636 | 
 637 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 638 |         """Apply the DFL module to input tensor and return transformed output."""
 639 |         b, _, a = x.shape  # batch, channels, anchors
 640 |         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
 641 |         # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
 642 | 
 643 | 
 644 | class WeightedAdd(nn.Module):
 645 |     """BiFPN-style weighted sum for already aligned feature maps."""
 646 | 
 647 |     def __init__(self, n_inputs: int = 2, eps: float = 1e-4):
 648 |         """Initialize learnable non-negative fusion weights."""
 649 |         super().__init__()
 650 |         self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
 651 |         self.eps = eps
 652 | 
 653 |     def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
 654 |         """Fuse tensors with normalized positive weights."""
 655 |         weights = torch.relu(self.weights)
 656 |         weights = weights / (weights.sum() + self.eps)
 657 |         out = inputs[0] * weights[0]
 658 |         for i, x in enumerate(inputs[1:], start=1):
 659 |             out = out + x * weights[i]
 660 |         return out
 661 | 
 662 | 
 663 | class ScalSeq(nn.Module):
 664 |     """Attentional Scale Fusion sequence block for multi-scale feature fusion."""
 665 | 
 666 |     def __init__(self, ch: list[int] | tuple[int, ...], c2: int, kernel_size: int = 3):
 667 |         """Initialize lateral projections and a 3D convolution over the scale dimension."""
 668 |         super().__init__()
 669 |         self.lateral = nn.ModuleList(Conv(c, c2, 1, 1) for c in ch)
 670 |         padding = kernel_size // 2
 671 |         self.conv3d = nn.Conv3d(c2, c2, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), bias=False)
 672 |         self.bn = nn.BatchNorm3d(c2)
 673 |         self.act = nn.LeakyReLU(0.1, inplace=True)
 674 | 
 675 |     def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
 676 |         """Fuse multi-scale tensors into a single [B, C, H, W] feature map."""
 677 |         if not isinstance(inputs, (list, tuple)):
 678 |             inputs = [inputs]
 679 |         target_size = inputs[0].shape[2:]
 680 |         features = []
 681 |         for x, lateral in zip(inputs, self.lateral):
 682 |             x = lateral(x)
 683 |             if x.shape[2:] != target_size:
 684 |                 x = F.interpolate(x, size=target_size, mode="nearest")
 685 |             features.append(x)
 686 |         x = torch.stack(features, dim=2)
 687 |         x = self.act(self.bn(self.conv3d(x)))
 688 |         return F.max_pool3d(x, kernel_size=(x.shape[2], 1, 1)).squeeze(2)
 689 | 
 690 | 
 691 | class ASFAttention(nn.Module):
 692 |     """ASF refinement block with channel attention and local spatial attention."""
 693 | 
 694 |     def __init__(self, c1: int, reduction: int = 16, local_kernel: int = 7):
 695 |         """Initialize channel and local attention branches."""
 696 |         super().__init__()
 697 |         hidden = max(c1 // reduction, 8)
 698 |         padding = local_kernel // 2
 699 |         self.channel_mlp = nn.Sequential(
 700 |             nn.Conv2d(c1, hidden, 1, bias=False),
 701 |             nn.LeakyReLU(0.1, inplace=True),
 702 |             nn.Conv2d(hidden, c1, 1, bias=False),
 703 |         )
 704 |         self.local = nn.Conv2d(2, 1, local_kernel, padding=padding, bias=False)
 705 |         self.sigmoid = nn.Sigmoid()
 706 | 
 707 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 708 |         """Refine important channels and local regions without changing shape."""
 709 |         avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
 710 |         max_attn = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
 711 |         x = x * self.sigmoid(avg_attn + max_attn)
 712 |         avg_map = torch.mean(x, dim=1, keepdim=True)
 713 |         max_map = torch.amax(x, dim=1, keepdim=True)
 714 |         return x * self.sigmoid(self.local(torch.cat((avg_map, max_map), dim=1)))
 715 | 
 716 | 
 717 | class EnSimAM(nn.Module):
 718 |     """Enhanced SimAM attention with global, local-variance, and edge-response branches."""
 719 | 
 720 |     def __init__(self, lambd: float = 1e-4, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
 721 |         """Initialize parameter-free attention coefficients and fixed Sobel kernels."""
 722 |         super().__init__()
 723 |         self.lambd = lambd
 724 |         self.alpha = alpha
 725 |         self.beta = beta
 726 |         self.eps = eps
 727 |         sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
 728 |         sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
 729 |         self.register_buffer("sobel_x", sobel_x, persistent=False)
 730 |         self.register_buffer("sobel_y", sobel_y, persistent=False)
 731 | 
 732 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 733 |         """Apply parameter-free attention without changing tensor shape."""
 734 |         mean = x.mean(dim=(2, 3), keepdim=True)
 735 |         var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
 736 |         energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
 737 |         a_global = torch.sigmoid(1.0 / energy)
 738 | 
 739 |         local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
 740 |         local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
 741 |         a_local = torch.sigmoid(self.alpha * local_var)
 742 | 
 743 |         c = x.shape[1]
 744 |         sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
 745 |         sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
 746 |         gx = F.conv2d(x, sobel_x, padding=1, groups=c)
 747 |         gy = F.conv2d(x, sobel_y, padding=1, groups=c)
 748 |         edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
 749 |         a_edge = torch.sigmoid(self.beta * edge)
 750 | 
 751 |         attention = 0.5 * a_global + 0.25 * a_local + 0.25 * a_edge
 752 |         return x * attention
 753 | 
 754 | 
 755 | 
 756 | def _choose_attention_heads(channels: int, requested_heads: int) -> int:
 757 |     """Pick a valid attention head count that divides channels."""
 758 |     requested_heads = max(1, min(int(requested_heads), int(channels)))
 759 |     for heads in range(requested_heads, 0, -1):
 760 |         if channels % heads == 0:
 761 |             return heads
 762 |     return 1
 763 | 
 764 | 
 765 | class KVCompressedAttention(nn.Module):
 766 |     """Self-attention with full-resolution queries and spatially compressed keys/values.
 767 | 
 768 |     Supports multiple K/V spatial compression modes:
 769 |     - avg: AvgPool + GroupNorm for K and V.
 770 |     - avg_dwk/dwconv: AvgPool + DWConv + GroupNorm for K, AvgPool + GroupNorm for V.
 771 |     - dw_stride: stride depthwise Conv + GroupNorm for K and V.
 772 |     - group_weight: learned softmax weighting inside each sr_ratio x sr_ratio group.
 773 |     Attention is computed via ``F.scaled_dot_product_attention`` which automatically
 774 |     dispatches to FlashAttention v2 on supported CUDA hardware.
 775 |     """
 776 | 
 777 |     def __init__(
 778 |         self,
 779 |         c1: int,
 780 |         c2: int,
 781 |         num_heads: int = 4,
 782 |         sr_ratio: int = 2,
 783 |         mode: str = "dwconv",
 784 |         attn_drop: float = 0.0,
 785 |         residual: bool = True,
 786 |     ):
 787 |         """Initialize KV-compressed attention.
 788 | 
 789 |         Args:
 790 |             c1: Input channels.
 791 |             c2: Output channels.
 792 |             num_heads: Requested attention heads. Reduced if it does not divide c2.
 793 |             sr_ratio: Spatial compression ratio for K/V tokens.
 794 |             mode: ``avg``, ``avg_dwk``, ``dw_stride``, ``dwconv``, or ``group_weight`` compression.
 795 |             attn_drop: Dropout probability applied to attention weights during training.
 796 |             residual: Whether to add the projected attention output back to the input projection.
 797 |         """
 798 |         super().__init__()
 799 |         if mode == "dwconv":
 800 |             mode = "avg_dwk"
 801 |         if mode not in {"avg", "avg_dwk", "dw_stride", "group_weight"}:
 802 |             raise ValueError(f"Unsupported KV compression mode: {mode}")
 803 | 
 804 |         self.c2 = c2
 805 |         self.num_heads = _choose_attention_heads(c2, num_heads)
 806 |         self.head_dim = c2 // self.num_heads
 807 |         self.scale = self.head_dim**-0.5
 808 |         self.sr_ratio = max(1, int(sr_ratio))
 809 |         self.mode = mode
 810 |         self.attn_drop_p = attn_drop
 811 |         self.residual = residual
 812 | 
 813 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
 814 |         self.q = nn.Conv2d(c2, c2, 1, bias=False)
 815 |         self.q_norm = nn.LayerNorm(self.head_dim)  # stabilize Q logits before SDPA
 816 | 
 817 |         if self.sr_ratio > 1 and self.mode == "avg":
 818 |             self.k_compress = nn.Sequential(
 819 |                 nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
 820 |                 nn.GroupNorm(min(32, c2), c2),
 821 |             )
 822 |             self.v_compress = nn.Sequential(
 823 |                 nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
 824 |                 nn.GroupNorm(min(32, c2), c2),
 825 |             )
 826 |         elif self.sr_ratio > 1 and self.mode == "avg_dwk":
 827 |             # No activation - K must stay linear for well-formed attention logits.
 828 |             self.k_compress = nn.Sequential(
 829 |                 nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
 830 |                 nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
 831 |                 nn.GroupNorm(min(32, c2), c2),
 832 |             )
 833 |             self.v_compress = nn.Sequential(
 834 |                 nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
 835 |                 nn.GroupNorm(min(32, c2), c2),
 836 |             )
 837 |         elif self.sr_ratio > 1 and self.mode == "dw_stride":
 838 |             self.k_compress = nn.Sequential(
 839 |                 nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
 840 |                 nn.GroupNorm(min(32, c2), c2),
 841 |             )
 842 |             self.v_compress = nn.Sequential(
 843 |                 nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
 844 |                 nn.GroupNorm(min(32, c2), c2),
 845 |             )
 846 |         else:
 847 |             self.k_compress = nn.Identity()
 848 |             self.v_compress = nn.Identity()
 849 | 
 850 |         # Separate linear projections for K and V (no shared kv conv, no activation)
 851 |         self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
 852 |         self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)
 853 | 
 854 |         # group_weight path keeps its own scorer (unchanged)
 855 |         self.group_score = nn.Linear(c2, 1) if self.mode == "group_weight" and self.sr_ratio > 1 else None
 856 |         # Shared kv conv kept for group_weight compatibility
 857 |         self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False) if self.mode == "group_weight" else None
 858 | 
 859 |         self.proj = nn.Conv2d(c2, c2, 1, bias=False)
 860 |         self.proj_bn = nn.BatchNorm2d(c2)
 861 | 
 862 |     def _compress_group_weight(self, x: torch.Tensor) -> torch.Tensor:
 863 |         """Compress each sr_ratio x sr_ratio token group with learned softmax weights."""
 864 |         if self.sr_ratio <= 1:
 865 |             return x
 866 |         b, c, h, w = x.shape
 867 |         pad_h = (self.sr_ratio - h % self.sr_ratio) % self.sr_ratio
 868 |         pad_w = (self.sr_ratio - w % self.sr_ratio) % self.sr_ratio
 869 |         if pad_h or pad_w:
 870 |             x = F.pad(x, (0, pad_w, 0, pad_h))
 871 |         hp, wp = x.shape[-2:]
 872 |         gh, gw = hp // self.sr_ratio, wp // self.sr_ratio
 873 |         tokens = x.view(b, c, gh, self.sr_ratio, gw, self.sr_ratio).permute(0, 2, 4, 3, 5, 1).contiguous()
 874 |         tokens = tokens.view(b, gh, gw, self.sr_ratio * self.sr_ratio, c)
 875 |         weights = self.group_score(tokens).softmax(dim=3)
 876 |         compressed = (tokens * weights).sum(dim=3)
 877 |         return compressed.permute(0, 3, 1, 2).contiguous()
 878 | 
 879 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 880 |         """Apply KV-compressed attention and return a BCHW tensor."""
 881 |         x = self.input_proj(x)
 882 |         b, c, h, w = x.shape
 883 | 
 884 |         # Q: full-resolution, normalized per head
 885 |         q = self.q(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
 886 |         q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, H*W, hd]
 887 |         q = self.q_norm(q)
 888 | 
 889 |         if self.mode == "group_weight" and self.group_score is not None:
 890 |             # group_weight path: unchanged (shared kv compress)
 891 |             kv_source = self._compress_group_weight(x)
 892 |             kv = self.kv(kv_source).flatten(2).transpose(1, 2)
 893 |             kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
 894 |             k, v = kv[0], kv[1]
 895 |         else:
 896 |             # Non-group-weight paths use separate K and V compression and projection.
 897 |             k_src = self.k_compress(x)  # [B, C, H/sr, W/sr]
 898 |             v_src = self.v_compress(x)  # [B, C, H/sr, W/sr]
 899 |             k = self.k_proj(k_src)  # no activation
 900 |             v = self.v_proj(v_src)  # no activation
 901 |             # reshape to [B, nh, tokens, hd]
 902 |             def _to_heads(t):
 903 |                 n = t.shape[2] * t.shape[3]
 904 |                 return t.flatten(2).transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
 905 |             k = _to_heads(k)
 906 |             v = _to_heads(v)
 907 | 
 908 |         # Flash Attention via PyTorch SDPA (dispatches to FlashAttn v2 on CUDA)
 909 |         out = F.scaled_dot_product_attention(
 910 |             q, k, v,
 911 |             dropout_p=self.attn_drop_p if self.training else 0.0,
 912 |             scale=self.scale,
 913 |         )
 914 | 
 915 |         out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
 916 |         out = self.proj_bn(self.proj(out))
 917 |         return x + out if self.residual else out
 918 | 
 919 | 
 920 | class KVCompressedTransformerEncoder(nn.Module):
 921 |     """Pre-norm transformer encoder block using KV-compressed self-attention and a DW-PW convolutional FFN.
 922 | 
 923 |     FFN structure: PW-expand → DW-spatial-mix → PW-project, giving each token access to its
 924 |     8-connected spatial neighborhood while keeping the channel mixing role of the outer PWs.
 925 |     """
 926 | 
 927 |     def __init__(
 928 |         self,
 929 |         c1: int,
 930 |         c2: int,
 931 |         num_heads: int = 4,
 932 |         sr_ratio: int = 2,
 933 |         mode: str = "dwconv",
 934 |         attn_drop: float = 0.0,
 935 |         mlp_ratio: float = 2.0,
 936 |     ):
 937 |         """Initialize LayerNorm-KVCA and LayerNorm-DW-FFN residual branches.
 938 | 
 939 |         Args:
 940 |             c1: Input channels.
 941 |             c2: Output channels.
 942 |             num_heads: Requested attention heads.
 943 |             sr_ratio: Spatial compression ratio passed to KVCompressedAttention.
 944 |             mode: KV compression mode (``dwconv`` or ``group_weight``).
 945 |             attn_drop: Attention dropout probability (training only).
 946 |             mlp_ratio: Hidden dim multiplier for the FFN.
 947 |         """
 948 |         super().__init__()
 949 |         hidden = max(c2, int(c2 * mlp_ratio))
 950 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
 951 |         self.norm1 = LayerNorm2d(c2)
 952 |         self.attn = KVCompressedAttention(c2, c2, num_heads, sr_ratio, mode, attn_drop=attn_drop, residual=False)
 953 |         self.norm2 = LayerNorm2d(c2)
 954 |         # DW-PW FFN: channel expand → spatial mix → channel project
 955 |         self.ffn = nn.Sequential(
 956 |             Conv(c2, hidden, 1, 1),                    # PW: channel expand
 957 |             Conv(hidden, hidden, 3, 1, g=hidden),      # DW: spatial mix in 3×3 neighborhood
 958 |             Conv(hidden, c2, 1, 1, act=False),         # PW: channel project
 959 |         )
 960 | 
 961 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 962 |         """Apply LN -> KVCA -> residual, then LN -> DW-FFN -> residual."""
 963 |         x = self.input_proj(x)
 964 |         x = x + self.attn(self.norm1(x))
 965 |         return x + self.ffn(self.norm2(x))
 966 | 
 967 | 
 968 | class KVCompressedAttentionPartial(nn.Module):
 969 |     """PSA-style partial KV-compressed attention.
 970 | 
 971 |     Splits input channels in half: one half passes through ``KVCompressedAttention``
 972 |     (with a residual connection), the other half bypasses unchanged.  The two halves
 973 |     are concatenated and projected back to ``c2`` channels with a pointwise Conv.
 974 | 
 975 |     Benefits at high-resolution feature maps (P2/P3):
 976 |     - Attention head_dim is halved → compute reduced ~50 %.
 977 |     - Bypass half retains fine-grained local texture untouched by attention.
 978 |     - Parameter overhead is small: only an extra ``Conv(c2, c2, 1)`` output projection.
 979 |     """
 980 | 
 981 |     def __init__(
 982 |         self,
 983 |         c1: int,
 984 |         c2: int,
 985 |         num_heads: int = 4,
 986 |         sr_ratio: int = 2,
 987 |         mode: str = "dwconv",
 988 |         attn_drop: float = 0.0,
 989 |     ):
 990 |         """Initialize partial KVCA.
 991 | 
 992 |         Args:
 993 |             c1: Input channels.
 994 |             c2: Output channels (must be even).
 995 |             num_heads: Attention heads for the *attention* half; clipped to c2//2.
 996 |             sr_ratio: Spatial compression ratio passed to ``KVCompressedAttention``.
 997 |             mode: KV compression mode (``dwconv`` or ``group_weight``).
 998 |             attn_drop: Attention dropout probability (training only).
 999 |         """
1000 |         super().__init__()
1001 |         if c2 % 2 != 0:
1002 |             raise ValueError(f"KVCompressedAttentionPartial requires even c2, got {c2}")
1003 |         c_attn = c2 // 2
1004 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
1005 |         self.attn = KVCompressedAttention(
1006 |             c_attn,
1007 |             c_attn,
1008 |             num_heads=max(1, num_heads // 2),  # half heads for half channels
1009 |             sr_ratio=sr_ratio,
1010 |             mode=mode,
1011 |             attn_drop=attn_drop,
1012 |             residual=True,
1013 |         )
1014 |         # Pointwise mix after concat – no activation to stay linear
1015 |         self.out_proj = Conv(c2, c2, 1, 1, act=False)
1016 | 
1017 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1018 |         """Apply KVCA on first half of channels, bypass second half, mix outputs."""
1019 |         x = self.input_proj(x)
1020 |         x_attn, x_bypass = x.chunk(2, dim=1)
1021 |         x_attn = self.attn(x_attn)
1022 |         return self.out_proj(torch.cat([x_attn, x_bypass], dim=1))
1023 | 
1024 | 
1025 | class _TopKGroupKVAttentionBase(nn.Module):
1026 |     """Shared utilities for top-k grouped K/V attention blocks."""
1027 | 
1028 |     def __init__(self, c1: int, c2: int, num_heads: int, group_size: int):
1029 |         super().__init__()
1030 |         self.c2 = c2
1031 |         self.num_heads = _choose_attention_heads(c2, num_heads)
1032 |         self.head_dim = c2 // self.num_heads
1033 |         self.scale = self.head_dim**-0.5
1034 |         self.group_size = max(1, int(group_size))
1035 | 
1036 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
1037 |         self.q = nn.Conv2d(c2, c2, 1, bias=False)
1038 |         self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False)
1039 |         self.score = nn.Conv2d(c2, 1, 3, padding=1, bias=True)
1040 |         self.proj = nn.Conv2d(c2, c2, 1, bias=False)
1041 |         self.proj_bn = nn.BatchNorm2d(c2)
1042 | 
1043 |     @staticmethod
1044 |     def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
1045 |         """Pad H/W to a multiple and return the original H/W."""
1046 |         h, w = x.shape[-2:]
1047 |         pad_h = (multiple - h % multiple) % multiple
1048 |         pad_w = (multiple - w % multiple) % multiple
1049 |         if pad_h or pad_w:
1050 |             x = F.pad(x, (0, pad_w, 0, pad_h))
1051 |         return x, h, w
1052 | 
1053 |     def _to_groups(self, x: torch.Tensor, group_h: int, group_w: int) -> torch.Tensor:
1054 |         """Convert BCHW into B x groups x group_tokens x C."""
1055 |         b, c, _, _ = x.shape
1056 |         g = self.group_size
1057 |         return (
1058 |             x.view(b, c, group_h, g, group_w, g)
1059 |             .permute(0, 2, 4, 3, 5, 1)
1060 |             .contiguous()
1061 |             .view(b, group_h * group_w, g * g, c)
1062 |         )
1063 | 
1064 |     @staticmethod
1065 |     def _to_regions(x: torch.Tensor, region_size: int, region_h: int, region_w: int) -> torch.Tensor:
1066 |         """Convert BCHW into B x regions x region_tokens x C."""
1067 |         b, c, _, _ = x.shape
1068 |         r = region_size
1069 |         return (
1070 |             x.view(b, c, region_h, r, region_w, r)
1071 |             .permute(0, 2, 4, 3, 5, 1)
1072 |             .contiguous()
1073 |             .view(b, region_h * region_w, r * r, c)
1074 |         )
1075 | 
1076 |     def _compress_kv_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
1077 |         """Compress projected K/V maps into one weighted token per spatial group."""
1078 |         padded, _, _ = self._pad_to_multiple(x, self.group_size)
1079 |         _, _, hp, wp = padded.shape
1080 |         group_h, group_w = hp // self.group_size, wp // self.group_size
1081 | 
1082 |         k_map, v_map = self.kv(padded).chunk(2, dim=1)
1083 |         score_map = self.score(padded)
1084 |         k_tokens = self._to_groups(k_map, group_h, group_w)
1085 |         v_tokens = self._to_groups(v_map, group_h, group_w)
1086 |         score_tokens = self._to_groups(score_map, group_h, group_w).squeeze(-1)
1087 |         weights = score_tokens.softmax(dim=2).unsqueeze(-1)
1088 | 
1089 |         k_groups = (k_tokens * weights).sum(dim=2)
1090 |         v_groups = (v_tokens * weights).sum(dim=2)
1091 |         group_scores = score_tokens.mean(dim=2)
1092 |         return k_groups, v_groups, group_scores
1093 | 
1094 |     def _format_full_q(self, q_map: torch.Tensor) -> torch.Tensor:
1095 |         """Convert full-resolution Q map to B x heads x tokens x head_dim."""
1096 |         b, c, h, w = q_map.shape
1097 |         q = q_map.flatten(2).transpose(1, 2)
1098 |         return q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
1099 | 
1100 |     def _format_selected_kv(self, selected: torch.Tensor) -> torch.Tensor:
1101 |         """Convert B x tokens x C selected K/V tokens to B x heads x tokens x head_dim."""
1102 |         b, n, _ = selected.shape
1103 |         return selected.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
1104 | 
1105 | 
1106 | class TopKGlobalGroupKVAttention(_TopKGroupKVAttentionBase):
1107 |     """Full-query attention over a global top-k set of compressed K/V groups."""
1108 | 
1109 |     def __init__(self, c1: int, c2: int, num_heads: int = 4, group_size: int = 4, topk: int = 100):
1110 |         """Initialize global top-k grouped K/V attention."""
1111 |         super().__init__(c1, c2, num_heads, group_size)
1112 |         self.topk = max(1, int(topk))
1113 | 
1114 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1115 |         """Select one top-k group set per image and attend all query tokens to it."""
1116 |         x = self.input_proj(x)
1117 |         b, c, h, w = x.shape
1118 |         q = self._format_full_q(self.q(x))
1119 |         k_groups, v_groups, group_scores = self._compress_kv_groups(x)
1120 | 
1121 |         route_count = min(self.topk, k_groups.shape[1])
1122 |         route_idx = group_scores.topk(route_count, dim=-1).indices
1123 |         batch_idx = torch.arange(b, device=x.device)[:, None]
1124 |         k = self._format_selected_kv(k_groups[batch_idx, route_idx])
1125 |         v = self._format_selected_kv(v_groups[batch_idx, route_idx])
1126 | 
1127 |         with torch.cuda.amp.autocast(enabled=False):
1128 |             attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
1129 |             attn = attn.softmax(dim=-1)
1130 |         out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
1131 |         return x + self.proj_bn(self.proj(out))
1132 | 
1133 | 
1134 | class TopKAdaptiveGroupKVAttention(_TopKGroupKVAttentionBase):
1135 |     """Region-wise query attention over adaptive top-k compressed K/V groups."""
1136 | 
1137 |     def __init__(
1138 |         self,
1139 |         c1: int,
1140 |         c2: int,
1141 |         num_heads: int = 4,
1142 |         group_size: int = 4,
1143 |         query_region_size: int = 10,
1144 |         topk: int = 8,
1145 |     ):
1146 |         """Initialize adaptive top-k grouped K/V attention."""
1147 |         super().__init__(c1, c2, num_heads, group_size)
1148 |         self.query_region_size = max(1, int(query_region_size))
1149 |         self.topk = max(1, int(topk))
1150 | 
1151 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1152 |         """Select top-k K/V groups per query region, then attend region tokens to them."""
1153 |         x = self.input_proj(x)
1154 |         b, c, _, _ = x.shape
1155 |         q_map = self.q(x)
1156 |         q_padded, orig_h, orig_w = self._pad_to_multiple(q_map, self.query_region_size)
1157 |         _, _, hp, wp = q_padded.shape
1158 |         region_h, region_w = hp // self.query_region_size, wp // self.query_region_size
1159 |         num_regions = region_h * region_w
1160 |         tokens_per_region= [REDACTED_SECRET] * self.query_region_size
1161 | 
1162 |         q_regions = self._to_regions(q_padded, self.query_region_size, region_h, region_w)
1163 |         q_repr = q_regions.mean(dim=2)
1164 |         k_groups, v_groups, _ = self._compress_kv_groups(x)
1165 |         affinity = (q_repr @ k_groups.transpose(-2, -1)) * (c**-0.5)
1166 | 
1167 |         route_count = min(self.topk, k_groups.shape[1])
1168 |         route_idx = affinity.topk(route_count, dim=-1).indices
1169 |         batch_idx = torch.arange(b, device=x.device)[:, None]
1170 | 
1171 |         outputs = []
1172 |         for region_idx in range(num_regions):
1173 |             selected = route_idx[:, region_idx]
1174 |             q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
1175 |             q_tokens = q_tokens.permute(0, 2, 1, 3)
1176 |             k_tokens= [REDACTED_SECRET](k_groups[batch_idx, selected])
1177 |             v_tokens= [REDACTED_SECRET](v_groups[batch_idx, selected])
1178 |             with torch.cuda.amp.autocast(enabled=False):
1179 |                 attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
1180 |                 attn = attn.softmax(dim=-1)
1181 |             out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
1182 |             outputs.append(out)
1183 | 
1184 |         out_regions = torch.stack(outputs, dim=1)
1185 |         r = self.query_region_size
1186 |         out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
1187 |         out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
1188 |         return x + self.proj_bn(self.proj(out))
1189 | 
1190 | 
1191 | class BiLevelRoutingAttention(nn.Module):
1192 |     """NCHW bi-level routing attention adapted for YOLO feature maps."""
1193 | 
1194 |     def __init__(
1195 |         self,
1196 |         c1: int,
1197 |         c2: int,
1198 |         num_heads: int = 4,
1199 |         n_win: int = 7,
1200 |         topk: int = 4,
1201 |         side_dwconv: int = 3,
1202 |     ):
1203 |         """Initialize bi-level routing attention."""
1204 |         super().__init__()
1205 |         self.c2 = c2
1206 |         self.num_heads = _choose_attention_heads(c2, num_heads)
1207 |         self.head_dim = c2 // self.num_heads
1208 |         self.scale = c2**-0.5
1209 |         self.n_win = max(1, int(n_win))
1210 |         self.topk = max(1, int(topk))
1211 | 
1212 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
1213 |         self.qkv_linear = nn.Conv2d(c2, c2 * 3, 1)
1214 |         self.lepe = (
1215 |             nn.Conv2d(c2, c2, side_dwconv, stride=1, padding=side_dwconv // 2, groups=c2)
1216 |             if side_dwconv > 0
1217 |             else None
1218 |         )
1219 |         self.output_linear = nn.Conv2d(c2, c2, 1, bias=False)
1220 |         self.output_bn = nn.BatchNorm2d(c2)
1221 | 
1222 |     @staticmethod
1223 |     def _pad_to_region_size(x: torch.Tensor, region_size: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
1224 |         """Pad H/W to multiples of region_size and return original H/W."""
1225 |         h, w = x.shape[-2:]
1226 |         pad_h = (region_size[0] - h % region_size[0]) % region_size[0]
1227 |         pad_w = (region_size[1] - w % region_size[1]) % region_size[1]
1228 |         if pad_h or pad_w:
1229 |             x = F.pad(x, (0, pad_w, 0, pad_h))
1230 |         return x, h, w
1231 | 
1232 |     @staticmethod
1233 |     def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
1234 |         """Convert BCHW to B x heads x regions x region_tokens x head_dim."""
1235 |         b, c, h, w = x.shape
1236 |         rh, rw = region_size
1237 |         region_h, region_w = h // rh, w // rw
1238 |         x = x.view(b, num_heads, c // num_heads, region_h, rh, region_w, rw)
1239 |         x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
1240 |         return x.view(b, num_heads, region_h * region_w, rh * rw, c // num_heads), region_h, region_w
1241 | 
1242 |     @staticmethod
1243 |     def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
1244 |         """Convert B x heads x regions x region_tokens x head_dim to BCHW."""
1245 |         b, num_heads, _, _, head_dim = x.shape
1246 |         rh, rw = region_size
1247 |         x = x.view(b, num_heads, region_h, region_w, rh, rw, head_dim)
1248 |         x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
1249 |         return x.view(b, num_heads * head_dim, region_h * rh, region_w * rw)
1250 | 
1251 |     def _regional_routing_attention(
1252 |         self,
1253 |         query: torch.Tensor,
1254 |         key: torch.Tensor,
1255 |         value: torch.Tensor,
1256 |         region_graph: torch.Tensor,
1257 |         region_size: tuple[int, int],
1258 |     ) -> torch.Tensor:
1259 |         """Apply token attention from each query region to selected key/value regions."""
1260 |         query, orig_h, orig_w = self._pad_to_region_size(query, region_size)
1261 |         key, _, _ = self._pad_to_region_size(key, region_size)
1262 |         value, _, _ = self._pad_to_region_size(value, region_size)
1263 | 
1264 |         query, q_region_h, q_region_w = self._grid2seq(query, region_size, self.num_heads)
1265 |         key, _, _ = self._grid2seq(key, region_size, self.num_heads)
1266 |         value, _, _ = self._grid2seq(value, region_size, self.num_heads)
1267 | 
1268 |         b, num_heads, q_regions, topk = region_graph.shape
1269 |         _, _, kv_regions, kv_region_tokens, head_dim = key.shape
1270 |         index = region_graph.view(b, num_heads, q_regions, topk, 1, 1).expand(
1271 |             -1, -1, -1, -1, kv_region_tokens, head_dim
1272 |         )
1273 |         key_g = torch.gather(
1274 |             key.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
1275 |                 -1, -1, q_regions, -1, -1, -1
1276 |             ),
1277 |             dim=3,
1278 |             index=index,
1279 |         )
1280 |         value_g = torch.gather(
1281 |             value.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
1282 |                 -1, -1, q_regions, -1, -1, -1
1283 |             ),
1284 |             dim=3,
1285 |             index=index,
1286 |         )
1287 | 
1288 |         with torch.cuda.amp.autocast(enabled=False):
1289 |             attn = (query.float() * self.scale) @ key_g.flatten(-3, -2).float().transpose(-1, -2)
1290 |             attn = attn.softmax(dim=-1)
1291 |         out = attn.to(value_g.dtype) @ value_g.flatten(-3, -2)
1292 |         out = self._seq2grid(out, q_region_h, q_region_w, region_size)
1293 |         return out[:, :, :orig_h, :orig_w]
1294 | 
1295 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1296 |         """Route query regions to top-k key/value regions and apply token attention."""
1297 |         x = self.input_proj(x)
1298 |         _, c, h, w = x.shape
1299 |         region_size = (max(1, h // self.n_win), max(1, w // self.n_win))
1300 | 
1301 |         qkv = self.qkv_linear(x)
1302 |         q, k, v = qkv.chunk(3, dim=1)
1303 | 
1304 |         q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
1305 |         k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
1306 |         q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)
1307 |         k_r = k_r.flatten(2, 3)
1308 |         affinity = q_r @ k_r
1309 |         route_count = min(self.topk, k_r.shape[-1])
1310 |         route_idx = affinity.topk(route_count, dim=-1).indices
1311 |         route_idx = route_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
1312 | 
1313 |         out = self._regional_routing_attention(q, k, v, route_idx, region_size)
1314 |         lepe = self.lepe(v) if self.lepe is not None else torch.zeros_like(v)
1315 |         out = out + lepe
1316 |         return x + self.output_bn(self.output_linear(out.reshape(-1, c, h, w)))
1317 | 
1318 | 
1319 | class RegionRoutingAttentionLite(nn.Module):
1320 |     """Bi-level routing attention for YOLO feature maps using pure PyTorch ops."""
1321 | 
1322 |     def __init__(self, c1: int, c2: int, num_heads: int = 4, region_size: int = 10, topk: int = 4):
1323 |         """Initialize region-routed token attention."""
1324 |         super().__init__()
1325 |         self.c2 = c2
1326 |         self.num_heads = _choose_attention_heads(c2, num_heads)
1327 |         self.head_dim = c2 // self.num_heads
1328 |         self.scale = self.head_dim**-0.5
1329 |         self.region_size = max(1, int(region_size))
1330 |         self.topk = max(1, int(topk))
1331 | 
1332 |         self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
1333 |         self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=False)
1334 |         self.proj = nn.Conv2d(c2, c2, 1, bias=False)
1335 |         self.proj_bn = nn.BatchNorm2d(c2)
1336 | 
1337 |     def _pad_to_regions(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
1338 |         """Pad H/W so both are divisible by region_size."""
1339 |         h, w = x.shape[-2:]
1340 |         pad_h = (self.region_size - h % self.region_size) % self.region_size
1341 |         pad_w = (self.region_size - w % self.region_size) % self.region_size
1342 |         if pad_h or pad_w:
1343 |             x = F.pad(x, (0, pad_w, 0, pad_h))
1344 |         return x, h, w
1345 | 
1346 |     def _to_regions(self, x: torch.Tensor, region_h: int, region_w: int) -> torch.Tensor:
1347 |         """Convert BCHW into B x regions x tokens x C."""
1348 |         b, c, _, _ = x.shape
1349 |         r = self.region_size
1350 |         return (
1351 |             x.view(b, c, region_h, r, region_w, r)
1352 |             .permute(0, 2, 4, 3, 5, 1)
1353 |             .contiguous()
1354 |             .view(b, region_h * region_w, r * r, c)
1355 |         )
1356 | 
1357 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1358 |         """Route each query region to top-k key/value regions, then attend locally."""
1359 |         x = self.input_proj(x)
1360 |         b, c, h, w = x.shape
1361 |         padded, orig_h, orig_w = self._pad_to_regions(x)
1362 |         _, _, hp, wp = padded.shape
1363 |         region_h, region_w = hp // self.region_size, wp // self.region_size
1364 |         num_regions = region_h * region_w
1365 |         tokens_per_region = self.region_size * self.region_size
1366 | 
1367 |         qkv = self.qkv(padded)
1368 |         q, k, v = qkv.chunk(3, dim=1)
1369 |         q_regions = self._to_regions(q, region_h, region_w)
1370 |         k_regions = self._to_regions(k, region_h, region_w)
1371 |         v_regions = self._to_regions(v, region_h, region_w)
1372 | 
1373 |         q_repr = q_regions.mean(dim=2)
1374 |         k_repr = k_regions.mean(dim=2)
1375 |         affinity = (q_repr @ k_repr.transpose(-2, -1)) * (c**-0.5)
1376 |         route_count = min(self.topk, num_regions)
1377 |         route_idx = affinity.topk(route_count, dim=-1).indices
1378 | 
1379 |         outputs = []
1380 |         batch_idx = torch.arange(b, device=x.device)[:, None]
1381 |         for region_idx in range(num_regions):
1382 |             selected = route_idx[:, region_idx]
1383 |             q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
1384 |             q_tokens = q_tokens.permute(0, 2, 1, 3)
1385 |             k_tokens = k_regions[batch_idx, selected].reshape(
1386 |                 b, route_count * tokens_per_region, self.num_heads, self.head_dim
1387 |             )
1388 |             v_tokens = v_regions[batch_idx, selected].reshape(
1389 |                 b, route_count * tokens_per_region, self.num_heads, self.head_dim
1390 |             )
1391 |             k_tokens = k_tokens.permute(0, 2, 1, 3)
1392 |             v_tokens = v_tokens.permute(0, 2, 1, 3)
1393 |             with torch.cuda.amp.autocast(enabled=False):
1394 |                 attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
1395 |                 attn = attn.softmax(dim=-1)
1396 |             out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
1397 |             outputs.append(out)
1398 | 
1399 |         out_regions = torch.stack(outputs, dim=1)
1400 |         r = self.region_size
1401 |         out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
1402 |         out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
1403 |         return x + self.proj_bn(self.proj(out))
1404 | 
1405 | 
1406 | class Proto(nn.Module):
1407 |     """Ultralytics YOLO models mask Proto module for segmentation models."""
1408 | 
1409 |     def __init__(self, c1: int, c_: int = 256, c2: int = 32):
1410 |         """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.
1411 | 
1412 |         Args:
1413 |             c1 (int): Input channels.
1414 |             c_ (int): Intermediate channels.
1415 |             c2 (int): Output channels (number of protos).
1416 |         """
1417 |         super().__init__()
1418 |         self.cv1 = Conv(c1, c_, k=3)
1419 |         self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
1420 |         self.cv2 = Conv(c_, c_, k=3)
1421 |         self.cv3 = Conv(c_, c2)
1422 | 
1423 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1424 |         """Perform a forward pass through layers using an upsampled input image."""
1425 |         return self.cv3(self.cv2(self.upsample(self.cv1(x))))
1426 | 
1427 | 
1428 | class HGStem(nn.Module):
1429 |     """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
1430 | 
1431 |     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
1432 |     """
1433 | 
1434 |     def __init__(self, c1: int, cm: int, c2: int):
1435 |         """Initialize the StemBlock of PPHGNetV2.
1436 | 
1437 |         Args:
1438 |             c1 (int): Input channels.
1439 |             cm (int): Middle channels.
1440 |             c2 (int): Output channels.
1441 |         """
1442 |         super().__init__()
1443 |         self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
1444 |         self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
1445 |         self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
1446 |         self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
1447 |         self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
1448 |         self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
1449 | 
1450 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1451 |         """Forward pass of a PPHGNetV2 backbone layer."""
1452 |         x = self.stem1(x)
1453 |         x = F.pad(x, [0, 1, 0, 1])
1454 |         x2 = self.stem2a(x)
1455 |         x2 = F.pad(x2, [0, 1, 0, 1])
1456 |         x2 = self.stem2b(x2)
1457 |         x1 = self.pool(x)
1458 |         x = torch.cat([x1, x2], dim=1)
1459 |         x = self.stem3(x)
1460 |         x = self.stem4(x)
1461 |         return x
1462 | 
1463 | 
1464 | class HGBlock(nn.Module):
1465 |     """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
1466 | 
1467 |     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
1468 |     """
1469 | 
1470 |     def __init__(
1471 |         self,
1472 |         c1: int,
1473 |         cm: int,
1474 |         c2: int,
1475 |         k: int = 3,
1476 |         n: int = 6,
1477 |         lightconv: bool = False,
1478 |         shortcut: bool = False,
1479 |         act: nn.Module = nn.ReLU(),
1480 |     ):
1481 |         """Initialize HGBlock with specified parameters.
1482 | 
1483 |         Args:
1484 |             c1 (int): Input channels.
1485 |             cm (int): Middle channels.
1486 |             c2 (int): Output channels.
1487 |             k (int): Kernel size.
1488 |             n (int): Number of LightConv or Conv blocks.
1489 |             lightconv (bool): Whether to use LightConv.
1490 |             shortcut (bool): Whether to use shortcut connection.
1491 |             act (nn.Module): Activation function.
1492 |         """
1493 |         super().__init__()
1494 |         block = LightConv if lightconv else Conv
1495 |         self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
1496 |         self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
1497 |         self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
1498 |         self.add = shortcut and c1 == c2
1499 | 
1500 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1501 |         """Forward pass of a PPHGNetV2 backbone layer."""
1502 |         y = [x]
1503 |         y.extend(m(y[-1]) for m in self.m)
1504 |         y = self.ec(self.sc(torch.cat(y, 1)))
1505 |         return y + x if self.add else y
1506 | 
1507 | 
1508 | class SPP(nn.Module):
1509 |     """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""
1510 | 
1511 |     def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
1512 |         """Initialize the SPP layer with input/output channels and pooling kernel sizes.
1513 | 
1514 |         Args:
1515 |             c1 (int): Input channels.
1516 |             c2 (int): Output channels.
1517 |             k (tuple): Kernel sizes for max pooling.
1518 |         """
1519 |         super().__init__()
1520 |         c_ = c1 // 2  # hidden channels
1521 |         self.cv1 = Conv(c1, c_, 1, 1)
1522 |         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
1523 |         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
1524 | 
1525 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1526 |         """Forward pass of the SPP layer, performing spatial pyramid pooling."""
1527 |         x = self.cv1(x)
1528 |         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
1529 | 
1530 | 
1531 | class SPPF(nn.Module):
1532 |     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
1533 | 
1534 |     def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
1535 |         """Initialize the SPPF layer with given input/output channels and kernel size.
1536 | 
1537 |         Args:
1538 |             c1 (int): Input channels.
1539 |             c2 (int): Output channels.
1540 |             k (int): Kernel size.
1541 |             n (int): Number of pooling iterations.
1542 |             shortcut (bool): Whether to use shortcut connection.
1543 | 
1544 |         Notes:
1545 |             This module is equivalent to SPP(k=(5, 9, 13)).
1546 |         """
1547 |         super().__init__()
1548 |         c_ = c1 // 2  # hidden channels
1549 |         self.cv1 = Conv(c1, c_, 1, 1, act=False)
1550 |         self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
1551 |         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
1552 |         self.n = n
1553 |         self.add = shortcut and c1 == c2
1554 | 
1555 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1556 |         """Apply sequential pooling operations to input and return concatenated feature maps."""
1557 |         y = [self.cv1(x)]
1558 |         y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
1559 |         y = self.cv2(torch.cat(y, 1))
1560 |         return y + x if getattr(self, "add", False) else y
1561 | 
1562 | 
1563 | class C1(nn.Module):
1564 |     """CSP Bottleneck with 1 convolution."""
1565 | 
1566 |     def __init__(self, c1: int, c2: int, n: int = 1):
1567 |         """Initialize the CSP Bottleneck with 1 convolution.
1568 | 
1569 |         Args:
1570 |             c1 (int): Input channels.
1571 |             c2 (int): Output channels.
1572 |             n (int): Number of convolutions.
1573 |         """
1574 |         super().__init__()
1575 |         self.cv1 = Conv(c1, c2, 1, 1)
1576 |         self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))
1577 | 
1578 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1579 |         """Apply convolution and residual connection to input tensor."""
1580 |         y = self.cv1(x)
1581 |         return self.m(y) + y
1582 | 
1583 | 
1584 | class C2(nn.Module):
1585 |     """CSP Bottleneck with 2 convolutions."""
1586 | 
1587 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
1588 |         """Initialize a CSP Bottleneck with 2 convolutions.
1589 | 
1590 |         Args:
1591 |             c1 (int): Input channels.
1592 |             c2 (int): Output channels.
1593 |             n (int): Number of Bottleneck blocks.
1594 |             shortcut (bool): Whether to use shortcut connections.
1595 |             g (int): Groups for convolutions.
1596 |             e (float): Expansion ratio.
1597 |         """
1598 |         super().__init__()
1599 |         self.c = int(c2 * e)  # hidden channels
1600 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
1601 |         self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
1602 |         # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
1603 |         self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
1604 | 
1605 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1606 |         """Forward pass through the CSP bottleneck with 2 convolutions."""
1607 |         a, b = self.cv1(x).chunk(2, 1)
1608 |         return self.cv2(torch.cat((self.m(a), b), 1))
1609 | 
1610 | 
1611 | class C2f(nn.Module):
1612 |     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
1613 | 
1614 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
1615 |         """Initialize a CSP bottleneck with 2 convolutions.
1616 | 
1617 |         Args:
1618 |             c1 (int): Input channels.
1619 |             c2 (int): Output channels.
1620 |             n (int): Number of Bottleneck blocks.
1621 |             shortcut (bool): Whether to use shortcut connections.
1622 |             g (int): Groups for convolutions.
1623 |             e (float): Expansion ratio.
1624 |         """
1625 |         super().__init__()
1626 |         self.c = int(c2 * e)  # hidden channels
1627 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
1628 |         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
1629 |         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
1630 | 
1631 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1632 |         """Forward pass through C2f layer."""
1633 |         y = list(self.cv1(x).chunk(2, 1))
1634 |         y.extend(m(y[-1]) for m in self.m)
1635 |         return self.cv2(torch.cat(y, 1))
1636 | 
1637 |     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
1638 |         """Forward pass using split() instead of chunk()."""
1639 |         y = self.cv1(x).split((self.c, self.c), 1)
1640 |         y = [y[0], y[1]]
1641 |         y.extend(m(y[-1]) for m in self.m)
1642 |         return self.cv2(torch.cat(y, 1))
1643 | 
1644 | 
1645 | class C2fCBAM(C2f):
1646 |     """C2f block followed by CBAM channel and spatial attention."""
1647 | 
1648 |     def __init__(
1649 |         self,
1650 |         c1: int,
1651 |         c2: int,
1652 |         n: int = 1,
1653 |         shortcut: bool = False,
1654 |         g: int = 1,
1655 |         e: float = 0.5,
1656 |         kernel_size: int = 7,
1657 |     ):
1658 |         """Initialize C2f with CBAM refinement on the output feature map."""
1659 |         super().__init__(c1, c2, n, shortcut, g, e)
1660 |         self.attn = CBAM(c2, kernel_size)
1661 | 
1662 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1663 |         """Apply C2f and refine its output with CBAM attention."""
1664 |         return self.attn(super().forward(x))
1665 | 
1666 |     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
1667 |         """Apply split-based C2f and refine its output with CBAM attention."""
1668 |         return self.attn(super().forward_split(x))
1669 | 
1670 | 
1671 | class RepC2f(C2f):
1672 |     """C2f module that replaces internal bottlenecks with RepBottleneck blocks."""
1673 | 
1674 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
1675 |         """Initialize RepC2f with the same interface and output shape as C2f."""
1676 |         super().__init__(c1, c2, n, shortcut, g, e)
1677 |         self.m = nn.ModuleList(RepBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
1678 | 
1679 | 
1680 | class EnSimAMEdgeRepC2f(RepC2f):
1681 |     """RepC2f with EnSimAM applied only to the second split branch."""
1682 | 
1683 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
1684 |         """Initialize branch-local EnSimAM refinement."""
1685 |         super().__init__(c1, c2, n, shortcut, g, e)
1686 |         self.edge = EnSimAM()
1687 | 
1688 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1689 |         """Apply edge refinement only on y[1], preserving the RepC2f topology."""
1690 |         y = list(self.cv1(x).chunk(2, 1))
1691 |         y[1] = self.edge(y[1])
1692 |         y.extend(m(y[-1]) for m in self.m)
1693 |         return self.cv2(torch.cat(y, 1))
1694 | 
1695 | 
1696 | class PoolingEdgeRepC2f(RepC2f):
1697 |     """RepC2f with pooling edge enhancement applied only to the second split branch."""
1698 | 
1699 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
1700 |         """Initialize branch-local pooling edge refinement."""
1701 |         super().__init__(c1, c2, n, shortcut, g, e)
1702 |         self.pool = nn.AvgPool2d(3, stride=1, padding=1)
1703 |         self.edge_conv = Conv(self.c, self.c, 3)
1704 | 
1705 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1706 |         """Apply edge refinement only on y[1], preserving the RepC2f topology."""
1707 |         y = list(self.cv1(x).chunk(2, 1))
1708 |         edge = y[1] - self.pool(y[1])
1709 |         edge = self.edge_conv(edge)
1710 |         y[1] = y[1] + edge
1711 |         y.extend(m(y[-1]) for m in self.m)
1712 |         return self.cv2(torch.cat(y, 1))
1713 | 
1714 | 
1715 | class C2fKV(nn.Module):
1716 |     """C2f block with PSA-style split and gated KV-compressed attention on the final hidden feature."""
1717 | 
1718 |     def __init__(
1719 |         self,
1720 |         c1: int,
1721 |         c2: int,
1722 |         n: int = 1,
1723 |         shortcut: bool = False,
1724 |         g: int = 1,
1725 |         e: float = 0.5,
1726 |         num_heads: int = 4,
1727 |         sr_ratio: int = 2,
1728 |         mode: str = "dwconv",
1729 |     ):
1730 |         """Initialize a C2f-style block with a lightweight PSA-style KV attention branch."""
1731 |         super().__init__()
1732 |         self.c = int(c2 * e)  # hidden channels
1733 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
1734 |         self.cv2 = Conv((2 + n) * self.c, c2, 1)
1735 |         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
1736 |         self.kv_c = max(1, self.c // 2)
1737 |         self.bypass_c = self.c - self.kv_c
1738 |         self.kv = KVCompressedAttention(self.kv_c, self.kv_c, num_heads, sr_ratio, mode)
1739 |         self.gamma = nn.Parameter(torch.zeros(1))
1740 | 
1741 |     def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
1742 |         """Refine one split of the final hidden feature with gated KV attention."""
1743 |         if self.bypass_c == 0:
1744 |             return x + self.gamma * (self.kv(x) - x)
1745 |         a, b = x.split((self.bypass_c, self.kv_c), dim=1)
1746 |         b = b + self.gamma * (self.kv(b) - b)
1747 |         return torch.cat((a, b), dim=1)
1748 | 
1749 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1750 |         """Forward pass through C2f with PSA-style KV refinement on the final hidden feature."""
1751 |         y = list(self.cv1(x).chunk(2, 1))
1752 |         y.extend(m(y[-1]) for m in self.m)
1753 |         y[-1] = self._refine_last(y[-1])
1754 |         return self.cv2(torch.cat(y, 1))
1755 | 
1756 |     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
1757 |         """Forward pass using split() instead of chunk()."""
1758 |         y = self.cv1(x).split((self.c, self.c), 1)
1759 |         y = [y[0], y[1]]
1760 |         y.extend(m(y[-1]) for m in self.m)
1761 |         y[-1] = self._refine_last(y[-1])
1762 |         return self.cv2(torch.cat(y, 1))
1763 | 
1764 | 
1765 | class C2fNAT(nn.Module):
1766 |     """C2f block with gated Neighborhood Attention on the final hidden feature."""
1767 | 
1768 |     def __init__(
1769 |         self,
1770 |         c1: int,
1771 |         c2: int,
1772 |         n: int = 1,
1773 |         shortcut: bool = False,
1774 |         g: int = 1,
1775 |         e: float = 0.5,
1776 |         num_heads: int = 4,
1777 |         kernel_size: int = 3,
1778 |     ):
1779 |         """Initialize a C2f-style block with a lightweight NAT refinement branch."""
1780 |         super().__init__()
1781 |         try:
1782 |             from natten import NeighborhoodAttention2D
1783 |         except ImportError as exc:
1784 |             raise ImportError(
1785 |                 "C2fNAT requires the 'natten' package. Install natten in the training environment before using "
1786 |                 "YAMLs that reference C2fNAT."
1787 |             ) from exc
1788 | 
1789 |         self.c = int(c2 * e)  # hidden channels
1790 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
1791 |         self.cv2 = Conv((2 + n) * self.c, c2, 1)
1792 |         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
1793 |         self.num_heads = _choose_attention_heads(self.c, num_heads)
1794 |         self.norm1 = nn.LayerNorm(self.c)
1795 |         self.attn = NeighborhoodAttention2D(embed_dim=self.c, num_heads=self.num_heads, kernel_size=int(kernel_size))
1796 |         self.norm2 = nn.LayerNorm(self.c)
1797 |         self.mlp = nn.Sequential(
1798 |             nn.Linear(self.c, 2 * self.c),
1799 |             nn.GELU(),
1800 |             nn.Dropout(0.1),
1801 |             nn.Linear(2 * self.c, self.c),
1802 |             nn.Dropout(0.1),
1803 |         )
1804 |         self.gamma = nn.Parameter(torch.zeros(1))
1805 | 
1806 |     def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
1807 |         """Refine one hidden split with NHWC Neighborhood Attention."""
1808 |         if x.device.type == "cpu" and x.requires_grad:
1809 |             return x
1810 |         x_nhwc = x.permute(0, 2, 3, 1).contiguous()
1811 |         att = self.attn(self.norm1(x_nhwc)) + x_nhwc
1812 |         refined = self.mlp(self.norm2(att)) + att
1813 |         refined = refined.permute(0, 3, 1, 2).contiguous()
1814 |         return x + self.gamma * (refined - x)
1815 | 
1816 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1817 |         """Forward pass through C2f with NAT refinement on the final hidden feature."""
1818 |         y = list(self.cv1(x).chunk(2, 1))
1819 |         y.extend(m(y[-1]) for m in self.m)
1820 |         y[-1] = self._refine_last(y[-1])
1821 |         return self.cv2(torch.cat(y, 1))
1822 | 
1823 |     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
1824 |         """Forward pass using split() instead of chunk()."""
1825 |         y = self.cv1(x).split((self.c, self.c), 1)
1826 |         y = [y[0], y[1]]
1827 |         y.extend(m(y[-1]) for m in self.m)
1828 |         y[-1] = self._refine_last(y[-1])
1829 |         return self.cv2(torch.cat(y, 1))
1830 | 
1831 | 
1832 | class NATBlock(nn.Module):
1833 |     """Neighborhood Attention Transformer block for YOLO."""
1834 | 
1835 |     def __init__(self, c1: int, c2: int, num_heads: int = 4, kernel_size: int = 7):
1836 |         """Initialize a Neighborhood Attention block."""
1837 |         super().__init__()
1838 |         try:
1839 |             from natten import NeighborhoodAttention2D
1840 |         except ImportError as exc:
1841 |             raise ImportError(
1842 |                 "NATBlock requires the 'natten' package. Install natten in the training environment before using "
1843 |                 "YAMLs that reference NATBlock."
1844 |             ) from exc
1845 | 
1846 |         self.c = c1
1847 |         self.num_heads = _choose_attention_heads(c1, num_heads)
1848 |         self.norm1 = nn.LayerNorm(c1)
1849 |         self.attn = NeighborhoodAttention2D(embed_dim=c1, num_heads=self.num_heads, kernel_size=int(kernel_size))
1850 |         self.norm2 = nn.LayerNorm(c1)
1851 |         self.mlp = nn.Sequential(
1852 |             nn.Linear(c1, 2 * c1),
1853 |             nn.GELU(),
1854 |             nn.Dropout(0.1),
1855 |             nn.Linear(2 * c1, c1),
1856 |             nn.Dropout(0.1),
1857 |         )
1858 |         self.gamma = nn.Parameter(torch.zeros(1))
1859 |         self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
1860 | 
1861 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1862 |         """Forward pass through NATBlock."""
1863 |         if x.device.type == "cpu" and x.requires_grad:
1864 |             return self.proj(x)
1865 |         x_nhwc = x.permute(0, 2, 3, 1).contiguous()
1866 |         att = self.attn(self.norm1(x_nhwc)) + x_nhwc
1867 |         refined = self.mlp(self.norm2(att)) + att
1868 |         refined = refined.permute(0, 3, 1, 2).contiguous()
1869 |         out = x + self.gamma * (refined - x)
1870 |         return self.proj(out)
1871 | 
1872 | 
1873 | class M3NATFuse(nn.Module):
1874 |     """Fuse P2/P3/P4 features into a P3-resolution feature and refine it with NAT."""
1875 | 
1876 |     def __init__(self, c1: list[int], c2: int, num_heads: int = 4, kernel_size: int = 3):
1877 |         """Initialize three-scale fusion.
1878 | 
1879 |         Args:
1880 |             c1: Input channels for [P2, P3, P4].
1881 |             c2: Output channels at P3 resolution.
1882 |             num_heads: Requested NAT heads.
1883 |             kernel_size: NAT neighborhood kernel size.
1884 |         """
1885 |         super().__init__()
1886 |         if len(c1) != 3:
1887 |             raise ValueError(f"M3NATFuse expects 3 input channel values for [P2, P3, P4], got {c1}.")
1888 |         try:
1889 |             from natten import NeighborhoodAttention2D
1890 |         except ImportError as exc:
1891 |             raise ImportError(
1892 |                 "M3NATFuse requires the 'natten' package. Install natten in the training environment before using "
1893 |                 "YAMLs that reference M3NATFuse."
1894 |             ) from exc
1895 | 
1896 |         self.p2_down = Conv(c1[0], c2, 3, 2)
1897 |         self.p3_proj = Conv(c1[1], c2, 3, 1)
1898 |         self.p4_proj = Conv(c1[2], c2, 3, 1)
1899 |         self.fuse = Conv(3 * c2, c2, 3, 1)
1900 |         self.num_heads = _choose_attention_heads(c2, num_heads)
1901 |         self.norm1 = nn.LayerNorm(c2)
1902 |         self.attn = NeighborhoodAttention2D(embed_dim=c2, num_heads=self.num_heads, kernel_size=int(kernel_size))
1903 |         self.norm2 = nn.LayerNorm(c2)
1904 |         self.mlp = nn.Sequential(
1905 |             nn.Linear(c2, 2 * c2),
1906 |             nn.GELU(),
1907 |             nn.Dropout(0.1),
1908 |             nn.Linear(2 * c2, c2),
1909 |             nn.Dropout(0.1),
1910 |         )
1911 |         self.gamma = nn.Parameter(torch.zeros(1))
1912 | 
1913 |     def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
1914 |         """Fuse [P2, P3, P4] into a P3-resolution feature map."""
1915 |         if len(x) != 3:
1916 |             raise ValueError(f"M3NATFuse expects [P2, P3, P4], got {len(x)} inputs.")
1917 |         p2, p3, p4 = x
1918 |         target_size = p3.shape[-2:]
1919 |         p2 = self.p2_down(p2)
1920 |         if p2.shape[-2:] != target_size:
1921 |             p2 = F.interpolate(p2, size=target_size, mode="nearest")
1922 |         p3 = self.p3_proj(p3)
1923 |         p4 = F.interpolate(p4, size=target_size, mode="nearest")
1924 |         p4 = self.p4_proj(p4)
1925 |         fused = self.fuse(torch.cat((p2, p3, p4), dim=1))
1926 |         if fused.device.type == "cpu" and fused.requires_grad:
1927 |             return fused
1928 | 
1929 |         fused_nhwc = fused.permute(0, 2, 3, 1).contiguous()
1930 |         att = self.attn(self.norm1(fused_nhwc)) + fused_nhwc
1931 |         refined = self.mlp(self.norm2(att)) + att
1932 |         refined = refined.permute(0, 3, 1, 2).contiguous()
1933 |         return fused + self.gamma * (refined - fused)
1934 | 
1935 | 
1936 | class C3(nn.Module):
1937 |     """CSP Bottleneck with 3 convolutions."""
1938 | 
1939 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
1940 |         """Initialize the CSP Bottleneck with 3 convolutions.
1941 | 
1942 |         Args:
1943 |             c1 (int): Input channels.
1944 |             c2 (int): Output channels.
1945 |             n (int): Number of Bottleneck blocks.
1946 |             shortcut (bool): Whether to use shortcut connections.
1947 |             g (int): Groups for convolutions.
1948 |             e (float): Expansion ratio.
1949 |         """
1950 |         super().__init__()
1951 |         c_ = int(c2 * e)  # hidden channels
1952 |         self.cv1 = Conv(c1, c_, 1, 1)
1953 |         self.cv2 = Conv(c1, c_, 1, 1)
1954 |         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
1955 |         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
1956 | 
1957 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1958 |         """Forward pass through the CSP bottleneck with 3 convolutions."""
1959 |         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
1960 | 
1961 | 
1962 | class C3CBAM(C3):
1963 |     """C3 block followed by CBAM channel and spatial attention."""
1964 | 
1965 |     def __init__(
1966 |         self,
1967 |         c1: int,
1968 |         c2: int,
1969 |         n: int = 1,
1970 |         shortcut: bool = True,
1971 |         g: int = 1,
1972 |         e: float = 0.5,
1973 |         kernel_size: int = 7,
1974 |     ):
1975 |         """Initialize C3 with CBAM refinement on the output feature map."""
1976 |         super().__init__(c1, c2, n, shortcut, g, e)
1977 |         self.attn = CBAM(c2, kernel_size)
1978 | 
1979 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
1980 |         """Apply C3 and refine its output with CBAM attention."""
1981 |         return self.attn(super().forward(x))
1982 | 
1983 | 
1984 | class C3x(C3):
1985 |     """C3 module with cross-convolutions."""
1986 | 
1987 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
1988 |         """Initialize C3 module with cross-convolutions.
1989 | 
1990 |         Args:
1991 |             c1 (int): Input channels.
1992 |             c2 (int): Output channels.
1993 |             n (int): Number of Bottleneck blocks.
1994 |             shortcut (bool): Whether to use shortcut connections.
1995 |             g (int): Groups for convolutions.
1996 |             e (float): Expansion ratio.
1997 |         """
1998 |         super().__init__(c1, c2, n, shortcut, g, e)
1999 |         self.c_ = int(c2 * e)
2000 |         self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))
2001 | 
2002 | 
2003 | class RepC3(nn.Module):
2004 |     """Rep C3."""
2005 | 
2006 |     def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
2007 |         """Initialize RepC3 module with RepConv blocks.
2008 | 
2009 |         Args:
2010 |             c1 (int): Input channels.
2011 |             c2 (int): Output channels.
2012 |             n (int): Number of RepConv blocks.
2013 |             e (float): Expansion ratio.
2014 |         """
2015 |         super().__init__()
2016 |         c_ = int(c2 * e)  # hidden channels
2017 |         self.cv1 = Conv(c1, c_, 1, 1)
2018 |         self.cv2 = Conv(c1, c_, 1, 1)
2019 |         self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
2020 |         self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
2021 | 
2022 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2023 |         """Forward pass of RepC3 module."""
2024 |         return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
2025 | 
2026 | 
2027 | class C3TR(C3):
2028 |     """C3 module with TransformerBlock()."""
2029 | 
2030 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
2031 |         """Initialize C3 module with TransformerBlock.
2032 | 
2033 |         Args:
2034 |             c1 (int): Input channels.
2035 |             c2 (int): Output channels.
2036 |             n (int): Number of Transformer blocks.
2037 |             shortcut (bool): Whether to use shortcut connections.
2038 |             g (int): Groups for convolutions.
2039 |             e (float): Expansion ratio.
2040 |         """
2041 |         super().__init__(c1, c2, n, shortcut, g, e)
2042 |         c_ = int(c2 * e)
2043 |         self.m = TransformerBlock(c_, c_, 4, n)
2044 | 
2045 | 
2046 | class C3Ghost(C3):
2047 |     """C3 module with GhostBottleneck()."""
2048 | 
2049 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
2050 |         """Initialize C3 module with GhostBottleneck.
2051 | 
2052 |         Args:
2053 |             c1 (int): Input channels.
2054 |             c2 (int): Output channels.
2055 |             n (int): Number of Ghost bottleneck blocks.
2056 |             shortcut (bool): Whether to use shortcut connections.
2057 |             g (int): Groups for convolutions.
2058 |             e (float): Expansion ratio.
2059 |         """
2060 |         super().__init__(c1, c2, n, shortcut, g, e)
2061 |         c_ = int(c2 * e)  # hidden channels
2062 |         self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
2063 | 
2064 | 
2065 | class GhostBottleneck(nn.Module):
2066 |     """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""
2067 | 
2068 |     def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
2069 |         """Initialize Ghost Bottleneck module.
2070 | 
2071 |         Args:
2072 |             c1 (int): Input channels.
2073 |             c2 (int): Output channels.
2074 |             k (int): Kernel size.
2075 |             s (int): Stride.
2076 |         """
2077 |         super().__init__()
2078 |         c_ = c2 // 2
2079 |         self.conv = nn.Sequential(
2080 |             GhostConv(c1, c_, 1, 1),  # pw
2081 |             DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
2082 |             GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
2083 |         )
2084 |         self.shortcut = (
2085 |             nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
2086 |         )
2087 | 
2088 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2089 |         """Apply skip connection and addition to input tensor."""
2090 |         return self.conv(x) + self.shortcut(x)
2091 | 
2092 | 
2093 | class Bottleneck(nn.Module):
2094 |     """Standard bottleneck."""
2095 | 
2096 |     def __init__(
2097 |         self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
2098 |     ):
2099 |         """Initialize a standard bottleneck module.
2100 | 
2101 |         Args:
2102 |             c1 (int): Input channels.
2103 |             c2 (int): Output channels.
2104 |             shortcut (bool): Whether to use shortcut connection.
2105 |             g (int): Groups for convolutions.
2106 |             k (tuple): Kernel sizes for convolutions.
2107 |             e (float): Expansion ratio.
2108 |         """
2109 |         super().__init__()
2110 |         c_ = int(c2 * e)  # hidden channels
2111 |         self.cv1 = Conv(c1, c_, k[0], 1)
2112 |         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
2113 |         self.add = shortcut and c1 == c2
2114 | 
2115 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2116 |         """Apply bottleneck with optional shortcut connection."""
2117 |         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
2118 | 
2119 | 
2120 | class BottleneckCSP(nn.Module):
2121 |     """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""
2122 | 
2123 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
2124 |         """Initialize CSP Bottleneck.
2125 | 
2126 |         Args:
2127 |             c1 (int): Input channels.
2128 |             c2 (int): Output channels.
2129 |             n (int): Number of Bottleneck blocks.
2130 |             shortcut (bool): Whether to use shortcut connections.
2131 |             g (int): Groups for convolutions.
2132 |             e (float): Expansion ratio.
2133 |         """
2134 |         super().__init__()
2135 |         c_ = int(c2 * e)  # hidden channels
2136 |         self.cv1 = Conv(c1, c_, 1, 1)
2137 |         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
2138 |         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
2139 |         self.cv4 = Conv(2 * c_, c2, 1, 1)
2140 |         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
2141 |         self.act = nn.SiLU()
2142 |         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
2143 | 
2144 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2145 |         """Apply CSP bottleneck with 4 convolutions."""
2146 |         y1 = self.cv3(self.m(self.cv1(x)))
2147 |         y2 = self.cv2(x)
2148 |         return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
2149 | 
2150 | 
2151 | class ResNetBlock(nn.Module):
2152 |     """ResNet block with standard convolution layers."""
2153 | 
2154 |     def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
2155 |         """Initialize ResNet block.
2156 | 
2157 |         Args:
2158 |             c1 (int): Input channels.
2159 |             c2 (int): Output channels.
2160 |             s (int): Stride.
2161 |             e (int): Expansion ratio.
2162 |         """
2163 |         super().__init__()
2164 |         c3 = e * c2
2165 |         self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
2166 |         self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
2167 |         self.cv3 = Conv(c2, c3, k=1, act=False)
2168 |         self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()
2169 | 
2170 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2171 |         """Forward pass through the ResNet block."""
2172 |         return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))
2173 | 
2174 | 
2175 | class ResNetLayer(nn.Module):
2176 |     """ResNet layer with multiple ResNet blocks."""
2177 | 
2178 |     def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
2179 |         """Initialize ResNet layer.
2180 | 
2181 |         Args:
2182 |             c1 (int): Input channels.
2183 |             c2 (int): Output channels.
2184 |             s (int): Stride.
2185 |             is_first (bool): Whether this is the first layer.
2186 |             n (int): Number of ResNet blocks.
2187 |             e (int): Expansion ratio.
2188 |         """
2189 |         super().__init__()
2190 |         self.is_first = is_first
2191 | 
2192 |         if self.is_first:
2193 |             self.layer = nn.Sequential(
2194 |                 Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
2195 |             )
2196 |         else:
2197 |             blocks = [ResNetBlock(c1, c2, s, e=e)]
2198 |             blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
2199 |             self.layer = nn.Sequential(*blocks)
2200 | 
2201 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2202 |         """Forward pass through the ResNet layer."""
2203 |         return self.layer(x)
2204 | 
2205 | 
2206 | class MaxSigmoidAttnBlock(nn.Module):
2207 |     """Max Sigmoid attention block."""
2208 | 
2209 |     def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
2210 |         """Initialize MaxSigmoidAttnBlock.
2211 | 
2212 |         Args:
2213 |             c1 (int): Input channels.
2214 |             c2 (int): Output channels.
2215 |             nh (int): Number of heads.
2216 |             ec (int): Embedding channels.
2217 |             gc (int): Guide channels.
2218 |             scale (bool): Whether to use learnable scale parameter.
2219 |         """
2220 |         super().__init__()
2221 |         self.nh = nh
2222 |         self.hc = c2 // nh
2223 |         self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
2224 |         self.gl = nn.Linear(gc, ec)
2225 |         self.bias = nn.Parameter(torch.zeros(nh))
2226 |         self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
2227 |         self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
2228 | 
2229 |     def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
2230 |         """Forward pass of MaxSigmoidAttnBlock.
2231 | 
2232 |         Args:
2233 |             x (torch.Tensor): Input tensor.
2234 |             guide (torch.Tensor): Guide tensor.
2235 | 
2236 |         Returns:
2237 |             (torch.Tensor): Output tensor after attention.
2238 |         """
2239 |         bs, _, h, w = x.shape
2240 | 
2241 |         guide = self.gl(guide)
2242 |         guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
2243 |         embed = self.ec(x) if self.ec is not None else x
2244 |         embed = embed.view(bs, self.nh, self.hc, h, w)
2245 | 
2246 |         aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
2247 |         aw = aw.max(dim=-1)[0]
2248 |         aw = aw / (self.hc**0.5)
2249 |         aw = aw + self.bias[None, :, None, None]
2250 |         aw = aw.sigmoid() * self.scale
2251 | 
2252 |         x = self.proj_conv(x)
2253 |         x = x.view(bs, self.nh, -1, h, w)
2254 |         x = x * aw.unsqueeze(2)
2255 |         return x.view(bs, -1, h, w)
2256 | 
2257 | 
2258 | class C2fAttn(nn.Module):
2259 |     """C2f module with an additional attn module."""
2260 | 
2261 |     def __init__(
2262 |         self,
2263 |         c1: int,
2264 |         c2: int,
2265 |         n: int = 1,
2266 |         ec: int = 128,
2267 |         nh: int = 1,
2268 |         gc: int = 512,
2269 |         shortcut: bool = False,
2270 |         g: int = 1,
2271 |         e: float = 0.5,
2272 |     ):
2273 |         """Initialize C2f module with attention mechanism.
2274 | 
2275 |         Args:
2276 |             c1 (int): Input channels.
2277 |             c2 (int): Output channels.
2278 |             n (int): Number of Bottleneck blocks.
2279 |             ec (int): Embedding channels for attention.
2280 |             nh (int): Number of heads for attention.
2281 |             gc (int): Guide channels for attention.
2282 |             shortcut (bool): Whether to use shortcut connections.
2283 |             g (int): Groups for convolutions.
2284 |             e (float): Expansion ratio.
2285 |         """
2286 |         super().__init__()
2287 |         self.c = int(c2 * e)  # hidden channels
2288 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
2289 |         self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
2290 |         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
2291 |         self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
2292 | 
2293 |     def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
2294 |         """Forward pass through C2f layer with attention.
2295 | 
2296 |         Args:
2297 |             x (torch.Tensor): Input tensor.
2298 |             guide (torch.Tensor): Guide tensor for attention.
2299 | 
2300 |         Returns:
2301 |             (torch.Tensor): Output tensor after processing.
2302 |         """
2303 |         y = list(self.cv1(x).chunk(2, 1))
2304 |         y.extend(m(y[-1]) for m in self.m)
2305 |         y.append(self.attn(y[-1], guide))
2306 |         return self.cv2(torch.cat(y, 1))
2307 | 
2308 |     def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
2309 |         """Forward pass using split() instead of chunk().
2310 | 
2311 |         Args:
2312 |             x (torch.Tensor): Input tensor.
2313 |             guide (torch.Tensor): Guide tensor for attention.
2314 | 
2315 |         Returns:
2316 |             (torch.Tensor): Output tensor after processing.
2317 |         """
2318 |         y = list(self.cv1(x).split((self.c, self.c), 1))
2319 |         y.extend(m(y[-1]) for m in self.m)
2320 |         y.append(self.attn(y[-1], guide))
2321 |         return self.cv2(torch.cat(y, 1))
2322 | 
2323 | 
2324 | class ImagePoolingAttn(nn.Module):
2325 |     """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""
2326 | 
2327 |     def __init__(
2328 |         self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
2329 |     ):
2330 |         """Initialize ImagePoolingAttn module.
2331 | 
2332 |         Args:
2333 |             ec (int): Embedding channels.
2334 |             ch (tuple): Channel dimensions for feature maps.
2335 |             ct (int): Channel dimension for text embeddings.
2336 |             nh (int): Number of attention heads.
2337 |             k (int): Kernel size for pooling.
2338 |             scale (bool): Whether to use learnable scale parameter.
2339 |         """
2340 |         super().__init__()
2341 | 
2342 |         nf = len(ch)
2343 |         self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
2344 |         self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
2345 |         self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
2346 |         self.proj = nn.Linear(ec, ct)
2347 |         self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
2348 |         self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
2349 |         self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
2350 |         self.ec = ec
2351 |         self.nh = nh
2352 |         self.nf = nf
2353 |         self.hc = ec // nh
2354 |         self.k = k
2355 | 
2356 |     def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
2357 |         """Forward pass of ImagePoolingAttn.
2358 | 
2359 |         Args:
2360 |             x (list[torch.Tensor]): List of input feature maps.
2361 |             text (torch.Tensor): Text embeddings.
2362 | 
2363 |         Returns:
2364 |             (torch.Tensor): Enhanced text embeddings.
2365 |         """
2366 |         bs = x[0].shape[0]
2367 |         assert len(x) == self.nf
2368 |         num_patches = self.k**2
2369 |         x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
2370 |         x = torch.cat(x, dim=-1).transpose(1, 2)
2371 |         q = self.query(text)
2372 |         k = self.key(x)
2373 |         v = self.value(x)
2374 | 
2375 |         # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
2376 |         q = q.reshape(bs, -1, self.nh, self.hc)
2377 |         k = k.reshape(bs, -1, self.nh, self.hc)
2378 |         v = v.reshape(bs, -1, self.nh, self.hc)
2379 | 
2380 |         aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
2381 |         aw = aw / (self.hc**0.5)
2382 |         aw = F.softmax(aw, dim=-1)
2383 | 
2384 |         x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
2385 |         x = self.proj(x.reshape(bs, -1, self.ec))
2386 |         return x * self.scale + text
2387 | 
2388 | 
2389 | class ContrastiveHead(nn.Module):
2390 |     """Implements contrastive learning head for region-text similarity in vision-language models."""
2391 | 
2392 |     def __init__(self):
2393 |         """Initialize ContrastiveHead with region-text similarity parameters."""
2394 |         super().__init__()
2395 |         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
2396 |         self.bias = nn.Parameter(torch.tensor([-10.0]))
2397 |         self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
2398 | 
2399 |     def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
2400 |         """Forward function of contrastive learning.
2401 | 
2402 |         Args:
2403 |             x (torch.Tensor): Image features.
2404 |             w (torch.Tensor): Text features.
2405 | 
2406 |         Returns:
2407 |             (torch.Tensor): Similarity scores.
2408 |         """
2409 |         x = F.normalize(x, dim=1, p=2)
2410 |         w = F.normalize(w, dim=-1, p=2)
2411 |         x = torch.einsum("bchw,bkc->bkhw", x, w)
2412 |         return x * self.logit_scale.exp() + self.bias
2413 | 
2414 | 
2415 | class BNContrastiveHead(nn.Module):
2416 |     """Batch Norm Contrastive Head using batch norm instead of l2-normalization.
2417 | 
2418 |     Args:
2419 |         embed_dims (int): Embed dimensions of text and image features.
2420 |     """
2421 | 
2422 |     def __init__(self, embed_dims: int):
2423 |         """Initialize BNContrastiveHead.
2424 | 
2425 |         Args:
2426 |             embed_dims (int): Embedding dimensions for features.
2427 |         """
2428 |         super().__init__()
2429 |         self.norm = nn.BatchNorm2d(embed_dims)
2430 |         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
2431 |         self.bias = nn.Parameter(torch.tensor([-10.0]))
2432 |         # use -1.0 is more stable
2433 |         self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
2434 | 
2435 |     def fuse(self):
2436 |         """Fuse the batch normalization layer in the BNContrastiveHead module."""
2437 |         del self.norm
2438 |         del self.bias
2439 |         del self.logit_scale
2440 |         self.forward = self.forward_fuse
2441 | 
2442 |     @staticmethod
2443 |     def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
2444 |         """Passes image features through unchanged after fusing."""
2445 |         return x
2446 | 
2447 |     def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
2448 |         """Forward function of contrastive learning with batch normalization.
2449 | 
2450 |         Args:
2451 |             x (torch.Tensor): Image features.
2452 |             w (torch.Tensor): Text features.
2453 | 
2454 |         Returns:
2455 |             (torch.Tensor): Similarity scores.
2456 |         """
2457 |         x = self.norm(x)
2458 |         w = F.normalize(w, dim=-1, p=2)
2459 | 
2460 |         x = torch.einsum("bchw,bkc->bkhw", x, w)
2461 |         return x * self.logit_scale.exp() + self.bias
2462 | 
2463 | 
2464 | class RepBottleneck(Bottleneck):
2465 |     """Rep bottleneck."""
2466 | 
2467 |     def __init__(
2468 |         self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
2469 |     ):
2470 |         """Initialize RepBottleneck.
2471 | 
2472 |         Args:
2473 |             c1 (int): Input channels.
2474 |             c2 (int): Output channels.
2475 |             shortcut (bool): Whether to use shortcut connection.
2476 |             g (int): Groups for convolutions.
2477 |             k (tuple): Kernel sizes for convolutions.
2478 |             e (float): Expansion ratio.
2479 |         """
2480 |         super().__init__(c1, c2, shortcut, g, k, e)
2481 |         c_ = int(c2 * e)  # hidden channels
2482 |         self.cv1 = RepConv(c1, c_, k[0], 1)
2483 | 
2484 | 
2485 | class RepCSP(C3):
2486 |     """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""
2487 | 
2488 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
2489 |         """Initialize RepCSP layer.
2490 | 
2491 |         Args:
2492 |             c1 (int): Input channels.
2493 |             c2 (int): Output channels.
2494 |             n (int): Number of RepBottleneck blocks.
2495 |             shortcut (bool): Whether to use shortcut connections.
2496 |             g (int): Groups for convolutions.
2497 |             e (float): Expansion ratio.
2498 |         """
2499 |         super().__init__(c1, c2, n, shortcut, g, e)
2500 |         c_ = int(c2 * e)  # hidden channels
2501 |         self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
2502 | 
2503 | 
2504 | class RepNCSPELAN4(nn.Module):
2505 |     """CSP-ELAN."""
2506 | 
2507 |     def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
2508 |         """Initialize CSP-ELAN layer.
2509 | 
2510 |         Args:
2511 |             c1 (int): Input channels.
2512 |             c2 (int): Output channels.
2513 |             c3 (int): Intermediate channels.
2514 |             c4 (int): Intermediate channels for RepCSP.
2515 |             n (int): Number of RepCSP blocks.
2516 |         """
2517 |         super().__init__()
2518 |         self.c = c3 // 2
2519 |         self.cv1 = Conv(c1, c3, 1, 1)
2520 |         self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
2521 |         self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
2522 |         self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
2523 | 
2524 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2525 |         """Forward pass through RepNCSPELAN4 layer."""
2526 |         y = list(self.cv1(x).chunk(2, 1))
2527 |         y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
2528 |         return self.cv4(torch.cat(y, 1))
2529 | 
2530 |     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
2531 |         """Forward pass using split() instead of chunk()."""
2532 |         y = list(self.cv1(x).split((self.c, self.c), 1))
2533 |         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
2534 |         return self.cv4(torch.cat(y, 1))
2535 | 
2536 | 
2537 | class ELAN1(RepNCSPELAN4):
2538 |     """ELAN1 module with 4 convolutions."""
2539 | 
2540 |     def __init__(self, c1: int, c2: int, c3: int, c4: int):
2541 |         """Initialize ELAN1 layer.
2542 | 
2543 |         Args:
2544 |             c1 (int): Input channels.
2545 |             c2 (int): Output channels.
2546 |             c3 (int): Intermediate channels.
2547 |             c4 (int): Intermediate channels for convolutions.
2548 |         """
2549 |         super().__init__(c1, c2, c3, c4)
2550 |         self.c = c3 // 2
2551 |         self.cv1 = Conv(c1, c3, 1, 1)
2552 |         self.cv2 = Conv(c3 // 2, c4, 3, 1)
2553 |         self.cv3 = Conv(c4, c4, 3, 1)
2554 |         self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
2555 | 
2556 | 
2557 | class AConv(nn.Module):
2558 |     """AConv."""
2559 | 
2560 |     def __init__(self, c1: int, c2: int):
2561 |         """Initialize AConv module.
2562 | 
2563 |         Args:
2564 |             c1 (int): Input channels.
2565 |             c2 (int): Output channels.
2566 |         """
2567 |         super().__init__()
2568 |         self.cv1 = Conv(c1, c2, 3, 2, 1)
2569 | 
2570 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2571 |         """Forward pass through AConv layer."""
2572 |         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
2573 |         return self.cv1(x)
2574 | 
2575 | 
2576 | class ADown(nn.Module):
2577 |     """ADown."""
2578 | 
2579 |     def __init__(self, c1: int, c2: int):
2580 |         """Initialize ADown module.
2581 | 
2582 |         Args:
2583 |             c1 (int): Input channels.
2584 |             c2 (int): Output channels.
2585 |         """
2586 |         super().__init__()
2587 |         self.c = c2 // 2
2588 |         self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
2589 |         self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
2590 | 
2591 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2592 |         """Forward pass through ADown layer."""
2593 |         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
2594 |         x1, x2 = x.chunk(2, 1)
2595 |         x1 = self.cv1(x1)
2596 |         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
2597 |         x2 = self.cv2(x2)
2598 |         return torch.cat((x1, x2), 1)
2599 | 
2600 | 
2601 | class SPPELAN(nn.Module):
2602 |     """SPP-ELAN."""
2603 | 
2604 |     def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
2605 |         """Initialize SPP-ELAN block.
2606 | 
2607 |         Args:
2608 |             c1 (int): Input channels.
2609 |             c2 (int): Output channels.
2610 |             c3 (int): Intermediate channels.
2611 |             k (int): Kernel size for max pooling.
2612 |         """
2613 |         super().__init__()
2614 |         self.c = c3
2615 |         self.cv1 = Conv(c1, c3, 1, 1)
2616 |         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
2617 |         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
2618 |         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
2619 |         self.cv5 = Conv(4 * c3, c2, 1, 1)
2620 | 
2621 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2622 |         """Forward pass through SPPELAN layer."""
2623 |         y = [self.cv1(x)]
2624 |         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
2625 |         return self.cv5(torch.cat(y, 1))
2626 | 
2627 | 
2628 | class CBLinear(nn.Module):
2629 |     """CBLinear."""
2630 | 
2631 |     def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
2632 |         """Initialize CBLinear module.
2633 | 
2634 |         Args:
2635 |             c1 (int): Input channels.
2636 |             c2s (list[int]): List of output channel sizes.
2637 |             k (int): Kernel size.
2638 |             s (int): Stride.
2639 |             p (int | None): Padding.
2640 |             g (int): Groups.
2641 |         """
2642 |         super().__init__()
2643 |         self.c2s = c2s
2644 |         self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)
2645 | 
2646 |     def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
2647 |         """Forward pass through CBLinear layer."""
2648 |         return self.conv(x).split(self.c2s, dim=1)
2649 | 
2650 | 
2651 | class CBFuse(nn.Module):
2652 |     """CBFuse."""
2653 | 
2654 |     def __init__(self, idx: list[int]):
2655 |         """Initialize CBFuse module.
2656 | 
2657 |         Args:
2658 |             idx (list[int]): Indices for feature selection.
2659 |         """
2660 |         super().__init__()
2661 |         self.idx = idx
2662 | 
2663 |     def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
2664 |         """Forward pass through CBFuse layer.
2665 | 
2666 |         Args:
2667 |             xs (list[torch.Tensor]): List of input tensors.
2668 | 
2669 |         Returns:
2670 |             (torch.Tensor): Fused output tensor.
2671 |         """
2672 |         target_size = xs[-1].shape[2:]
2673 |         res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
2674 |         return torch.sum(torch.stack(res + xs[-1:]), dim=0)
2675 | 
2676 | 
2677 | class C3f(nn.Module):
2678 |     """Faster Implementation of CSP Bottleneck with 3 convolutions."""
2679 | 
2680 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
2681 |         """Initialize CSP bottleneck layer with three convolutions.
2682 | 
2683 |         Args:
2684 |             c1 (int): Input channels.
2685 |             c2 (int): Output channels.
2686 |             n (int): Number of Bottleneck blocks.
2687 |             shortcut (bool): Whether to use shortcut connections.
2688 |             g (int): Groups for convolutions.
2689 |             e (float): Expansion ratio.
2690 |         """
2691 |         super().__init__()
2692 |         c_ = int(c2 * e)  # hidden channels
2693 |         self.cv1 = Conv(c1, c_, 1, 1)
2694 |         self.cv2 = Conv(c1, c_, 1, 1)
2695 |         self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
2696 |         self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
2697 | 
2698 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2699 |         """Forward pass through C3f layer."""
2700 |         y = [self.cv2(x), self.cv1(x)]
2701 |         y.extend(m(y[-1]) for m in self.m)
2702 |         return self.cv3(torch.cat(y, 1))
2703 | 
2704 | 
2705 | class C3k2(C2f):
2706 |     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
2707 | 
2708 |     def __init__(
2709 |         self,
2710 |         c1: int,
2711 |         c2: int,
2712 |         n: int = 1,
2713 |         c3k: bool = False,
2714 |         e: float = 0.5,
2715 |         attn: bool = False,
2716 |         g: int = 1,
2717 |         shortcut: bool = True,
2718 |     ):
2719 |         """Initialize C3k2 module.
2720 | 
2721 |         Args:
2722 |             c1 (int): Input channels.
2723 |             c2 (int): Output channels.
2724 |             n (int): Number of blocks.
2725 |             c3k (bool): Whether to use C3k blocks.
2726 |             e (float): Expansion ratio.
2727 |             attn (bool): Whether to use attention blocks.
2728 |             g (int): Groups for convolutions.
2729 |             shortcut (bool): Whether to use shortcut connections.
2730 |         """
2731 |         super().__init__(c1, c2, n, shortcut, g, e)
2732 |         self.m = nn.ModuleList(
2733 |             nn.Sequential(
2734 |                 Bottleneck(self.c, self.c, shortcut, g),
2735 |                 PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
2736 |             )
2737 |             if attn
2738 |             else C3k(self.c, self.c, 2, shortcut, g)
2739 |             if c3k
2740 |             else Bottleneck(self.c, self.c, shortcut, g)
2741 |             for _ in range(n)
2742 |         )
2743 | 
2744 | 
2745 | class C3k(C3):
2746 |     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
2747 | 
2748 |     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
2749 |         """Initialize C3k module.
2750 | 
2751 |         Args:
2752 |             c1 (int): Input channels.
2753 |             c2 (int): Output channels.
2754 |             n (int): Number of Bottleneck blocks.
2755 |             shortcut (bool): Whether to use shortcut connections.
2756 |             g (int): Groups for convolutions.
2757 |             e (float): Expansion ratio.
2758 |             k (int): Kernel size.
2759 |         """
2760 |         super().__init__(c1, c2, n, shortcut, g, e)
2761 |         c_ = int(c2 * e)  # hidden channels
2762 |         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
2763 |         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
2764 | 
2765 | 
2766 | class RepVGGDW(torch.nn.Module):
2767 |     """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""
2768 | 
2769 |     def __init__(self, ed: int) -> None:
2770 |         """Initialize RepVGGDW module.
2771 | 
2772 |         Args:
2773 |             ed (int): Input and output channels.
2774 |         """
2775 |         super().__init__()
2776 |         self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
2777 |         self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
2778 |         self.dim = ed
2779 |         self.act = nn.SiLU()
2780 | 
2781 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2782 |         """Perform a forward pass of the RepVGGDW block.
2783 | 
2784 |         Args:
2785 |             x (torch.Tensor): Input tensor.
2786 | 
2787 |         Returns:
2788 |             (torch.Tensor): Output tensor after applying the depth-wise convolution.
2789 |         """
2790 |         return self.act(self.conv(x) + self.conv1(x))
2791 | 
2792 |     def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
2793 |         """Perform a forward pass of the fused RepVGGDW block.
2794 | 
2795 |         Args:
2796 |             x (torch.Tensor): Input tensor.
2797 | 
2798 |         Returns:
2799 |             (torch.Tensor): Output tensor after applying the depth-wise convolution.
2800 |         """
2801 |         return self.act(self.conv(x))
2802 | 
2803 |     @torch.no_grad()
2804 |     def fuse(self):
2805 |         """Fuse the convolutional layers in the RepVGGDW block.
2806 | 
2807 |         This method fuses the convolutional layers and updates the weights and biases accordingly.
2808 |         """
2809 |         if not hasattr(self, "conv1"):
2810 |             return  # already fused
2811 |         conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
2812 |         conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
2813 | 
2814 |         conv_w = conv.weight
2815 |         conv_b = conv.bias
2816 |         conv1_w = conv1.weight
2817 |         conv1_b = conv1.bias
2818 | 
2819 |         conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
2820 | 
2821 |         final_conv_w = conv_w + conv1_w
2822 |         final_conv_b = conv_b + conv1_b
2823 | 
2824 |         conv.weight.data.copy_(final_conv_w)
2825 |         conv.bias.data.copy_(final_conv_b)
2826 | 
2827 |         self.conv = conv
2828 |         del self.conv1
2829 | 
2830 | 
2831 | class CIB(nn.Module):
2832 |     """Compact Inverted Block (CIB) module.
2833 | 
2834 |     Args:
2835 |         c1 (int): Number of input channels.
2836 |         c2 (int): Number of output channels.
2837 |         shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
2838 |         e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
2839 |         lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
2840 |     """
2841 | 
2842 |     def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
2843 |         """Initialize the CIB module.
2844 | 
2845 |         Args:
2846 |             c1 (int): Input channels.
2847 |             c2 (int): Output channels.
2848 |             shortcut (bool): Whether to use shortcut connection.
2849 |             e (float): Expansion ratio.
2850 |             lk (bool): Whether to use RepVGGDW.
2851 |         """
2852 |         super().__init__()
2853 |         c_ = int(c2 * e)  # hidden channels
2854 |         self.cv1 = nn.Sequential(
2855 |             Conv(c1, c1, 3, g=c1),
2856 |             Conv(c1, 2 * c_, 1),
2857 |             RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
2858 |             Conv(2 * c_, c2, 1),
2859 |             Conv(c2, c2, 3, g=c2),
2860 |         )
2861 | 
2862 |         self.add = shortcut and c1 == c2
2863 | 
2864 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2865 |         """Forward pass of the CIB module.
2866 | 
2867 |         Args:
2868 |             x (torch.Tensor): Input tensor.
2869 | 
2870 |         Returns:
2871 |             (torch.Tensor): Output tensor.
2872 |         """
2873 |         return x + self.cv1(x) if self.add else self.cv1(x)
2874 | 
2875 | 
2876 | class C2fCIB(C2f):
2877 |     """C2fCIB class represents a convolutional block with C2f and CIB modules.
2878 | 
2879 |     Args:
2880 |         c1 (int): Number of input channels.
2881 |         c2 (int): Number of output channels.
2882 |         n (int, optional): Number of CIB modules to stack. Defaults to 1.
2883 |         shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
2884 |         lk (bool, optional): Whether to use large kernel. Defaults to False.
2885 |         g (int, optional): Number of groups for grouped convolution. Defaults to 1.
2886 |         e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
2887 |     """
2888 | 
2889 |     def __init__(
2890 |         self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
2891 |     ):
2892 |         """Initialize C2fCIB module.
2893 | 
2894 |         Args:
2895 |             c1 (int): Input channels.
2896 |             c2 (int): Output channels.
2897 |             n (int): Number of CIB modules.
2898 |             shortcut (bool): Whether to use shortcut connection.
2899 |             lk (bool): Whether to use large kernel.
2900 |             g (int): Groups for convolutions.
2901 |             e (float): Expansion ratio.
2902 |         """
2903 |         super().__init__(c1, c2, n, shortcut, g, e)
2904 |         self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
2905 | 
2906 | 
2907 | class Attention(nn.Module):
2908 |     """Attention module that performs self-attention on the input tensor.
2909 | 
2910 |     Args:
2911 |         dim (int): The input tensor dimension.
2912 |         num_heads (int): The number of attention heads.
2913 |         attn_ratio (float): The ratio of the attention key dimension to the head dimension.
2914 | 
2915 |     Attributes:
2916 |         num_heads (int): The number of attention heads.
2917 |         head_dim (int): The dimension of each attention head.
2918 |         key_dim (int): The dimension of the attention key.
2919 |         scale (float): The scaling factor for the attention scores.
2920 |         qkv (Conv): Convolutional layer for computing the query, key, and value.
2921 |         proj (Conv): Convolutional layer for projecting the attended values.
2922 |         pe (Conv): Convolutional layer for positional encoding.
2923 |     """
2924 | 
2925 |     def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
2926 |         """Initialize multi-head attention module.
2927 | 
2928 |         Args:
2929 |             dim (int): Input dimension.
2930 |             num_heads (int): Number of attention heads.
2931 |             attn_ratio (float): Attention ratio for key dimension.
2932 |         """
2933 |         super().__init__()
2934 |         self.num_heads = num_heads
2935 |         self.head_dim = dim // num_heads
2936 |         self.key_dim = int(self.head_dim * attn_ratio)
2937 |         self.scale = self.key_dim**-0.5
2938 |         nh_kd = self.key_dim * num_heads
2939 |         h = dim + nh_kd * 2
2940 |         self.qkv = Conv(dim, h, 1, act=False)
2941 |         self.proj = Conv(dim, dim, 1, act=False)
2942 |         self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
2943 | 
2944 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
2945 |         """Forward pass of the Attention module.
2946 | 
2947 |         Args:
2948 |             x (torch.Tensor): The input tensor.
2949 | 
2950 |         Returns:
2951 |             (torch.Tensor): The output tensor after self-attention.
2952 |         """
2953 |         B, C, H, W = x.shape
2954 |         N = H * W
2955 |         qkv = self.qkv(x)
2956 |         q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
2957 |             [self.key_dim, self.key_dim, self.head_dim], dim=2
2958 |         )
2959 | 
2960 |         attn = (q.transpose(-2, -1) @ k) * self.scale
2961 |         attn = attn.softmax(dim=-1)
2962 |         x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
2963 |         x = self.proj(x)
2964 |         return x
2965 | 
2966 | 
2967 | class PSABlock(nn.Module):
2968 |     """PSABlock class implementing a Position-Sensitive Attention block for neural networks.
2969 | 
2970 |     This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
2971 |     with optional shortcut connections.
2972 | 
2973 |     Attributes:
2974 |         attn (Attention): Multi-head attention module.
2975 |         ffn (nn.Sequential): Feed-forward neural network module.
2976 |         add (bool): Flag indicating whether to add shortcut connections.
2977 | 
2978 |     Methods:
2979 |         forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
2980 | 
2981 |     Examples:
2982 |         Create a PSABlock and perform a forward pass
2983 |         >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
2984 |         >>> input_tensor = torch.randn(1, 128, 32, 32)
2985 |         >>> output_tensor = psablock(input_tensor)
2986 |     """
2987 | 
2988 |     def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
2989 |         """Initialize the PSABlock.
2990 | 
2991 |         Args:
2992 |             c (int): Input and output channels.
2993 |             attn_ratio (float): Attention ratio for key dimension.
2994 |             num_heads (int): Number of attention heads.
2995 |             shortcut (bool): Whether to use shortcut connections.
2996 |         """
2997 |         super().__init__()
2998 | 
2999 |         self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
3000 |         self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
3001 |         self.add = shortcut
3002 | 
3003 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3004 |         """Execute a forward pass through PSABlock.
3005 | 
3006 |         Args:
3007 |             x (torch.Tensor): Input tensor.
3008 | 
3009 |         Returns:
3010 |             (torch.Tensor): Output tensor after attention and feed-forward processing.
3011 |         """
3012 |         x = x + self.attn(x) if self.add else self.attn(x)
3013 |         x = x + self.ffn(x) if self.add else self.ffn(x)
3014 |         return x
3015 | 
3016 | 
3017 | class PSA(nn.Module):
3018 |     """PSA class for implementing Position-Sensitive Attention in neural networks.
3019 | 
3020 |     This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
3021 |     input tensors, enhancing feature extraction and processing capabilities.
3022 | 
3023 |     Attributes:
3024 |         c (int): Number of hidden channels after applying the initial convolution.
3025 |         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
3026 |         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
3027 |         attn (Attention): Attention module for position-sensitive attention.
3028 |         ffn (nn.Sequential): Feed-forward network for further processing.
3029 | 
3030 |     Methods:
3031 |         forward: Applies position-sensitive attention and feed-forward network to the input tensor.
3032 | 
3033 |     Examples:
3034 |         Create a PSA module and apply it to an input tensor
3035 |         >>> psa = PSA(c1=128, c2=128, e=0.5)
3036 |         >>> input_tensor = torch.randn(1, 128, 64, 64)
3037 |         >>> output_tensor = psa.forward(input_tensor)
3038 |     """
3039 | 
3040 |     def __init__(self, c1: int, c2: int, e: float = 0.5):
3041 |         """Initialize PSA module.
3042 | 
3043 |         Args:
3044 |             c1 (int): Input channels.
3045 |             c2 (int): Output channels.
3046 |             e (float): Expansion ratio.
3047 |         """
3048 |         super().__init__()
3049 |         assert c1 == c2
3050 |         self.c = int(c1 * e)
3051 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
3052 |         self.cv2 = Conv(2 * self.c, c1, 1)
3053 | 
3054 |         self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
3055 |         self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))
3056 | 
3057 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3058 |         """Execute forward pass in PSA module.
3059 | 
3060 |         Args:
3061 |             x (torch.Tensor): Input tensor.
3062 | 
3063 |         Returns:
3064 |             (torch.Tensor): Output tensor after attention and feed-forward processing.
3065 |         """
3066 |         a, b = self.cv1(x).split((self.c, self.c), dim=1)
3067 |         b = b + self.attn(b)
3068 |         b = b + self.ffn(b)
3069 |         return self.cv2(torch.cat((a, b), 1))
3070 | 
3071 | 
3072 | class C2PSA(nn.Module):
3073 |     """C2PSA module with attention mechanism for enhanced feature extraction and processing.
3074 | 
3075 |     This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
3076 |     capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
3077 | 
3078 |     Attributes:
3079 |         c (int): Number of hidden channels.
3080 |         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
3081 |         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
3082 |         m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
3083 | 
3084 |     Methods:
3085 |         forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
3086 | 
3087 |     Examples:
3088 |         >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
3089 |         >>> input_tensor = torch.randn(1, 256, 64, 64)
3090 |         >>> output_tensor = c2psa(input_tensor)
3091 | 
3092 |     Notes:
3093 |         This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
3094 |     """
3095 | 
3096 |     def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
3097 |         """Initialize C2PSA module.
3098 | 
3099 |         Args:
3100 |             c1 (int): Input channels.
3101 |             c2 (int): Output channels.
3102 |             n (int): Number of PSABlock modules.
3103 |             e (float): Expansion ratio.
3104 |         """
3105 |         super().__init__()
3106 |         assert c1 == c2
3107 |         self.c = int(c1 * e)
3108 |         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
3109 |         self.cv2 = Conv(2 * self.c, c1, 1)
3110 | 
3111 |         self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
3112 | 
3113 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3114 |         """Process the input tensor through a series of PSA blocks.
3115 | 
3116 |         Args:
3117 |             x (torch.Tensor): Input tensor.
3118 | 
3119 |         Returns:
3120 |             (torch.Tensor): Output tensor after processing.
3121 |         """
3122 |         a, b = self.cv1(x).split((self.c, self.c), dim=1)
3123 |         b = self.m(b)
3124 |         return self.cv2(torch.cat((a, b), 1))
3125 | 
3126 | 
3127 | class C2fPSA(C2f):
3128 |     """C2fPSA module with enhanced feature extraction using PSA blocks.
3129 | 
3130 |     This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
3131 |     extraction.
3132 | 
3133 |     Attributes:
3134 |         c (int): Number of hidden channels.
3135 |         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
3136 |         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
3137 |         m (nn.ModuleList): List of PSABlock modules for feature extraction.
3138 | 
3139 |     Methods:
3140 |         forward: Performs a forward pass through the C2fPSA module.
3141 |         forward_split: Performs a forward pass using split() instead of chunk().
3142 | 
3143 |     Examples:
3144 |         >>> import torch
3145 |         >>> from ultralytics.nn.modules.block import C2fPSA
3146 |         >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
3147 |         >>> x = torch.randn(1, 64, 128, 128)
3148 |         >>> output = model(x)
3149 |         >>> print(output.shape)
3150 |     """
3151 | 
3152 |     def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
3153 |         """Initialize C2fPSA module.
3154 | 
3155 |         Args:
3156 |             c1 (int): Input channels.
3157 |             c2 (int): Output channels.
3158 |             n (int): Number of PSABlock modules.
3159 |             e (float): Expansion ratio.
3160 |         """
3161 |         assert c1 == c2
3162 |         super().__init__(c1, c2, n=n, e=e)
3163 |         self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
3164 | 
3165 | 
3166 | class SCDown(nn.Module):
3167 |     """SCDown module for downsampling with separable convolutions.
3168 | 
3169 |     This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
3170 |     efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.
3171 | 
3172 |     Attributes:
3173 |         cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
3174 |         cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.
3175 | 
3176 |     Methods:
3177 |         forward: Applies the SCDown module to the input tensor.
3178 | 
3179 |     Examples:
3180 |         >>> import torch
3181 |         >>> from ultralytics.nn.modules.block import SCDown
3182 |         >>> model = SCDown(c1=64, c2=128, k=3, s=2)
3183 |         >>> x = torch.randn(1, 64, 128, 128)
3184 |         >>> y = model(x)
3185 |         >>> print(y.shape)
3186 |         torch.Size([1, 128, 64, 64])
3187 |     """
3188 | 
3189 |     def __init__(self, c1: int, c2: int, k: int, s: int):
3190 |         """Initialize SCDown module.
3191 | 
3192 |         Args:
3193 |             c1 (int): Input channels.
3194 |             c2 (int): Output channels.
3195 |             k (int): Kernel size.
3196 |             s (int): Stride.
3197 |         """
3198 |         super().__init__()
3199 |         self.cv1 = Conv(c1, c2, 1, 1)
3200 |         self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
3201 | 
3202 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3203 |         """Apply convolution and downsampling to the input tensor.
3204 | 
3205 |         Args:
3206 |             x (torch.Tensor): Input tensor.
3207 | 
3208 |         Returns:
3209 |             (torch.Tensor): Downsampled output tensor.
3210 |         """
3211 |         return self.cv2(self.cv1(x))
3212 | 
3213 | 
3214 | class TorchVision(nn.Module):
3215 |     """TorchVision module to allow loading any torchvision model.
3216 | 
3217 |     This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
3218 |     customize the model by truncating or unwrapping layers.
3219 | 
3220 |     Args:
3221 |         model (str): Name of the torchvision model to load.
3222 |         weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
3223 |         unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
3224 |         truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
3225 |         split (bool, optional): Returns output from intermediate child modules as list. Default is False.
3226 | 
3227 |     Attributes:
3228 |         m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
3229 |     """
3230 | 
3231 |     def __init__(
3232 |         self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
3233 |     ):
3234 |         """Load the model and weights from torchvision.
3235 | 
3236 |         Args:
3237 |             model (str): Name of the torchvision model to load.
3238 |             weights (str): Pre-trained weights to load.
3239 |             unwrap (bool): Whether to unwrap the model.
3240 |             truncate (int): Number of layers to truncate.
3241 |             split (bool): Whether to split the output.
3242 |         """
3243 |         import torchvision  # scope for faster 'import ultralytics'
3244 | 
3245 |         super().__init__()
3246 |         if hasattr(torchvision.models, "get_model"):
3247 |             self.m = torchvision.models.get_model(model, weights=weights)
3248 |         else:
3249 |             self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
3250 |         if unwrap:
3251 |             layers = list(self.m.children())
3252 |             if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
3253 |                 layers = [*list(layers[0].children()), *layers[1:]]
3254 |             self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
3255 |             self.split = split
3256 |         else:
3257 |             self.split = False
3258 |             self.m.head = self.m.heads = nn.Identity()
3259 | 
3260 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3261 |         """Forward pass through the model.
3262 | 
3263 |         Args:
3264 |             x (torch.Tensor): Input tensor.
3265 | 
3266 |         Returns:
3267 |             (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
3268 |         """
3269 |         if self.split:
3270 |             y = [x]
3271 |             y.extend(m(y[-1]) for m in self.m)
3272 |         else:
3273 |             y = self.m(x)
3274 |         return y
3275 | 
3276 | 
3277 | class AAttn(nn.Module):
3278 |     """Area-attention module for YOLO models, providing efficient attention mechanisms.
3279 | 
3280 |     This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
3281 |     making it particularly effective for object detection tasks.
3282 | 
3283 |     Attributes:
3284 |         area (int): Number of areas the feature map is divided into.
3285 |         num_heads (int): Number of heads into which the attention mechanism is divided.
3286 |         head_dim (int): Dimension of each attention head.
3287 |         qkv (Conv): Convolution layer for computing query, key and value tensors.
3288 |         proj (Conv): Projection convolution layer.
3289 |         pe (Conv): Position encoding convolution layer.
3290 | 
3291 |     Methods:
3292 |         forward: Applies area-attention to input tensor.
3293 | 
3294 |     Examples:
3295 |         >>> attn = AAttn(dim=256, num_heads=8, area=4)
3296 |         >>> x = torch.randn(1, 256, 32, 32)
3297 |         >>> output = attn(x)
3298 |         >>> print(output.shape)
3299 |         torch.Size([1, 256, 32, 32])
3300 |     """
3301 | 
3302 |     def __init__(self, dim: int, num_heads: int, area: int = 1):
3303 |         """Initialize an Area-attention module for YOLO models.
3304 | 
3305 |         Args:
3306 |             dim (int): Number of hidden channels.
3307 |             num_heads (int): Number of heads into which the attention mechanism is divided.
3308 |             area (int): Number of areas the feature map is divided into.
3309 |         """
3310 |         super().__init__()
3311 |         self.area = area
3312 | 
3313 |         self.num_heads = num_heads
3314 |         self.head_dim = head_dim = dim // num_heads
3315 |         self.all_head_dim = all_head_dim = head_dim * self.num_heads
3316 | 
3317 |         self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
3318 |         self.proj = Conv(all_head_dim, dim, 1, act=False)
3319 |         self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)
3320 | 
3321 |     def __setstate__(self, state):
3322 |         """Add missing all_head_dim attribute to old checkpoints."""
3323 |         super().__setstate__(state)
3324 |         if not hasattr(self, "all_head_dim"):
3325 |             self.all_head_dim = self.head_dim * self.num_heads
3326 | 
3327 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3328 |         """Process the input tensor through the area-attention.
3329 | 
3330 |         Args:
3331 |             x (torch.Tensor): Input tensor.
3332 | 
3333 |         Returns:
3334 |             (torch.Tensor): Output tensor after area-attention.
3335 |         """
3336 |         B, _, H, W = x.shape
3337 |         N = H * W
3338 | 
3339 |         qkv = self.qkv(x).flatten(2).transpose(1, 2)
3340 |         if self.area > 1:
3341 |             qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
3342 |             B, N, _ = qkv.shape
3343 |         q, k, v = (
3344 |             qkv.view(B, N, self.num_heads, self.head_dim * 3)
3345 |             .permute(0, 2, 3, 1)
3346 |             .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
3347 |         )
3348 |         attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
3349 |         attn = attn.softmax(dim=-1)
3350 |         x = v @ attn.transpose(-2, -1)
3351 |         x = x.permute(0, 3, 1, 2)
3352 |         v = v.permute(0, 3, 1, 2)
3353 | 
3354 |         if self.area > 1:
3355 |             x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
3356 |             v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
3357 |             B, N, _ = x.shape
3358 | 
3359 |         x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
3360 |         v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
3361 | 
3362 |         x = x + self.pe(v)
3363 |         return self.proj(x)
3364 | 
3365 | 
3366 | class ABlock(nn.Module):
3367 |     """Area-attention block module for efficient feature extraction in YOLO models.
3368 | 
3369 |     This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
3370 |     It uses a novel area-based attention approach that is more efficient than traditional self-attention while
3371 |     maintaining effectiveness.
3372 | 
3373 |     Attributes:
3374 |         attn (AAttn): Area-attention module for processing spatial features.
3375 |         mlp (nn.Sequential): Multi-layer perceptron for feature transformation.
3376 | 
3377 |     Methods:
3378 |         _init_weights: Initializes module weights using truncated normal distribution.
3379 |         forward: Applies area-attention and feed-forward processing to input tensor.
3380 | 
3381 |     Examples:
3382 |         >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
3383 |         >>> x = torch.randn(1, 256, 32, 32)
3384 |         >>> output = block(x)
3385 |         >>> print(output.shape)
3386 |         torch.Size([1, 256, 32, 32])
3387 |     """
3388 | 
3389 |     def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
3390 |         """Initialize an Area-attention block module.
3391 | 
3392 |         Args:
3393 |             dim (int): Number of input channels.
3394 |             num_heads (int): Number of heads into which the attention mechanism is divided.
3395 |             mlp_ratio (float): Expansion ratio for MLP hidden dimension.
3396 |             area (int): Number of areas the feature map is divided into.
3397 |         """
3398 |         super().__init__()
3399 | 
3400 |         self.attn = AAttn(dim, num_heads=num_heads, area=area)
3401 |         mlp_hidden_dim = int(dim * mlp_ratio)
3402 |         self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
3403 | 
3404 |         self.apply(self._init_weights)
3405 | 
3406 |     @staticmethod
3407 |     def _init_weights(m: nn.Module):
3408 |         """Initialize weights using a truncated normal distribution.
3409 | 
3410 |         Args:
3411 |             m (nn.Module): Module to initialize.
3412 |         """
3413 |         if isinstance(m, nn.Conv2d):
3414 |             nn.init.trunc_normal_(m.weight, std=0.02)
3415 |             if m.bias is not None:
3416 |                 nn.init.constant_(m.bias, 0)
3417 | 
3418 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3419 |         """Forward pass through ABlock.
3420 | 
3421 |         Args:
3422 |             x (torch.Tensor): Input tensor.
3423 | 
3424 |         Returns:
3425 |             (torch.Tensor): Output tensor after area-attention and feed-forward processing.
3426 |         """
3427 |         x = x + self.attn(x)
3428 |         return x + self.mlp(x)
3429 | 
3430 | 
3431 | class A2C2f(nn.Module):
3432 |     """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.
3433 | 
3434 |     This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
3435 |     processing. It supports both area-attention and standard convolution modes.
3436 | 
3437 |     Attributes:
3438 |         cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
3439 |         cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
3440 |         gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
3441 |         m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.
3442 | 
3443 |     Methods:
3444 |         forward: Processes input through area-attention or standard convolution pathway.
3445 | 
3446 |     Examples:
3447 |         >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
3448 |         >>> x = torch.randn(1, 512, 32, 32)
3449 |         >>> output = m(x)
3450 |         >>> print(output.shape)
3451 |         torch.Size([1, 512, 32, 32])
3452 |     """
3453 | 
3454 |     def __init__(
3455 |         self,
3456 |         c1: int,
3457 |         c2: int,
3458 |         n: int = 1,
3459 |         a2: bool = True,
3460 |         area: int = 1,
3461 |         residual: bool = False,
3462 |         mlp_ratio: float = 2.0,
3463 |         e: float = 0.5,
3464 |         g: int = 1,
3465 |         shortcut: bool = True,
3466 |     ):
3467 |         """Initialize Area-Attention C2f module.
3468 | 
3469 |         Args:
3470 |             c1 (int): Number of input channels.
3471 |             c2 (int): Number of output channels.
3472 |             n (int): Number of ABlock or C3k modules to stack.
3473 |             a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
3474 |             area (int): Number of areas the feature map is divided into.
3475 |             residual (bool): Whether to use residual connections with learnable gamma parameter.
3476 |             mlp_ratio (float): Expansion ratio for MLP hidden dimension.
3477 |             e (float): Channel expansion ratio for hidden channels.
3478 |             g (int): Number of groups for grouped convolutions.
3479 |             shortcut (bool): Whether to use shortcut connections in C3k blocks.
3480 |         """
3481 |         super().__init__()
3482 |         c_ = int(c2 * e)  # hidden channels
3483 |         assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."
3484 | 
3485 |         self.cv1 = Conv(c1, c_, 1, 1)
3486 |         self.cv2 = Conv((1 + n) * c_, c2, 1)
3487 | 
3488 |         self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
3489 |         self.m = nn.ModuleList(
3490 |             nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
3491 |             if a2
3492 |             else C3k(c_, c_, 2, shortcut, g)
3493 |             for _ in range(n)
3494 |         )
3495 | 
3496 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3497 |         """Forward pass through A2C2f layer.
3498 | 
3499 |         Args:
3500 |             x (torch.Tensor): Input tensor.
3501 | 
3502 |         Returns:
3503 |             (torch.Tensor): Output tensor after processing.
3504 |         """
3505 |         y = [self.cv1(x)]
3506 |         y.extend(m(y[-1]) for m in self.m)
3507 |         y = self.cv2(torch.cat(y, 1))
3508 |         if self.gamma is not None:
3509 |             return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
3510 |         return y
3511 | 
3512 | 
3513 | class SwiGLUFFN(nn.Module):
3514 |     """SwiGLU Feed-Forward Network for transformer-based architectures."""
3515 | 
3516 |     def __init__(self, gc: int, ec: int, e: int = 4) -> None:
3517 |         """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.
3518 | 
3519 |         Args:
3520 |             gc (int): Guide channels.
3521 |             ec (int): Embedding channels.
3522 |             e (int): Expansion factor.
3523 |         """
3524 |         super().__init__()
3525 |         self.w12 = nn.Linear(gc, e * ec)
3526 |         self.w3 = nn.Linear(e * ec // 2, ec)
3527 | 
3528 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3529 |         """Apply SwiGLU transformation to input features."""
3530 |         x12 = self.w12(x)
3531 |         x1, x2 = x12.chunk(2, dim=-1)
3532 |         hidden = F.silu(x1) * x2
3533 |         return self.w3(hidden)
3534 | 
3535 | 
3536 | class Residual(nn.Module):
3537 |     """Residual connection wrapper for neural network modules."""
3538 | 
3539 |     def __init__(self, m: nn.Module) -> None:
3540 |         """Initialize residual module with the wrapped module.
3541 | 
3542 |         Args:
3543 |             m (nn.Module): Module to wrap with residual connection.
3544 |         """
3545 |         super().__init__()
3546 |         self.m = m
3547 |         nn.init.zeros_(self.m.w3.bias)
3548 |         # For models with l scale, please change the initialization to
3549 |         # nn.init.constant_(self.m.w3.weight, 1e-6)
3550 |         nn.init.zeros_(self.m.w3.weight)
3551 | 
3552 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
3553 |         """Apply residual connection to input features."""
3554 |         return x + self.m(x)
3555 | 
3556 | 
3557 | class SAVPE(nn.Module):
3558 |     """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""
3559 | 
3560 |     def __init__(self, ch: list[int], c3: int, embed: int):
3561 |         """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.
3562 | 
3563 |         Args:
3564 |             ch (list[int]): List of input channel dimensions.
3565 |             c3 (int): Intermediate channels.
3566 |             embed (int): Embedding dimension.
3567 |         """
3568 |         super().__init__()
3569 |         self.cv1 = nn.ModuleList(
3570 |             nn.Sequential(
3571 |                 Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
3572 |             )
3573 |             for i, x in enumerate(ch)
3574 |         )
3575 | 
3576 |         self.cv2 = nn.ModuleList(
3577 |             nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
3578 |             for i, x in enumerate(ch)
3579 |         )
3580 | 
3581 |         self.c = 16
3582 |         self.cv3 = nn.Conv2d(3 * c3, embed, 1)
3583 |         self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
3584 |         self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
3585 |         self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))
3586 | 
3587 |     def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
3588 |         """Process input features and visual prompts to generate enhanced embeddings."""
3589 |         y = [self.cv2[i](xi) for i, xi in enumerate(x)]
3590 |         y = self.cv4(torch.cat(y, dim=1))
3591 | 
3592 |         x = [self.cv1[i](xi) for i, xi in enumerate(x)]
3593 |         x = self.cv3(torch.cat(x, dim=1))
3594 | 
3595 |         B, C, H, W = x.shape
3596 | 
3597 |         Q = vp.shape[1]
3598 | 
3599 |         x = x.view(B, C, -1)
3600 | 
3601 |         y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
3602 |         vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)
3603 | 
3604 |         y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))
3605 | 
3606 |         y = y.reshape(B, Q, self.c, -1)
3607 |         vp = vp.reshape(B, Q, 1, -1)
3608 | 
3609 |         score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
3610 |         score = F.softmax(score, dim=-1).to(y.dtype)
3611 |         aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)
3612 | 
3613 |         return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
3614 | 
3615 | 
3616 | class Proto26(Proto):
3617 |     """Ultralytics YOLO26 models mask Proto module for segmentation models."""
3618 | 
3619 |     def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
3620 |         """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.
3621 | 
3622 |         Args:
3623 |             ch (tuple): Tuple of channel sizes from backbone feature maps.
3624 |             c_ (int): Intermediate channels.
3625 |             c2 (int): Output channels (number of protos).
3626 |             nc (int): Number of classes for semantic segmentation.
3627 |         """
3628 |         super().__init__(c_, c_, c2)
3629 |         self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
3630 |         self.feat_fuse = Conv(ch[0], c_, k=3)
3631 |         self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))
3632 | 
3633 |     def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
3634 |         """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
3635 |         feat = x[0]
3636 |         for i, f in enumerate(self.feat_refine):
3637 |             up_feat = f(x[i + 1])
3638 |             up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
3639 |             feat = feat + up_feat
3640 |         p = super().forward(self.feat_fuse(feat))
3641 |         if self.training and return_semantic:
3642 |             semantic = self.semseg(feat)
3643 |             return (p, semantic)
3644 |         return p
3645 | 
3646 |     def fuse(self):
3647 |         """Fuse the model for inference by removing the semantic segmentation head."""
3648 |         self.semseg = None
3649 | 
3650 | 
3651 | class RealNVP(nn.Module):
3652 |     """RealNVP: a flow-based generative model.
3653 | 
3654 |     References:
3655 |         https://arxiv.org/abs/1605.08803
3656 |         https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
3657 |     """
3658 | 
3659 |     @staticmethod
3660 |     def nets():
3661 |         """Get the scale model in a single invertible mapping."""
3662 |         return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())
3663 | 
3664 |     @staticmethod
3665 |     def nett():
3666 |         """Get the translation model in a single invertible mapping."""
3667 |         return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
3668 | 
3669 |     @property
3670 |     def prior(self):
3671 |         """The prior distribution."""
3672 |         return torch.distributions.MultivariateNormal(self.loc, self.cov)
3673 | 
3674 |     def __init__(self):
3675 |         super().__init__()
3676 | 
3677 |         self.register_buffer("loc", torch.zeros(2))
3678 |         self.register_buffer("cov", torch.eye(2))
3679 |         self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
3680 | 
3681 |         self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
3682 |         self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
3683 |         self.init_weights()
3684 | 
3685 |     def init_weights(self):
3686 |         """Initialize model weights."""
3687 |         for m in self.modules():
3688 |             if isinstance(m, nn.Linear):
3689 |                 nn.init.xavier_uniform_(m.weight, gain=0.01)
3690 | 
3691 |     def backward_p(self, x):
3692 |         """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
3693 |         matrix.
3694 |         """
3695 |         log_det_jacob, z = x.new_zeros(x.shape[0]), x
3696 |         for i in reversed(range(len(self.t))):
3697 |             z_ = self.mask[i] * z
3698 |             s = self.s[i](z_) * (1 - self.mask[i])
3699 |             t = self.t[i](z_) * (1 - self.mask[i])
3700 |             z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
3701 |             log_det_jacob -= s.sum(dim=1)
3702 |         return z, log_det_jacob
3703 | 
3704 |     def log_prob(self, x):
3705 |         """Calculate the log probability of given sample in data space."""
3706 |         if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
3707 |             self.float()
3708 |         z, log_det = self.backward_p(x)
3709 |         return self.prior.log_prob(z) + log_det
3710 | 
```

### models_related/ultralytics/ultralytics/nn/tasks.py

Bytes: 99564
SHA-256: 3ce3d3dbd444737e588810188920ed904bee481f32be54153594ee424dfab41c
Lines: 1-2426 of 2426

```python
   1 | # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
   2 | 
   3 | import contextlib
   4 | import pickle
   5 | import re
   6 | import threading
   7 | from copy import deepcopy
   8 | from pathlib import Path
   9 | 
  10 | import torch
  11 | import torch.nn as nn
  12 | 
  13 | from ultralytics.nn.autobackend import check_class_names
  14 | from ultralytics.nn.modules import (
  15 |     AIFI,
  16 |     C1,
  17 |     C2,
  18 |     C2PSA,
  19 |     C3,
  20 |     C3TR,
  21 |     ELAN1,
  22 |     OBB,
  23 |     OBB26,
  24 |     PSA,
  25 |     SPP,
  26 |     SPPELAN,
  27 |     SPPF,
  28 |     A2C2f,
  29 |     AConv,
  30 |     ADown,
  31 |     AdversarialPerturbationInjection,
  32 |     ASFAttention,
  33 |     BiLevelRoutingAttention,
  34 |     BoundaryFeatureBlock,
  35 |     Bottleneck,
  36 |     BottleneckCSP,
  37 |     C2f,
  38 |     C2fCBAM,
  39 |     C2fAttn,
  40 |     C2fCIB,
  41 |     C2fKV,
  42 |     C2fNAT,
  43 |     C2fPSA,
  44 |     EnSimAM,
  45 |     EnSimAMEdgeRepC2f,
  46 |     FeatureDGFE,
  47 |     C3CBAM,
  48 |     C3Ghost,
  49 |     C3k2,
  50 |     C3x,
  51 |     CBFuse,
  52 |     CBLinear,
  53 |     Classify,
  54 |     Concat,
  55 |     Conv,
  56 |     Conv2,
  57 |     ConvTranspose,
  58 |     Detect,
  59 |     DWConv,
  60 |     DWConvTranspose2d,
  61 |     Focus,
  62 |     GhostBottleneck,
  63 |     GhostConv,
  64 |     HGBlock,
  65 |     HGStem,
  66 |     ImagePoolingAttn,
  67 |     Index,
  68 |     KVCompressedAttention,
  69 |     KVCompressedAttentionPartial,
  70 |     KVCompressedTransformerEncoder,
  71 |     LRPCHead,
  72 |     M3NATFuse,
  73 |     NATBlock,
  74 |     Pose,
  75 |     Pose26,
  76 |     RegionRoutingAttentionLite,
  77 |     RepC3,
  78 |     RepC2f,
  79 |     PoolingEdgeRepC2f,
  80 |     RepConv,
  81 |     RepNCSPELAN4,
  82 |     RepVGGDW,
  83 |     ResNetLayer,
  84 |     RTDETRDecoder,
  85 |     SCDown,
  86 |     ScalSeq,
  87 |     Segment,
  88 |     Segment26,
  89 |     SemanticSegment,
  90 |     TorchVision,
  91 |     TopKAdaptiveGroupKVAttention,
  92 |     TopKGlobalGroupKVAttention,
  93 |     WeightedAdd,
  94 |     WorldDetect,
  95 |     YOLOEDetect,
  96 |     YOLOESegment,
  97 |     YOLOESegment26,
  98 |     clear_boundary_context,
  99 |     set_boundary_context,
 100 |     v10Detect,
 101 | )
 102 | from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, SAFE_LOAD, SETTINGS, WINDOWS, YAML, colorstr, emojis
 103 | from ultralytics.utils.checks import REMOTE_FILE_PREFIXES, check_file, check_requirements, check_suffix, check_yaml
 104 | from ultralytics.utils.loss import (
 105 |     E2ELoss,
 106 |     PoseLoss26,
 107 |     SemanticSegmentationLoss,
 108 |     v8ClassificationLoss,
 109 |     v8DetectionLoss,
 110 |     v8OBBLoss,
 111 |     v8PoseLoss,
 112 |     v8SegmentationLoss,
 113 | )
 114 | from ultralytics.utils.ops import make_divisible
 115 | from ultralytics.utils.patches import torch_load
 116 | from ultralytics.utils.plotting import feature_visualization
 117 | from ultralytics.utils.torch_utils import (
 118 |     fuse_conv_and_bn,
 119 |     fuse_deconv_and_bn,
 120 |     initialize_weights,
 121 |     intersect_dicts,
 122 |     model_info,
 123 |     scale_img,
 124 |     smart_inference_mode,
 125 |     time_sync,
 126 | )
 127 | 
 128 | 
 129 | class BaseModel(torch.nn.Module):
 130 |     """Base class for all YOLO models in the Ultralytics family.
 131 | 
 132 |     This class provides common functionality for YOLO models including forward pass handling, model fusion, information
 133 |     display, and weight loading capabilities.
 134 | 
 135 |     Attributes:
 136 |         model (torch.nn.Sequential): The neural network model.
 137 |         save (list): List of layer indices to save outputs from.
 138 |         stride (torch.Tensor): Model stride values.
 139 | 
 140 |     Methods:
 141 |         forward: Perform forward pass for training or inference.
 142 |         predict: Perform inference on input tensor.
 143 |         fuse: Fuse Conv/BatchNorm layers and reparameterize for optimization.
 144 |         info: Print model information.
 145 |         load: Load weights into the model.
 146 |         loss: Compute loss for training.
 147 | 
 148 |     Examples:
 149 |         Create a BaseModel instance
 150 |         >>> model = BaseModel()
 151 |         >>> model.info()  # Display model information
 152 |     """
 153 | 
 154 |     def forward(self, x, *args, **kwargs):
 155 |         """Perform forward pass of the model for either training or inference.
 156 | 
 157 |         If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.
 158 | 
 159 |         Args:
 160 |             x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
 161 |             *args (Any): Variable length argument list.
 162 |             **kwargs (Any): Arbitrary keyword arguments.
 163 | 
 164 |         Returns:
 165 |             (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
 166 |         """
 167 |         if isinstance(x, dict):  # for cases of training and validating while training.
 168 |             return self.loss(x, *args, **kwargs)
 169 |         return self.predict(x, *args, **kwargs)
 170 | 
 171 |     def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
 172 |         """Perform a forward pass through the network.
 173 | 
 174 |         Args:
 175 |             x (torch.Tensor): The input tensor to the model.
 176 |             profile (bool): Print the computation time of each layer if True.
 177 |             visualize (bool): Save the feature maps of the model if True.
 178 |             augment (bool): Augment image during prediction.
 179 |             embed (list, optional): A list of layer indices to return embeddings from.
 180 | 
 181 |         Returns:
 182 |             (torch.Tensor): The last output of the model.
 183 |         """
 184 |         if augment:
 185 |             return self._predict_augment(x)
 186 |         return self._predict_once(x, profile, visualize, embed)
 187 | 
 188 |     @staticmethod
 189 |     def _attach_dgfe_aux(x, dgfe_aux):
 190 |         """Attach collected DGFE auxiliary tensors to raw Detect predictions."""
 191 |         if not dgfe_aux:
 192 |             return x
 193 |         if isinstance(x, dict):
 194 |             x["dgfe_aux"] = dgfe_aux
 195 |         elif isinstance(x, tuple) and len(x) > 1 and isinstance(x[1], dict):
 196 |             x[1]["dgfe_aux"] = dgfe_aux
 197 |         return x
 198 | 
 199 |     def _predict_once(self, x, profile=False, visualize=False, embed=None):
 200 |         """Perform a forward pass through the network.
 201 | 
 202 |         Args:
 203 |             x (torch.Tensor): The input tensor to the model.
 204 |             profile (bool): Print the computation time of each layer if True.
 205 |             visualize (bool): Save the feature maps of the model if True.
 206 |             embed (list, optional): A list of layer indices to return embeddings from.
 207 | 
 208 |         Returns:
 209 |             (torch.Tensor): The last output of the model.
 210 |         """
 211 |         img0 = x
 212 |         y, dt, embeddings, dgfe_aux = [], [], [], []  # outputs
 213 |         embed = frozenset(embed) if embed is not None else {-1}
 214 |         max_idx = max(embed)
 215 |         for m in self.model:
 216 |             if m.f != -1:  # if not from previous layer
 217 |                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
 218 |             if profile and not isinstance(m, FeatureDGFE):
 219 |                 self._profile_one_layer(m, x, dt)
 220 |             if isinstance(m, FeatureDGFE):
 221 |                 x = m(x, img0)
 222 |                 if m.last_aux is not None:
 223 |                     dgfe_aux.append(m.last_aux)
 224 |             else:
 225 |                 x = m(x)  # run
 226 |             y.append(x if m.i in self.save else None)  # save output
 227 |             if visualize:
 228 |                 feature_visualization(x, m.type, m.i, save_dir=visualize)
 229 |             if m.i in embed:
 230 |                 embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
 231 |                 if m.i == max_idx:
 232 |                     return torch.unbind(torch.cat(embeddings, 1), dim=0)
 233 |         return self._attach_dgfe_aux(x, dgfe_aux)
 234 | 
 235 |     def _predict_once_full_y(self, x):
 236 |         """Like _predict_once but also returns the save-filtered y-list for partial forward caching.
 237 | 
 238 |         Returns:
 239 |             (torch.Tensor, list): Final prediction tensor and the y-list where y[i] is the
 240 |                 output of self.model[i] when i is in self.save, else None.
 241 |         """
 242 |         img0 = x
 243 |         y, dgfe_aux = [], []
 244 |         for m in self.model:
 245 |             if m.f != -1:
 246 |                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
 247 |             if isinstance(m, FeatureDGFE):
 248 |                 x = m(x, img0)
 249 |                 if m.last_aux is not None:
 250 |                     dgfe_aux.append(m.last_aux)
 251 |             else:
 252 |                 x = m(x)
 253 |             y.append(x if m.i in self.save else None)
 254 |         return self._attach_dgfe_aux(x, dgfe_aux), y
 255 | 
 256 |     def _predict_perturbed_partial(self, api, img0) -> torch.Tensor:
 257 |         """Run the perturbed forward pass only from api.layer_idx onward.
 258 | 
 259 |         All layers before api.layer_idx are skipped; their outputs are sourced from the
 260 |         clean y-list cached in ``api._cached_clean_y``.  The API layer itself is run in
 261 |         perturb mode (``api._clean_input`` feeds its input unchanged, and the module adds
 262 |         the stored perturbation).  Subsequent layers are run normally using either the
 263 |         cached (pre-API) or freshly computed (post-API) y entries.
 264 | 
 265 |         Args:
 266 |             api (AdversarialPerturbationInjection): The API module, which must have
 267 |                 ``layer_idx``, ``_clean_input``, and ``_cached_clean_y`` populated.
 268 | 
 269 |         Returns:
 270 |             (torch.Tensor): Model output from the partial perturbed forward pass.
 271 |         """
 272 |         assert api._cached_clean_y is not None, "Partial forward requires api._cached_clean_y (set in _api_adversarial_loss)."
 273 |         assert api._clean_input is not None, "Partial forward requires api._clean_input (set in API capture forward)."
 274 |         assert api.layer_idx is not None, "api.layer_idx must be bound during model initialisation."
 275 | 
 276 |         # Start from the clean y-list; we overwrite entries from api.layer_idx onward.
 277 |         y = list(api._cached_clean_y)  # shallow copy; length == len(self.model)
 278 |         x = api._clean_input  # clean input to the API layer (identical to clean pass)
 279 | 
 280 |         dgfe_aux = []
 281 |         for m in self.model:
 282 |             if m.i < api.layer_idx:
 283 |                 continue  # skip backbone and early neck — already in y
 284 |             if m.i == api.layer_idx:
 285 |                 # API module is in perturb mode: returns x + perturbation (+ optional FGSM dropout)
 286 |                 x = m(x)
 287 |             else:
 288 |                 if m.f != -1:
 289 |                     x = (y[m.f] if isinstance(m.f, int)
 290 |                          else [x if j == -1 else y[j] for j in m.f])
 291 |                 if isinstance(m, FeatureDGFE):
 292 |                     x = m(x, img0)
 293 |                     if m.last_aux is not None:
 294 |                         dgfe_aux.append(m.last_aux)
 295 |                 else:
 296 |                     x = m(x)
 297 |             if m.i in self.save:
 298 |                 y[m.i] = x  # overwrite with freshly computed output
 299 |         return self._attach_dgfe_aux(x, dgfe_aux)
 300 | 
 301 |     def _predict_augment(self, x):
 302 |         """Perform augmentations on input image x and return augmented inference."""
 303 |         LOGGER.warning(
 304 |             f"{self.__class__.__name__} does not support 'augment=True' prediction. "
 305 |             f"Reverting to single-scale prediction."
 306 |         )
 307 |         return self._predict_once(x)
 308 | 
 309 |     def _profile_one_layer(self, m, x, dt):
 310 |         """Profile the computation time and FLOPs of a single layer of the model on a given input.
 311 | 
 312 |         Args:
 313 |             m (torch.nn.Module): The layer to be profiled.
 314 |             x (torch.Tensor): The input data to the layer.
 315 |             dt (list): A list to store the computation time of the layer.
 316 |         """
 317 |         try:
 318 |             import thop
 319 |         except ImportError:
 320 |             thop = None  # conda support without 'ultralytics-thop' installed
 321 | 
 322 |         c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
 323 |         flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
 324 |         t = time_sync()
 325 |         for _ in range(10):
 326 |             m(x.copy() if c else x)
 327 |         dt.append((time_sync() - t) * 100)
 328 |         if m == self.model[0]:
 329 |             LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
 330 |         LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
 331 |         if c:
 332 |             LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
 333 | 
 334 |     def fuse(self, verbose=True):
 335 |         """Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency.
 336 | 
 337 |         Args:
 338 |             verbose (bool): Whether to print model information after fusion.
 339 | 
 340 |         Returns:
 341 |             (torch.nn.Module): The fused model is returned.
 342 |         """
 343 |         if not self.is_fused():
 344 |             for m in self.model.modules():
 345 |                 if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
 346 |                     if isinstance(m, Conv2):
 347 |                         m.fuse_convs()
 348 |                     m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
 349 |                     delattr(m, "bn")  # remove batchnorm
 350 |                     m.forward = m.forward_fuse  # update forward
 351 |                 if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
 352 |                     m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
 353 |                     delattr(m, "bn")  # remove batchnorm
 354 |                     m.forward = m.forward_fuse  # update forward
 355 |                 if isinstance(m, RepConv):
 356 |                     m.fuse_convs()
 357 |                     m.forward = m.forward_fuse  # update forward
 358 |                 if isinstance(m, RepVGGDW):
 359 |                     m.fuse()
 360 |                     m.forward = m.forward_fuse
 361 |                 if isinstance(m, Detect) and getattr(m, "end2end", False):
 362 |                     m.fuse()  # remove one2many head
 363 |             self.info(verbose=verbose)
 364 | 
 365 |         return self
 366 | 
 367 |     def is_fused(self, thresh=10):
 368 |         """Check if the model has less than a certain threshold of normalization layers.
 369 | 
 370 |         Args:
 371 |             thresh (int, optional): The threshold number of normalization layers.
 372 | 
 373 |         Returns:
 374 |             (bool): True if the number of normalization layers in the model is less than the threshold, False otherwise.
 375 |         """
 376 |         bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
 377 |         return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
 378 | 
 379 |     def info(self, detailed=False, verbose=True, imgsz=640):
 380 |         """Print model information.
 381 | 
 382 |         Args:
 383 |             detailed (bool): If True, prints out detailed information about the model.
 384 |             verbose (bool): If True, prints out the model information.
 385 |             imgsz (int): The size of the image used for computing model information.
 386 |         """
 387 |         return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
 388 | 
 389 |     def _apply(self, fn):
 390 |         """Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.
 391 | 
 392 |         Args:
 393 |             fn (function): The function to apply to the model.
 394 | 
 395 |         Returns:
 396 |             (BaseModel): An updated BaseModel object.
 397 |         """
 398 |         self = super()._apply(fn)
 399 |         m = self.model[-1]  # Detect()
 400 |         if isinstance(
 401 |             m, Detect
 402 |         ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
 403 |             m.stride = fn(m.stride)
 404 |             m.anchors = fn(m.anchors)
 405 |             m.strides = fn(m.strides)
 406 |         return self
 407 | 
 408 |     def load(self, weights, verbose=True, smart_transfer=True):
 409 |         """Load weights into the model.
 410 | 
 411 |         Args:
 412 |             weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
 413 |             verbose (bool, optional): Whether to log the transfer progress.
 414 |             smart_transfer (bool, optional): Whether to use smart matching for shifted layers.
 415 |         """
 416 |         model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
 417 |         csd = model.float().state_dict()  # checkpoint state_dict as FP32
 418 |         updated_csd = intersect_dicts(csd, self.state_dict(), smart_transfer=smart_transfer)  # intersect
 419 |         self.load_state_dict(updated_csd, strict=False)  # load
 420 |         len_updated_csd = len(updated_csd)
 421 |         first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
 422 |         # mostly used to boost multi-channel training
 423 |         state_dict = self.state_dict()
 424 |         if first_conv not in updated_csd and first_conv in state_dict:
 425 |             c1, c2, h, w = state_dict[first_conv].shape
 426 |             cc1, cc2, ch, cw = csd[first_conv].shape
 427 |             if ch == h and cw == w:
 428 |                 c1, c2 = min(c1, cc1), min(c2, cc2)
 429 |                 state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
 430 |                 len_updated_csd += 1
 431 |         if verbose:
 432 |             LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
 433 | 
 434 |     def _api_modules(self) -> list[AdversarialPerturbationInjection]:
 435 |         """Return train-time API perturbation modules in this model."""
 436 | 
 437 |         return [m for m in self.modules() if isinstance(m, AdversarialPerturbationInjection)]
 438 | 
 439 |     def _clear_api_modules(self) -> None:
 440 |         """Clear state from all API perturbation modules."""
 441 | 
 442 |         for module in self._api_modules():
 443 |             module.clear_state()
 444 | 
 445 |     @staticmethod
 446 |     def _api_foreground_target(batch, feature: torch.Tensor) -> torch.Tensor:
 447 |         """Build a P2-resolution foreground target from normalized xywh labels."""
 448 | 
 449 |         target = torch.zeros((feature.shape[0], 1, feature.shape[2], feature.shape[3]), device=feature.device, dtype=feature.dtype)
 450 |         bboxes = batch.get("bboxes")
 451 |         batch_idx = batch.get("batch_idx")
 452 |         if bboxes is None or batch_idx is None or bboxes.numel() == 0:
 453 |             return target
 454 | 
 455 |         bboxes = bboxes.to(device=feature.device, dtype=feature.dtype)
 456 |         batch_idx = batch_idx.to(device=feature.device, dtype=torch.long).view(-1)
 457 |         h, w = feature.shape[2:]
 458 |         for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
 459 |             if idx < 0 or idx >= feature.shape[0]:
 460 |                 continue
 461 |             xc, yc, bw, bh = [float(v) for v in box]
 462 |             x1 = max(0, min(w - 1, int((xc - bw / 2.0) * w)))
 463 |             y1 = max(0, min(h - 1, int((yc - bh / 2.0) * h)))
 464 |             x2 = max(x1 + 1, min(w, int((xc + bw / 2.0) * w) + 1))
 465 |             y2 = max(y1 + 1, min(h, int((yc + bh / 2.0) * h) + 1))
 466 |             target[idx, :, y1:y2, x1:x2] = 1.0
 467 |         return target
 468 | 
 469 |     @staticmethod
 470 |     def _api_boundary_target(batch, feature: torch.Tensor, ring: float = 1.0, shrinkage: float = 0.25) -> torch.Tensor:
 471 |         """Build a P2-resolution target that marks GT box boundary bands."""
 472 | 
 473 |         target = torch.zeros((feature.shape[0], 1, feature.shape[2], feature.shape[3]), device=feature.device, dtype=feature.dtype)
 474 |         bboxes = batch.get("bboxes")
 475 |         batch_idx = batch.get("batch_idx")
 476 |         if bboxes is None or batch_idx is None or bboxes.numel() == 0:
 477 |             return target
 478 | 
 479 |         bboxes = bboxes.to(device=feature.device, dtype=feature.dtype)
 480 |         batch_idx = batch_idx.to(device=feature.device, dtype=torch.long).view(-1)
 481 |         h, w = feature.shape[2:]
 482 |         for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
 483 |             if idx < 0 or idx >= feature.shape[0]:
 484 |                 continue
 485 | 
 486 |             xc, yc, bw, bh = [float(v) for v in box]
 487 |             x1 = (xc - bw / 2.0) * w
 488 |             y1 = (yc - bh / 2.0) * h
 489 |             x2 = (xc + bw / 2.0) * w
 490 |             y2 = (yc + bh / 2.0) * h
 491 | 
 492 |             pad_x = max(ring, bw * w * 0.5 * ring)
 493 |             pad_y = max(ring, bh * h * 0.5 * ring)
 494 |             outer_x1 = max(0, min(w - 1, int(x1 - pad_x)))
 495 |             outer_y1 = max(0, min(h - 1, int(y1 - pad_y)))
 496 |             outer_x2 = max(outer_x1 + 1, min(w, int(x2 + pad_x) + 1))
 497 |             outer_y2 = max(outer_y1 + 1, min(h, int(y2 + pad_y) + 1))
 498 | 
 499 |             shrink_x = bw * w * 0.5 * shrinkage
 500 |             shrink_y = bh * h * 0.5 * shrinkage
 501 |             inner_x1 = max(0, min(w - 1, int(x1 + shrink_x)))
 502 |             inner_y1 = max(0, min(h - 1, int(y1 + shrink_y)))
 503 |             inner_x2 = max(inner_x1 + 1, min(w, int(x2 - shrink_x) + 1))
 504 |             inner_y2 = max(inner_y1 + 1, min(h, int(y2 - shrink_y) + 1))
 505 | 
 506 |             target[idx, :, outer_y1:outer_y2, outer_x1:outer_x2] = 1.0
 507 |             target[idx, :, inner_y1:inner_y2, inner_x1:inner_x2] = 0.0
 508 |         return target
 509 | 
 510 |     def _api_target(self, batch, api: AdversarialPerturbationInjection) -> torch.Tensor:
 511 |         """Build the auxiliary API target selected by the module configuration."""
 512 | 
 513 |         if api.captured is None:
 514 |             raise RuntimeError("API target requires a captured feature.")
 515 |         if api.target_mode in {"boundary", "boundary_ring", "ring"}:
 516 |             return self._api_boundary_target(batch, api.captured)
 517 |         return self._api_foreground_target(batch, api.captured)
 518 | 
 519 |     @staticmethod
 520 |     def _api_is_boxgrad(api: AdversarialPerturbationInjection) -> bool:
 521 |         """Return whether API should use localization-loss gradients."""
 522 | 
 523 |         return api.target_mode in {"boxgrad", "locgrad", "localization", "bbox", "box"}
 524 | 
 525 |     def _api_adversarial_loss(self, batch):
 526 |         """Compute clean detection loss plus SET-style API regularization."""
 527 | 
 528 |         api_modules = self._api_modules()
 529 |         if not api_modules or not self.training or not torch.is_grad_enabled():
 530 |             return None
 531 | 
 532 |         api = api_modules[0]  # API is intentionally applied to one P2/P3 feature in the compare YAML.
 533 |         if api.rho == 0 or api.api_weight == 0:
 534 |             return None
 535 | 
 536 |         try:
 537 |             for module in api_modules:
 538 |                 module.capture()
 539 | 
 540 |             set_boundary_context(batch.get("batch_idx"), batch.get("bboxes"), tuple(batch["img"].shape))
 541 |             try:
 542 |                 if api.use_partial_forward:
 543 |                     clean_preds, cached_y = self._predict_once_full_y(batch["img"])
 544 |                     api._cached_clean_y = cached_y
 545 |                 else:
 546 |                     clean_preds = self.forward(batch["img"])
 547 |             finally:
 548 |                 clear_boundary_context()
 549 | 
 550 |             clean_loss, clean_items = self.criterion(clean_preds, batch)
 551 |             if api.captured is None or clean_loss.numel() < 2:
 552 |                 return clean_loss, clean_items
 553 | 
 554 |             if self._api_is_boxgrad(api):
 555 |                 if clean_loss.numel() < 3:
 556 |                     return clean_loss, clean_items
 557 |                 loc_loss = clean_loss[0] + clean_loss[2]
 558 |                 grad = torch.autograd.grad(
 559 |                     loc_loss,
 560 |                     api.captured,
 561 |                     retain_graph=True,
 562 |                     create_graph=False,
 563 |                     allow_unused=True,
 564 |                 )[0]
 565 |                 if not api.set_perturbation_from_grad(
 566 |                     grad,
 567 |                     bboxes=batch.get("bboxes") if api.use_per_box_norm else None,
 568 |                     batch_idx=batch.get("batch_idx") if api.use_per_box_norm else None,
 569 |                 ):
 570 |                     return clean_loss, clean_items
 571 | 
 572 |                 for module in api_modules:
 573 |                     module.perturb()
 574 | 
 575 |                 set_boundary_context(batch.get("batch_idx"), batch.get("bboxes"), tuple(batch["img"].shape))
 576 |                 try:
 577 |                     if api.use_partial_forward:
 578 |                         perturbed_preds = self._predict_perturbed_partial(api, batch["img"])
 579 |                     else:
 580 |                         perturbed_preds = self.forward(batch["img"])
 581 |                 finally:
 582 |                     clear_boundary_context()
 583 | 
 584 |                 perturbed_loss, perturbed_items = self.criterion(perturbed_preds, batch)
 585 |                 total_loss = clean_loss.clone()
 586 |                 total_items = clean_items.clone()
 587 |                 w = api.current_api_weight
 588 |                 total_loss[0] = total_loss[0] + w * perturbed_loss[0]
 589 |                 total_loss[2] = total_loss[2] + w * perturbed_loss[2]
 590 |                 total_items[0] = total_items[0] + w * perturbed_items[0].detach()
 591 |                 total_items[2] = total_items[2] + w * perturbed_items[2].detach()
 592 |                 return total_loss, total_items.detach()
 593 | 
 594 |             target = self._api_target(batch, api)
 595 |             aux_clean_loss = api.auxiliary_loss(target, feature=api.captured)
 596 |             grad = torch.autograd.grad(
 597 |                 aux_clean_loss,
 598 |                 api.captured,
 599 |                 retain_graph=True,
 600 |                 create_graph=False,
 601 |                 allow_unused=True,
 602 |             )[0]
 603 |             if not api.set_perturbation_from_grad(grad):
 604 |                 return clean_loss, clean_items
 605 | 
 606 |             aux_loss = api.adversarial_auxiliary_loss(target)
 607 |             total_loss = clean_loss.clone()
 608 |             total_loss[1] = total_loss[1] + api.current_api_weight * aux_loss
 609 |             total_items = clean_items.clone()
 610 |             total_items[1] = total_items[1] + api.current_api_weight * aux_loss.detach()
 611 |             return total_loss, total_items.detach()
 612 |         finally:
 613 |             self._clear_api_modules()
 614 | 
 615 |     def loss(self, batch, preds=None):
 616 |         """Compute loss.
 617 | 
 618 |         Args:
 619 |             batch (dict): Batch to compute loss on.
 620 |             preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
 621 |         """
 622 |         if getattr(self, "criterion", None) is None:
 623 |             self.criterion = self.init_criterion()
 624 | 
 625 |         if preds is None:
 626 |             api_loss = self._api_adversarial_loss(batch)
 627 |             if api_loss is not None:
 628 |                 return api_loss
 629 |             set_boundary_context(batch.get("batch_idx"), batch.get("bboxes"), tuple(batch["img"].shape))
 630 |             try:
 631 |                 preds = self.forward(batch["img"])
 632 |             finally:
 633 |                 clear_boundary_context()
 634 |         return self.criterion(preds, batch)
 635 | 
 636 |     def init_criterion(self):
 637 |         """Initialize the loss criterion for the BaseModel."""
 638 |         raise NotImplementedError("compute_loss() needs to be implemented by task heads")
 639 | 
 640 | 
 641 | def _initialize_yolo_model(model, cfg, ch, nc, verbose):
 642 |     """Initialize common YOLO model attributes from a YAML config."""
 643 |     model.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
 644 |     if model.yaml["backbone"][0][2] == "Silence":
 645 |         LOGGER.warning(
 646 |             "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
 647 |             "Please delete local *.pt file and re-download the latest model checkpoint."
 648 |         )
 649 |         model.yaml["backbone"][0][2] = "nn.Identity"
 650 | 
 651 |     model.yaml["channels"] = ch  # save channels
 652 |     if nc and nc != model.yaml["nc"]:
 653 |         LOGGER.info(f"Overriding model.yaml nc={model.yaml['nc']} with nc={nc}")
 654 |         model.yaml["nc"] = nc  # override YAML value
 655 |     model.model, model.save = parse_model(deepcopy(model.yaml), ch=ch, verbose=verbose)  # model, savelist
 656 |     model.names = {i: f"{i}" for i in range(model.yaml["nc"])}  # default names dict
 657 |     model.inplace = model.yaml.get("inplace", True)
 658 | 
 659 | 
 660 | class DetectionModel(BaseModel):
 661 |     """YOLO detection model.
 662 | 
 663 |     This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
 664 |     inference, and loss computation for object detection tasks.
 665 | 
 666 |     Attributes:
 667 |         yaml (dict): Model configuration dictionary.
 668 |         model (torch.nn.Sequential): The neural network model.
 669 |         save (list): List of layer indices to save outputs from.
 670 |         names (dict): Class names dictionary.
 671 |         inplace (bool): Whether to use inplace operations.
 672 |         end2end (bool): Whether the model uses end-to-end detection.
 673 |         stride (torch.Tensor): Model stride values.
 674 | 
 675 |     Methods:
 676 |         __init__: Initialize the YOLO detection model.
 677 |         _predict_augment: Perform augmented inference.
 678 |         _descale_pred: De-scale predictions following augmented inference.
 679 |         _clip_augmented: Clip YOLO augmented inference tails.
 680 |         init_criterion: Initialize the loss criterion.
 681 | 
 682 |     Examples:
 683 |         Initialize a detection model
 684 |         >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
 685 |         >>> results = model.predict(image_tensor)
 686 |     """
 687 | 
 688 |     def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
 689 |         """Initialize the YOLO detection model with the given config and parameters.
 690 | 
 691 |         Args:
 692 |             cfg (str | dict): Model configuration file path or dictionary.
 693 |             ch (int): Number of input channels.
 694 |             nc (int, optional): Number of classes.
 695 |             verbose (bool): Whether to display model information.
 696 |         """
 697 |         super().__init__()
 698 |         _initialize_yolo_model(self, cfg, ch, nc, verbose)
 699 | 
 700 |         # Bind API layer indices so partial forward can skip backbone layers.
 701 |         for m in self.model:
 702 |             if isinstance(m, AdversarialPerturbationInjection):
 703 |                 m.layer_idx = m.i
 704 | 
 705 |         # Build strides
 706 |         m = self.model[-1]  # Detect()
 707 |         if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
 708 |             s = 256  # 2x min stride
 709 |             m.inplace = self.inplace
 710 | 
 711 |             def _forward(x):
 712 |                 """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
 713 |                 output = self.forward(x)
 714 |                 if self.end2end:
 715 |                     output = output["one2many"]
 716 |                 return output["feats"]
 717 | 
 718 |             self.model.eval()  # Avoid changing batch statistics until training begins
 719 |             m.training = True  # Setting it to True to properly return strides
 720 |             m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
 721 |             self.stride = m.stride
 722 |             self.model.train()  # Set model back to training(default) mode
 723 |             m.bias_init()  # only run once
 724 |         else:
 725 |             self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR
 726 | 
 727 |         # Init weights, biases
 728 |         initialize_weights(self)
 729 |         if verbose:
 730 |             self.info()
 731 |             LOGGER.info("")
 732 | 
 733 |     @property
 734 |     def end2end(self):
 735 |         """Return whether the model uses end-to-end NMS-free detection."""
 736 |         return getattr(self.model[-1], "end2end", False)
 737 | 
 738 |     @end2end.setter
 739 |     def end2end(self, value):
 740 |         """Override the end-to-end detection mode."""
 741 |         self.set_head_attr(end2end=value)
 742 | 
 743 |     def set_head_attr(self, **kwargs):
 744 |         """Set attributes of the model head (last layer).
 745 | 
 746 |         Args:
 747 |             **kwargs (Any): Arbitrary keyword arguments representing attributes to set.
 748 |         """
 749 |         head = self.model[-1]
 750 |         for k, v in kwargs.items():
 751 |             if not hasattr(head, k):
 752 |                 LOGGER.warning(f"Head has no attribute '{k}'.")
 753 |                 continue
 754 |             setattr(head, k, v)
 755 | 
 756 |     def _predict_augment(self, x):
 757 |         """Perform augmentations on input image x and return augmented inference and train outputs.
 758 | 
 759 |         Args:
 760 |             x (torch.Tensor): Input image tensor.
 761 | 
 762 |         Returns:
 763 |             (tuple[torch.Tensor, None]): Augmented inference output and None for train output.
 764 |         """
 765 |         if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
 766 |             LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
 767 |             return self._predict_once(x)
 768 |         img_size = x.shape[-2:]  # height, width
 769 |         s = [1, 0.83, 0.67]  # scales
 770 |         f = [None, 3, None]  # flips (2-ud, 3-lr)
 771 |         y = []  # outputs
 772 |         for si, fi in zip(s, f):
 773 |             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
 774 |             yi = super().predict(xi)[0]  # forward
 775 |             yi = self._descale_pred(yi, fi, si, img_size)
 776 |             y.append(yi)
 777 |         y = self._clip_augmented(y)  # clip augmented tails
 778 |         return torch.cat(y, -1), None  # augmented inference, train
 779 | 
 780 |     @staticmethod
 781 |     def _descale_pred(p, flips, scale, img_size, dim=1):
 782 |         """De-scale predictions following augmented inference (inverse operation).
 783 | 
 784 |         Args:
 785 |             p (torch.Tensor): Predictions tensor.
 786 |             flips (int | None): Flip type (None=none, 2=ud, 3=lr).
 787 |             scale (float): Scale factor.
 788 |             img_size (tuple): Original image size (height, width).
 789 |             dim (int): Dimension to split at.
 790 | 
 791 |         Returns:
 792 |             (torch.Tensor): De-scaled predictions.
 793 |         """
 794 |         p[:, :4] /= scale  # de-scale
 795 |         x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
 796 |         if flips == 2:
 797 |             y = img_size[0] - y  # de-flip ud
 798 |         elif flips == 3:
 799 |             x = img_size[1] - x  # de-flip lr
 800 |         return torch.cat((x, y, wh, cls), dim)
 801 | 
 802 |     def _clip_augmented(self, y):
 803 |         """Clip YOLO augmented inference tails.
 804 | 
 805 |         Args:
 806 |             y (list[torch.Tensor]): List of detection tensors.
 807 | 
 808 |         Returns:
 809 |             (list[torch.Tensor]): Clipped detection tensors.
 810 |         """
 811 |         nl = self.model[-1].nl  # number of detection layers (P3-P5)
 812 |         g = sum(4**x for x in range(nl))  # grid points
 813 |         e = 1  # exclude layer count
 814 |         i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
 815 |         y[0] = y[0][..., :-i]  # large
 816 |         i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
 817 |         y[-1] = y[-1][..., i:]  # small
 818 |         return y
 819 | 
 820 |     def init_criterion(self):
 821 |         """Initialize the loss criterion for the DetectionModel."""
 822 |         return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
 823 | 
 824 | 
 825 | class OBBModel(DetectionModel):
 826 |     """YOLO Oriented Bounding Box (OBB) model.
 827 | 
 828 |     This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
 829 |     computation for rotated object detection.
 830 | 
 831 |     Methods:
 832 |         __init__: Initialize YOLO OBB model.
 833 |         init_criterion: Initialize the loss criterion for OBB detection.
 834 | 
 835 |     Examples:
 836 |         Initialize an OBB model
 837 |         >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
 838 |         >>> results = model.predict(image_tensor)
 839 |     """
 840 | 
 841 |     def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
 842 |         """Initialize YOLO OBB model with given config and parameters.
 843 | 
 844 |         Args:
 845 |             cfg (str | dict): Model configuration file path or dictionary.
 846 |             ch (int): Number of input channels.
 847 |             nc (int, optional): Number of classes.
 848 |             verbose (bool): Whether to display model information.
 849 |         """
 850 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
 851 | 
 852 |     def init_criterion(self):
 853 |         """Initialize the loss criterion for the model."""
 854 |         return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)
 855 | 
 856 | 
 857 | class SegmentationModel(DetectionModel):
 858 |     """YOLO segmentation model.
 859 | 
 860 |     This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
 861 |     pixel-level object detection and segmentation.
 862 | 
 863 |     Methods:
 864 |         __init__: Initialize YOLO segmentation model.
 865 |         init_criterion: Initialize the loss criterion for segmentation.
 866 | 
 867 |     Examples:
 868 |         Initialize a segmentation model
 869 |         >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
 870 |         >>> results = model.predict(image_tensor)
 871 |     """
 872 | 
 873 |     def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
 874 |         """Initialize Ultralytics YOLO segmentation model with given config and parameters.
 875 | 
 876 |         Args:
 877 |             cfg (str | dict): Model configuration file path or dictionary.
 878 |             ch (int): Number of input channels.
 879 |             nc (int, optional): Number of classes.
 880 |             verbose (bool): Whether to display model information.
 881 |         """
 882 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
 883 | 
 884 |     def init_criterion(self):
 885 |         """Initialize the loss criterion for the SegmentationModel."""
 886 |         return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)
 887 | 
 888 | 
 889 | class SemanticSegmentationModel(BaseModel):
 890 |     """YOLO semantic segmentation model.
 891 | 
 892 |     This class implements a semantic segmentation model that produces per-pixel class predictions. Unlike
 893 |     SegmentationModel (instance segmentation), this does not produce bounding boxes.
 894 | 
 895 |     Methods:
 896 |         __init__: Initialize the semantic segmentation model.
 897 |         init_criterion: Initialize the loss criterion for semantic segmentation.
 898 | 
 899 |     Examples:
 900 |         Initialize a semantic segmentation model
 901 |         >>> model = SemanticSegmentationModel("yolo26n-sem.yaml", ch=3, nc=19)
 902 |     """
 903 | 
 904 |     def __init__(self, cfg="yolo26n-sem.yaml", ch=3, nc=None, verbose=True):
 905 |         """Initialize the YOLO semantic segmentation model.
 906 | 
 907 |         Args:
 908 |             cfg (str | dict): Model configuration file path or dictionary.
 909 |             ch (int): Number of input channels.
 910 |             nc (int, optional): Number of classes.
 911 |             verbose (bool): Whether to display model information.
 912 |         """
 913 |         super().__init__()
 914 |         _initialize_yolo_model(self, cfg, ch, nc, verbose)
 915 | 
 916 |         # Build strides: track smallest spatial size across all layers to find the deepest
 917 |         # backbone stride (e.g. P5/32). Head input alone is insufficient: the FPN upsamples
 918 |         # P5 away before the head, but the encoder still requires inputs aligned to that
 919 |         # deepest stride or FPN concats fail on rounding mismatches.
 920 |         m = self.model[-1]
 921 |         if isinstance(m, SemanticSegment):
 922 |             s = 256
 923 |             self.model.eval()
 924 |             m.training = True  # get training output (stride-4)
 925 |             min_h = [s]
 926 | 
 927 |             def _record(_m, _inp, out, _h=min_h):
 928 |                 if isinstance(out, torch.Tensor) and out.ndim == 4:
 929 |                     _h[0] = min(_h[0], out.shape[-2])
 930 | 
 931 |             hooks = [layer.register_forward_hook(_record) for layer in self.model]
 932 |             try:
 933 |                 self.forward(torch.zeros(1, ch, s, s))
 934 |             finally:
 935 |                 for h in hooks:
 936 |                     h.remove()
 937 |             m.stride = torch.tensor([s / min_h[0]], dtype=torch.float32)  # e.g., 256/8 = 32
 938 |             self.stride = m.stride
 939 |             self.model.train()
 940 |         else:
 941 |             self.stride = torch.Tensor([32])
 942 | 
 943 |         initialize_weights(self)
 944 |         if verbose:
 945 |             self.info()
 946 |             LOGGER.info("")
 947 | 
 948 |     def init_criterion(self):
 949 |         """Initialize the loss criterion for semantic segmentation."""
 950 |         return SemanticSegmentationLoss(self)
 951 | 
 952 |     def _apply(self, fn):
 953 |         """Apply a function to all tensors in the model."""
 954 |         self = super()._apply(fn)
 955 |         m = self.model[-1]
 956 |         if isinstance(m, SemanticSegment):
 957 |             m.stride = fn(m.stride)
 958 |         return self
 959 | 
 960 | 
 961 | class PoseModel(DetectionModel):
 962 |     """YOLO pose model.
 963 | 
 964 |     This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
 965 |     keypoint detection and pose estimation.
 966 | 
 967 |     Attributes:
 968 |         kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).
 969 | 
 970 |     Methods:
 971 |         __init__: Initialize YOLO pose model.
 972 |         init_criterion: Initialize the loss criterion for pose estimation.
 973 | 
 974 |     Examples:
 975 |         Initialize a pose model
 976 |         >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
 977 |         >>> results = model.predict(image_tensor)
 978 |     """
 979 | 
 980 |     def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
 981 |         """Initialize Ultralytics YOLO Pose model.
 982 | 
 983 |         Args:
 984 |             cfg (str | dict): Model configuration file path or dictionary.
 985 |             ch (int): Number of input channels.
 986 |             nc (int, optional): Number of classes.
 987 |             data_kpt_shape (tuple): Shape of keypoints data.
 988 |             verbose (bool): Whether to display model information.
 989 |         """
 990 |         if not isinstance(cfg, dict):
 991 |             cfg = yaml_model_load(cfg)  # load model YAML
 992 |         if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
 993 |             LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
 994 |             cfg["kpt_shape"] = data_kpt_shape
 995 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
 996 | 
 997 |     def init_criterion(self):
 998 |         """Initialize the loss criterion for the PoseModel."""
 999 |         return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)
1000 | 
1001 | 
1002 | class ClassificationModel(BaseModel):
1003 |     """YOLO classification model.
1004 | 
1005 |     This class implements the YOLO classification architecture for image classification tasks, providing model
1006 |     initialization, configuration, and output reshaping capabilities.
1007 | 
1008 |     Attributes:
1009 |         yaml (dict): Model configuration dictionary.
1010 |         model (torch.nn.Sequential): The neural network model.
1011 |         stride (torch.Tensor): Model stride values.
1012 |         names (dict): Class names dictionary.
1013 | 
1014 |     Methods:
1015 |         __init__: Initialize ClassificationModel.
1016 |         _from_yaml: Set model configurations and define architecture.
1017 |         reshape_outputs: Update model to specified class count.
1018 |         init_criterion: Initialize the loss criterion.
1019 | 
1020 |     Examples:
1021 |         Initialize a classification model
1022 |         >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
1023 |         >>> results = model.predict(image_tensor)
1024 |     """
1025 | 
1026 |     def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
1027 |         """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.
1028 | 
1029 |         Args:
1030 |             cfg (str | dict): Model configuration file path or dictionary.
1031 |             ch (int): Number of input channels.
1032 |             nc (int, optional): Number of classes.
1033 |             verbose (bool): Whether to display model information.
1034 |         """
1035 |         super().__init__()
1036 |         self._from_yaml(cfg, ch, nc, verbose)
1037 | 
1038 |     def _from_yaml(self, cfg, ch, nc, verbose):
1039 |         """Set Ultralytics YOLO model configurations and define the model architecture.
1040 | 
1041 |         Args:
1042 |             cfg (str | dict): Model configuration file path or dictionary.
1043 |             ch (int): Number of input channels.
1044 |             nc (int, optional): Number of classes.
1045 |             verbose (bool): Whether to display model information.
1046 |         """
1047 |         self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
1048 | 
1049 |         # Define model
1050 |         ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
1051 |         if nc and nc != self.yaml["nc"]:
1052 |             LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
1053 |             self.yaml["nc"] = nc  # override YAML value
1054 |         elif not nc and not self.yaml.get("nc", None):
1055 |             raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
1056 |         self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
1057 |         self.stride = torch.Tensor([1])  # no stride constraints
1058 |         self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
1059 |         self.info()
1060 | 
1061 |     @staticmethod
1062 |     def reshape_outputs(model, nc):
1063 |         """Update a TorchVision classification model to class count 'nc' if required.
1064 | 
1065 |         Args:
1066 |             model (torch.nn.Module): Model to update.
1067 |             nc (int): New number of classes.
1068 |         """
1069 |         name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
1070 |         if isinstance(m, Classify):  # YOLO Classify() head
1071 |             if m.linear.out_features != nc:
1072 |                 m.linear = torch.nn.Linear(m.linear.in_features, nc)
1073 |         elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
1074 |             if m.out_features != nc:
1075 |                 setattr(model, name, torch.nn.Linear(m.in_features, nc))
1076 |         elif isinstance(m, torch.nn.Sequential):
1077 |             types = [type(x) for x in m]
1078 |             if torch.nn.Linear in types:
1079 |                 i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
1080 |                 if m[i].out_features != nc:
1081 |                     m[i] = torch.nn.Linear(m[i].in_features, nc)
1082 |             elif torch.nn.Conv2d in types:
1083 |                 i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
1084 |                 if m[i].out_channels != nc:
1085 |                     m[i] = torch.nn.Conv2d(
1086 |                         m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
1087 |                     )
1088 | 
1089 |     def init_criterion(self):
1090 |         """Initialize the loss criterion for the ClassificationModel."""
1091 |         return v8ClassificationLoss()
1092 | 
1093 | 
1094 | class RTDETRDetectionModel(DetectionModel):
1095 |     """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.
1096 | 
1097 |     This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
1098 |     the training and inference processes. RTDETR is an object detection and tracking model that extends from the
1099 |     DetectionModel base class.
1100 | 
1101 |     Attributes:
1102 |         nc (int): Number of classes for detection.
1103 |         criterion (RTDETRDetectionLoss): Loss function for training.
1104 | 
1105 |     Methods:
1106 |         __init__: Initialize the RTDETRDetectionModel.
1107 |         init_criterion: Initialize the loss criterion.
1108 |         loss: Compute loss for training.
1109 |         predict: Perform forward pass through the model.
1110 | 
1111 |     Examples:
1112 |         Initialize an RTDETR model
1113 |         >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
1114 |         >>> results = model.predict(image_tensor)
1115 |     """
1116 | 
1117 |     def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
1118 |         """Initialize the RTDETRDetectionModel.
1119 | 
1120 |         Args:
1121 |             cfg (str | dict): Configuration file name or path.
1122 |             ch (int): Number of input channels.
1123 |             nc (int, optional): Number of classes.
1124 |             verbose (bool): Print additional information during initialization.
1125 |         """
1126 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
1127 | 
1128 |     def _apply(self, fn):
1129 |         """Apply a function to all tensors in the model, including decoder anchors and valid mask.
1130 | 
1131 |         Args:
1132 |             fn (function): The function to apply to the model.
1133 | 
1134 |         Returns:
1135 |             (RTDETRDetectionModel): An updated RTDETRDetectionModel object.
1136 |         """
1137 |         self = super()._apply(fn)
1138 |         m = self.model[-1]
1139 |         m.anchors = fn(m.anchors)
1140 |         m.valid_mask = fn(m.valid_mask)
1141 |         return self
1142 | 
1143 |     def init_criterion(self):
1144 |         """Initialize the loss criterion for the RTDETRDetectionModel."""
1145 |         from ultralytics.models.utils.loss import RTDETRDetectionLoss
1146 | 
1147 |         return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
1148 | 
1149 |     def loss(self, batch, preds=None):
1150 |         """Compute the loss for the given batch of data.
1151 | 
1152 |         Args:
1153 |             batch (dict): Dictionary containing image and label data.
1154 |             preds (tuple, optional): Precomputed model predictions.
1155 | 
1156 |         Returns:
1157 |             (torch.Tensor): Total loss value.
1158 |             (torch.Tensor): Main three losses in a tensor.
1159 |         """
1160 |         if not hasattr(self, "criterion"):
1161 |             self.criterion = self.init_criterion()
1162 | 
1163 |         img = batch["img"]
1164 |         # NOTE: preprocess gt_bbox and gt_labels to list.
1165 |         bs = img.shape[0]
1166 |         batch_idx = batch["batch_idx"]
1167 |         gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
1168 |         targets = {
1169 |             "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
1170 |             "bboxes": batch["bboxes"].to(device=img.device),
1171 |             "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
1172 |             "gt_groups": gt_groups,
1173 |         }
1174 | 
1175 |         if preds is None:
1176 |             preds = self.predict(img, batch=targets)
1177 |         dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
1178 |         if dn_meta is None:
1179 |             dn_bboxes, dn_scores = None, None
1180 |         else:
1181 |             dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
1182 |             dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
1183 | 
1184 |         dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
1185 |         dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
1186 | 
1187 |         loss = self.criterion(
1188 |             (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
1189 |         )
1190 |         # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
1191 |         return sum(loss.values()), torch.as_tensor(
1192 |             [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
1193 |         )
1194 | 
1195 |     def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
1196 |         """Perform a forward pass through the model.
1197 | 
1198 |         Args:
1199 |             x (torch.Tensor): The input tensor.
1200 |             profile (bool): If True, profile the computation time for each layer.
1201 |             visualize (bool): If True, save feature maps for visualization.
1202 |             batch (dict, optional): Ground truth data for evaluation.
1203 |             augment (bool): If True, perform data augmentation during inference.
1204 |             embed (list, optional): A list of layer indices to return embeddings from.
1205 | 
1206 |         Returns:
1207 |             (torch.Tensor): Model's output tensor.
1208 |         """
1209 |         y, dt, embeddings = [], [], []  # outputs
1210 |         embed = frozenset(embed) if embed is not None else {-1}
1211 |         max_idx = max(embed)
1212 |         for m in self.model[:-1]:  # except the head part
1213 |             if m.f != -1:  # if not from previous layer
1214 |                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
1215 |             if profile:
1216 |                 self._profile_one_layer(m, x, dt)
1217 |             x = m(x)  # run
1218 |             y.append(x if m.i in self.save else None)  # save output
1219 |             if visualize:
1220 |                 feature_visualization(x, m.type, m.i, save_dir=visualize)
1221 |             if m.i in embed:
1222 |                 embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
1223 |                 if m.i == max_idx:
1224 |                     return torch.unbind(torch.cat(embeddings, 1), dim=0)
1225 |         head = self.model[-1]
1226 |         x = head([y[j] for j in head.f], batch)  # head inference
1227 |         return x
1228 | 
1229 | 
1230 | class WorldModel(DetectionModel):
1231 |     """YOLOv8 World Model.
1232 | 
1233 |     This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
1234 |     specification and CLIP model integration for zero-shot detection capabilities.
1235 | 
1236 |     Attributes:
1237 |         txt_feats (torch.Tensor): Text feature embeddings for classes.
1238 |         clip_model (torch.nn.Module): CLIP model for text encoding.
1239 | 
1240 |     Methods:
1241 |         __init__: Initialize YOLOv8 world model.
1242 |         set_classes: Set classes for offline inference.
1243 |         get_text_pe: Get text positional embeddings.
1244 |         predict: Perform forward pass with text features.
1245 |         loss: Compute loss with text features.
1246 | 
1247 |     Examples:
1248 |         Initialize a world model
1249 |         >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
1250 |         >>> model.set_classes(["person", "car", "bicycle"])
1251 |         >>> results = model.predict(image_tensor)
1252 |     """
1253 | 
1254 |     def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
1255 |         """Initialize YOLOv8 world model with given config and parameters.
1256 | 
1257 |         Args:
1258 |             cfg (str | dict): Model configuration file path or dictionary.
1259 |             ch (int): Number of input channels.
1260 |             nc (int, optional): Number of classes.
1261 |             verbose (bool): Whether to display model information.
1262 |         """
1263 |         self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
1264 |         self.clip_model = None  # CLIP model placeholder
1265 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
1266 | 
1267 |     def set_classes(self, text, batch=80, cache_clip_model=True):
1268 |         """Set classes in advance so that model could do offline-inference without clip model.
1269 | 
1270 |         Args:
1271 |             text (list[str]): List of class names.
1272 |             batch (int): Batch size for processing text tokens.
1273 |             cache_clip_model (bool): Whether to cache the CLIP model.
1274 |         """
1275 |         self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
1276 |         self.model[-1].nc = len(text)
1277 | 
1278 |     def get_text_pe(self, text, batch=80, cache_clip_model=True):
1279 |         """Get text positional embeddings using the CLIP model.
1280 | 
1281 |         Args:
1282 |             text (list[str]): List of class names.
1283 |             batch (int): Batch size for processing text tokens.
1284 |             cache_clip_model (bool): Whether to cache the CLIP model.
1285 | 
1286 |         Returns:
1287 |             (torch.Tensor): Text positional embeddings.
1288 |         """
1289 |         from ultralytics.nn.text_model import build_text_model
1290 | 
1291 |         device = next(self.model.parameters()).device
1292 |         if not getattr(self, "clip_model", None) and cache_clip_model:
1293 |             # For backwards compatibility of models lacking clip_model attribute
1294 |             self.clip_model = build_text_model("clip:ViT-B/32", device=device)
1295 |         model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
1296 |         text_token = model.tokenize(text)
1297 |         txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
1298 |         txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
1299 |         return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
1300 | 
1301 |     def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
1302 |         """Perform a forward pass through the model.
1303 | 
1304 |         Args:
1305 |             x (torch.Tensor): The input tensor.
1306 |             profile (bool): If True, profile the computation time for each layer.
1307 |             visualize (bool): If True, save feature maps for visualization.
1308 |             txt_feats (torch.Tensor, optional): The text features, use it if it's given.
1309 |             augment (bool): If True, perform data augmentation during inference.
1310 |             embed (list, optional): A list of layer indices to return embeddings from.
1311 | 
1312 |         Returns:
1313 |             (torch.Tensor): Model's output tensor.
1314 |         """
1315 |         txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
1316 |         if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
1317 |             txt_feats = txt_feats.expand(x.shape[0], -1, -1)
1318 |         ori_txt_feats = txt_feats.clone()
1319 |         y, dt, embeddings = [], [], []  # outputs
1320 |         embed = frozenset(embed) if embed is not None else {-1}
1321 |         max_idx = max(embed)
1322 |         for m in self.model:  # except the head part
1323 |             if m.f != -1:  # if not from previous layer
1324 |                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
1325 |             if profile:
1326 |                 self._profile_one_layer(m, x, dt)
1327 |             if isinstance(m, C2fAttn):
1328 |                 x = m(x, txt_feats)
1329 |             elif isinstance(m, WorldDetect):
1330 |                 x = m(x, ori_txt_feats)
1331 |             elif isinstance(m, ImagePoolingAttn):
1332 |                 txt_feats = m(x, txt_feats)
1333 |             else:
1334 |                 x = m(x)  # run
1335 | 
1336 |             y.append(x if m.i in self.save else None)  # save output
1337 |             if visualize:
1338 |                 feature_visualization(x, m.type, m.i, save_dir=visualize)
1339 |             if m.i in embed:
1340 |                 embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
1341 |                 if m.i == max_idx:
1342 |                     return torch.unbind(torch.cat(embeddings, 1), dim=0)
1343 |         return x
1344 | 
1345 |     def loss(self, batch, preds=None):
1346 |         """Compute loss.
1347 | 
1348 |         Args:
1349 |             batch (dict): Batch to compute loss on.
1350 |             preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
1351 |         """
1352 |         if not hasattr(self, "criterion"):
1353 |             self.criterion = self.init_criterion()
1354 | 
1355 |         if preds is None:
1356 |             preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
1357 |         return self.criterion(preds, batch)
1358 | 
1359 | 
1360 | class YOLOEModel(DetectionModel):
1361 |     """YOLOE detection model.
1362 | 
1363 |     This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
1364 |     both prompt-based and prompt-free inference modes.
1365 | 
1366 |     Attributes:
1367 |         pe (torch.Tensor): Prompt embeddings for classes.
1368 |         clip_model (torch.nn.Module): CLIP model for text encoding.
1369 | 
1370 |     Methods:
1371 |         __init__: Initialize YOLOE model.
1372 |         get_text_pe: Get text positional embeddings.
1373 |         get_visual_pe: Get visual embeddings.
1374 |         set_vocab: Set vocabulary for prompt-free model.
1375 |         get_vocab: Get fused vocabulary layer.
1376 |         set_classes: Set classes for offline inference.
1377 |         get_cls_pe: Get class positional embeddings.
1378 |         predict: Perform forward pass with prompts.
1379 |         loss: Compute loss with prompts.
1380 | 
1381 |     Examples:
1382 |         Initialize a YOLOE model
1383 |         >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
1384 |         >>> results = model.predict(image_tensor, tpe=text_embeddings)
1385 |     """
1386 | 
1387 |     def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
1388 |         """Initialize YOLOE model with given config and parameters.
1389 | 
1390 |         Args:
1391 |             cfg (str | dict): Model configuration file path or dictionary.
1392 |             ch (int): Number of input channels.
1393 |             nc (int, optional): Number of classes.
1394 |             verbose (bool): Whether to display model information.
1395 |         """
1396 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
1397 |         self.text_model = self.yaml.get("text_model", "mobileclip:blt")
1398 | 
1399 |     @smart_inference_mode()
1400 |     def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
1401 |         """Get text positional embeddings using the CLIP model.
1402 | 
1403 |         Args:
1404 |             text (list[str]): List of class names.
1405 |             batch (int): Batch size for processing text tokens.
1406 |             cache_clip_model (bool): Whether to cache the CLIP model.
1407 |             without_reprta (bool): Whether to return text embeddings without reprta module processing.
1408 | 
1409 |         Returns:
1410 |             (torch.Tensor): Text positional embeddings.
1411 |         """
1412 |         from ultralytics.nn.text_model import build_text_model
1413 | 
1414 |         device = next(self.model.parameters()).device
1415 |         if not getattr(self, "clip_model", None) and cache_clip_model:
1416 |             # For backwards compatibility of models lacking clip_model attribute
1417 |             self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
1418 | 
1419 |         model = (
1420 |             self.clip_model
1421 |             if cache_clip_model
1422 |             else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
1423 |         )
1424 |         text_token = model.tokenize(text)
1425 |         txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
1426 |         txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
1427 |         txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
1428 |         if without_reprta:
1429 |             return txt_feats
1430 | 
1431 |         head = self.model[-1]
1432 |         assert isinstance(head, YOLOEDetect)
1433 |         return head.get_tpe(txt_feats)  # run auxiliary text head
1434 | 
1435 |     @smart_inference_mode()
1436 |     def get_visual_pe(self, img, visual):
1437 |         """Get visual positional embeddings.
1438 | 
1439 |         Args:
1440 |             img (torch.Tensor): Input image tensor.
1441 |             visual (torch.Tensor): Visual features.
1442 | 
1443 |         Returns:
1444 |             (torch.Tensor): Visual positional embeddings.
1445 |         """
1446 |         return self(img, vpe=visual, return_vpe=True)
1447 | 
1448 |     def set_vocab(self, vocab, names):
1449 |         """Set vocabulary for the prompt-free model.
1450 | 
1451 |         Args:
1452 |             vocab (nn.ModuleList): List of vocabulary items.
1453 |             names (list[str]): List of class names.
1454 |         """
1455 |         assert not self.training
1456 |         head = self.model[-1]
1457 |         assert isinstance(head, YOLOEDetect)
1458 | 
1459 |         # Cache anchors for head
1460 |         device = next(self.parameters()).device
1461 |         self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup
1462 | 
1463 |         cv3 = getattr(head, "one2one_cv3", head.cv3)
1464 |         cv2 = getattr(head, "one2one_cv2", head.cv2)
1465 | 
1466 |         # re-parameterization for prompt-free model
1467 |         self.model[-1].lrpc = nn.ModuleList(
1468 |             LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
1469 |         )
1470 |         for loc_head, cls_head in zip(head.cv2, head.cv3):
1471 |             assert isinstance(loc_head, nn.Sequential)
1472 |             assert isinstance(cls_head, nn.Sequential)
1473 |             del loc_head[-1]
1474 |             del cls_head[-1]
1475 |         self.model[-1].nc = len(names)
1476 |         self.names = check_class_names(names)
1477 | 
1478 |     def get_vocab(self, names):
1479 |         """Get fused vocabulary layer from the model.
1480 | 
1481 |         Args:
1482 |             names (list[str]): List of class names.
1483 | 
1484 |         Returns:
1485 |             (nn.ModuleList): List of vocabulary modules.
1486 |         """
1487 |         assert not self.training
1488 |         head = self.model[-1]
1489 |         assert isinstance(head, YOLOEDetect)
1490 |         assert not head.is_fused
1491 | 
1492 |         tpe = self.get_text_pe(names)
1493 |         self.set_classes(names, tpe)
1494 |         device = next(self.model.parameters()).device
1495 |         head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head
1496 | 
1497 |         cv3 = getattr(head, "one2one_cv3", head.cv3)
1498 |         vocab = nn.ModuleList()
1499 |         for cls_head in cv3:
1500 |             assert isinstance(cls_head, nn.Sequential)
1501 |             vocab.append(cls_head[-1])
1502 |         return vocab
1503 | 
1504 |     def set_classes(self, names, embeddings):
1505 |         """Set classes in advance so that model could do offline-inference without clip model.
1506 | 
1507 |         Args:
1508 |             names (list[str]): List of class names.
1509 |             embeddings (torch.Tensor): Embeddings tensor.
1510 |         """
1511 |         assert not hasattr(self.model[-1], "lrpc"), (
1512 |             "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
1513 |         )
1514 |         assert embeddings.ndim == 3
1515 |         self.pe = embeddings
1516 |         self.model[-1].nc = len(names)
1517 |         self.names = check_class_names(names)
1518 | 
1519 |     def get_cls_pe(self, tpe, vpe):
1520 |         """Get class positional embeddings.
1521 | 
1522 |         Args:
1523 |             tpe (torch.Tensor | None): Text positional embeddings.
1524 |             vpe (torch.Tensor | None): Visual positional embeddings.
1525 | 
1526 |         Returns:
1527 |             (torch.Tensor): Class positional embeddings.
1528 |         """
1529 |         all_pe = []
1530 |         if tpe is not None:
1531 |             assert tpe.ndim == 3
1532 |             all_pe.append(tpe)
1533 |         if vpe is not None:
1534 |             assert vpe.ndim == 3
1535 |             all_pe.append(vpe)
1536 |         if not all_pe:
1537 |             all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
1538 |         return torch.cat(all_pe, dim=1)
1539 | 
1540 |     def predict(
1541 |         self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
1542 |     ):
1543 |         """Perform a forward pass through the model.
1544 | 
1545 |         Args:
1546 |             x (torch.Tensor): The input tensor.
1547 |             profile (bool): If True, profile the computation time for each layer.
1548 |             visualize (bool): If True, save feature maps for visualization.
1549 |             tpe (torch.Tensor, optional): Text positional embeddings.
1550 |             augment (bool): If True, perform data augmentation during inference.
1551 |             embed (list, optional): A list of layer indices to return embeddings from.
1552 |             vpe (torch.Tensor, optional): Visual positional embeddings.
1553 |             return_vpe (bool): If True, return visual positional embeddings.
1554 | 
1555 |         Returns:
1556 |             (torch.Tensor): Model's output tensor.
1557 |         """
1558 |         y, dt, embeddings = [], [], []  # outputs
1559 |         b = x.shape[0]
1560 |         embed = frozenset(embed) if embed is not None else {-1}
1561 |         max_idx = max(embed)
1562 |         for m in self.model:  # except the head part
1563 |             if m.f != -1:  # if not from previous layer
1564 |                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
1565 |             if profile:
1566 |                 self._profile_one_layer(m, x, dt)
1567 |             if isinstance(m, YOLOEDetect):
1568 |                 vpe = m.get_vpe(x, vpe) if vpe is not None else None
1569 |                 if return_vpe:
1570 |                     assert vpe is not None
1571 |                     assert not self.training
1572 |                     return vpe
1573 |                 cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
1574 |                 if cls_pe.shape[0] != b or m.export:
1575 |                     cls_pe = cls_pe.expand(b, -1, -1)
1576 |                 x.append(cls_pe)  # adding cls embedding
1577 |             x = m(x)  # run
1578 | 
1579 |             y.append(x if m.i in self.save else None)  # save output
1580 |             if visualize:
1581 |                 feature_visualization(x, m.type, m.i, save_dir=visualize)
1582 |             if m.i in embed:
1583 |                 embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
1584 |                 if m.i == max_idx:
1585 |                     return torch.unbind(torch.cat(embeddings, 1), dim=0)
1586 |         return x
1587 | 
1588 |     def loss(self, batch, preds=None):
1589 |         """Compute loss.
1590 | 
1591 |         Args:
1592 |             batch (dict): Batch to compute loss on.
1593 |             preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
1594 |         """
1595 |         if not hasattr(self, "criterion"):
1596 |             from ultralytics.utils.loss import TVPDetectLoss
1597 | 
1598 |             visual_prompt = batch.get("visuals", None) is not None  # TODO
1599 |             self.criterion = (
1600 |                 (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
1601 |                 if visual_prompt
1602 |                 else self.init_criterion()
1603 |             )
1604 |         if preds is None:
1605 |             preds = self.forward(
1606 |                 batch["img"],
1607 |                 tpe=None if "visuals" in batch else batch.get("txt_feats", None),
1608 |                 vpe=batch.get("visuals", None),
1609 |             )
1610 |         return self.criterion(preds, batch)
1611 | 
1612 | 
1613 | class YOLOESegModel(YOLOEModel, SegmentationModel):
1614 |     """YOLOE segmentation model.
1615 | 
1616 |     This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
1617 |     specialized loss computation for pixel-level object detection and segmentation.
1618 | 
1619 |     Methods:
1620 |         __init__: Initialize YOLOE segmentation model.
1621 |         loss: Compute loss with prompts for segmentation.
1622 | 
1623 |     Examples:
1624 |         Initialize a YOLOE segmentation model
1625 |         >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
1626 |         >>> results = model.predict(image_tensor, tpe=text_embeddings)
1627 |     """
1628 | 
1629 |     def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
1630 |         """Initialize YOLOE segmentation model with given config and parameters.
1631 | 
1632 |         Args:
1633 |             cfg (str | dict): Model configuration file path or dictionary.
1634 |             ch (int): Number of input channels.
1635 |             nc (int, optional): Number of classes.
1636 |             verbose (bool): Whether to display model information.
1637 |         """
1638 |         super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
1639 | 
1640 |     def loss(self, batch, preds=None):
1641 |         """Compute loss.
1642 | 
1643 |         Args:
1644 |             batch (dict): Batch to compute loss on.
1645 |             preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
1646 |         """
1647 |         if not hasattr(self, "criterion"):
1648 |             from ultralytics.utils.loss import TVPSegmentLoss
1649 | 
1650 |             visual_prompt = batch.get("visuals", None) is not None  # TODO
1651 |             self.criterion = (
1652 |                 (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
1653 |                 if visual_prompt
1654 |                 else self.init_criterion()
1655 |             )
1656 | 
1657 |         if preds is None:
1658 |             preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
1659 |         return self.criterion(preds, batch)
1660 | 
1661 | 
1662 | class Ensemble(torch.nn.ModuleList):
1663 |     """Ensemble of models.
1664 | 
1665 |     This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
1666 |     or other ensemble techniques.
1667 | 
1668 |     Methods:
1669 |         __init__: Initialize an ensemble of models.
1670 |         forward: Generate predictions from all models in the ensemble.
1671 | 
1672 |     Examples:
1673 |         Create an ensemble of models
1674 |         >>> ensemble = Ensemble()
1675 |         >>> ensemble.append(model1)
1676 |         >>> ensemble.append(model2)
1677 |         >>> results = ensemble(image_tensor)
1678 |     """
1679 | 
1680 |     def __init__(self):
1681 |         """Initialize an ensemble of models."""
1682 |         super().__init__()
1683 | 
1684 |     def forward(self, x, augment=False, profile=False, visualize=False):
1685 |         """Run ensemble forward pass and concatenate predictions from all models.
1686 | 
1687 |         Args:
1688 |             x (torch.Tensor): Input tensor.
1689 |             augment (bool): Whether to augment the input.
1690 |             profile (bool): Whether to profile the model.
1691 |             visualize (bool): Whether to visualize the features.
1692 | 
1693 |         Returns:
1694 |             (torch.Tensor): Concatenated predictions from all models.
1695 |             (None): Always None for ensemble inference.
1696 |         """
1697 |         y = [module(x, augment, profile, visualize)[0] for module in self]
1698 |         # y = torch.stack(y).max(0)[0]  # max ensemble
1699 |         # y = torch.stack(y).mean(0)  # mean ensemble
1700 |         y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C*num_models)
1701 |         return y, None  # inference, train output
1702 | 
1703 | 
1704 | # Functions ------------------------------------------------------------------------------------------------------------
1705 | 
1706 | 
1707 | @contextlib.contextmanager
1708 | def temporary_modules(modules=None, attributes=None):
1709 |     """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).
1710 | 
1711 |     This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
1712 |     moved a module from one location to another, but you still want to support the old import paths for backwards
1713 |     compatibility.
1714 | 
1715 |     Args:
1716 |         modules (dict, optional): A dictionary mapping old module paths to new module paths.
1717 |         attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.
1718 | 
1719 |     Examples:
1720 |         >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
1721 |         >>> import old.module  # this will now import new.module
1722 |         >>> from old.module import attribute  # this will now import new.module.attribute
1723 | 
1724 |     Notes:
1725 |         The changes are only in effect inside the context manager and are undone once the context manager exits.
1726 |         Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
1727 |         applications or libraries. Use this function with caution.
1728 |     """
1729 |     if modules is None:
1730 |         modules = {}
1731 |     if attributes is None:
1732 |         attributes = {}
1733 |     import sys
1734 |     from importlib import import_module
1735 | 
1736 |     try:
1737 |         # Set attributes in sys.modules under their old name
1738 |         for old, new in attributes.items():
1739 |             old_module, old_attr = old.rsplit(".", 1)
1740 |             new_module, new_attr = new.rsplit(".", 1)
1741 |             setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))
1742 | 
1743 |         # Set modules in sys.modules under their old name
1744 |         for old, new in modules.items():
1745 |             sys.modules[old] = import_module(new)
1746 | 
1747 |         yield
1748 |     finally:
1749 |         # Remove the temporary module paths
1750 |         for old in modules:
1751 |             if old in sys.modules:
1752 |                 del sys.modules[old]
1753 | 
1754 | 
1755 | class _SafeLoad:
1756 |     """Opt-in restricted checkpoint loading: reconstruct only known model classes (`weights_only=True` plus an
1757 |     allow-list) and build models without `eval()`.
1758 | 
1759 |     Enabled per-process by the `ULTRALYTICS_SAFE_LOAD` env flag, or per-call by `torch_safe_load(..., safe_only=True)`.
1760 |     Default loading (flag off) is unchanged.
1761 |     """
1762 | 
1763 |     # Restricted loading reconstructs allow-listed classes via the torch.serialization.safe_globals context manager,
1764 |     # added in torch 2.5. On older torch it is unavailable, so restricted loading degrades to a standard load there.
1765 |     SUPPORTED = hasattr(torch.serialization, "safe_globals")
1766 |     _globals = None  # cached allow-list, built once
1767 |     _local = threading.local()  # per-thread flag set while a weights_only load is in progress
1768 | 
1769 |     @classmethod
1770 |     def restricted(cls):
1771 |         """Whether model construction should use the no-eval, known-layer path (env flag or an in-progress load)."""
1772 |         return cls.SUPPORTED and (SAFE_LOAD or getattr(cls._local, "active", False))
1773 | 
1774 |     @classmethod
1775 |     @contextlib.contextmanager
1776 |     def loading(cls):
1777 |         """Load with `weights_only=True`: scope the allow-list to this load and mark the thread restricted, so a
1778 |         checkpoint that reaches model construction (parse_model) also uses the no-eval, known-layer path.
1779 |         """
1780 |         if cls._globals is None:
1781 |             cls._globals = cls._build()
1782 |         cls._local.active = True
1783 |         try:
1784 |             with torch.serialization.safe_globals(cls._globals):
1785 |                 yield
1786 |         finally:
1787 |             cls._local.active = False
1788 | 
1789 |     @staticmethod
1790 |     def activation(act):
1791 |         """Resolve a model-yaml `activation` spec to a `torch.nn` module instance without `eval()`.
1792 | 
1793 |         Accepts only the documented `[torch.]nn.<Class>(literal args)` shape (e.g. `nn.SiLU()`,
1794 |         `torch.nn.LeakyReLU(0.1)`) with literal arguments, and rejects anything else.
1795 |         """
1796 |         import ast
1797 | 
1798 |         try:
1799 |             call = ast.parse(act.strip(), mode="eval").body
1800 |             assert isinstance(call, ast.Call)
1801 |             attrs = []
1802 |             node = call.func
1803 |             while isinstance(node, ast.Attribute):  # unwind e.g. torch.nn.SiLU -> ["SiLU","nn","torch"]
1804 |                 attrs.append(node.attr)
1805 |                 node = node.value
1806 |             assert isinstance(node, ast.Name)
1807 |             attrs.append(node.id)  # e.g. ["SiLU", "nn"] or ["SiLU", "nn", "torch"]
1808 |             assert attrs[1:] in (["nn"], ["nn", "torch"]), "activation must be a torch.nn class"
1809 |             klass = getattr(nn, attrs[0])
1810 |             assert isinstance(klass, type) and issubclass(klass, nn.Module)
1811 |             args = [ast.literal_eval(a) for a in call.args]
1812 |             kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
1813 |             return klass(*args, **kwargs)
1814 |         except Exception as e:
1815 |             raise TypeError(
1816 |                 emojis(f"ERROR ❌️ unsupported activation '{act}' blocked during restricted model load.")
1817 |             ) from e
1818 | 
1819 |     @classmethod
1820 |     def _build(cls):
1821 |         """Auto-discover `nn.Module` subclasses across `torch.nn` and the ultralytics model families, registered under
1822 |         every namespace path they are reachable from (covering re-exports such as `block.RealNVP` as
1823 |         `head.RealNVP`), plus torchvision transforms and legacy aliases.
1824 | 
1825 |         Returns:
1826 |             (list): Items for `torch.serialization.safe_globals` — classes and `(obj, "module.Name")` aliases.
1827 |         """
1828 |         import importlib
1829 |         import inspect
1830 |         import pathlib
1831 |         import pkgutil
1832 | 
1833 |         import torch.nn.modules as torch_nn
1834 | 
1835 |         import ultralytics.nn.modules as ul_nn
1836 |         import ultralytics.nn.tasks as ul_tasks
1837 | 
1838 |         allow = []
1839 | 
1840 |         def _scan(pkg):
1841 |             mods = [pkg]
1842 |             if hasattr(pkg, "__path__"):  # package: include all submodules
1843 |                 for info in pkgutil.iter_modules(pkg.__path__, f"{pkg.__name__}."):
1844 |                     try:
1845 |                         mods.append(importlib.import_module(info.name))
1846 |                     except Exception:  # optional/oddball submodule — skip
1847 |                         continue
1848 |             for mod in mods:
1849 |                 for name, klass in inspect.getmembers(mod, inspect.isclass):
1850 |                     if issubclass(klass, nn.Module):
1851 |                         # Register under the path the class is reachable from — matches how a checkpoint pickled it.
1852 |                         allow.append((klass, f"{mod.__name__}.{name}"))
1853 | 
1854 |         _scan(torch_nn)  # PyTorch nn modules
1855 |         _scan(ul_nn)  # ultralytics block/conv/head/transformer
1856 |         _scan(ul_tasks)  # ultralytics task models
1857 | 
1858 |         # Non-nn.Module data globals in official checkpoints (classification preprocessing transforms).
1859 |         try:
1860 |             import torchvision.transforms.transforms as tvt
1861 |             from torchvision.transforms.functional import InterpolationMode
1862 | 
1863 |             allow += [tvt.Compose, tvt.Normalize, tvt.Resize, tvt.CenterCrop, tvt.ToTensor, InterpolationMode]
1864 |         except ImportError:
1865 |             pass
1866 | 
1867 |         # Legacy/cross-platform aliases (pickled paths with no current class namespace), mirroring temporary_modules().
1868 |         from ultralytics.utils.loss import E2EDetectLoss
1869 | 
1870 |         allow += [
1871 |             (nn.Identity, "ultralytics.nn.modules.block.Silence"),  # YOLOv9e
1872 |             (DetectionModel, "ultralytics.nn.tasks.YOLOv10DetectionModel"),  # YOLOv10
1873 |             (E2EDetectLoss, "ultralytics.utils.loss.v10DetectLoss"),  # YOLOv10
1874 |         ]
1875 |         if WINDOWS:
1876 |             allow += [pathlib.WindowsPath, (pathlib.WindowsPath, "pathlib.PosixPath")]
1877 |         else:
1878 |             allow += [pathlib.PosixPath, (pathlib.PosixPath, "pathlib.WindowsPath")]
1879 |         return allow
1880 | 
1881 | 
1882 | def torch_safe_load(weight, safe_only=None):
1883 |     """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
1884 |     the error, logs a warning message, and attempts to install the missing module via the check_requirements()
1885 |     function. After installation, the function again attempts to load the model using torch.load().
1886 | 
1887 |     Args:
1888 |         weight (str | Path): The file path of the PyTorch model.
1889 |         safe_only (bool, optional): Load with `torch.load(weights_only=True)`, reconstructing only the known
1890 |             Ultralytics/torch model classes on the allow-list. Defaults to the `ULTRALYTICS_SAFE_LOAD` environment
1891 |             variable (off), so standard usage is unchanged; set the env to opt in.
1892 | 
1893 |     Returns:
1894 |         (dict): The loaded model checkpoint.
1895 |         (str): The loaded filename.
1896 | 
1897 |     Examples:
1898 |         >>> from ultralytics.nn.tasks import torch_safe_load
1899 |         >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
1900 |     """
1901 |     from ultralytics.utils.downloads import attempt_download_asset
1902 | 
1903 |     if safe_only is None:
1904 |         safe_only = SAFE_LOAD
1905 |     if safe_only and not _SafeLoad.SUPPORTED:
1906 |         LOGGER.warning("Restricted model loading requires torch>=2.5; loading without restriction.")
1907 |         safe_only = False
1908 |     check_suffix(file=weight, suffix=".pt")
1909 |     file = attempt_download_asset(weight)  # search online if missing locally
1910 | 
1911 |     def _load():
1912 |         with temporary_modules(
1913 |             modules={
1914 |                 "ultralytics.yolo.utils": "ultralytics.utils",
1915 |                 "ultralytics.yolo.v8": "ultralytics.models.yolo",
1916 |                 "ultralytics.yolo.data": "ultralytics.data",
1917 |             },
1918 |             attributes={
1919 |                 "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
1920 |                 "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
1921 |                 "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
1922 |                 # resolve cross-platform pathlib pickle incompatibility
1923 |                 **(
1924 |                     {"pathlib.PosixPath": "pathlib.WindowsPath"}
1925 |                     if WINDOWS
1926 |                     else {"pathlib.WindowsPath": "pathlib.PosixPath"}
1927 |                 ),
1928 |             },
1929 |         ):
1930 |             if safe_only:
1931 |                 with _SafeLoad.loading():  # weights_only load scoped to the known-class allow-list
1932 |                     return torch_load(file, map_location="cpu", weights_only=True)
1933 |             return torch_load(file, map_location="cpu")
1934 | 
1935 |     try:
1936 |         ckpt = _load()
1937 | 
1938 |     except RuntimeError as e:
1939 |         # Corrupt downloaded weight (e.g. truncated); skip user-supplied local paths to avoid destructive unlink.
1940 |         if "PytorchStreamReader" not in str(e) or Path(str(weight)).exists():
1941 |             raise
1942 |         LOGGER.warning(f"Corrupt cache {file}, re-downloading {weight}...")
1943 |         Path(file).unlink(missing_ok=True)
1944 |         file = attempt_download_asset(weight)
1945 |         ckpt = _load()
1946 | 
1947 |     except ModuleNotFoundError as e:  # e.name is missing module name
1948 |         if e.name == "models":
1949 |             raise TypeError(
1950 |                 emojis(
1951 |                     f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
1952 |                     f"with https://github.com/ultralytics/yolov5. This model is NOT forwards compatible with "
1953 |                     f"YOLOv8 at https://github.com/ultralytics/ultralytics."
1954 |                     f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
1955 |                     f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
1956 |                 )
1957 |             ) from e
1958 |         elif e.name == "numpy._core":
1959 |             raise ModuleNotFoundError(
1960 |                 emojis(
1961 |                     f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
1962 |                 )
1963 |             ) from e
1964 |         elif e.name and e.name.startswith("ultralytics."):
1965 |             raise ModuleNotFoundError(
1966 |                 emojis(
1967 |                     f"ERROR ❌️ {weight} requires missing Ultralytics module '{e.name}'. "
1968 |                     "Train a new model using the latest 'ultralytics' package or run a command with an official "
1969 |                     "Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
1970 |                 )
1971 |             ) from e
1972 |         if safe_only:
1973 |             # Under weights_only loading, do not auto-install a module named by the checkpoint or fall back to a
1974 |             # weights_only=False reload.
1975 |             raise
1976 |         LOGGER.warning(
1977 |             f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
1978 |             f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
1979 |             f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
1980 |             f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
1981 |         )
1982 |         check_requirements(e.name)  # install missing module
1983 |         ckpt = torch_load(file, map_location="cpu")
1984 | 
1985 |     except pickle.UnpicklingError as e:
1986 |         # weights_only=True encountered a global outside the allow-list. The default (weights_only=False) path can also
1987 |         # raise this for a corrupt or legacy file, so re-raise verbatim there to preserve existing behavior.
1988 |         if not safe_only:
1989 |             raise
1990 |         raise TypeError(
1991 |             emojis(
1992 |                 f"ERROR ❌️ {weight} references types outside the supported Ultralytics checkpoint format. "
1993 |                 f"Use an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
1994 |             )
1995 |         ) from e
1996 | 
1997 |     if not isinstance(ckpt, dict):
1998 |         # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
1999 |         LOGGER.warning(
2000 |             f"The file '{weight}' appears to be improperly saved or formatted. "
2001 |             f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
2002 |         )
2003 |         ckpt = {"model": ckpt.model}
2004 | 
2005 |     return ckpt, file
2006 | 
2007 | 
2008 | def load_checkpoint(weight, device=None, inplace=True, fuse=False):
2009 |     """Load single model weights.
2010 | 
2011 |     Args:
2012 |         weight (str | Path): Model weight path.
2013 |         device (torch.device, optional): Device to load model to.
2014 |         inplace (bool): Whether to do inplace operations.
2015 |         fuse (bool): Whether to fuse model.
2016 | 
2017 |     Returns:
2018 |         (torch.nn.Module): Loaded model.
2019 |         (dict): Model checkpoint dictionary.
2020 |     """
2021 |     if str(weight).lower().startswith(REMOTE_FILE_PREFIXES):
2022 |         weight = check_file(weight, download_dir=SETTINGS["weights_dir"])
2023 |     ckpt, weight = torch_safe_load(weight)  # load ckpt
2024 |     args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
2025 |     model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model
2026 | 
2027 |     # Model compatibility updates
2028 |     model.args = args  # attach args to model
2029 |     model.pt_path = str(weight)  # attach *.pt file path to model as string (avoids WindowsPath pickle issues)
2030 |     model.task = getattr(model, "task", guess_model_task(model))
2031 |     if not hasattr(model, "stride"):
2032 |         model.stride = torch.tensor([32.0])
2033 | 
2034 |     model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode
2035 | 
2036 |     # Module updates
2037 |     for m in model.modules():
2038 |         if hasattr(m, "inplace"):
2039 |             m.inplace = inplace
2040 |         elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
2041 |             m.recompute_scale_factor = None  # torch 1.11.0 compatibility
2042 | 
2043 |     # Return model and ckpt
2044 |     return model, ckpt
2045 | 
2046 | 
2047 | def parse_model(d, ch, verbose=True):
2048 |     """Parse a YOLO model.yaml dictionary into a PyTorch model.
2049 | 
2050 |     Args:
2051 |         d (dict): Model dictionary.
2052 |         ch (int): Input channels.
2053 |         verbose (bool): Whether to print model details.
2054 | 
2055 |     Returns:
2056 |         (torch.nn.Sequential): PyTorch model.
2057 |         (list): Sorted list of layer indices whose outputs need to be saved.
2058 |     """
2059 |     import ast
2060 | 
2061 |     # Args
2062 |     legacy = True  # backward compatibility for v3/v5/v8/v9 models
2063 |     max_channels = float("inf")
2064 |     nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
2065 |     reg_max = d.get("reg_max", 16)
2066 |     cls_geometry_fuse = d.get("cls_geometry_fuse", False)
2067 |     cls_geometry_mode = d.get("cls_geometry_mode", "add")
2068 |     cls_geometry_detach = d.get("cls_geometry_detach", True)
2069 |     cls_deform_geometry = d.get("cls_deform_geometry", False)
2070 |     quality_head = d.get("quality_head", False)
2071 |     quality_score_mode = d.get("quality_score_mode", "cls_mul_q")
2072 |     quality_box_features = d.get("quality_box_features", False)
2073 |     quality_box_detach = d.get("quality_box_detach", True)
2074 |     depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
2075 |     scale = d.get("scale")
2076 |     if scales:
2077 |         if not scale:
2078 |             scale = next(iter(scales.keys()))
2079 |             LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
2080 |         depth, width, max_channels = scales[scale]
2081 | 
2082 |     restricted = _SafeLoad.restricted()
2083 |     if act:
2084 |         # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU(). Under restricted loading, resolve the
2085 |         # spec without eval() (see _SafeLoad.activation).
2086 |         Conv.default_act = _SafeLoad.activation(act) if restricted else eval(act)
2087 |         if verbose:
2088 |             LOGGER.info(f"{colorstr('activation:')} {act}")  # print
2089 | 
2090 |     if verbose:
2091 |         LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
2092 |     ch = [ch]
2093 |     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
2094 |     base_modules = frozenset(
2095 |         {
2096 |             Classify,
2097 |             Conv,
2098 |             ConvTranspose,
2099 |             GhostConv,
2100 |             Bottleneck,
2101 |             GhostBottleneck,
2102 |             SPP,
2103 |             SPPF,
2104 |             C2fPSA,
2105 |             C2PSA,
2106 |             DWConv,
2107 |             Focus,
2108 |             BottleneckCSP,
2109 |             C1,
2110 |             C2,
2111 |             C2f,
2112 |             C2fCBAM,
2113 |             C2fKV,
2114 |             C2fNAT,
2115 |             C3k2,
2116 |             RepNCSPELAN4,
2117 |             ELAN1,
2118 |             ADown,
2119 |             AConv,
2120 |             SPPELAN,
2121 |             C2fAttn,
2122 |             C3,
2123 |             C3CBAM,
2124 |             C3TR,
2125 |             C3Ghost,
2126 |             torch.nn.ConvTranspose2d,
2127 |             DWConvTranspose2d,
2128 |             C3x,
2129 |             RepC3,
2130 |             RepC2f,
2131 |             EnSimAMEdgeRepC2f,
2132 |             PoolingEdgeRepC2f,
2133 |             PSA,
2134 |             SCDown,
2135 |             C2fCIB,
2136 |             A2C2f,
2137 |             BiLevelRoutingAttention,
2138 |             KVCompressedAttention,
2139 |             KVCompressedAttentionPartial,
2140 |             KVCompressedTransformerEncoder,
2141 |             RegionRoutingAttentionLite,
2142 |             NATBlock,
2143 |             TopKAdaptiveGroupKVAttention,
2144 |             TopKGlobalGroupKVAttention,
2145 |         }
2146 |     )
2147 |     repeat_modules = frozenset(  # modules with 'repeat' arguments
2148 |         {
2149 |             BottleneckCSP,
2150 |             C1,
2151 |             C2,
2152 |             C2f,
2153 |             C2fCBAM,
2154 |             C2fKV,
2155 |             C2fNAT,
2156 |             C3k2,
2157 |             C2fAttn,
2158 |             C3,
2159 |             C3CBAM,
2160 |             C3TR,
2161 |             C3Ghost,
2162 |             C3x,
2163 |             RepC3,
2164 |             RepC2f,
2165 |             EnSimAMEdgeRepC2f,
2166 |             PoolingEdgeRepC2f,
2167 |             C2fPSA,
2168 |             C2fCIB,
2169 |             C2PSA,
2170 |             A2C2f,
2171 |         }
2172 |     )
2173 |     for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
2174 |         m = (
2175 |             getattr(torch.nn, m[3:])
2176 |             if m.startswith("nn.")
2177 |             else getattr(__import__("torchvision").ops, m[16:])
2178 |             if m.startswith("torchvision.ops.")
2179 |             else globals()[m]
2180 |         )  # get module
2181 |         if restricted and not (isinstance(m, type) and issubclass(m, torch.nn.Module)):
2182 |             # Under restricted loading, only known model layers may be named here.
2183 |             raise TypeError(emojis(f"ERROR ❌️ module '{m}' is not a permitted model layer under restricted loading."))
2184 |         for j, a in enumerate(args):
2185 |             if isinstance(a, str):
2186 |                 with contextlib.suppress(ValueError):
2187 |                     args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
2188 |         n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
2189 |         if m in base_modules:
2190 |             c1, c2 = ch[f], args[0]
2191 |             if c2 != nc:  # if c2 != nc (e.g., Classify() output)
2192 |                 c2 = make_divisible(min(c2, max_channels) * width, 8)
2193 |             if m is C2fAttn:  # set 1) embed channels and 2) num heads
2194 |                 args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
2195 |                 args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
2196 | 
2197 |             args = [c1, c2, *args[1:]]
2198 |             # KVCompressed* modules: no legacy args to append (force_fp32_attention removed)
2199 |             if m in repeat_modules:
2200 |                 args.insert(2, n)  # number of repeats
2201 |                 n = 1
2202 |             if m is C3k2:  # for M/L/X sizes
2203 |                 legacy = False
2204 |                 if scale in "mlx":
2205 |                     args[3] = True
2206 |             if m is A2C2f:
2207 |                 legacy = False
2208 |                 if scale in "lx":  # for L/X sizes
2209 |                     args.extend((True, 1.2))
2210 |             if m is C2fCIB:
2211 |                 legacy = False
2212 |         elif m is M3NATFuse:
2213 |             c1, c2 = [ch[x] for x in f], args[0]
2214 |             if c2 != nc:
2215 |                 c2 = make_divisible(min(c2, max_channels) * width, 8)
2216 |             args = [c1, c2, *args[1:]]
2217 |         elif m is ScalSeq:
2218 |             c1, c2 = [ch[x] for x in f], args[0]
2219 |             if c2 != nc:
2220 |                 c2 = make_divisible(min(c2, max_channels) * width, 8)
2221 |             args = [c1, c2, *args[1:]]
2222 |         elif m is ASFAttention:
2223 |             c2 = ch[f]
2224 |             args = [c2, *args]
2225 |         elif m in frozenset({AdversarialPerturbationInjection, BoundaryFeatureBlock, FeatureDGFE}):
2226 |             c2 = ch[f]
2227 |             args = [c2, *args]
2228 |         elif m is EnSimAM:
2229 |             c2 = ch[f]
2230 |         elif m is WeightedAdd:
2231 |             c2 = ch[f[0]] if isinstance(f, list) else ch[f]
2232 |         elif m is AIFI:
2233 |             args = [ch[f], *args]
2234 |         elif m in frozenset({HGStem, HGBlock}):
2235 |             c1, cm, c2 = ch[f], args[0], args[1]
2236 |             args = [c1, cm, c2, *args[2:]]
2237 |             if m is HGBlock:
2238 |                 args.insert(4, n)  # number of repeats
2239 |                 n = 1
2240 |         elif m is ResNetLayer:
2241 |             c2 = args[1] if args[3] else args[1] * 4
2242 |         elif m is torch.nn.BatchNorm2d:
2243 |             args = [ch[f]]
2244 |         elif m is Concat:
2245 |             c2 = sum(ch[x] for x in f)
2246 |         elif m in frozenset(
2247 |             {
2248 |                 Detect,
2249 |                 WorldDetect,
2250 |                 YOLOEDetect,
2251 |                 Segment,
2252 |                 Segment26,
2253 |                 YOLOESegment,
2254 |                 YOLOESegment26,
2255 |                 Pose,
2256 |                 Pose26,
2257 |                 OBB,
2258 |                 OBB26,
2259 |             }
2260 |         ):
2261 |             args.extend([reg_max, end2end, [ch[x] for x in f]])
2262 |             if m is Detect:
2263 |                 args.extend(
2264 |                     [
2265 |                         cls_geometry_fuse,
2266 |                         cls_geometry_mode,
2267 |                         cls_geometry_detach,
2268 |                         cls_deform_geometry,
2269 |                         quality_head,
2270 |                         quality_score_mode,
2271 |                         quality_box_features,
2272 |                         quality_box_detach,
2273 |                     ]
2274 |                 )
2275 |             if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
2276 |                 args[2] = make_divisible(min(args[2], max_channels) * width, 8)
2277 |             if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
2278 |                 m.legacy = legacy
2279 |         elif m is SemanticSegment:
2280 |             args.append([ch[x] for x in f])  # nc, ch tuple
2281 |         elif m is v10Detect:
2282 |             args.append([ch[x] for x in f])
2283 |         elif m is ImagePoolingAttn:
2284 |             args.insert(1, [ch[x] for x in f])  # channels as second arg
2285 |         elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
2286 |             args.insert(1, [ch[x] for x in f])
2287 |         elif m is CBLinear:
2288 |             c2 = args[0]
2289 |             c1 = ch[f]
2290 |             args = [c1, c2, *args[1:]]
2291 |         elif m is CBFuse:
2292 |             c2 = ch[f[-1]]
2293 |         elif m in frozenset({TorchVision, Index}):
2294 |             c2 = args[0]
2295 |             c1 = ch[f]
2296 |             args = [*args[1:]]
2297 |         else:
2298 |             c2 = ch[f]
2299 | 
2300 |         m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
2301 |         t = str(m)[8:-2].replace("__main__.", "")  # module type
2302 |         m_.np = sum(x.numel() for x in m_.parameters())  # number params
2303 |         m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
2304 |         if verbose:
2305 |             LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
2306 |         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
2307 |         layers.append(m_)
2308 |         if i == 0:
2309 |             ch = []
2310 |         ch.append(c2)
2311 |     return torch.nn.Sequential(*layers), sorted(save)
2312 | 
2313 | 
2314 | def yaml_model_load(path):
2315 |     """Load a YOLO model from a YAML file.
2316 | 
2317 |     Args:
2318 |         path (str | Path): Path to the YAML file.
2319 | 
2320 |     Returns:
2321 |         (dict): Model dictionary.
2322 |     """
2323 |     path = Path(path)
2324 |     if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
2325 |         new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
2326 |         LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
2327 |         path = path.with_name(new_stem + path.suffix)
2328 | 
2329 |     unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
2330 |     yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
2331 |     d = YAML.load(yaml_file)  # model dict
2332 |     d["scale"] = d.get("scale") or guess_model_scale(path)
2333 |     d["yaml_file"] = str(path)
2334 |     return d
2335 | 
2336 | 
2337 | def guess_model_scale(model_path):
2338 |     """Extract the size character n, s, m, l, or x of the model's scale from the model path.
2339 | 
2340 |     Args:
2341 |         model_path (str | Path): The path to the YOLO model's YAML file.
2342 | 
2343 |     Returns:
2344 |         (str): The size character of the model's scale (n, s, m, l, or x), or empty string if not found.
2345 |     """
2346 |     try:
2347 |         return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
2348 |     except AttributeError:
2349 |         return ""
2350 | 
2351 | 
2352 | def guess_model_task(model):
2353 |     """Guess the task of a PyTorch model from its architecture or configuration.
2354 | 
2355 |     Args:
2356 |         model (torch.nn.Module | dict | str | Path): PyTorch model, model configuration dict, or model file path.
2357 | 
2358 |     Returns:
2359 |         (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb', 'semantic').
2360 |     """
2361 | 
2362 |     def cfg2task(cfg):
2363 |         """Guess from YAML dictionary."""
2364 |         m = cfg["head"][-1][-2].lower()  # output module name
2365 |         if m in {"classify", "classifier", "cls", "fc"}:
2366 |             return "classify"
2367 |         if "detect" in m:
2368 |             return "detect"
2369 |         if "semanticsegment" in m:
2370 |             return "semantic"
2371 |         if "segment" in m:
2372 |             return "segment"
2373 |         if "pose" in m:
2374 |             return "pose"
2375 |         if "obb" in m:
2376 |             return "obb"
2377 | 
2378 |     # Guess from model cfg
2379 |     if isinstance(model, dict):
2380 |         with contextlib.suppress(Exception):
2381 |             return cfg2task(model)
2382 |     # Guess from PyTorch model
2383 |     if isinstance(model, torch.nn.Module):  # PyTorch model
2384 |         for x in "model.args", "model.model.args", "model.model.model.args":
2385 |             with contextlib.suppress(Exception):
2386 |                 return eval(x)["task"]  # nosec B307: safe eval of known attribute paths
2387 |         for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
2388 |             with contextlib.suppress(Exception):
2389 |                 return cfg2task(eval(x))  # nosec B307: safe eval of known attribute paths
2390 |         for m in model.modules():
2391 |             if isinstance(m, SemanticSegment):
2392 |                 return "semantic"
2393 |             elif isinstance(m, (Segment, YOLOESegment)):
2394 |                 return "segment"
2395 |             elif isinstance(m, Classify):
2396 |                 return "classify"
2397 |             elif isinstance(m, Pose):
2398 |                 return "pose"
2399 |             elif isinstance(m, OBB):
2400 |                 return "obb"
2401 |             elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
2402 |                 return "detect"
2403 | 
2404 |     # Guess from model filename
2405 |     if isinstance(model, (str, Path)):
2406 |         model = Path(model)
2407 |         if "-sem" in model.stem or "semantic" in model.parts:
2408 |             return "semantic"
2409 |         elif "-seg" in model.stem or "segment" in model.parts:
2410 |             return "segment"
2411 |         elif "-cls" in model.stem or "classify" in model.parts:
2412 |             return "classify"
2413 |         elif "-pose" in model.stem or "pose" in model.parts:
2414 |             return "pose"
2415 |         elif "-obb" in model.stem or "obb" in model.parts:
2416 |             return "obb"
2417 |         elif "detect" in model.parts:
2418 |             return "detect"
2419 | 
2420 |     # Unable to determine task from model
2421 |     LOGGER.warning(
2422 |         "Unable to automatically guess model task, assuming 'task=detect'. "
2423 |         "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', 'pose', 'obb' or 'semantic'."
2424 |     )
2425 |     return "detect"  # assume detect
2426 | 
```

## Skipped Files

None.
