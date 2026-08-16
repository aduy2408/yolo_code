import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
if str(ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS))

from ultralytics import YOLO

def test_variant(config_path, name):
    print(f"=== Testing {name} ===")
    try:
        model = YOLO(config_path)
        # Create dummy input of shape (batch, channels, height, width) -> (2, 3, 512, 512)
        dummy_input = torch.randn(2, 3, 512, 512)
        out = model.model(dummy_input)
        print(f"{name} loaded successfully!")
        
        # Check params
        params = sum(p.numel() for p in model.model.parameters())
        print(f"Parameters: {params:,}")
    except Exception as e:
        print(f"Error in {name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    configs = {
        "C1 (Cross Injection)": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c1_cross_injection.yaml",
        "C2 (Agreement)": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c2_agreement.yaml",
        "C3 (Polarity)": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c3_polarity.yaml",
        "C4 (Low Rank)": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c4_rank4.yaml",
    }
    
    for name, path in configs.items():
        test_variant(path, name)
