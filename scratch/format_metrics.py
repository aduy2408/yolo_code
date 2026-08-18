import json

with open("scratch/all_extracted_metrics.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("| Repo / Run | Split | Precision | Recall | mAP50 | mAP75 | mAP50-95 | nms_iou |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for res in results:
    repo = res["repo"]
    run = res["run"]
    data = res["data"]
    
    # short repo name
    short_repo = repo.replace("duyle2408/", "")
    
    nms_iou = data.get("nms_iou", "0.5") # default is 0.5 as per rules
    
    for split in ["val", "test"]:
        p = data.get(f"{split}/metrics/precision(B)", "N/A")
        r = data.get(f"{split}/metrics/recall(B)", "N/A")
        map50 = data.get(f"{split}/metrics/mAP50(B)", "N/A")
        map75 = data.get(f"{split}/metrics/mAP75(B)", "N/A")
        map50_95 = data.get(f"{split}/metrics/mAP50-95(B)", "N/A")
        
        # format values
        p_str = f"{p:.4f}" if isinstance(p, (float, int)) else str(p)
        r_str = f"{r:.4f}" if isinstance(r, (float, int)) else str(r)
        map50_str = f"{map50:.4f}" if isinstance(map50, (float, int)) else str(map50)
        map75_str = f"{map75:.4f}" if isinstance(map75, (float, int)) else str(map75)
        map50_95_str = f"{map50_95:.4f}" if isinstance(map50_95, (float, int)) else str(map50_95)
        
        print(f"| `{short_repo}` ({run}) | **{split}** | {p_str} | {r_str} | {map50_str} | {map75_str} | {map50_95_str} | {nms_iou} |")
