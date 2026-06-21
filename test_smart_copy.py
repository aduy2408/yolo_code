import re
import torch

da = {
    'model.15.cv1.weight': torch.zeros(256, 128),
    'model.16.conv.weight': torch.zeros(256, 256),
    'model.18.cv1.weight': torch.zeros(512, 256),
    'model.19.conv.weight': torch.zeros(512, 512),
}

db = {
    'model.15.cv1.weight': torch.zeros(256, 128), # exact
    'model.19.conv.weight': torch.zeros(128, 128), # different shape
    'model.22.conv.weight': torch.zeros(256, 256), # should map to 16
    'model.24.cv1.weight': torch.zeros(512, 256),  # should map to 18
    'model.25.conv.weight': torch.zeros(512, 512), # should map to 19
}

def intersect_dicts(da, db, exclude=()):
    res = {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
    da_unmatched = [k for k in da if k not in res and all(x not in k for x in exclude)]
    db_unmatched = [k for k in db if k not in res and all(x not in k for x in exclude)]
    
    def sig(k, d):
        m = re.match(r'^model\.(\d+)\.(.*)', k)
        suffix = m.group(2) if m else k
        return (suffix, tuple(d[k].shape))
        
    db_sigs = [sig(k, db) for k in db_unmatched]
    
    for ka in da_unmatched:
        s_a = sig(ka, da)
        if s_a in db_sigs:
            idx = db_sigs.index(s_a)
            kb = db_unmatched[idx]
            res[kb] = da[ka]
            db_sigs.pop(idx)
            db_unmatched.pop(idx)
            
    return res

res = intersect_dicts(da, db)
for k in res:
    print(k, "mapped")
