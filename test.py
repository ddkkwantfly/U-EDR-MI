import numpy as np
import json
from pathlib import Path

p = Path(r"processed_final\win2.0s\A01E_train.npz")
pack = np.load(p, allow_pickle=True)
meta = json.loads(pack["meta"].item())
print("cue_unique:", meta["cue_eid_unique"])
print("sfreq:", meta["sfreq"])
