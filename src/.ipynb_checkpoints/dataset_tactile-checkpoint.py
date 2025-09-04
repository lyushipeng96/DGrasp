import os, json
from typing import Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _load_image_rgb(path: str, image_size: int):
    im = Image.open(path).convert("RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.BICUBIC)
    x = T.ToTensor()(im)      # [0,1], float32
    x = x * 2.0 - 1.0         # -> [-1,1]
    return x

class TactilePairDataset(Dataset):
    """
    meta jsonl 的每行应包含：
      - begin_path, after_path : 相对 data_root_prefix 的路径或绝对路径
      - 可选 mass_value(float), texture_id(int)
    """
    def __init__(self, meta_file: str, image_size: int, root_prefix: str = ""):
        super().__init__()
        self.meta: List[Dict[str, Any]] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))
        self.N = len(self.meta)
        self.root = root_prefix
        self.image_size = int(image_size)

    def __len__(self): return self.N

    def _abs(self, p: str):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, idx):
        m = self.meta[idx]
        begin = _load_image_rgb(self._abs(m["begin_path"]), self.image_size)  # [-1,1]
        after = _load_image_rgb(self._abs(m["after_path"]), self.image_size)  # [-1,1]
        mass = float(m.get("mass_value", 0.0))
        tex  = int(m.get("texture_id", 0))
        return {
            "begin_pixel": begin,         # [-1,1], [3,H,W]
            "after_pixel": after,         # [-1,1], [3,H,W]
            "mass_value": torch.tensor(mass, dtype=torch.float32),
            "texture_id": torch.tensor(tex, dtype=torch.long),
        }
