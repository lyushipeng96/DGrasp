import json, random, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class TactilePairDataset(Dataset):
    """
    读取 JSONL（begin_path/after_path/mass/texture），产出：
      - begin_pixel: [0,1]  供 VAE 编码为 begin_latent
      - after_pixel: [-1,1] 供 VAE 编码为 target_latent（训练监督）
      - mass_value:  float(grams)
      - texture_id:  long
    """
    def __init__(self, meta_file, image_size=256, root_prefix=None, texture_to_id=None):
        self.items = [json.loads(l) for l in open(meta_file,'r',encoding='utf-8')]
        self.root_prefix = Path(root_prefix).expanduser().resolve() if root_prefix else None
        self.texture_to_id = texture_to_id or {"<unk>":0}

        self.tf_after = T.Compose([
            T.Resize((image_size,image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # [-1,1]
        ])
        self.tf_begin = T.Compose([
            T.Resize((image_size,image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()  # [0,1]
        ])

    def _resolve(self, p: str) -> Path:
        pth = Path(p)
        return (self.root_prefix / pth) if (not pth.is_absolute() and self.root_prefix) else pth

    def _open_rgb(self, p: Path):
        im = Image.open(p)
        return im.convert("RGB") if im.mode!="RGB" else im

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        begin = self._open_rgb(self._resolve(it["begin_path"]))
        after = self._open_rgb(self._resolve(it["after_path"]))

        begin_pixel = self.tf_begin(begin)  # [0,1]
        after_pixel = self.tf_after(after)  # [-1,1]

        # 数值化
        from src.mt_embedder import parse_mass_to_grams
        mass_val = float(parse_mass_to_grams(it.get("mass","0")))
        tex = str(it.get("texture","")).strip().lower()
        tex_id = self.texture_to_id.get(tex, self.texture_to_id.get("<unk>",0))

        return {
            "begin_pixel": begin_pixel,
            "after_pixel": after_pixel,
            "mass_value": torch.tensor(mass_val, dtype=torch.float32),
            "texture_id": torch.tensor(tex_id, dtype=torch.long),
        }
