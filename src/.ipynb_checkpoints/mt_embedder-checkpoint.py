import re, json, torch, torch.nn as nn
from pathlib import Path

def parse_mass_to_grams(mass_str: str) -> float:
    if not mass_str: return 0.0
    s = str(mass_str).strip().lower()
    m = re.match(r'^\s*([0-9]*\.?[0-9]+)\s*([a-z]*)', s)
    if not m: return 0.0
    v = float(m.group(1)); u = m.group(2)
    if u in ("", "g", "gram", "grams"): return v
    if u in ("kg", "kilogram", "kilograms"): return v * 1000.0
    if u in ("mg",): return v / 1000.0
    return v

def build_or_load_texture_vocab(jsonl_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    vf = out / "texture_vocab.json"
    if vf.exists():
        return json.loads(vf.read_text(encoding="utf-8"))
    tex_set = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                t = str(o.get("texture","")).strip().lower()
                if t: tex_set.add(t)
            except Exception:
                pass
    mapping = {"<unk>": 0}
    for i, t in enumerate(sorted(tex_set), start=1):
        mapping[t] = i
    vf.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping

class MassTextureEmbedder(nn.Module):
    """
    将 (mass_in_grams, texture_id) -> cross-attention 条件 token 序列
    """
    def __init__(self, num_textures: int, embed_dim: int = 384, seq_len: int = 12):
        super().__init__()
        self.seq_len = seq_len
        hid = 256
        self.mass_mlp = nn.Sequential(
            nn.Linear(1, hid//2), nn.SiLU(), nn.Linear(hid//2, hid//2), nn.SiLU()
        )
        self.tex_emb = nn.Embedding(num_textures, hid//2)  # 0 为 <unk>
        self.fuse = nn.Sequential(
            nn.Linear(hid, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.null_token = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, mass_vals, tex_ids, cond_drop_prob=0.0):
        if mass_vals.dim() == 1: mass_vals = mass_vals.unsqueeze(-1)
        mass_feat = self.mass_mlp(mass_vals / 1000.0)
        tex_feat  = self.tex_emb(tex_ids)
        fused = torch.cat([mass_feat, tex_feat], dim=-1)
        emb = self.fuse(fused)  # [B, D]
        cond = emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, D]

        # 无条件 token（用于CFG）
        B = emb.size(0)
        uncond = self.null_token.unsqueeze(0).unsqueeze(1).repeat(B, self.seq_len, 1)

        # 训练期随机丢弃条件
        if self.training and cond_drop_prob > 0:
            mask = torch.rand(B, device=emb.device) < cond_drop_prob
            cond[mask] = uncond[mask]

        return cond, uncond
