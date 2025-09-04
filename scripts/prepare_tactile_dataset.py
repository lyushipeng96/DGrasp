# 调用示例
# python tools/prepare_tactile_dataset.py \
#   --root /data/LyuShipeng/Digit_sensor/Digit_dataset \
#   --out  ./data \
#   --train_ratio 0.9 --val_ratio 0.1 \
#   --normalize
#!/usr/bin/env python3
import argparse, os, json, random, re
from pathlib import Path

def read_text(p: Path):
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return p.read_text(errors="ignore").strip()

def normalize_mass(s: str):
    """把质量统一到 'xxx g' 文本，保留原样也可以；按需开 --normalize 开关。"""
    s = s.strip()
    m = re.match(r'^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)', s)
    if not m:  # 解析失败就原样返回
        return s
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("", "g", "gram", "grams"):
        return f"{val:g} g"
    if unit in ("kg", "kilogram", "kilograms"):
        return f"{val*1000:g} g"
    if unit == "mg":
        return f"{val/1000:g} g"
    return s

def normalize_texture(s: str):
    return s.strip()

def index_images(dirpath: Path, exts):
    d = {}
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            d[p.stem] = p
    return d

def main(args):
    root = Path(args.root).expanduser().resolve()
    after_dir   = root / "After"
    begin_dir   = root / "Begin"
    mass_dir    = root / "Mass"
    texture_dir = root / "Texture"

    assert after_dir.is_dir() and begin_dir.is_dir() and mass_dir.is_dir() and texture_dir.is_dir(), \
        f"目录不存在：{after_dir}, {begin_dir}, {mass_dir}, {texture_dir}"

    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    after   = index_images(after_dir, img_exts)
    begin   = index_images(begin_dir, img_exts)
    mass    = {p.stem: p for p in mass_dir.glob("*.txt")}
    texture = {p.stem: p for p in texture_dir.glob("*.txt")}

    keys = set(after) & set(begin) & set(mass) & set(texture)
    missing_report = {
        "only_after":   len(set(after)   - keys),
        "only_begin":   len(set(begin)   - keys),
        "only_mass":    len(set(mass)    - keys),
        "only_texture": len(set(texture) - keys),
    }
    print(f"[INFO] 匹配到样本数: {len(keys)}\n[INFO] 缺失统计: {missing_report}")

    items = []
    for k in sorted(keys):
        mass_txt = read_text(mass[k])
        tex_txt  = read_text(texture[k])
        if args.normalize:
            mass_txt = normalize_mass(mass_txt)
            tex_txt  = normalize_texture(tex_txt)

        rec = {
            # 相对/绝对路径任选；建议相对，训练时传 root_prefix 还原
            "begin_path":  (begin[k].as_posix()  if args.absolute else begin[k].relative_to(root).as_posix()),
            "after_path":  (after[k].as_posix()  if args.absolute else after[k].relative_to(root).as_posix()),
            "mass":        mass_txt,
            "texture":     tex_txt,
            "stem":        k
        }
        items.append(rec)

    # 随机打乱 & 划分
    random.seed(args.seed)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    train, val, test = items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    def dump(name, subset):
        with open(outdir / name, "w", encoding="utf-8") as f:
            for r in subset:
                r2 = dict(r)
                r2["split"] = name.split(".")[0]
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")
        print(f"[INFO] 写出 {name}: {len(subset)}")

    dump("train.jsonl", train)
    dump("val.jsonl",   val)
    if test:
        dump("test.jsonl",  test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="数据根：/data/LyuShipeng/Digit_sensor/Digit_dataset")
    parser.add_argument("--out", type=str, default="./data",
                        help="输出 JSONL 目录（例如 ./data）")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--absolute", action="store_true",
                        help="把 JSONL 里的路径写成绝对路径（默认相对）")
    parser.add_argument("--normalize", action="store_true",
                        help="规范化 mass/texture 字符串（质量统一成 'xxx g' 文本）")
    args = parser.parse_args()
    assert 0 < args.train_ratio <= 1 and 0 <= args.val_ratio <= 1 and args.train_ratio + args.val_ratio <= 1
    main(args)
