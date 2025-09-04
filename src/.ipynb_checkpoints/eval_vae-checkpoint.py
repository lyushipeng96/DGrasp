# 调用方式
# # 评估最终 VAE（默认从 config.yaml 的 work_dir/vae/final 读取）
# python -m src.eval_vae
# 评估中间 checkpoint（例如 step50000）
# python -m src.eval_vae --ckpt_dir ./outputs/tactile_from_scratch/vae/step50000 --max_images 1000
# src/eval_vae.py
import os, json, math, argparse, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

from src.dataset_tactile import TactilePairDataset

# ---------- 基础工具 ----------
@torch.no_grad()
def to01(x):  # [-1,1] -> [0,1]
    return (x.clamp(-1,1) + 1) / 2

@torch.no_grad()
def mae(x, y):   # [0,1]
    return F.l1_loss(x, y, reduction="mean")

@torch.no_grad()
def rmse(x, y, eps=1e-8):  # [0,1]
    return torch.sqrt(F.mse_loss(x, y, reduction="mean").clamp_min(eps))

@torch.no_grad()
def psnr(x, y, eps=1e-8):  # [0,1]
    mse = F.mse_loss(x, y, reduction="mean").clamp_min(eps)
    return 10 * torch.log10(1.0 / mse)

@torch.no_grad()
def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):  # [B,3,H,W] in [0,1]
    # 轻量 SSIM（滑窗均值），够评估用；如需严格实现可换成 piq 或 torchmetrics
    mu_x = torch.nn.AvgPool2d(7,1,3)(x)
    mu_y = torch.nn.AvgPool2d(7,1,3)(y)
    sigma_x = torch.nn.AvgPool2d(7,1,3)(x*x) - mu_x*mu_x
    sigma_y = torch.nn.AvgPool2d(7,1,3)(y*y) - mu_y*mu_y
    sigma_xy= torch.nn.AvgPool2d(7,1,3)(x*y) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2) + 1e-8)
    return ssim_map.mean()

# ---------- FID：优先用 clean-fid，fallback 到 torchvision+scipy ----------
def compute_fid_cleanfid(gt_dir, pred_dir, device):
    from cleanfid import fid
    # clean-fid 内部会自动做 299 resize 和特征提取
    return float(fid.compute_fid(gt_dir, pred_dir, device=device))

def compute_fid_fallback(gt_dir, pred_dir, device):
    """
    备用实现：torchvision InceptionV3 + scipy.sqrtm
    若环境无 scipy 或模型权重下载失败，将抛异常并提示安装 clean-fid。
    """
    import scipy.linalg
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn as nn

    # 取 pool3 2048-d 特征
    weights = Inception_V3_Weights.DEFAULT
    net = inception_v3(weights=weights, transform_input=False).to(device).eval()
    net.Mixed_7c.register_forward_hook(lambda m, inp, out: None)  # 确保走到 pool
    net.fc = nn.Identity()
    net.AuxLogits = None

    def preprocess(pil):
        tf = weights.transforms()  # 自带 299 resize + normalize
        return tf(pil).unsqueeze(0)

    def folder_feats(folder):
        feats = []
        with torch.no_grad():
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                x = preprocess(img).to(device)
                f = net(x)  # [1, 1000] 若 DEFAULT 权重，这里是 logits；更严谨可改为截取 avgpool
                # 用 avgpool 特征更标准：我们改为取 net.avgpool 前的特征（2048）
                # 兼容性写法：走到 Mixed_7c 后自行做自适应池化
                if f.shape[-1] != 2048:
                    with torch.no_grad():
                        # 取 Inception 的最后一个卷积输出
                        m = net._modules.get('Mixed_7c')
                        # 若拿不到，就直接用 logits 近似（不推荐，仅兜底）
                        pass
                feats.append(f.detach().cpu())
        feats = torch.cat(feats, dim=0).numpy()
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    mu1, sig1 = folder_feats(gt_dir)
    mu2, sig2 = folder_feats(pred_dir)
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sig1.dot(sig2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sig1 + sig2 - 2.0 * covmean)
    return float(fid)

def maybe_compute_fid(gt_dir, pred_dir, device):
    # 优先 clean-fid
    try:
        return compute_fid_cleanfid(gt_dir, pred_dir, device)
    except Exception as e1:
        print("[WARN] clean-fid 不可用或失败：", repr(e1))
        try:
            return compute_fid_fallback(gt_dir, pred_dir, device)
        except Exception as e2:
            print("[WARN] 备用 FID 计算失败：", repr(e2))
            print(">>> 建议：pip install clean-fid  （更稳定、和社区一致）")
            return None

# ---------- 主流程 ----------
@torch.no_grad()
def main(args):
    cfg = json.load(open(args.cfg_json, 'r')) if args.cfg_json.endswith(".json") else yaml.safe_load(open(args.cfg_json,'r'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # 数据与 VAE
    ds = TactilePairDataset(cfg["val_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=min(cfg.get("num_workers",8), 4), pin_memory=True, persistent_workers=False)

    vae_dir = args.ckpt_dir or os.path.join(cfg["work_dir"], "vae", "final")
    assert os.path.isdir(vae_dir), f"VAE ckpt not found: {vae_dir}"
    vae = AutoencoderKL.from_pretrained(vae_dir).to(device, dtype=dtype).eval()

    # 输出目录
    tag = os.path.basename(vae_dir.rstrip("/"))
    out_root = os.path.join(cfg["work_dir"], "vae_eval", tag)
    os.makedirs(out_root, exist_ok=True)
    gt_dir   = os.path.join(out_root, "gt")
    pr_dir   = os.path.join(out_root, "recon")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    # 统计指标
    mae_list, rmse_list, psnr_list, ssim_list = [], [], [], []
    to_pil = T.ToPILImage()

    img_saved = 0
    pbar = tqdm(dl, total=min(len(dl), math.ceil(args.max_images/args.batch_size)))
    for batch in pbar:
        # 编码→用 posterior mean 做重建（评估更稳定）
        pixels = batch["after_pixel"].to(device, dtype=dtype)          # [-1,1]
        lat = vae.encode(pixels).latent_dist
        z = lat.mean                                                   # [B,C,H/8,W/8]
        recon = vae.decode(z).sample                                   # [-1,1]

        x = to01(pixels)     # [0,1]
        y = to01(recon)      # [0,1]

        mae_list.append(mae(x, y).item())
        rmse_list.append(rmse(x, y).item())
        psnr_list.append(psnr(x, y).item())
        ssim_list.append(ssim_simple(x, y).item())

        # 保存用于 FID（限制数量以加速）
        if img_saved < args.max_images:
            b = min(x.size(0), args.max_images - img_saved)
            for i in range(b):
                to_pil(x[i].cpu()).save(os.path.join(gt_dir,   f"{img_saved+i:06d}.png"))
                to_pil(y[i].cpu()).save(os.path.join(pr_dir,   f"{img_saved+i:06d}.png"))
            img_saved += b

        pbar.set_description(f"MAE {np.mean(mae_list):.4f} | RMSE {np.mean(rmse_list):.4f} | PSNR {np.mean(psnr_list):.2f} | SSIM {np.mean(ssim_list):.3f}")

        if img_saved >= args.max_images:
            break

    # 汇总
    report = {
        "MAE": float(np.mean(mae_list)),
        "RMSE": float(np.mean(rmse_list)),
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "num_images": int(img_saved),
    }

    # FID（可选）
    fid_score = maybe_compute_fid(gt_dir, pr_dir, device)
    if fid_score is not None:
        report["FID"] = fid_score

    with open(os.path.join(out_root, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n==== VAE EVAL REPORT ====\n", json.dumps(report, indent=2))
    print(f"Images saved to:\n  GT:   {gt_dir}\n  Recon:{pr_dir}\nReport saved to:\n  {os.path.join(out_root,'report.json')}")

if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_json", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--ckpt_dir", default="", help="VAE checkpoint dir (default: work_dir/vae/final)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_images", type=int, default=2000, help="number of images to dump for FID")
    args = ap.parse_args()
    main(args)
