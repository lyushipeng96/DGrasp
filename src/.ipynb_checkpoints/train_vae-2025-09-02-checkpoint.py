# 该版本VAE训练结果除FID外所有指标均可接受，但是相对于以前计算结果FID值过高！
# src/train_vae.py
import os, time, json, yaml, math
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
import torchvision.transforms as TV
from PIL import Image

from src.dataset_tactile_v1 import TactilePairDataset

# ----------------- 基础工具 -----------------
def to01(x):  # [-1,1] -> [0,1]
    return (x.clamp(-1,1) + 1) / 2

def to_float32_01(x):
    # 将任意 dtype 的 [-1,1]/[0,1] 张量安全转为 float32 的 [0,1]
    return x.detach().to(torch.float32).clamp(0.0, 1.0)

def save_png01(img01_chw, path):
    """
    img01_chw: [C,H,W]，值域 [0,1]，任意 dtype
    将其安全保存为 8-bit PNG
    """
    arr = (to_float32_01(img01_chw).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    from PIL import Image
    Image.fromarray(arr).save(path)

def mae(x, y):
    return F.l1_loss(x, y, reduction="mean")

def rmse(x, y, eps=1e-8):
    return torch.sqrt(F.mse_loss(x, y, reduction="mean").clamp_min(eps))

def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction="mean").clamp_min(eps)
    return 10 * torch.log10(1.0 / mse)

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    # x,y: [B,3,H,W] in [0,1]；轻量 SSIM，评估足够
    avg = torch.nn.AvgPool2d(7, 1, 3)
    mu_x, mu_y = avg(x), avg(y)
    sigma_x = avg(x*x) - mu_x*mu_x
    sigma_y = avg(y*y) - mu_y*mu_y
    sigma_xy= avg(x*y) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2))/((mu_x**2+mu_y**2+C1)*(sigma_x+sigma_y+C2)+1e-8)
    return ssim_map.mean()

def compute_fid_cleanfid(gt_dir, pr_dir, device="cuda"):
    from cleanfid import fid
    return float(fid.compute_fid(gt_dir, pr_dir, device=device))

def compute_fid_fallback(gt_dir, pr_dir, device="cuda"):
    import scipy.linalg
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn as nn
    weights = Inception_V3_Weights.DEFAULT
    net = inception_v3(weights=weights, transform_input=False).to(device).eval()
    net.fc = nn.Identity(); net.AuxLogits = None

    tf = weights.transforms()
    def folder_feats(folder):
        feats = []
        with torch.no_grad():
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                x = tf(img).unsqueeze(0).to(device)
                f = net(x)           # [1,1000]（权重默认是 logits），作为兜底近似
                feats.append(f.cpu())
        feats = torch.cat(feats, 0).numpy()
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    mu1, s1 = folder_feats(gt_dir)
    mu2, s2 = folder_feats(pr_dir)
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(s1.dot(s2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(s1 + s2 - 2.0 * covmean)
    return float(fid)

def maybe_compute_fid(gt_dir, pr_dir, device="cuda"):
    try:
        return compute_fid_cleanfid(gt_dir, pr_dir, device)
    except Exception as e1:
        print("[WARN] clean-fid 不可用：", repr(e1))
        try:
            return compute_fid_fallback(gt_dir, pr_dir, device)
        except Exception as e2:
            print("[WARN] 备用 FID 失败：", repr(e2), "\n建议 pip install clean-fid")
            return None

# ----------------- 模型与前向 -----------------
def build_vae(cfg, dtype):
    v = cfg["vae"]
    vae = AutoencoderKL(
        sample_size=v["sample_size"],
        in_channels=v["in_channels"],
        out_channels=v["out_channels"],
        block_out_channels=v["block_out_channels"],
        down_block_types=v["down_block_types"],
        up_block_types=v["up_block_types"],
        latent_channels=v["latent_channels"],
    ).to(dtype=dtype)
    return vae

def encode_decode(vae, pixels):
    # 统一 encode→decode，跨 diffusers 版本最稳
    lat_dist = vae.encode(pixels).latent_dist
    mu, logvar = lat_dist.mean, lat_dist.logvar
    # 评估时通常用 mean；训练时我们也可以 sample，差异不大
    recon = vae.decode(lat_dist.sample()).sample
    return recon, mu, logvar

# ----------------- 验证评估（带保存/可选 FID） -----------------
@torch.no_grad()
def run_validation(cfg, vae, step, device, dtype):
    vae.eval()
    # dataloader
    ds_val = TactilePairDataset(cfg["val_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"])
    bs_val = int(cfg["eval"]["val_batch_size"])
    dl_val = DataLoader(ds_val, batch_size=bs_val, shuffle=False,
                        num_workers=min(cfg.get("num_workers",8), 4),
                        pin_memory=True, persistent_workers=False)

    # 输出路径
    tag = f"step{step:06d}"
    out_root = os.path.join(cfg["work_dir"], "vae_eval", tag)
    gt_dir = os.path.join(out_root, "gt"); pr_dir = os.path.join(out_root, "recon")
    os.makedirs(gt_dir, exist_ok=True); os.makedirs(pr_dir, exist_ok=True)

    # 累计指标
    mae_list, rmse_list, psnr_list, ssim_list = [], [], [], []
    to_pil = TV.ToPILImage()

    saved = 0
    max_imgs = int(cfg["eval"]["val_max_images"])
    pbar = tqdm(dl_val, total=math.ceil(min(len(dl_val)*bs_val, max_imgs)/bs_val), desc=f"[VAL step {step}]")
    for batch in pbar:
        pixels = batch["after_pixel"].to(device, dtype=dtype)          # [-1,1]
        # posterior mean 更平稳
        lat = vae.encode(pixels).latent_dist
        z = lat.mean
        recon = vae.decode(z).sample

#         x = to01(pixels)
#         y = to01(recon)
        x = to_float32_01(to01(pixels))   # [B,3,H,W], float32, [0,1]
        y = to_float32_01(to01(recon))    # [B,3,H,W], float32, [0,1]

        mae_list.append(mae(x,y).item())
        rmse_list.append(rmse(x,y).item())
        psnr_list.append(psnr(x,y).item())
        ssim_list.append(ssim_simple(x,y).item())

        if saved < max_imgs:
            b = min(x.size(0), max_imgs - saved)
            for i in range(b):
#                 to_pil(x[i].cpu()).save(os.path.join(gt_dir, f"{saved+i:06d}.png"))
#                 to_pil(y[i].cpu()).save(os.path.join(pr_dir, f"{saved+i:06d}.png"))
                save_png01(x[i], os.path.join(gt_dir, f"{saved+i:06d}.png"))
                save_png01(y[i], os.path.join(pr_dir, f"{saved+i:06d}.png"))
            saved += b

        pbar.set_postfix(MAE=np.mean(mae_list), RMSE=np.mean(rmse_list),
                         PSNR=np.mean(psnr_list), SSIM=np.mean(ssim_list))
        if saved >= max_imgs:
            break

    # 汇总
    report = {
        "MAE": float(np.mean(mae_list)),
        "RMSE": float(np.mean(rmse_list)),
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "num_images": int(saved),
    }

    fid_score = None
    if bool(cfg["eval"].get("compute_fid", True)) and saved >= 50:  # 少量样本的 FID 没意义
        fid_score = maybe_compute_fid(gt_dir, pr_dir, device=device)
        if fid_score is not None:
            report["FID"] = fid_score

    with open(os.path.join(out_root, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 可视化网格（前 16 对）
    grid_imgs = []
    for i in range(min(16, saved)):
        grid_imgs.append(Image.open(os.path.join(gt_dir, f"{i:06d}.png")).convert("RGB"))
        grid_imgs.append(Image.open(os.path.join(pr_dir, f"{i:06d}.png")).convert("RGB"))
    if grid_imgs:
        w, h = grid_imgs[0].size
        cols = 4
        rows = math.ceil(len(grid_imgs) / cols)
        board = Image.new("RGB", (cols*w, rows*h))
        for j, im in enumerate(grid_imgs):
            r, c = divmod(j, cols)
            board.paste(im, (c*w, r*h))
        board.save(os.path.join(out_root, "grid_gt_recon.png"))

    vae.train()
    return report, out_root

# ----------------- 训练主过程（带 wandb） -----------------
def main(cfg):
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision","no"))
    device = accelerator.device
    dtype = torch.bfloat16 if accelerator.mixed_precision=="bf16" else (torch.float16 if accelerator.mixed_precision=="fp16" else torch.float32)

    # dataloader（train）
    ds = TactilePairDataset(cfg["train_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"])
    dl = DataLoader(ds, batch_size=int(cfg["vae"]["batch_size"]), shuffle=True,
                    num_workers=int(cfg.get("num_workers",8)), pin_memory=True, persistent_workers=False)

    vae = build_vae(cfg, dtype)
    opt = torch.optim.AdamW(vae.parameters(), lr=float(cfg["vae"]["lr"]))
    vae, opt, dl = accelerator.prepare(vae, opt, dl)
    vae.train()

    out_dir = os.path.join(cfg["work_dir"], "vae"); os.makedirs(out_dir, exist_ok=True)
    total_steps = int(cfg["vae"]["num_steps"])
    save_every  = int(cfg["vae"]["save_every"])
    val_every   = int(cfg.get("eval",{}).get("val_interval_steps", 2000))
    kl_w        = float(cfg["vae"]["kl_weight"])

    # 性能：A800 建议开 TF32
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE","1")
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # wandb
    use_wandb = bool(cfg.get("logging",{}).get("use_wandb", False))
    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.init(
            project=cfg["logging"].get("project","tactile_vae"),
            entity=cfg["logging"].get("entity") or None,
            name=cfg["logging"].get("run_name") or f"vae_{int(time.time())}",
            config=cfg
        )

    pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    step = 0
    while step < total_steps:
        for batch in dl:
            pixels = batch["after_pixel"].to(device, dtype=dtype)  # [-1,1]
            recon, mu, logvar = encode_decode(vae, pixels)
            rec_loss = F.l1_loss(recon.float(), pixels.float())
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + kl_w * kl

            accelerator.backward(loss)
            opt.step(); opt.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                pbar.set_description(f"vae {step} | rec {rec_loss.item():.4f} | kl {kl.item():.4f}")
                if use_wandb:
                    import wandb
                    wandb.log({"train/rec_l1": rec_loss.item(), "train/kl": kl.item(),
                               "train/loss": (rec_loss.item()+kl_w*kl.item()),
                               "global_step": step}, step=step)

            # save ckpt
            if accelerator.is_main_process and (step>0 and step % save_every == 0):
                sd = os.path.join(out_dir, f"step{step}")
                (vae.module if hasattr(vae,"module") else vae).save_pretrained(sd)

            # validate
            if accelerator.is_main_process and (step>0 and step % val_every == 0):
                report, eval_dir = run_validation(cfg, (vae.module if hasattr(vae,"module") else vae), step, device, dtype)
                # 控制台打印
                print(f"\n[VAL step {step}] {json.dumps(report, indent=2)}\n")
                # wandb 记录
                if use_wandb:
                    import wandb
                    wandb.log({
                        "val/MAE": report["MAE"],
                        "val/RMSE": report["RMSE"],
                        "val/PSNR": report["PSNR"],
                        "val/SSIM": report["SSIM"],
                        **({"val/FID": report["FID"]} if "FID" in report else {}),
                        "global_step": step
                    }, step=step)
                    grid_path = os.path.join(eval_dir, "grid_gt_recon.png")
                    if os.path.exists(grid_path):
                        wandb.log({"val/grid_gt_recon": wandb.Image(grid_path)}, step=step)

            step += 1
            pbar.update(1)
            if step >= total_steps:
                break

    # final save
    if accelerator.is_main_process:
        fd = os.path.join(out_dir, "final")
        (vae.module if hasattr(vae,"module") else vae).save_pretrained(fd)
        print("VAE saved:", fd)
        if use_wandb:
            import wandb; wandb.finish()

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config_v1.yaml","r"))
    main(cfg)
