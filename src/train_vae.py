# src/train_vae_two_stage.py
# -*- coding: utf-8 -*-
import os, time, json, yaml, math
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from tqdm import tqdm
from diffusers import AutoencoderKL
from PIL import Image

# 你的数据集：要求 __getitem__ 返回 {"after_pixel": tensor in [-1,1], ...}
from src.dataset_tactile import TactilePairDataset

# ========== 工具 ==========
def to01(x):  # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) / 2

def to_float32_01(x):
    x = x.detach().to(torch.float32)
    x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
    return x.clamp(0.0, 1.0)

def save_png01(img01_chw, path):
    arr = (to_float32_01(img01_chw).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)

@torch.no_grad()
def mae(x, y):   return F.l1_loss(x, y, reduction="mean")
@torch.no_grad()
def rmse(x, y, eps=1e-8):  return torch.sqrt(F.mse_loss(x, y, reduction="mean").clamp_min(eps))
@torch.no_grad()
def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction="mean").clamp_min(eps)
    return 10 * torch.log10(1.0 / mse)
@torch.no_grad()
def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    avg = torch.nn.AvgPool2d(7, 1, 3)
    mu_x, mu_y = avg(x), avg(y)
    sigma_x = avg(x * x) - mu_x * mu_x
    sigma_y = avg(y * y) - mu_y * mu_y
    sigma_xy = avg(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    return ssim_map.mean()

def fft_mag_l1(x01, y01):
    X = torch.fft.rfft2(x01, norm='ortho'); Y = torch.fft.rfft2(y01, norm='ortho')
    return F.l1_loss(torch.log1p(torch.abs(X)), torch.log1p(torch.abs(Y)))

def compute_fid_cleanfid(gt_dir, pr_dir, device="cuda"):
    from cleanfid import fid
    return float(fid.compute_fid(gt_dir, pr_dir, mode="clean", device=device))
def maybe_compute_fid(gt_dir, pr_dir, device="cuda"):
    try:    return compute_fid_cleanfid(gt_dir, pr_dir, device)
    except Exception as e:
        print("[WARN] clean-fid 失败：", repr(e)); print("       请 pip install -U clean-fid"); return None

# ========== 模型 ==========
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
    vae.enable_slicing()   # 需要更省显存可再开 enable_tiling()
    return vae

def encode_decode(vae, pixels, use_mean: bool = False):
    lat = vae.encode(pixels).latent_dist
    mu, logvar = lat.mean, lat.logvar
    z = mu if use_mean else (mu + torch.exp(0.5 * logvar) * torch.randn_like(mu))
    try:
        recon = vae.decode(z).sample
    except RuntimeError as e:
        if "INTERNAL ASSERT" in str(e) or "out of memory" in str(e).lower():
            try:
                vae.disable_slicing(); torch.cuda.empty_cache()
                recon = vae.decode(z).sample
            finally:
                vae.enable_slicing()
        else:
            raise
    return recon, mu, logvar

# ========== 两阶段调度 ==========
def interp(a, b, t): return a + (b - a) * float(t)

def get_phase_weights(cfg, step):
    total = int(cfg["vae"]["num_steps"])
    split = int(cfg["train_schedule"]["stageA_steps"])
    # Stage A（v1 同构：纯 L1@[-1,1] + KL=0 + 均值训练）
    if step < split:
        return dict(
            phase="A",
            # A 阶段仅用 raw L1 作为优化目标
            rec_l1_raw_w=1.0,      # 仅在 A 阶段使用
            rec_l1_raw_add=0.0,    # 常留 0
            rec_l1_01_w=0.0,       # A 阶段不用 [0,1] 口径
            rec_mse_raw_w=0.0,     # A 阶段不用 MSE
            eff_kl_w=0.0,
            use_mean=True,
            lpips_w=0.0,
            fft_w=0.0,
        )
    # Stage B（v2：混合损失@ [-1,1]，KL warmup）
    t = (step - split) / max(1, (total - split))
    eff_kl_w = interp(0.0, float(cfg["stageB"].get("kl_weight", 2e-4)), t)
    return dict(
        phase="B",
        rec_l1_raw_w=float(cfg["stageB"].get("rec_l1_raw_w", 0.45)),   # = 0.9 @ [0,1] 换算到 [-1,1] 的 0.5 倍
        rec_mse_raw_w=float(cfg["stageB"].get("rec_mse_raw_w", 0.025)),# = 0.1 @ [0,1] 换算到 [-1,1] 的 0.25 倍
        rec_l1_raw_add=0.0,   # 如需额外 L1 可加在这
        rec_l1_01_w=0.0,      # 若想仍观察/搭配 [0,1] 口径，可调非零，但默认 0
        eff_kl_w=eff_kl_w,
        use_mean=(step - split) < int(cfg["stageB"].get("recon_use_mean_steps", 30000)),
        lpips_w=float(cfg["stageB"].get("lpips_w", 0.0)),
        fft_w=float(cfg["stageB"].get("fft_w", 0.0)),
    )

# ========== 验证（Mean 指标 + 采样 FID） ==========
@torch.no_grad()
def run_validation(cfg, vae, step, device, dtype, wb=None):
    vae.eval(); torch.cuda.empty_cache()
    ds_val = TactilePairDataset(cfg["val_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"])
    bs_val = int(cfg["eval"]["val_batch_size"])
    dl_val = DataLoader(ds_val, batch_size=bs_val, shuffle=False,
                        num_workers=min(int(cfg.get("num_workers", 8)), 4),
                        pin_memory=True, persistent_workers=False)

    tag = f"step{step:06d}"
    out_root = os.path.join(cfg["work_dir"], "vae_eval", tag)
    gt_dir = os.path.join(out_root, "gt")
    pr_dir = os.path.join(out_root, "recon")
    pr_fid_dir = os.path.join(out_root, "recon_fid")
    os.makedirs(gt_dir, exist_ok=True); os.makedirs(pr_dir, exist_ok=True); os.makedirs(pr_fid_dir, exist_ok=True)

    compute_fid = bool(cfg["eval"].get("compute_fid", True))
    fid_start = int(cfg["eval"].get("fid_start_step", int(cfg["train_schedule"]["stageA_steps"])))
    if step < fid_start: compute_fid = False

    use_post_samp = bool(cfg["eval"].get("fid_use_posterior_sampling", True))
    tau = float(cfg["eval"].get("fid_temperature", 0.20))
    max_imgs = int(cfg["eval"]["val_max_images"])

    mae_list, rmse_list, psnr_list, ssim_list = [], [], [], []
    saved = 0

    pbar = tqdm(dl_val, total=math.ceil(min(len(dl_val)*bs_val, max_imgs)/bs_val), desc=f"[VAL step {step}]")
    for batch in pbar:
        pixels = batch["after_pixel"].to(device, dtype=dtype)  # 期望 ∈ [-1,1]

        # Mean 重建用于指标
        lat = vae.encode(pixels).latent_dist
        z_mean = lat.mean
        recon_mean = vae.decode(z_mean).sample

        # 指标在 [0,1] 计算（更直观）
        x = to_float32_01(to01(pixels))
        y = to_float32_01(to01(recon_mean))

        mae_list.append(mae(x, y).item())
        rmse_list.append(rmse(x, y).item())
        psnr_list.append(psnr(x, y).item())
        ssim_list.append(ssim_simple(x, y).item())

        # FID: posterior 采样 + 简单异常回退
        if compute_fid and use_post_samp:
            eps = torch.randn_like(lat.mean)
            z_samp = lat.mean + tau * torch.exp(0.5 * lat.logvar) * eps
            recon_fid = vae.decode(z_samp).sample
            y_fid = to_float32_01(to01(recon_fid))
            std = y_fid.float().std().item()
            if std < 0.02 or std > 0.30:  # 防止异常值破坏 FID
                y_fid = y
        else:
            y_fid = y

        # 保存图片
        if saved < max_imgs:
            b = min(x.size(0), max_imgs - saved)
            for i in range(b):
                save_png01(x[i], os.path.join(gt_dir, f"{saved+i:06d}.png"))
                save_png01(y[i], os.path.join(pr_dir, f"{saved+i:06d}.png"))
                save_png01(y_fid[i], os.path.join(pr_fid_dir, f"{saved+i:06d}.png"))
            saved += b

        pbar.set_postfix(MAE=np.mean(mae_list), RMSE=np.mean(rmse_list),
                         PSNR=np.mean(psnr_list), SSIM=np.mean(ssim_list))
        if saved >= max_imgs: break

    report = {
        "MAE": float(np.mean(mae_list)) if mae_list else float("nan"),
        "RMSE": float(np.mean(rmse_list)) if rmse_list else float("nan"),
        "PSNR": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "SSIM": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "num_images": int(saved),
    }

    fid_score = None
    if compute_fid and saved >= 50:
        fid_score = maybe_compute_fid(gt_dir, pr_fid_dir, device=device)
        if fid_score is not None: report["FID"] = fid_score

    with open(os.path.join(out_root, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 网格（前 16 对）
    grid_imgs = []
    k = min(16, saved)
    for i in range(k):
        grid_imgs.append(Image.open(os.path.join(gt_dir, f"{i:06d}.png")).convert("RGB"))
        grid_imgs.append(Image.open(os.path.join(pr_dir, f"{i:06d}.png")).convert("RGB"))
    if grid_imgs:
        w, h = grid_imgs[0].size; cols = 4; rows = math.ceil(len(grid_imgs) / cols)
        board = Image.new("RGB", (cols*w, rows*h))
        for j, im in enumerate(grid_imgs):
            r, c = divmod(j, cols); board.paste(im, (c*w, r*h))
        board.save(os.path.join(out_root, "grid_gt_recon.png"))

    vae.train(); return report, out_root

# ========== 训练 ==========
def main(cfg):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "no"))
    device = accelerator.device
    dtype = (torch.bfloat16 if accelerator.mixed_precision == "bf16"
             else (torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32))

    # 数据
    ds_full = TactilePairDataset(cfg["train_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"])
    overfit_n = int(cfg.get("debug", {}).get("overfit_n", 0))
    ds = Subset(ds_full, list(range(min(overfit_n, len(ds_full))))) if overfit_n > 0 else ds_full
    if overfit_n > 0: print(f"[debug] overfit on {len(ds)} samples")

    dl = DataLoader(ds, batch_size=int(cfg["vae"]["batch_size"]),
                    shuffle=True, num_workers=int(cfg.get("num_workers", 8)),
                    pin_memory=True, persistent_workers=False)

    # 模型与优化器
    vae = build_vae(cfg, dtype)
    opt = torch.optim.AdamW(vae.parameters(), lr=float(cfg["vae"]["lr"]), betas=(0.9, 0.999), weight_decay=0.0)
    vae, opt, dl = accelerator.prepare(vae, opt, dl)
    vae.train()

    # 性能设置
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # wandb（可选）
    use_wandb = bool(cfg.get("logging", {}).get("use_wandb", False))
    wb = None
    if accelerator.is_main_process and use_wandb:
        try:
            import wandb
            wb = wandb.init(
                project=cfg["logging"].get("project", "tactile_vae"),
                entity=cfg["logging"].get("entity") or None,
                name=cfg["logging"].get("run_name") or f"vae_{int(time.time())}",
                config=cfg,
                settings=wandb.Settings(init_timeout=int(cfg["logging"].get("init_timeout", 180)),
                                        start_method="thread"),
            )
            print("[wandb] online logging enabled.")
        except Exception as e:
            print("[wandb] init failed -> local only:", repr(e))
            os.environ["WANDB_DISABLED"] = "true"
            wb = None

    out_dir = os.path.join(cfg["work_dir"], "vae"); os.makedirs(out_dir, exist_ok=True)
    total_steps = int(cfg["vae"]["num_steps"])
    save_every  = int(cfg["vae"]["save_every"])
    # **每 1000 步验证一次**
    val_every   = int(cfg.get("eval", {}).get("val_interval_steps", 1000))

    pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    step = 0
    while step < total_steps:
        for batch in dl:
            pixels = batch["after_pixel"].to(device, dtype=dtype)  # 期望 ∈ [-1,1]

            # —— 像素范围体检（每 200 步一次）——
            if accelerator.is_main_process and (step % 200 == 0):
                px = pixels.detach().to(torch.float32)
#                 print(f"[stats] pixels range: min={px.min().item():.3f} max={px.max().item():.3f} mean={px.mean().item():.3f} std={px.std().item():.3f}")

            # 权重/阶段
            w = get_phase_weights(cfg, step)
            recon, mu, logvar = encode_decode(vae, pixels, use_mean=w["use_mean"])

            # A) raw 域（[-1,1]）——与 v1 对齐；B 阶段也在 raw 域优化
            rec_l1_raw = F.l1_loss(recon.float(), pixels.float())
            rec_mse_raw = F.mse_loss(recon.float(), pixels.float())

            # B) 01 域（[0,1]）——仅做记录/可选感知项
            x01 = to_float32_01(to01(pixels))
            y01 = to_float32_01(to01(recon))
            rec_l1_01 = F.l1_loss(y01, x01)
            rec_mse_01 = F.mse_loss(y01, x01)

            # 组合优化目标（A：raw L1；B：raw L1+raw MSE）
            if w["phase"] == "A":
                rec_pix = w["rec_l1_raw_w"] * rec_l1_raw + w["rec_l1_raw_add"] * rec_l1_raw
            else:
                rec_pix = w["rec_l1_raw_w"] * rec_l1_raw + w["rec_mse_raw_w"] * rec_mse_raw

            # KL
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_pix + w["eff_kl_w"] * kl

            # 可选感知/频域（这些项在 [0,1] 上计算）
            if w["fft_w"] > 0:
                loss = loss + w["fft_w"] * fft_mag_l1(x01, y01)

            accelerator.backward(loss)

            # 梯度裁剪（默认 0：欠拟合时建议关闭）
            max_gn = float(cfg.get("max_grad_norm", 0))
            if max_gn and max_gn > 0: accelerator.clip_grad_norm_(vae.parameters(), max_gn)

            opt.step(); opt.zero_grad(set_to_none=True)

            # 训练日志
            if accelerator.is_main_process:
                pbar.set_description(
                    f"[{w['phase']}] {step} | rec {rec_pix.item():.4f} "
                    f"(rawL1 {rec_l1_raw.item():.4f} rawMSE {rec_mse_raw.item():.4f} | L1_01 {rec_l1_01.item():.4f}) "
                    f"| kl {kl.item():.4f} | eff_kl {w['eff_kl_w']:.6f} | mean={w['use_mean']}"
                )
                if wb is not None:
                    import wandb
                    wandb.log({
                        "phase": 0 if w["phase"]=="A" else 1,
                        "train/rec_pix": rec_pix.item(),
                        "train/rec_l1_raw": rec_l1_raw.item(),
                        "train/rec_mse_raw": rec_mse_raw.item(),
                        "train/rec_l1_01":  rec_l1_01.item(),
                        "train/rec_mse_01":  rec_mse_01.item(),
                        "train/kl":         kl.item(),
                        "train/eff_kl_w":   w["eff_kl_w"],
                        "train/is_mean":    float(w["use_mean"]),
                        "global_step":      step
                    }, step=step)

            # 保存权重
            if accelerator.is_main_process and (step > 0 and step % save_every == 0):
                sd = os.path.join(out_dir, f"step{step}")
                (vae.module if hasattr(vae, "module") else vae).save_pretrained(sd)

            # —— 每 1k 步验证并保存 —— #
            if accelerator.is_main_process and (step > 0 and step % val_every == 0):
                report, eval_dir = run_validation(cfg, (vae.module if hasattr(vae, "module") else vae),
                                                  step, device, dtype, wb=wb)
                print(f"\n[VAL step {step}] {json.dumps(report, indent=2)}\n")
                if wb is not None:
                    import wandb
                    logd = {"val/MAE": report["MAE"], "val/RMSE": report["RMSE"],
                            "val/PSNR": report["PSNR"], "val/SSIM": report["SSIM"],
                            "global_step": step}
                    if "FID" in report: logd["val/FID"] = report["FID"]
                    wandb.log(logd, step=step)
                    grid_path = os.path.join(eval_dir, "grid_gt_recon.png")
                    if os.path.exists(grid_path):
                        wandb.log({"val/grid_gt_recon": wandb.Image(grid_path)}, step=step)

            step += 1; pbar.update(1)
            if step >= total_steps: break

    if accelerator.is_main_process:
        fd = os.path.join(out_dir, "final")
        (vae.module if hasattr(vae, "module") else vae).save_pretrained(fd)
        print("VAE saved:", fd)
        if wb is not None:
            import wandb; wandb.finish()

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    main(cfg)
