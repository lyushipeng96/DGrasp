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

# 你的数据集：要求 __getitem__ 返回 {"after_pixel": tensor in [-1,1]}
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

# ========== LPIPS（可选） ==========
try:
    from piq import LPIPS
    _has_piq = True
except Exception:
    _has_piq = False
    LPIPS = None

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
    vae.enable_slicing()
    return vae

def encode_stats(vae, pixels):
    lat = vae.encode(pixels).latent_dist
    return lat.mean, lat.logvar

def decode_with(vae, mu, logvar, mode: str, tau: float = 0.0):
    """mode: 'mean' / 'posterior' / 'posterior_tau'"""
    if mode == "mean":
        z = mu
    elif mode == "posterior":
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    else:  # posterior_tau
        z = mu + (tau * torch.exp(0.5 * logvar)) * torch.randn_like(mu)
    return vae.decode(z).sample

# ========== 两阶段调度（仅在 Stage B 引入增强）==========
def cosine_ramp(t):  # t in [0,1]
    return 0.5 - 0.5*math.cos(math.pi * max(0.0, min(1.0, t)))

def get_phase_weights(cfg, step):
    total = int(cfg["vae"]["num_steps"])
    split = int(cfg["train_schedule"]["stageA_steps"])
    if step < split:
        # Stage A：v1同构（不引入任何“降FID增强”）
        return dict(
            phase="A",
            rec_l1_raw_w=1.0, rec_mse_raw_w=0.0, eff_kl_w=0.0,
            train_mode="mean",  # mean 训练
            use_late_polish=False, tau_train=0.0
        )
    # Stage B：v2逻辑 + 仅在此阶段启用增强
    t = (step - split) / max(1, (total - split))
    eff_kl_w = float(cfg["stageB"]["kl_weight"]) * (cosine_ramp(t) if cfg["stageB"].get("kl_cosine", True) else t)
    # 末段与评测口径一致的 τ 采样训练（仅最后 last_k 步）
    last_k = int(cfg["stageB"].get("last_k_tau_train_steps", 20000))
    use_tau_train = (step >= total - last_k)
    train_mode = "posterior_tau" if use_tau_train else ("mean" if (step - split) < int(cfg["stageB"]["recon_use_mean_steps"]) else "posterior")
    return dict(
        phase="B",
        rec_l1_raw_w=float(cfg["stageB"]["rec_l1_raw_w"]),
        rec_mse_raw_w=float(cfg["stageB"]["rec_mse_raw_w"]),
        eff_kl_w=eff_kl_w,
        train_mode=train_mode,
        tau_train=float(cfg["stageB"].get("train_tau", 0.20)),
        # 晚启极小 LPIPS/FFT（仅 Stage B 后 40%）
        use_late_polish=(step >= split + int(0.6 * (total - split)))
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

    tau = float(cfg["eval"].get("fid_temperature", 0.20))
    max_imgs = int(cfg["eval"]["val_max_images"])

    mae_list, rmse_list, psnr_list, ssim_list = [], [], [], []
    saved = 0

    pbar = tqdm(dl_val, total=math.ceil(min(len(dl_val)*bs_val, max_imgs)/bs_val), desc=f"[VAL step {step}]")
    for batch in pbar:
        pixels = batch["after_pixel"].to(device, dtype=dtype)  # ∈ [-1,1]

        mu, logvar = encode_stats(vae, pixels)
        # Mean 重建用于指标
        recon_mean = decode_with(vae, mu, logvar, mode="mean")

        # 指标在 [0,1] 计算
        x = to_float32_01(to01(pixels))
        y = to_float32_01(to01(recon_mean))
        mae_list.append(mae(x, y).item())
        rmse_list.append(rmse(x, y).item())
        psnr_list.append(psnr(x, y).item())
        ssim_list.append(ssim_simple(x, y).item())

        # FID: posterior τ 采样（仅 Stage B）
        if compute_fid:
            recon_fid = decode_with(vae, mu, logvar, mode="posterior_tau", tau=tau)
            y_fid = to_float32_01(to01(recon_fid))
            std = y_fid.float().std().item()
            if std < 0.02 or std > 0.30:  # 简单异常回退
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

    # 简易可视化网格
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

# ========== FID 温度网格（仅手动在 B 阶段末触发时使用）==========
@torch.no_grad()
def sweep_fid(cfg, vae, step, device, dtype, taus=(0.10,0.15,0.20,0.25,0.30)):
    results = {}
    base = os.path.join(cfg["work_dir"], "vae_eval", f"step{step:06d}_sweep")
    os.makedirs(base, exist_ok=True)
    for tau in taus:
        sub_cfg = yaml.safe_load(yaml.dump(cfg))  # 深拷贝
        sub_cfg["eval"]["fid_temperature"] = float(tau)
        report, out_root = run_validation(sub_cfg, vae, step, device, dtype, wb=None)
        results[float(tau)] = report.get("FID", None)
    with open(os.path.join(base, "fid_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

# ========== EMA ==========
def ema_update(ema_model, model, decay=0.9999):
    with torch.no_grad():
        msd = (model.module if hasattr(model,"module") else model).state_dict()
        esd = (ema_model.module if hasattr(ema_model,"module") else ema_model).state_dict()
        for k in esd.keys():
            esd[k].mul_(decay).add_(msd[k].to(esd[k].dtype), alpha=(1.0 - decay))

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

    # EMA
    ema_decay = float(cfg["ema"]["decay"])
    use_ema_eval = bool(cfg["ema"]["use_for_eval"])
    ema_vae = build_vae(cfg, dtype).to(device)
    ema_vae.load_state_dict(vae.state_dict(), strict=True)

    # LPIPS（可选）
    lpips_loss = None
    if _has_piq:
        try:
            lpips_loss = LPIPS(network='vgg').to(device)
        except Exception as e:
            print("[WARN] LPIPS init failed:", repr(e))
            lpips_loss = None

    vae, opt, dl, ema_vae = accelerator.prepare(vae, opt, dl, ema_vae)
    vae.train(); ema_vae.eval()

    # 性能选项
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
    val_every   = int(cfg.get("eval", {}).get("val_interval_steps", 1000))

    pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    step = 0
    split = int(cfg["train_schedule"]["stageA_steps"])
    stage_b_len = total_steps - split

    while step < total_steps:
        for batch in dl:
            pixels = batch["after_pixel"].to(device, dtype=dtype)  # 期望 ∈ [-1,1]

            # 范围体检（每 200 步一次）
            if accelerator.is_main_process and (step % 200 == 0):
                px = pixels.detach().to(torch.float32)
                print(f"[stats] pixels range: min={px.min().item():.3f} max={px.max().item():.3f} mean={px.mean().item():.3f} std={px.std().item():.3f}")

            w = get_phase_weights(cfg, step)

            # 编码
            mu, logvar = encode_stats(vae, pixels)

            # 训练解码模式（A: mean；B: mean→posterior→posterior_tau）
            if w["train_mode"] == "mean":
                recon = decode_with(vae, mu, logvar, mode="mean")
            elif w["train_mode"] == "posterior":
                recon = decode_with(vae, mu, logvar, mode="posterior")
            else:  # posterior_tau
                recon = decode_with(vae, mu, logvar, mode="posterior_tau", tau=w["tau_train"])

            # 主损失全部在 raw 域（[-1,1]）
            rec_l1_raw = F.l1_loss(recon.float(), pixels.float())
            rec_mse_raw = F.mse_loss(recon.float(), pixels.float())
            if w["phase"] == "A":
                rec_pix = rec_l1_raw  # 纯 L1；KL=0
            else:
                rec_pix = w["rec_l1_raw_w"] * rec_l1_raw + w["rec_mse_raw_w"] * rec_mse_raw

            # KL
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_pix + w["eff_kl_w"] * kl

            # 晚启“抛光”（只在 B 后段，并且权重大都很小）
            if w["phase"] == "B" and w["use_late_polish"]:
                x01 = to_float32_01(to01(pixels)); y01 = to_float32_01(to01(recon))
                if float(cfg["stageB"].get("fft_w", 0.0)) > 0:
                    loss = loss + float(cfg["stageB"]["fft_w"]) * fft_mag_l1(x01, y01)
                if float(cfg["stageB"].get("lpips_w", 0.0)) > 0 and lpips_loss is not None:
                    loss = loss + float(cfg["stageB"]["lpips_w"]) * lpips_loss(y01, x01).mean()

            accelerator.backward(loss)
            max_gn = float(cfg.get("max_grad_norm", 0))
            if max_gn and max_gn > 0: accelerator.clip_grad_norm_(vae.parameters(), max_gn)
            opt.step(); opt.zero_grad(set_to_none=True)

            # EMA
            ema_update(ema_vae, vae, decay=ema_decay)

            # 训练日志
            if accelerator.is_main_process:
                desc = (f"[{w['phase']}] {step} | rec {rec_pix.item():.4f} "
                        f"(rawL1 {rec_l1_raw.item():.4f} rawMSE {rec_mse_raw.item():.4f}) "
                        f"| kl {kl.item():.4f} | eff_kl {w['eff_kl_w']:.6f} | mode={w['train_mode']}")
                pbar.set_description(desc)
                if wb is not None:
                    import wandb
                    wandb.log({
                        "phase": 0 if w["phase"]=="A" else 1,
                        "train/rec_pix": rec_pix.item(),
                        "train/rec_l1_raw": rec_l1_raw.item(),
                        "train/rec_mse_raw": rec_mse_raw.item(),
                        "train/kl": kl.item(),
                        "train/eff_kl_w": w["eff_kl_w"],
                        "train/mode": 0 if w["train_mode"]=="mean" else (1 if w["train_mode"]=="posterior" else 2),
                        "global_step": step
                    }, step=step)

            # 保存权重
            if accelerator.is_main_process and (step > 0 and step % save_every == 0):
                sd = os.path.join(out_dir, f"step{step}")
                (vae.module if hasattr(vae, "module") else vae).save_pretrained(sd)
                (ema_vae.module if hasattr(ema_vae, "module") else ema_vae).save_pretrained(sd + "_ema")

            # 每 1k 步验证（使用 EMA 权重评估更稳）
            if accelerator.is_main_process and (step > 0 and step % val_every == 0):
                eval_target = (ema_vae.module if hasattr(ema_vae,"module") else ema_vae) if use_ema_eval else (vae.module if hasattr(vae,"module") else vae)
                report, eval_dir = run_validation(cfg, eval_target, step, device, dtype, wb=wb)
                print(f"\n[VAL step {step}] {json.dumps(report, indent=2)}\n")
                if wb is not None:
                    import wandb
                    logd = {"val/MAE": report["MAE"], "val/RMSE": report["RMSE"],
                            "val/PSNR": report["PSNR"], "val/SSIM": report["SSIM"],
                            "global_step": step}
                    if "FID" in report: logd["val/FID"] = report["FID"]
                    wandb.log(logd, step=step)

            # 末期可选 τ 网格（手动开关）
            if accelerator.is_main_process and bool(cfg.get("eval", {}).get("do_fid_sweep", False)):
                if step >= total_steps - 10000 and step % 5000 == 0:
                    eval_target = (ema_vae.module if hasattr(ema_vae,"module") else ema_vae) if use_ema_eval else (vae.module if hasattr(vae,"module") else vae)
                    results = sweep_fid(cfg, eval_target, step, device, dtype, taus=tuple(cfg["eval"].get("fid_sweep_taus",[0.10,0.15,0.20,0.25,0.30])))
                    print("[FID Sweep]", results)

            step += 1; pbar.update(1)
            if step >= total_steps: break

    if accelerator.is_main_process:
        fd = os.path.join(out_dir, "final")
        (vae.module if hasattr(vae, "module") else vae).save_pretrained(fd)
        (ema_vae.module if hasattr(ema_vae, "module") else ema_vae).save_pretrained(fd + "_ema")
        print("VAE saved:", fd)
        if wb is not None:
            import wandb; wandb.finish()

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config_two_stage.yaml", "r"))
    main(cfg)
