import os, yaml, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from src.dataset_tactile import TactilePairDataset
from src.mt_embedder import MassTextureEmbedder, build_or_load_texture_vocab

def build_vae(vae_dir, dtype):
    vae = AutoencoderKL.from_pretrained(vae_dir).to(dtype=dtype)
    vae.eval(); vae.requires_grad_(False)
    return vae

def build_unet(cfg, dtype):
    dcfg = cfg["diffusion"]
    unet = UNet2DConditionModel(
        sample_size=dcfg["sample_size"],
        in_channels=dcfg["in_channels"],
        out_channels=dcfg["out_channels"],
        down_block_types=dcfg["down_block_types"],
        up_block_types=dcfg["up_block_types"],
        block_out_channels=dcfg["block_out_channels"],
        cross_attention_dim=dcfg["cross_attention_dim"],
        attention_head_dim=dcfg["attention_head_dim"],
    ).to(dtype=dtype)
    return unet

def main(cfg):
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision","no"))
    device = accelerator.device
    dtype = torch.bfloat16 if accelerator.mixed_precision=="bf16" else (torch.float16 if accelerator.mixed_precision=="fp16" else torch.float32)

    work = cfg["work_dir"]
    vae_dir = os.path.join(work, "vae", "final")  # 用上一步训练好的VAE
    assert os.path.isdir(vae_dir), f"VAE not found: {vae_dir}"

    # 词表（texture）
    vocab = build_or_load_texture_vocab(cfg["mt_embed"]["vocab_from"], os.path.join(work, "emb"))
    mt = MassTextureEmbedder(num_textures=len(vocab), embed_dim=cfg["mt_embed"]["embed_dim"], seq_len=cfg["mt_embed"]["seq_len"])

    # 模型
    vae  = build_vae(vae_dir, dtype)
    unet = build_unet(cfg, dtype)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

    # 数据
    ds = TactilePairDataset(cfg["train_meta"], image_size=cfg["image_size"], root_prefix=cfg["data_root_prefix"], texture_to_id=vocab)
    dl = DataLoader(ds, batch_size=cfg["diffusion"]["batch_size"], shuffle=True, num_workers=cfg.get("num_workers",8), pin_memory=True, persistent_workers=True)

    # 优化器（仅训练 UNet + mt_embed）
    params = list(unet.parameters()) + list(mt.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["diffusion"]["lr"])

    (unet, mt, vae, opt, dl) = accelerator.prepare(unet, mt, vae, opt, dl)
    vae.eval(); unet.train(); mt.train()

    # 超参
    vae_sf = float(cfg["diffusion"]["vae_scaling_factor"])  # 我们自己定义，一致使用即可
    cond_drop = float(cfg["mt_embed"]["cond_drop_prob"])
    max_grad_norm = float(cfg.get("max_grad_norm",1.0))
    save_every = int(cfg["diffusion"]["save_every"])
    total_steps= int(cfg["diffusion"]["num_steps"])

    out_dir = os.path.join(work, "ldm")
    os.makedirs(out_dir, exist_ok=True)

    pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    step = 0
    while step < total_steps:
        for batch in dl:
            with accelerator.accumulate(unet):
                # 1) 编码为 latent
                with torch.no_grad():
                    begin_lat = vae.encode(batch["begin_pixel"].to(device, dtype=dtype)).latent_dist.sample() * vae_sf
                    after_lat = vae.encode(batch["after_pixel"].to(device, dtype=dtype)).latent_dist.sample() * vae_sf

                # 2) 条件嵌入（无文本）
                mass_vals = batch["mass_value"].to(device)
                tex_ids   = batch["texture_id"].to(device)
                cond_emb, uncond_emb = mt(mass_vals, tex_ids, cond_drop_prob=cond_drop)  # [B, T, D]

                # 3) 噪声与时间步
                noise = torch.randn_like(after_lat)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (after_lat.size(0),), device=device, dtype=torch.long)
                noisy = noise_scheduler.add_noise(after_lat, noise, timesteps)

                # 4) 模型输入：拼接 (noisy_target, begin_latent) → [B, 8, H, W]
                model_in = torch.cat([noisy, begin_lat], dim=1)

                # 5) 预测噪声（一次前向，内部已做条件drop）
                noise_pred = unet(
                    model_in, timesteps,
                    encoder_hidden_states=cond_emb.to(dtype=dtype)  # [B, T, D]
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process and (step % 100 == 0):
                pbar.set_description(f"ldm step {step} | loss {loss.item():.4f}")

            if accelerator.is_local_main_process and (step % save_every == 0 and step>0):
                sd = os.path.join(out_dir, f"step{step}")
                os.makedirs(sd, exist_ok=True)
                (unet.module if hasattr(unet,"module") else unet).save_pretrained(os.path.join(sd,"unet"))
                torch.save((mt.module if hasattr(mt,"module") else mt).state_dict(), os.path.join(sd,"mt_embedder.pt"))
                # 也把 vocab 复制一份
                import shutil, json
                vocab_file = os.path.join(work,"emb","texture_vocab.json")
                if os.path.exists(vocab_file):
                    shutil.copy(vocab_file, os.path.join(sd,"texture_vocab.json"))

            step += 1; pbar.update(1)
            if step >= total_steps: break

    if accelerator.is_local_main_process:
        fd = os.path.join(out_dir, "final")
        os.makedirs(fd, exist_ok=True)
        (unet.module if hasattr(unet,"module") else unet).save_pretrained(os.path.join(fd,"unet"))
        torch.save((mt.module if hasattr(mt,"module") else mt).state_dict(), os.path.join(fd,"mt_embedder.pt"))
        import shutil
        vf = os.path.join(work,"emb","texture_vocab.json")
        if os.path.exists(vf): shutil.copy(vf, os.path.join(fd,"texture_vocab.json"))
        print("Diffusion saved:", fd)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml","r"))
    main(cfg)
