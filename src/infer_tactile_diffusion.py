import argparse, os, torch, torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from src.mt_embedder import MassTextureEmbedder, parse_mass_to_grams
import json

@torch.no_grad()
def run(work_dir, begin_path, mass, texture, image_size=256, steps=40, guidance_scale=6.5, seed=42, out="./after.png"):
    device = "cuda"
    torch.manual_seed(seed)

    # 1) 加载 VAE + UNet + MT
    vae_dir  = os.path.join(work_dir, "vae", "final")
    ldm_dir  = os.path.join(work_dir, "ldm", "final")
    vae  = AutoencoderKL.from_pretrained(vae_dir).to(device, dtype=torch.float16).eval()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(ldm_dir,"unet")).to(device, dtype=torch.float16).eval()

    vocab = json.load(open(os.path.join(ldm_dir,"texture_vocab.json"),"r",encoding="utf-8"))
    mt = MassTextureEmbedder(num_textures=len(vocab), embed_dim=unet.config.cross_attention_dim, seq_len=12).to(device, dtype=torch.float16).eval()
    mt.load_state_dict(torch.load(os.path.join(ldm_dir,"mt_embedder.pt"), map_location="cpu"))

    vae_sf = 1.0  # 与训练保持一致
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    scheduler.set_timesteps(steps, device=device)

    # 2) 处理 Begin + 条件
    tf = T.Compose([T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()])
    begin = Image.open(begin_path).convert("RGB")
    begin = tf(begin).unsqueeze(0).to(device)  # [1,3,H,W]
    begin_lat = vae.encode((begin*2-1)).latent_dist.mode() * vae_sf  # 注意：VAE期望[-1,1]，所以做线性变换

    mass_val = torch.tensor([parse_mass_to_grams(mass)], dtype=torch.float32, device=device)
    tex_id   = torch.tensor([vocab.get(str(texture).strip().lower(), 0)], dtype=torch.long, device=device)
    cond_emb, uncond_emb = mt(mass_val, tex_id, cond_drop_prob=0.0)  # [1,T,D]

    # 3) 采样 latent
    lat = torch.randn_like(begin_lat)
    for t in scheduler.timesteps:
        # CFG: 批次拼2份
        lat_in = torch.cat([lat, lat], dim=0)
        begin_in = torch.cat([begin_lat, begin_lat], dim=0)
        model_in = torch.cat([lat_in, begin_in], dim=1)  # [2, 8, H/8, W/8]
        emb = torch.cat([uncond_emb, cond_emb], dim=0)

        eps = unet(model_in, t, encoder_hidden_states=emb).sample
        eps_u, eps_c = eps.chunk(2, dim=0)
        eps_hat = eps_u + guidance_scale * (eps_c - eps_u)

        lat = scheduler.step(eps_hat, t, lat).prev_sample

    # 4) 解码
    img = vae.decode(lat / vae_sf).sample
    img = (img.clamp(-1,1) + 1)/2
    img = (img*255).round().type(torch.uint8).cpu()[0]
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    T.ToPILImage()(img).save(out)
    print("Saved to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--begin_path", required=True)
    ap.add_argument("--mass", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--out", default="./after.png")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance_scale", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(**vars(args))
