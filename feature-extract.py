import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from nsd_access import NSDAccess
from einops import rearrange
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    
    return parser.parse_args()


def main():
    nsda = NSDAccess(f'../nsd/')
    resultsdir = '../nsd-results/latents-320'

    os.makedirs(f'../{resultsdir}/image-latent/', exist_ok=True)
    os.makedirs(f'../{resultsdir}/text-latent/', exist_ok=True)
    
    
    args = get_args()
    seed_everything(args.seed)
    
    resolution = 320
        
    precision_scope = autocast
    ckpt = '../StableDiffusionReconstruction-main/codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/weights.ckpt'
    config = OmegaConf.load('../StableDiffusionReconstruction-main/codes/diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
       
    torch.cuda.set_device(args.gpu)
    model = load_model_from_config(config, ckpt, args.gpu)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    image_stims_unique =  np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave.npy')
    image_stims_unique_map =  np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave_map.npy')
    selected_images = image_stims_unique[image_stims_unique_map]

    with open('prompts2.json', 'r') as json_file:
        prompts = json.load(json_file)

    for img_id in tqdm(selected_images):
        prompt = prompts[f'{img_id}']  
        img = get_image(nsda, img_id, resolution, device)
        
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(img)) 
        
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    c = model.get_learned_conditioning(prompt).mean(axis=0).unsqueeze(0)
                    
        init_latent = init_latent.cpu().detach().numpy().flatten()
        c = c.cpu().detach().numpy().flatten()
         
        np.save(f'../{resultsdir}/image-latent/{img_id:06}.npy',init_latent)
        np.save(f'../{resultsdir}/text-latent/{img_id:06}.npy',c)

def get_image(nsda, img_id, resolution, device):
    nsda_img = nsda.read_images(img_id)
    img = load_img_from_arr(nsda_img, resolution).to(device)
    img = repeat(img, '1 ... -> b ...', b=1)
    return img


# From stable diff example library, nothing fancy, little modifications. 
# I tweaked it for gpu usage, beware.

def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model


def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


if __name__ == "__main__":
    main()
