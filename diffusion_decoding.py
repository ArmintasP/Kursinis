import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, gpu, verbose=False):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    
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
    model = model.to(device)
    
    return model


def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--imgidx",
        required=True,
        type=int,
        nargs="*",
        help="img idx"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan",
    )
    return parser.parse_args()


def make_directories(subject, savedir, method):
    outdir = f'{savedir}/image-{method}/{subject}/'
    os.makedirs(outdir, exist_ok=True)

    sample_path = os.path.join(outdir, f"samples")
    os.makedirs(sample_path, exist_ok=True)


def save_original_image(save_path, subject, test_id):
    img_stims_unique = np.load(f'../nsd-results/mrifeat/{subject}/{subject}_stims_ave.npy')
    img_stims_all_split_indexes = np.load(f'../nsd-results/mrifeat/{subject}/{subject}_stims_all_split_ids.npy')
    img_stims_unique_test = img_stims_unique[np.isin(img_stims_unique, img_stims_all_split_indexes)]
    
    nsda = NSDAccess('../nsd/')
    sdataset = h5py.File(nsda.stimuli_file, 'r').get('imgBrick')
    
    # Load z (Image)
    img_index = img_stims_unique_test[test_id]
    Image.fromarray(np.squeeze(sdataset[img_index,:,:,:]).astype(np.uint8)).save(os.path.join(save_path, f"{test_id:05}_org.png"))  


def load_diffusion_model(gpu):
    ckpt = '../StableDiffusionReconstruction-main/codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/weights.ckpt'
    config = '../StableDiffusionReconstruction-main/codes/diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    
    config = OmegaConf.load(f"{config}")
    return load_model_from_config(config, f"{ckpt}", gpu)


def load_sampler(model, ddim_steps, ddim_eta):
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    return sampler


def get_image_latent_space(model,savedir, subject, imgidx):
    roi_latent = 'ventral'
    scores_latent = np.load(f'{savedir}/{subject}/final/{subject}_{roi_latent}_scores_image-latent.npy')
    imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,65,65)).unsqueeze(0).to('cuda')

    # Generate image from Z
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                x_samples = model.decode_first_stage(imgarr)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

    img = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
    img = np.array(img)

    init_image = load_img_from_arr(img).to('cuda')
    result = model.get_first_stage_encoding(model.encode_first_stage(init_image))

    print(result.shape)

    return result


def main():
    args = get_args()
    seed_everything(args.seed)
    
    gpu = args.gpu
    method = args.method
    subject= args.subject
    
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    
    savedir = f'../nsd-results/latents-520/'
    
    make_directories(subject, savedir, method)
    sample_path = f'{savedir}/image-{method}/{subject}/samples'
    # Load Stable Diffusion Model
    model = load_diffusion_model(gpu)
    
    batch_size = 1
    ddim_steps = 60
    strength = 0.8
    n_iter = 5
    scale = 5.0
    
    sampler = load_sampler(model, ddim_steps=ddim_steps, ddim_eta=0.0)
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # Generated latent space from Z
    for imgidx in range(args.imgidx[0], args.imgidx[1]):
        save_original_image(sample_path, subject, imgidx)

        image_latent_space = get_image_latent_space(model,savedir, subject, imgidx)

        # c data
        roi_c = 'ventral'
        scores_c = np.load(f'{savedir}/{subject}/final/{subject}_{roi_c}_scores_text-latent.npy')
        carr = scores_c[imgidx,:].reshape(77, 768)
        c = torch.Tensor(carr).unsqueeze(0).to('cuda')

        # Generate image from Z (image) + C (semantics)
        base_count = 0
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    for n in trange(n_iter, desc="Sampling"):
                        uc = model.get_learned_conditioning(batch_size * [""])

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(image_latent_space, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png"))    
                        base_count += 1


if __name__ == "__main__":
    main()
