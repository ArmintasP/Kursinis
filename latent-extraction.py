from tqdm import tqdm
from nsd_access import NSDAccess
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
import utilities
import torch
import numpy as np
import os
import json


def main():
    torch.cuda.manual_seed(0)
    torch_device = "cuda"

    nsda = NSDAccess('../nsd/')
    os.makedirs(f'../nsd-results/latents/image-latent/', exist_ok=True)
    os.makedirs(f'../nsd-results/latents/text-latent/', exist_ok=True)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
    img_proc = VaeImageProcessor()
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)

    vae.to(torch_device)
    text_encoder.to(torch_device)
    torch.cuda.device(torch_device)

    image_stims_unique =  np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave.npy')
    image_stims_unique_map =  np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave_map.npy')
    selected_images = image_stims_unique[image_stims_unique_map]

    with open('prompts.json', 'r') as json_file:
        prompts = json.load(json_file)

    for image_id in tqdm(selected_images):
        prompt = prompts[f'{image_id}']

        img = utilities.load_img_from_nsd(nsda, image_index=image_id, resolution=520)
        img = img_proc.preprocess(img).to(torch_device)

        with torch.no_grad():
            image_latent = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            text_embedding = text_encoder(text_input.input_ids.to(torch_device))[0]

        image_latent = image_latent.cpu().detach().numpy().flatten()
        text_embedding = text_embedding.cpu().detach().numpy().flatten()
        
        np.save(f'../nsd-results/latents/image-latent/{image_id:06}.npy', image_latent)
        np.save(f'../nsd-results/latents/text-latent/{image_id:06}.npy', text_embedding)

if __name__ == "__main__":
    main()