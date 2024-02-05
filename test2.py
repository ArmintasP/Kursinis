import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import utilities
import json
import diffusers.utils.torch_utils
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from PIL import Image


def dummy(images, **kwargs):
    return images, [False for i in images]


def main():
    #dtype
    guidance_scale = 7

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.safety_checker = dummy

    pipe = pipe.to("cuda")
    generator = torch.cuda.manual_seed(0)  # Seed generator to create the initial latent noise

    # Load z (Image)
    imgidx = 301

    img_stims_unique = np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave.npy')
    img_stims_all_split_indexes = np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_all_split_ids.npy')
    img_stims_unique_test = img_stims_unique[np.isin(img_stims_unique, img_stims_all_split_indexes)]

    img_index = img_stims_unique_test[imgidx]
    imm = utilities.load_img_from_nsd(img_index, resolution=320)
    imm.save(f'{imgidx:05}_org.png')

    # c data
    scores_c = np.load('../nsd-results/latents-320/subj01/final/subj01_ventral_scores_text-latent.npy')
    carr = scores_c[imgidx,:].reshape(77,768)
    text_latent = torch.Tensor(carr).unsqueeze(0).to('cuda')

    # x data
    scores_latent = np.load('../nsd-results/latents-320/subj01/final/subj01_early_scores_image-latent.npy')
    img_latent = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')

    image = pipe(
        image= img_latent,
        prompt_embeds=text_latent,
        generator = generator,
        guidance_scale=guidance_scale,
        strength=0.8,
    ).images[0]

    image.save("aha1.png")


    image = pipe(
        image= img_latent,
        prompt_embeds=text_latent,
        generator = generator,
        guidance_scale=guidance_scale,
        strength=0.8,
    ).images[0]

    image.save("aha2.png")

    image = pipe(
        image= img_latent,
        prompt_embeds=text_latent,
        generator = generator,
        guidance_scale=guidance_scale,
        strength=0.8,
    ).images[0]
    image.save("aha3.png")

    image = pipe(
        image= img_latent,
        prompt_embeds=text_latent,
        generator = generator,
        guidance_scale=guidance_scale,
        strength=0.8,
    ).images[0]

    image.save("aha4.png")


def smth(vae, latents):
    # scale and decode the image latents with vae
    latents = 1 /vae.config.scaling_factor * latents
    with torch.no_grad():
        image = vae.decode(latents).sample    

    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    
    image = Image.fromarray(image).resize((512,512))
    image.save(f'ssmm.png')
    return image


if __name__ == "__main__":
    main()