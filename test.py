from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import numpy as np
import utilities

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=False)

tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)

unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=False)
scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

torch_device = "cuda"

vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

height = 320  # default height of Stable Diffusion
width = 320  # default width of Stable Diffusion
num_inference_steps = 30 # Number of denoising steps
guidance_scale = 10  # Scale for classifier-free guidance
batch_size = 1

generator = torch.cuda.manual_seed(42)  # Seed generator to create the initial latent noise

# Load z (Image)
imgidx = 100
test_id = imgidx

img_stims_unique = np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_ave.npy')
img_stims_all_split_indexes = np.load(f'../nsd-results/mrifeat/subj01/subj01_stims_all_split_ids.npy')
img_stims_unique_test = img_stims_unique[np.isin(img_stims_unique, img_stims_all_split_indexes)]

img_index = img_stims_unique_test[test_id]
imm = utilities.load_img_from_nsd(img_index, resolution=width)
imm.save(f'{test_id:05}_org.png')

# c data
scores_c = np.load('../nsd-results/latents-320/final/subj01/subj01_ventral_scores_text-latent.npy')
carr = scores_c[imgidx,:].reshape(77,768)
c = torch.Tensor(carr).unsqueeze(0).to('cuda')

# x data
scores_latent = np.load('../nsd-results/latents-320/final/subj01/subj01_early_scores_image-latent.npy')
imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')

# 4,40,40
# 4 65 65

latents = imgarr
text_embeddings = c

max_length = 77
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


latents = latents * scheduler.init_noise_sigma

from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for i in range(0, 5):
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, int(t * 0.8), encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, int(t * 0.8), latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 /vae.config.scaling_factor * latents
    with torch.no_grad():
        image = vae.decode(latents).sample    

    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    images = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    image.save(f'{test_id:05}_dif_{i}.png')