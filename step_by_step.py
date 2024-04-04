import torch
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image

torch_device = "cuda"

YOUR_TOKEN='hf_lXrddNstPxRSTNYRKberJXvcUoLeatu EkG'

prompt = ["A big banana leaf with lemons in front of it"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 100           # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
batch_size = len(prompt)
UNET_INPUTS_CHANNEL=4
# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                    subfolder="vae",
                                    use_auth_token=YOUR_TOKEN)

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents in Float16 datatype.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            subfolder="unet",
                                            torch_dtype=torch.float16,
                                            revision="fp16",
                                            use_auth_token=YOUR_TOKEN)
scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                 beta_end=0.012, beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)
#Set the models to your inference device
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).half().cuda()

latents = torch.randn(
    (batch_size, UNET_INPUTS_CHANNEL, height // 8, width // 8))
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)

latents = latents * scheduler.sigmas[0]

scheduler.set_timesteps(num_inference_steps)
# Denoising Loop
with torch.inference_mode(), autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)

#Convert the image with PIL and save it
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save('image_generated.png')