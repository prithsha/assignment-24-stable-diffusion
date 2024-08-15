import os
import torch
import math

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from tqdm.auto import tqdm


class Config:
    GUIDANCE_SCALE = 7.5
    INFERENCE_STEPS = 50
    SEED = 30

class DiffuserModels:

    def __init__(self, model_name, torch_device):
        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        # The noise scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                        beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)

        # To the GPU we go!
        self.vae = self.vae.to(torch_device)
        self.text_encoder = self.text_encoder.to(torch_device)
        self.unet = self.unet.to(torch_device)
        self.torch_device = torch_device

def get_stable_diffusion_pipeline(pretrained_model_name, torch_device):
    pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float16,
            safety_checker=None).to(torch_device)

    return pipe

# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)

def latents_to_pil(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def generate_with_embs(models : DiffuserModels, text_embeddings, generator, text_input_max_length ):
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = Config.INFERENCE_STEPS            # Number of denoising steps
    guidance_scale = Config.GUIDANCE_SCALE                # Scale for classifier-free guidance
    batch_size = 1

    uncond_input = models.tokenizer(
                [""] * batch_size, padding="max_length",
                max_length=text_input_max_length, return_tensors="pt" )
    with torch.no_grad():
        uncond_embeddings = models.text_encoder(uncond_input.input_ids.to(models.torch_device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(models.scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn((batch_size, models.unet.in_channels, height // 8, width // 8), generator=generator)
    latents = latents.to(models.torch_device)
    latents = latents * models.scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(models.scheduler.timesteps), total=len(models.scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = models.scheduler.sigmas[i]
        latent_model_input = models.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = models.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = models.scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def get_output_embeds(models : DiffuserModels, input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = _create_4d_causal_attention_mask((bsz, seq_len),
                                                            dtype=input_embeddings.dtype,
                                                            device=models.torch_device)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = models.text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(models.torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = models.text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output