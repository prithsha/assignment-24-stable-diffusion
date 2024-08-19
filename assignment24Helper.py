import os
import torch
import math

import PIL
from PIL import Image
import torchvision.transforms as transforms

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn


class Config:
    GUIDANCE_SCALE = 7.5
    INFERENCE_STEPS = 50
    SEED = 150
    CUSTOM_LOSS_SCALE = 100

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
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.eval()
        self.clip_model.to(torch_device)

def load_style_image():    
    
    img = Image.open("style2.png").convert('RGB') 
    transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to desired dimensions
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    # print(f"img_tensor.shape : {img_tensor.shape}")
    return img_tensor

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

def blue_loss(images):
    print(f"images shape: {images.shape}")
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    print(f"blue error shape: {error.shape}")
    return error

def clip_loss(style_embedding, generated_embedding):
    return torch.nn.functional.cosine_similarity(style_embedding, generated_embedding, dim=-1).mean()

def generate_with_embs_blue_loss(models : DiffuserModels, text_embeddings, generator, text_input_max_length, apply_custom_loss_guidance = False ):
    height = 512                                            # default height of Stable Diffusion
    width = 512                                             # default width of Stable Diffusion
    num_inference_steps = Config.INFERENCE_STEPS            # Number of denoising steps
    guidance_scale = Config.GUIDANCE_SCALE                  # Scale for classifier-free guidance
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
    style_image = load_style_image()
    style_image = (style_image + 1) / 2

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

        if(apply_custom_loss_guidance):
            #### ADDITIONAL GUIDANCE ###
            if i%5 == 0:
                    # Requires grad on the latents
                    latents = latents.detach()                    
                    latents.requires_grad_(True)

                    # Get the predicted x0:
                    latents_x0 = latents - sigma * noise_pred

                    # Decode to image space
                    de_noised_images = models.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)
                    de_noised_images = torch.clamp(de_noised_images, min=0, max=1)

                    # Calculate loss
                    loss = blue_loss(de_noised_images) * Config.CUSTOM_LOSS_SCALE
                    
                    print(i, 'loss:', loss.item())
                    print(f"loss: {loss}")
                    
                    # Get gradient
                    cond_grad = torch.autograd.grad(loss, latents)[0]

                    # Modify the latents based on this gradient
                    latents = latents.detach() - cond_grad * sigma**2
                    print(f"latents.shape: {latents.shape}")


        # compute the previous noisy sample x_t -> x_t-1
        latents = models.scheduler.step(noise_pred, t, latents).prev_sample
        

    return latents

def generate_with_embs_with_clip_latents(models : DiffuserModels, text_embeddings, generator, text_input_max_length, apply_custom_loss_guidance = False ):
    height = 512                                            # default height of Stable Diffusion
    width = 512                                             # default width of Stable Diffusion
    num_inference_steps = Config.INFERENCE_STEPS            # Number of denoising steps
    guidance_scale = Config.GUIDANCE_SCALE                  # Scale for classifier-free guidance
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
    style_image = load_style_image()
    style_image = (style_image + 1) / 2
    style_image = style_image.to(models.torch_device)

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

        if(apply_custom_loss_guidance):
            #### ADDITIONAL GUIDANCE ###
            if i%1 == 0:
                    # Requires grad on the latents
                    latents = latents.detach()                    
                    # latents.requires_grad_(True)

                    # # Get the predicted x0:
                    latents_x0 = latents - sigma * noise_pred
                    # de_noised_images = models.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)
                    # de_noised_images = torch.clamp(de_noised_images, min=0, max=1)

                    # # Preprocess images                    
                    # generated_image_tensor = models.clip_processor(images=de_noised_images, return_tensors="pt").to(models.torch_device)
                    # generated_embedding = models.clip_model.get_image_features(**generated_image_tensor)

                    style_image_tensor = models.clip_processor(images=style_image, return_tensors="pt").to(models.torch_device)

                    # # Calculate CLIP embeddings
                    style_embedding = models.clip_model.get_image_features(**style_image_tensor)
                    # difference_mean = clip_loss(style_embedding, generated_embedding).mean()

                    projection = nn.Linear(768, 4*64*64).to(models.torch_device)
                    projected_features = projection(style_embedding)
                    projected_features = projected_features.view(latents.shape)
                    latents = latents_x0 + projected_features

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

def initialize_model(model_name, torch_device, prompt_replacement_text = "flower"):

    models = DiffuserModels(model_name=model_name, torch_device=torch_device)

    prompt = prompt_replacement_text
    tokens = models.tokenizer(prompt)
    # print('tokenizer(prompt):', tokens)

    selected_token = tokens["input_ids"][1]
    pos_emb_layer = models.text_encoder.text_model.embeddings.position_embedding
    token_emb_layer = models.text_encoder.text_model.embeddings.token_embedding
    position_ids = models.text_encoder.text_model.embeddings.position_ids[:, :77]
    position_embeddings = pos_emb_layer(position_ids)

    return models, selected_token, token_emb_layer, position_embeddings

def get_token_embedding_for_prompt(models : DiffuserModels, token_emb_layer, prompt="A cinematic hand made wall poster of girl with umbrella and red flower"):
    
    text_input = models.tokenizer(prompt, padding="max_length",
                              max_length=models.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(models.torch_device)
    token_embeddings = token_emb_layer(input_ids)
    # Only printing first 20 tokens
    # print('tokenizer(prompt):', input_ids[0][:20], input_ids[0].shape)
    return input_ids, token_embeddings

def get_embedding_for_concept_library(library_name, torch_device):
    embedding_directory = "embeddings"
    selected_name = library_name.split("/")[1]
    embeddings = torch.load(os.path.join(os.getcwd(), embedding_directory, selected_name + "-learned_embeds.bin"))
    replacement_token_embedding = embeddings[list(embeddings.keys())[0]].to(torch_device)
    return replacement_token_embedding

def execute_diffusion_for_concept_library(models, input_embeddings, text_input_max_length, seed, apply_custom_loss_guidance= False):

    generator = torch.Generator(device="cpu").manual_seed(seed) 
    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(models=models, input_embeddings= input_embeddings)    
    latents = generate_with_embs_with_clip_latents(models=models, text_embeddings= modified_output_embeddings,
                                     generator=generator,
                                     text_input_max_length=text_input_max_length,
                                     apply_custom_loss_guidance=apply_custom_loss_guidance
                                     )
    return latents_to_pil(latents, models.vae)[0]