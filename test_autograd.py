from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel


class DiffuserModels:
    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.eval()
        self.clip_model.to(self.torch_device)

def load_style_image(image_name):    
    img = Image.open(image_name).convert('RGB') 
    transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to desired dimensions
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    print(f"img_tensor.shape : {img_tensor.shape}")
    img_tensor = (img_tensor + 1) / 2
    return img_tensor

def blue_loss(images):
    print(f"images shape: {images.shape}")
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    print(f"blue error shape: {error.shape}")
    return error

def clip_loss(style_embedding, generated_embedding):
    return  torch.abs(style_embedding - generated_embedding).mean()
    return torch.nn.functional.cosine_similarity(style_embedding, generated_embedding, dim=-1).mean()


def calculate_auto_grad():
    models = DiffuserModels()

    latents = torch.rand(torch.Size([1, 4, 64, 64])).requires_grad_(True)
    style_image = load_style_image("style1.png")
    de_noised_images = load_style_image("style2.png")

    # # Preprocess images
    style_image_tensor = models.clip_processor(images=style_image, return_tensors="pt").to(models.torch_device)
    generated_image_tensor = models.clip_processor(images=de_noised_images, return_tensors="pt").to(models.torch_device)

    # print(style_image_tensor.keys())

    # # Calculate CLIP embeddings
    style_embedding = models.clip_model.get_image_features(**style_image_tensor)
    generated_embedding = models.clip_model.get_image_features(**generated_image_tensor)

    loss_1 = clip_loss(style_embedding, generated_embedding).requires_grad_(True)

    loss_2 = blue_loss(style_image).requires_grad_(True)

    print(loss_1)
    cond_grad = torch.autograd.grad(loss_2, latents)[0]
    print(cond_grad)


if __name__ == "__main__":
    calculate_auto_grad()

