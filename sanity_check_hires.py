import os
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    from pytorch_grad_cam import GuidedBackpropReLUModel
except ImportError:
    from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

def load_trained_model(weights_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def create_randomized_model(device):
    """Create an untrained model with completely randomized weights (Sanity Baseline)."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model = model.to(device)
    model.eval()
    return model

def clean_sketch(img_arr):
    pos = np.maximum(img_arr, 0)
    max_val = np.percentile(pos, 99.5)
    if max_val > 0:
        pos = pos / max_val
    return np.clip(pos, 0.0, 1.0)

def compute_visuals(model, input_tensor, target_category, device):
    # 1. Grad-CAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 2. Guided Backpropagation
    try:
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
    except TypeError:
        try:
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=(device.type == "cuda"))
        except Exception:
            gb_model = GuidedBackpropReLUModel(model=model)

    gb_result = gb_model(input_tensor, target_category=target_category)
    if isinstance(gb_result, torch.Tensor):
        gb_result = gb_result.detach().cpu().numpy()
    if gb_result.ndim == 4:
        gb_result = gb_result[0]
    if gb_result.shape[0] == 3 and gb_result.shape[2] != 3:
        gb_result = np.transpose(gb_result, (1, 2, 0))

    # 3. Guided Grad-CAM Fusion
    cam_3d = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2)
    guided_gradcam = gb_result * cam_3d

    ggc_vis = clean_sketch(guided_gradcam)
    return grayscale_cam, ggc_vis

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Running High-Res Sanity Check on: {device}")

    # 1. Load Image
    sample_img_path = "sample_cat_hires.jpg"
    if not os.path.exists(sample_img_path):
        url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(sample_img_path, 'wb') as f:
            f.write(resp.read())

    raw_pil = Image.open(sample_img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    input_tensor = preprocess(raw_pil).unsqueeze(0).to(device)

    # 2. Models
    trained_model = load_trained_model('weights/best_model_advanced.pth', device)
    random_model = create_randomized_model(device)

    with torch.no_grad():
        pred_trained = trained_model(input_tensor).softmax(dim=1)[0]
        conf_trained = pred_trained[3].item()  # Class 3: Cat
        pred_random = random_model(input_tensor).softmax(dim=1)[0]
        conf_random = pred_random[3].item()

    print(f"[+] Trained Model Cat Confidence   : {conf_trained * 100:.2f}%")
    print(f"[+] Randomized Model Cat Confidence: {conf_random * 100:.2f}% (Complete Amnesia)")

    # 3. Compute Visualizations (Target: Cat, Class Index 3)
    cam_trained, ggc_trained = compute_visuals(trained_model, input_tensor, 3, device)
    cam_random, ggc_random = compute_visuals(random_model, input_tensor, 3, device)

    # Unnormalize original image
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
    orig_np = np.transpose(input_tensor.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
    orig_unnorm = np.clip(orig_np * std + mean, 0.0, 1.0)

    trained_overlay = show_cam_on_image(orig_unnorm, cam_trained, use_rgb=True, colormap=cv2.COLORMAP_JET)
    random_overlay = show_cam_on_image(orig_unnorm, cam_random, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # 4. Generate 2x2 Comparative Benchmark Figure
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Row 1: Trained Model
    axes[0, 0].imshow(trained_overlay)
    axes[0, 0].set_title(f"Trained: Grad-CAM (Conf: {conf_trained*100:.1f}%)", fontsize=11, fontweight='bold', color='navy')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ggc_trained)
    axes[0, 1].set_title("Trained: Guided Grad-CAM", fontsize=11, fontweight='bold', color='forestgreen')
    axes[0, 1].axis('off')

    # Row 2: Randomized Model (Sanity Check)
    axes[1, 0].imshow(random_overlay)
    axes[1, 0].set_title(f"Randomized: Grad-CAM (Conf: {conf_random*100:.1f}%)", fontsize=11, fontweight='bold', color='crimson')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ggc_random)
    axes[1, 1].set_title("Randomized: Guided Grad-CAM (Artifact?)", fontsize=11, fontweight='bold', color='darkorange')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = 'results/sanity_check_hires.png'
    plt.savefig(output_path, dpi=200)
    print(f"\n[+] High-Res Sanity Check benchmark saved to {output_path}")

if __name__ == '__main__':
    main()