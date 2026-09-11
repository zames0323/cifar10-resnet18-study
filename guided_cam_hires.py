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

def load_model(weights_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def download_highres_sample(target_path):
    """Download a crisp, high-resolution cat photo if not present."""
    if not os.path.exists(target_path):
        print("[*] Downloading high-resolution cat image...")
        url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[+] Downloaded successfully to {target_path}")

def main():
    # 1. Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Running High-Res Guided Grad-CAM on: {device}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Acquire and Preprocess High-Res Image (224x224 from rich source)
    sample_img_path = "sample_cat_hires.jpg"
    download_highres_sample(sample_img_path)

    raw_pil = Image.open(sample_img_path).convert('RGB')
    
    # Center crop and resize to 224x224 maintaining crisp edge resolution
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    input_tensor = preprocess(raw_pil).unsqueeze(0).to(device)

    # 3. Load Trained Model
    weights_path = 'weights/best_model_advanced.pth'
    model = load_model(weights_path, device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        conf = probs[pred_idx].item()
    print(f"[+] Predicted Class: {classes[pred_idx]} ({conf * 100:.2f}%)")

    # 4. Compute Coarse Grad-CAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 5. Compute Guided Backpropagation
    try:
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
    except TypeError:
        try:
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=(device.type == "cuda"))
        except Exception:
            gb_model = GuidedBackpropReLUModel(model=model)

    gb_result = gb_model(input_tensor, target_category=pred_idx)
    if isinstance(gb_result, torch.Tensor):
        gb_result = gb_result.detach().cpu().numpy()
    if gb_result.ndim == 4:
        gb_result = gb_result[0]
    if gb_result.shape[0] == 3 and gb_result.shape[2] != 3:
        gb_result = np.transpose(gb_result, (1, 2, 0))

    # 6. Guided Grad-CAM Fusion (Masking fine edges with coarse CAM)
    cam_3d = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2)
    guided_gradcam = gb_result * cam_3d

    # Correct Contrast Normalization (Keep background pitch-black, highlight positive edges)
    def clean_sketch(img_arr):
        # Clip negative noise, preserve glowing positive attributions
        pos = np.maximum(img_arr, 0)
        max_val = np.percentile(pos, 99.5)  # Suppress extreme pixel outliers
        if max_val > 0:
            pos = pos / max_val
        return np.clip(pos, 0.0, 1.0)

    gb_vis = clean_sketch(gb_result)
    ggc_vis = clean_sketch(guided_gradcam)

    # Unnormalize original 224x224 crop for display
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
    orig_np = np.transpose(input_tensor.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
    orig_unnorm = np.clip(orig_np * std + mean, 0.0, 1.0)

    # Overlay Grad-CAM
    cam_overlay = show_cam_on_image(orig_unnorm, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # 7. Multi-panel Visualization
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_unnorm)
    axes[0].set_title(f"High-Res Input ({classes[pred_idx]}: {conf*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cam_overlay)
    axes[1].set_title("Grad-CAM (Semantic Region)", fontsize=12, fontweight='bold', color='crimson')
    axes[1].axis('off')

    axes[2].imshow(gb_vis)
    axes[2].set_title("Guided Backprop (All Sharp Edges)", fontsize=12, fontweight='bold', color='darkorange')
    axes[2].axis('off')

    axes[3].imshow(ggc_vis)
    axes[3].set_title("Guided Grad-CAM (Target Neon Sketch)", fontsize=12, fontweight='bold', color='forestgreen')
    axes[3].axis('off')

    plt.tight_layout()
    output_path = 'results/guided_gradcam_hires.png'
    plt.savefig(output_path, dpi=200)
    print(f"\n[+] High-Res Guided Grad-CAM analysis saved to {output_path}")

if __name__ == '__main__':
    main()