import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Safe import across pytorch-grad-cam versions
try:
    from pytorch_grad_cam import GuidedBackpropReLUModel
except ImportError:
    from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

def load_optimized_model(weights_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def normalize_attribution(attr):
    """Normalize attribution map to 0.0 - 1.0 for clean visualization."""
    attr = attr - attr.min()
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr

def main():
    # 1. Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Initializing Guided Grad-CAM Pipeline on: {device}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Input Pipeline (224x224 spatial resolution)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    # 3. Load Trained Model
    weights_path = 'weights/best_model_advanced.pth'
    model = load_optimized_model(weights_path, device)

    # 4. Search for High-Confidence Correct Sample
    print("[*] Extracting high-confidence evaluation target...")
    best_conf = 0.0
    target_img, target_label = None, None

    for inputs, labels in testloader:
        inputs_dev, labels_dev = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

        correct_mask = (preds == labels_dev)
        if correct_mask.any():
            matched_indices = torch.where(correct_mask)[0]
            for idx in matched_indices:
                conf = confs[idx].item()
                if conf > best_conf:
                    best_conf = conf
                    target_img = inputs[idx:idx+1]
                    target_label = labels_dev[idx].item()

        if best_conf > 0.90:
            break

    print(f"[+] Target Selected: {classes[target_label]} (Confidence: {best_conf * 100:.2f}%)")

    # 5. Compute Coarse Grad-CAM from Layer 4
    target_img_dev = target_img.to(device)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=target_img_dev, targets=None)[0, :]  # Shape: (224, 224)

    # 6. Compute Fine-grained Guided Backpropagation
    try:
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
    except TypeError:
        try:
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=(device.type == "cuda"))
        except Exception:
            gb_model = GuidedBackpropReLUModel(model=model)

    gb_result = gb_model(target_img_dev, target_category=target_label)

    if isinstance(gb_result, torch.Tensor):
        gb_result = gb_result.detach().cpu().numpy()
    if gb_result.ndim == 4:
        gb_result = gb_result[0]
    if gb_result.shape[0] == 3 and gb_result.shape[2] != 3:
        gb_result = np.transpose(gb_result, (1, 2, 0))  # Shape: (224, 224, 3)

    # 7. Guided Grad-CAM Fusion: Element-wise Multiplication
    # Expand 2D Grad-CAM to 3D to match RGB gradient dimensions
    cam_3d = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2)
    guided_gradcam = gb_result * cam_3d

    # Normalize attribution maps for visualization
    gb_vis = normalize_attribution(gb_result)
    ggc_vis = normalize_attribution(guided_gradcam)

    # Unnormalize original image
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
    orig_np = np.transpose(target_img.squeeze(0).numpy(), (1, 2, 0))
    orig_unnorm = np.clip(orig_np * std + mean, 0.0, 1.0)

    # Create Grad-CAM overlay
    cam_overlay = show_cam_on_image(orig_unnorm, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # 8. Generate 1x4 Comparative Plot
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_unnorm)
    axes[0].set_title(f"Original: {classes[target_label]} ({best_conf*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cam_overlay)
    axes[1].set_title("Grad-CAM (Coarse Location)", fontsize=12, fontweight='bold', color='crimson')
    axes[1].axis('off')

    axes[2].imshow(gb_vis)
    axes[2].set_title("Guided Backprop (All Edges)", fontsize=12, fontweight='bold', color='darkorange')
    axes[2].axis('off')

    axes[3].imshow(ggc_vis)
    axes[3].set_title("Guided Grad-CAM (Target Details)", fontsize=12, fontweight='bold', color='forestgreen')
    axes[3].axis('off')

    plt.tight_layout()
    output_path = 'results/guided_gradcam_analysis.png'
    plt.savefig(output_path, dpi=200)
    print(f"\n[+] Guided Grad-CAM analysis saved to {output_path}")

if __name__ == '__main__':
    main()