import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_model(weights_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    # 1. Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. 224x224 Upsampling Pipeline for High Spatial Resolution
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    # 3. Load Both Models (Baseline vs Optimized)
    baseline_path = 'weights/best_model.pth'
    advanced_path = 'weights/best_model_advanced.pth'

    if not os.path.exists(baseline_path) or not os.path.exists(advanced_path):
        print("[-] Checkpoint missing! Ensure both best_model.pth and best_model_advanced.pth exist.")
        return

    model_base = load_model(baseline_path, device)
    model_adv = load_model(advanced_path, device)
    print("[+] Successfully loaded both checkpoints!")

    cam_base = GradCAM(model=model_base, target_layers=[model_base.layer4[-1]])
    cam_adv = GradCAM(model=model_adv, target_layers=[model_adv.layer4[-1]])

    # 4. Search for 3 Comparative Samples
    # Condition: Baseline misclassified, Optimized classified correctly
    selected_samples = []
    print("[*] Searching for comparative evaluation samples...")

    for inputs, labels in testloader:
        inputs_dev, labels_dev = inputs.to(device), labels.to(device)
        out_base = model_base(inputs_dev)
        out_adv = model_adv(inputs_dev)

        pred_base = out_base.argmax(dim=1)
        pred_adv = out_adv.argmax(dim=1)

        for i in range(inputs.size(0)):
            if pred_base[i] != labels_dev[i] and pred_adv[i] == labels_dev[i]:
                selected_samples.append((inputs[i:i+1], labels_dev[i].item(), pred_base[i].item(), pred_adv[i].item()))
                if len(selected_samples) >= 3:
                    break

        if len(selected_samples) >= 3:
            break

    # Fallback if 3 flipped samples not found in immediate batches
    if len(selected_samples) < 3:
        for inputs, labels in testloader:
            inputs_dev, labels_dev = inputs.to(device), labels.to(device)
            out_base = model_base(inputs_dev)
            out_adv = model_adv(inputs_dev)
            pred_base = out_base.argmax(dim=1)
            pred_adv = out_adv.argmax(dim=1)
            for i in range(inputs.size(0)):
                selected_samples.append((inputs[i:i+1], labels_dev[i].item(), pred_base[i].item(), pred_adv[i].item()))
                if len(selected_samples) >= 3:
                    break
            if len(selected_samples) >= 3:
                break

    # 5. Generate Comparative Visualization Plot (3 Rows x 3 Columns)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))

    for idx, (img_tensor, label, p_base, p_adv) in enumerate(selected_samples[:3]):
        img_dev = img_tensor.to(device)

        cam_map_base = cam_base(input_tensor=img_dev, targets=None)[0, :]
        cam_map_base = np.maximum(cam_map_base, 0)
        if cam_map_base.max() > cam_map_base.min():
            cam_map_base = (cam_map_base - cam_map_base.min()) / (cam_map_base.max() - cam_map_base.min())

        cam_map_adv = cam_adv(input_tensor=img_dev, targets=None)[0, :]
        cam_map_adv = np.maximum(cam_map_adv, 0)
        if cam_map_adv.max() > cam_map_adv.min():
            cam_map_adv = (cam_map_adv - cam_map_adv.min()) / (cam_map_adv.max() - cam_map_adv.min())

        rgb_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        overlay_base = show_cam_on_image(rgb_img, cam_map_base, use_rgb=True, colormap=cv2.COLORMAP_JET)
        overlay_adv = show_cam_on_image(rgb_img, cam_map_adv, use_rgb=True, colormap=cv2.COLORMAP_JET)

        # Raw Image
        axes[idx, 0].imshow(rgb_img)
        axes[idx, 0].set_title(f"Target Label: {classes[label]}", fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        # Baseline CAM
        base_color = "green" if p_base == label else "red"
        axes[idx, 1].imshow(overlay_base)
        axes[idx, 1].set_title(f"Baseline (Pred: {classes[p_base]})", fontsize=12, color=base_color)
        axes[idx, 1].axis('off')

        # Optimized CAM
        adv_color = "green" if p_adv == label else "red"
        axes[idx, 2].imshow(overlay_adv)
        axes[idx, 2].set_title(f"Optimized (Pred: {classes[p_adv]})", fontsize=12, color=adv_color)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('results/xai_model_comparison.png', dpi=200)
    print("\n[+] Direct XAI comparison saved to results/xai_model_comparison.png!")

if __name__ == '__main__':
    main()