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

def main():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Preprocessing with 224x224 Upsampling
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # 3. Model Architecture & Trained Weights Load
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    checkpoint_path = 'weights/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Loaded weights from {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # 4. Grad-CAM Target Layer
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 5. Extract single sample
    inputs, labels = next(iter(testloader))
    inputs = inputs.to(device)
    label_idx = labels.item()

    outputs = model(inputs)
    pred_idx = outputs.argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=inputs, targets=None)[0, :]
    grayscale_cam = np.maximum(grayscale_cam, 0)
    c_min, c_max = grayscale_cam.min(), grayscale_cam.max()
    if c_max > c_min:
        grayscale_cam = (grayscale_cam - c_min) / (c_max - c_min)

    # Unnormalize
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    rgb_img = inputs.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # 6. Save Plot
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"True: {classes[label_idx]}", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM (Pred: {classes[pred_idx]})", fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('results/cam_sample.png')
    print("[+] Verified! Saved high-res sample to results/cam_sample.png")

if __name__ == '__main__':
    main()