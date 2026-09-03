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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 1. High-resolution input pipeline (Upsampling to 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    # 2. Model & Checkpoint Load
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    checkpoint_path = 'weights/best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Loaded weights from {checkpoint_path}")
    else:
        print("[-] Checkpoint not found! Proceeding with fallback mode.")

    model = model.to(device)
    model.eval()

    # 3. Target layer: Final Conv layer in Layer 4
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Extract samples
    correct_samples = []
    wrong_samples = []

    print("[*] Filtering distinct samples...")
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        for i in range(inputs.size(0)):
            if preds[i] == labels[i] and len(correct_samples) < 2:
                correct_samples.append((inputs[i:i+1], labels[i].item(), preds[i].item()))
            elif preds[i] != labels[i] and len(wrong_samples) < 2:
                wrong_samples.append((inputs[i:i+1], labels[i].item(), preds[i].item()))

            if len(correct_samples) == 2 and len(wrong_samples) == 2:
                break
        if len(correct_samples) == 2 and len(wrong_samples) == 2:
            break

    target_samples = correct_samples + wrong_samples
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (img_tensor, label, pred) in enumerate(target_samples):
        # Grad-CAM computation
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]

        # Normalize contrast to ensure saturated red highlights
        grayscale_cam = np.maximum(grayscale_cam, 0)
        c_min, c_max = grayscale_cam.min(), grayscale_cam.max()
        if c_max > c_min:
            grayscale_cam = (grayscale_cam - c_min) / (c_max - c_min)

        # Unnormalize RGB image
        rgb_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        # Overlay heatmap
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

        row_type = "Correct" if idx < 2 else "Misclassified"
        title_color = "green" if idx < 2 else "red"

        axes[0, idx].imshow(rgb_img)
        axes[0, idx].set_title(f"[{row_type}] True: {classes[label]}", fontsize=11, color=title_color)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(cam_image)
        axes[1, idx].set_title(f"Grad-CAM (Pred: {classes[pred]})", fontsize=11, color=title_color)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig('results/xai_gradcam_vibrant.png', dpi=200)
    print("[+] Complete! Vibrant heatmap saved to results/xai_gradcam_vibrant.png")

if __name__ == '__main__':
    main()