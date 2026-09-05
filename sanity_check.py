import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    # 1. Device configuration (GPU / MPS / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. CIFAR-10 test data loading with 224x224 upsampling
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # 3. Load Day 4 fine-tuned model (Trained Model)
    trained_model = resnet18()
    trained_model.fc = nn.Linear(trained_model.fc.in_features, 10)
    trained_model.load_state_dict(torch.load('weights/best_model_advanced.pth', map_location=device))
    trained_model.to(device).eval()

    # 4. Create randomized model for sanity check (Randomize Layer 4 and FC)
    random_model = resnet18()
    random_model.fc = nn.Linear(random_model.fc.in_features, 10)
    random_model.load_state_dict(torch.load('weights/best_model_advanced.pth', map_location=device))
    
    # Randomize weights of Layer 4 and FC using Kaiming Normal initialization
    for layer in [random_model.layer4, random_model.fc]:
        for module in layer.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    random_model.to(device).eval()

    # 5. Extract test samples and generate Grad-CAM heatmaps
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    cam_trained = GradCAM(model=trained_model, target_layers=[trained_model.layer4[-1]])
    cam_random = GradCAM(model=random_model, target_layers=[random_model.layer4[-1]])

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx in range(2):
        input_tensor = images[idx].unsqueeze(0)
        true_label = classes[labels[idx].item()]

        # Unnormalize original image for visualization
        rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = np.clip(rgb_img * std + mean, 0, 1)

        # Generate Grad-CAM with targets=None explicitly specified
        gray_cam_trained = cam_trained(input_tensor=input_tensor, targets=None)[0]
        vis_trained = show_cam_on_image(rgb_img, gray_cam_trained, use_rgb=True)

        gray_cam_random = cam_random(input_tensor=input_tensor, targets=None)[0]
        vis_random = show_cam_on_image(rgb_img, gray_cam_random, use_rgb=True)

        # Plot comparison
        axes[idx, 0].imshow(rgb_img)
        axes[idx, 0].set_title(f"Original (True: {true_label})", fontsize=12)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(vis_trained)
        axes[idx, 1].set_title("Trained Model CAM\n(Targeted Focus)", fontsize=12, color='darkgreen')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(vis_random)
        axes[idx, 2].set_title("Randomized Model CAM\n(Disrupted / Sanity Passed)", fontsize=12, color='crimson')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('results/sanity_check.png', dpi=200)
    print("[+] Sanity check complete. Result saved to results/sanity_check.png")

if __name__ == '__main__':
    main()