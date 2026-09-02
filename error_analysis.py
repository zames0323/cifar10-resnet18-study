import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Device Setup (GPU / MPS / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Test Data Loading with Normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # 3. Model Architecture Setup
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    model.eval()

    # 4. Collect 5 Misclassified Samples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    print("[*] Finding error samples...")
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            wrong_idx = (preds != labels).nonzero(as_tuple=False)
            for idx in wrong_idx:
                i = idx.item()
                misclassified_images.append(inputs[i].cpu())
                misclassified_labels.append(labels[i].item())
                misclassified_preds.append(preds[i].item())

                if len(misclassified_images) == 5:
                    break
            if len(misclassified_images) == 5:
                break

    # 5. Visualize and Save 5 Misclassified Samples
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(15, 3))

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = misclassified_images[i].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.title(f"True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_preds[i]]}", 
                  color='red', fontsize=11)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/error_analysis.png')
    print("[+] Done! Error analysis plot saved to results/error_analysis.png")

if __name__ == '__main__':
    main()