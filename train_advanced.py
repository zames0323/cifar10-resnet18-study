import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Advanced Training on: {device}")

    # 2. Advanced Augmentation Pipeline (Adding ColorJitter to mitigate Context Bias)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("[*] Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # 3. Model Architecture
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # 4. Loss, Optimizer, and Cosine Annealing LR Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_losses = []
    test_accuracies = []
    best_acc = 0.0

    print("[*] Starting Advanced Optimization Pipeline (12 Epochs)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step LR scheduler per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        test_accuracies.append(acc)
        print(f"[*] Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")

        # Save Best Advanced Weights
        if acc > best_acc:
            best_acc = acc
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/best_model_advanced.pth')
            print(f"[+] Saved optimal checkpoint: {best_acc:.2f}% -> weights/best_model_advanced.pth")

    # 5. Comparative Ablation Visualization
    os.makedirs('results', exist_ok=True)
    baseline_acc = 82.50  # Day 2 benchmark

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', color='tab:purple', label='Advanced (CosineLR+Jitter)')
    plt.title('Training Loss with Cosine Annealing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', color='tab:green', label='Optimized Model')
    plt.axhline(y=baseline_acc, color='tab:red', linestyle='--', label=f'Baseline Day 2 ({baseline_acc}%)')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/ablation_compare.png', dpi=200)
    print(f"\n[+] Ablation study completed! Peak Acc: {best_acc:.2f}%. Saved to results/ablation_compare.png")

if __name__ == '__main__':
    main()