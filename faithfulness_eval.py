# -*- coding: utf-8 -*-
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

def compute_auc(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)

def load_optimized_model(weights_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Executing Faithfulness Benchmark on: {device}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    weights_path = 'weights/best_model_advanced.pth'
    model = load_optimized_model(weights_path, device)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    print("[*] Sampling a representative high-confidence evaluation target...")
    best_conf = 0.0
    target_img, target_label = None, None

    for inputs, labels in testloader:
        inputs_dev, labels_dev = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

        # Correct predictions only
        correct_mask = (preds == labels_dev)
        if correct_mask.any():
            matched_indices = torch.where(correct_mask)[0]
            for idx in matched_indices:
                conf = confs[idx].item()
                if conf > best_conf:
                    best_conf = conf
                    target_img = inputs[idx:idx+1]
                    target_label = labels_dev[idx].item()

        # Stop immediately if we found a highly confident sample (> 90%)
        if best_conf > 0.90:
            break

    print(f"[+] Selected Target Label: {classes[target_label]} (Initial Confidence: {best_conf * 100:.2f}%)")

    target_img_dev = target_img.to(device)
    cam_mask = cam(input_tensor=target_img_dev, targets=None)[0, :]

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    orig_np = target_img.squeeze(0).numpy()
    orig_unnorm = np.clip(orig_np * std + mean, 0.0, 1.0)

    flat_cam = cam_mask.flatten()
    sorted_pixel_indices = np.argsort(flat_cam)[::-1]
    total_pixels = flat_cam.size

    blurred_chw = np.full_like(orig_unnorm, 0.5)

    def to_normalized_tensor(img_chw_np):
        norm_img = (img_chw_np - mean) / std
        return torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).to(device)

    steps = 11
    fractions = np.linspace(0.0, 1.0, steps)

    deletion_probs = []
    insertion_probs = []
    deletion_visuals = []
    insertion_visuals = []

    print("[*] Computing Deletion & Insertion Trajectories...")
    for frac in fractions:
        num_pixels_masked = int(frac * total_pixels)
        top_indices = sorted_pixel_indices[:num_pixels_masked]

        del_img = np.copy(orig_unnorm)
        for c in range(3):
            del_c = del_img[c].flatten()
            blur_c = blurred_chw[c].flatten()
            del_c[top_indices] = blur_c[top_indices]
            del_img[c] = del_c.reshape(224, 224)

        with torch.no_grad():
            out_del = model(to_normalized_tensor(del_img))
            prob_del = torch.softmax(out_del, dim=1)[0, target_label].item()
            deletion_probs.append(prob_del)

        ins_img = np.copy(blurred_chw)
        for c in range(3):
            ins_c = ins_img[c].flatten()
            orig_c = orig_unnorm[c].flatten()
            ins_c[top_indices] = orig_c[top_indices]
            ins_img[c] = ins_c.reshape(224, 224)

        with torch.no_grad():
            out_ins = model(to_normalized_tensor(ins_img))
            prob_ins = torch.softmax(out_ins, dim=1)[0, target_label].item()
            insertion_probs.append(prob_ins)

        if frac in [0.0, 0.2, 0.5, 0.8, 1.0]:
            deletion_visuals.append(np.transpose(del_img, (1, 2, 0)))
            insertion_visuals.append(np.transpose(ins_img, (1, 2, 0)))

    del_auc = compute_auc(fractions, deletion_probs)
    ins_auc = compute_auc(fractions, insertion_probs)

    print("\n" + "="*50)
    print(f"[*] XAI Faithfulness Evaluation Results (Label: {classes[target_label]})")
    print(f"[+] Deletion AUC Score : {del_auc:.4f}  (Lower is Better)")
    print(f"[+] Insertion AUC Score: {ins_auc:.4f}  (Higher is Better)")
    print("="*50)

    os.makedirs('results', exist_ok=True)
    fig = plt.figure(figsize=(15, 8))

    checkpoints = ["0%", "20%", "50%", "80%", "100%"]
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(deletion_visuals[i])
        ax.set_title(f"Del: {checkpoints[i]} Masked", fontsize=10)
        ax.axis('off')

    for i in range(5):
        ax = plt.subplot(3, 5, i + 6)
        ax.imshow(insertion_visuals[i])
        ax.set_title(f"Ins: {checkpoints[i]} Restored", fontsize=10)
        ax.axis('off')

    ax_plot = plt.subplot(3, 1, 3)
    ax_plot.plot(fractions * 100, deletion_probs, marker='o', color='tab:red', 
                 linewidth=2.5, label=f'Deletion Curve (AUC = {del_auc:.3f} Del)')
    ax_plot.plot(fractions * 100, insertion_probs, marker='s', color='tab:blue', 
                 linewidth=2.5, label=f'Insertion Curve (AUC = {ins_auc:.3f} Ins)')
    
    ax_plot.set_xlabel("Perturbed Pixels Fraction (%)", fontsize=12, fontweight='bold')
    ax_plot.set_ylabel(f"P({classes[target_label]}) Confidence", fontsize=12, fontweight='bold')
    ax_plot.set_title(f"Quantitative Faithfulness Evaluation: Target '{classes[target_label]}'", fontsize=13)
    ax_plot.set_xlim([0, 100])
    ax_plot.set_ylim([0.0, 1.05])
    ax_plot.grid(True, linestyle='--', alpha=0.6)
    ax_plot.legend(loc='center right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    output_path = 'results/faithfulness_benchmark.png'
    plt.savefig(output_path, dpi=200)
    print(f"\n[+] Faithfulness verification curve saved to {output_path}")

if __name__ == '__main__':
    main()