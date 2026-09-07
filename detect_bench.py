import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import urllib.request
import os

def download_sample_image(url, save_path):
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

def main():
    os.makedirs('results', exist_ok=True)
    
    # 1. Load lightweight YOLOv8 nano model (pre-trained on COCO)
    print("[*] Loading YOLOv8n backbone...")
    model = YOLO('yolov8n.pt')
    
    # 2. Download a high-quality road/traffic test image
    img_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    img_path = "results/input_sample.jpg"
    download_sample_image(img_url, img_path)
    
    # 3. Warm-up GPU/CPU
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy_img, verbose=False)
        
    # 4. Latency / FPS Benchmark (Iterate 30 times)
    print("[*] Benchmarking inference latency over 30 runs...")
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model(img_path, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms
        
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    print(f"\n[+] Benchmark Results:")
    print(f"    Average Latency: {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
    print(f"    Throughput (FPS): {fps:.2f} FPS\n")
    
    # 5. Run inference and plot bounding boxes
    results = model(img_path, verbose=False)
    res_bgr = results[0].plot() # BGR image with drawn bounding boxes
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    
    # 6. Save visualization with benchmark summary
    plt.figure(figsize=(10, 8))
    plt.imshow(res_rgb)
    plt.axis('off')
    title_text = f"YOLOv8n Object Detection\nLatency: {avg_latency:.2f}ms | Throughput: {fps:.1f} FPS"
    plt.title(title_text, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    
    output_path = "results/detection_sample.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Detection and benchmark visual saved to: {output_path}")

if __name__ == '__main__':
    main()