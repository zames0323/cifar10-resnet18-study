import os
from ultralytics import YOLO

def main():
    # 1. Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # 2. Load pre-trained YOLOv8 Nano backbone
    print("[*] Loading YOLOv8n pre-trained weights...")
    model = YOLO('yolov8n.pt')
    
    # 3. Fine-tune on mini-dataset (coco128.yaml) for 10 epochs
    # Ultralytics automatically downloads coco128 dataset if not cached
    print("[*] Starting domain fine-tuning for 10 epochs...")
    train_results = model.train(
        data='coco128.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        project='results',
        name='yolo_finetune',
        exist_ok=True,
        verbose=True
    )
    
    # 4. Evaluate detection metrics on validation set
    print("\n[*] Evaluating fine-tuned model performance...")
    metrics = model.val(project='results', name='yolo_finetune_val', exist_ok=True)
    
    # 5. Extract bounding box mAP metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print("\n" + "=" * 50)
    print("[+] YOLO Fine-tuning Completed Successfully!")
    print(f"    Validation mAP@50:    {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"    Validation mAP@50-95: {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print("    Trained weights:      results/yolo_finetune/weights/best.pt")
    print("=" * 50)

if __name__ == '__main__':
    main()