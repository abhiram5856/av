import time
import torch
import numpy as np
from PIL import Image
from backend.models.vision_training import build_model
from backend.models.explainability import GradCAM

def benchmark_inference():
    print("=========================================")
    print("NOVA ML Pipeline Validation & Benchmark")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")
    
    # 1. Cold Start Benchmark (Model Instantiation & Weight Loading)
    start_time = time.time()
    model = build_model(35).to(device)
    cold_start_time = time.time() - start_time
    print(f"Cold Start latency: {cold_start_time:.4f}s")
    
    # Check GPU VRAM allocation if CUDA is enabled
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM Allocation: {allocated:.2f} MB")
    
    # 2. Mock Image Creation
    fake_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(fake_img).unsqueeze(0).to(device)
    
    # 3. Warm Inference Benchmark (Average latency over 20 iterations)
    warm_latencies = []
    model.eval()
    
    # Warmup pass
    with torch.no_grad():
        _ = model(tensor)
        
    for i in range(20):
        t0 = time.time()
        with torch.no_grad():
            _ = model(tensor)
        warm_latencies.append(time.time() - t0)
        
    avg_latency = np.mean(warm_latencies)
    print(f"Average Warm Inference Latency: {avg_latency * 1000:.2f} ms")
    
    # 4. GradCAM Validation
    print("Validating GradCAM Heatmap generation...")
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    
    t0 = time.time()
    heatmap, _, _ = cam.generate_heatmap(tensor, class_idx=0)
    cam_latency = time.time() - t0
    print(f"GradCAM Generation Latency: {cam_latency * 1000:.2f} ms")
    
    coverage = cam.calculate_lesion_area_ratio(heatmap)
    print(f"Calculated Lesion Coverage Ratio: {coverage:.4f}")
    
    # Cleanup memory
    del model
    del cam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup executed successfully.")

if __name__ == "__main__":
    benchmark_inference()
