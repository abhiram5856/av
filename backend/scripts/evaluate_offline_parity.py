import os
import json
import torch
import numpy as np
import onnxruntime as ort
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def get_pytorch_model(weights_path, num_classes=34):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image_path):
    # PyTorch production transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def evaluate_parity():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(repo_root, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    onnx_path = os.path.join(repo_root, "frontend", "public", "models", "mobilenetv3_small.onnx")
    
    # We will use the sample images if they exist, or create a random dummy image for testing parity
    # Let's create a test image
    test_img_path = os.path.join(repo_root, "backend", "test_parity.jpg")
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    img.save(test_img_path)
    
    print("Loading PyTorch model...")
    pt_model = get_pytorch_model(weights_path)
    
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    print("Preprocessing image...")
    input_batch = preprocess_image(test_img_path)
    
    # PyTorch inference
    with torch.no_grad():
        pt_output = pt_model(input_batch).numpy()
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_batch.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]
    
    # Compare
    diff = np.abs(pt_output - onnx_output).max()
    print(f"Max absolute difference between PyTorch and ONNX: {diff}")
    
    pt_top1 = np.argmax(pt_output, axis=1)[0]
    onnx_top1 = np.argmax(onnx_output, axis=1)[0]
    
    match = pt_top1 == onnx_top1
    print(f"PyTorch top-1: {pt_top1}")
    print(f"ONNX top-1: {onnx_top1}")
    print(f"Top-1 match: {match}")
    
    parity_result = {
        "model_version": "nova_mobilenet_v3_34_classes.pth",
        "export_path": onnx_path,
        "input_dimensions": "[1, 3, 224, 224]",
        "class_count": 34,
        "server_top1": int(pt_top1),
        "offline_top1": int(onnx_top1),
        "max_numerical_difference": float(diff),
        "parity_achieved": bool(match and diff < 1e-4)
    }
    
    out_dir = os.path.join(repo_root, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "offline_inference_parity.json"), "w") as f:
        json.dump(parity_result, f, indent=4)
        
    print("Parity evaluation saved to evaluation/offline_inference_parity.json")

if __name__ == "__main__":
    evaluate_parity()
