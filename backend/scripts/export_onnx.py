import torch
import torchvision.models as models
import os

def export_to_onnx():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(repo_root, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    
    # 34 classes based on the dataset structure used in AgriVision-AI
    num_classes = 34
    
    print(f"Loading weights from {weights_path}")
    
    # Load MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Create dummy input: Batch size 1, 3 channels, 224x224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    export_path = os.path.join(repo_root, "frontend", "public", "models", "mobilenetv3_small.onnx")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    print(f"Exporting ONNX model to {export_path}")
    
    # Export the model
    torch.onnx.export(
        model,                       
        dummy_input,                 
        export_path,                 
        export_params=True,          
        opset_version=14,            
        do_constant_folding=True,    
        input_names=['input'],       
        output_names=['output'],     
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    export_to_onnx()
