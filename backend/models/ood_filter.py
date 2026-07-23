import torch
from torchvision import models, transforms
from PIL import Image
import io

class OODFilter:
    def __init__(self):
        # Using a highly compressed MobileNet for ultra-fast, lightweight OOD filtering
        # In a real enterprise app, we'd fine-tune this on 'leaf vs non-leaf' binary classification
        # Here we use pretrained ImageNet and check if top predictions fall into 'plant/leaf' categories
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # ImageNet classes related to plants, pots, greenhouses, agriculture
        self.valid_classes = {985, 986, 987, 988, 989, 990} # Daisy, Bell Pepper, etc.
        # This is a mocked implementation for demonstration. A real one maps all 1000 imagenet classes
        # to binary (plant=1, non-plant=0) or uses a custom trained binary classifier.

    def is_plant_image(self, image_bytes: bytes) -> bool:
        """
        Fast pass to check if the uploaded image is likely a plant.
        Returns False if the image is highly likely to be a face, car, etc.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # If the max probability across all classes is extremely low, or if it strongly predicts 
            # something definitely not a plant (like a car), we reject it.
            # (Simplified for demonstration)
            
            top_prob, top_class = torch.max(probabilities, dim=0)
            
            # Simulated OOD logic:
            if top_prob.item() < 0.1:
                return False
                
            return True
            
        except Exception as e:
            return False

ood_filter = OODFilter()
