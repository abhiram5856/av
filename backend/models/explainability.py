import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Explainable AI (XAI) module using Grad-CAM.
    Generates a heatmap highlighting the regions of the image that most heavily 
    influenced the model's classification decision.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t the output of the layer
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates the Grad-CAM heatmap for the given input.
        """
        self.model.eval()
        
        # Forward pass
        model_output = self.model(input_tensor)
        
        if class_idx is None:
            # If no target class specified, use the model's top prediction
            class_idx = torch.argmax(model_output, dim=1).item()
            
        self.model.zero_grad()
        
        # Target for backprop
        target = model_output[0][class_idx]
        target.backward(retain_graph=True)
        
        # Get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        activations = self.activations.detach()[0]
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1 range
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy(), model_output, class_idx

    def calculate_lesion_area_ratio(self, heatmap: np.ndarray, threshold: float = 0.6) -> float:
        """
        Calculates what percentage of the leaf is covered by the lesion/disease.
        This directly feeds into the SeverityScoringEngine.
        
        Args:
            heatmap: Normalized 2D numpy array (0.0 to 1.0)
            threshold: The heatmap intensity threshold to be considered a 'lesion'
        Returns:
            Float between 0.0 and 1.0 representing the affected area ratio.
        """
        # Count pixels in the heatmap that are above the threshold
        affected_pixels = np.sum(heatmap > threshold)
        total_pixels = heatmap.size
        
        ratio = affected_pixels / total_pixels
        return min(ratio, 1.0)

    def check_background_attention(self, heatmap: np.ndarray, original_img: np.ndarray, threshold: float = 0.6) -> float:
        """
        Calculates what percentage of the model's high attention falls OUTSIDE the leaf area.
        This helps detect if the model is 'cheating' by looking at the background (e.g., soil/hands).
        
        Args:
            heatmap: Normalized 2D numpy array (0.0 to 1.0)
            original_img: Original RGB image as numpy array (H, W, 3)
            threshold: The heatmap intensity threshold to be considered high attention
        Returns:
            Float between 0.0 and 1.0 representing the ratio of attention on the background.
        """
        if heatmap.size == 0 or original_img.size == 0:
            return 0.0

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        attention_mask = heatmap_resized > threshold
        
        # Convert RGB to HSV to find green (leaf) regions
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        
        # Define color range for leaves (green/yellow-green/brownish-green)
        # This is a broad range to capture various leaf states, including some necrotic tissue
        lower_leaf = np.array([15, 30, 30])
        upper_leaf = np.array([95, 255, 255])
        
        leaf_mask = cv2.inRange(hsv_img, lower_leaf, upper_leaf)
        
        # Dilate leaf mask slightly to account for edges and lesions that might have lost color
        kernel = np.ones((15, 15), np.uint8)
        leaf_mask = cv2.dilate(leaf_mask, kernel, iterations=2)
        
        # Background is everything not in the leaf mask
        background_mask = cv2.bitwise_not(leaf_mask)
        
        total_attention_pixels = np.sum(attention_mask)
        if total_attention_pixels == 0:
            return 0.0
            
        # Find attention pixels that are in the background
        background_attention = np.logical_and(attention_mask, background_mask > 0)
        background_attention_pixels = np.sum(background_attention)
        
        return float(background_attention_pixels / total_attention_pixels)

    @staticmethod
    def overlay_heatmap(img_path: str, heatmap: np.ndarray, output_path: str = None):
        """
        Utility to overlay the generated heatmap onto the original image.
        Used for sending visual proofs back to the frontend dashboard.
        """
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB heatmap (JET colormap)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on the original image (40% opacity)
        superimposed_img = heatmap_color * 0.4 + img * 0.6
        
        if output_path:
            cv2.imwrite(output_path, superimposed_img)
            
        return superimposed_img

# --- Integration Example ---
# If you are using the MobileNetV3 model from vision_training.py:
# 
# from vision_training import build_model
# model = build_model(num_classes=35)
# target_layer = model.features[-1] # The last conv block in MobileNetV3
# cam = GradCAM(model, target_layer)
# heatmap, output, class_idx = cam.generate_heatmap(input_tensor)
# lesion_ratio = cam.calculate_lesion_area_ratio(heatmap)
#
# Then feed `lesion_ratio` into your SeverityScoringEngine!
