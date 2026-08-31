# NOVA User Guide

This guide outlines how users, agronomists, and researchers can interact with the NOVA platform dashboard interfaces.

---

## 📸 Crop Disease Diagnosis Screen

1. **Upload Leaf**: Drag and drop a leaf image (e.g., JPEG/PNG format) or click to browse files.
2. **Configure Environmental Parameters**: Supply temperature, humidity, and soil pH parameters to allow accurate severity and climate risk evaluations.
3. **Run AI Analysis**: Click **Run AI Analysis**.
4. **View Diagnostics**:
   * Review the identified disease and confidence score percentage.
   * Toggle the **Grad-CAM visual heatmap overlay** directly over the leaf image to inspect where the vision network directed its focus.
   * View climate stress levels and risk indicators.
5. **Download PDF Report**: Click **Download PDF Report** to save a local document compiled with all vision parameters, GradCAM visuals, and recommendations.

---

## 💬 AI RAG Assistant Chat

1. Navigate to the **AI Assistant** screen in the dashboard sidebar.
2. Ask questions about crop care, pesticide guidelines, or diagnostic recoveries (e.g., *"How do I treat tomato late blight?"*).
3. The chatbot will retrieve context from the FAISS database (built from official ICAR agronomist guidelines) and output detailed treatment instructions.
