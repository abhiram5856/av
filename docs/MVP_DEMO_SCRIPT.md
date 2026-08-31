# AgriVision AI MVP - Demo Script

## Welcome & Introduction (1 min)
"Hello! Welcome to AgriVision AI, our AI-powered precision farming assistant. 
Our goal is to turn complex agricultural AI into simple, understandable, and actionable insights to empower farmers."

## 1. The Core Problem (1 min)
"Farmers often face crop diseases but lack immediate access to agronomists. Existing AI apps often act as black boxes—giving an answer without explaining *why*, or they ignore the local environment entirely."

## 2. Live Demo: Disease Diagnosis (2 mins)
* **Action:** Open the Next.js frontend on a mobile view. Show the multilingual selector.
* **Action:** Navigate to "Check Your Crop" and upload an image of a diseased Tomato leaf.
* **Talking Points:**
  - "The image is processed by our Vision Model on the FastAPI backend."
  - "Instead of a black box, we generate a Grad-CAM overlay to show the farmer exactly *where* the model sees the disease."
  - "We also pull live weather data to form our **Evidence Consistency Engine**."

## 3. Evidence Consistency Mismatch (1 min)
* **Action:** Highlight the "Evidence Mismatch" warning in the UI (if applicable).
* **Talking Points:**
  - "If the visual symptoms say 'fungus' but the weather is completely dry, the system warns the farmer about an Evidence Mismatch and lowers the confidence. This prevents over-reliance on the AI."

## 4. IoT & Predictive Risk (1 min)
* **Action:** Navigate to the "Hardware" dashboard.
* **Talking Points:**
  - "We integrate with IoT sensors. The dashboard shows a 24-hour telemetry graph."
  - "The system calculates predictive risks—like Fungal Risk or Irrigation Need—based on real-time soil moisture and humidity."

## 5. GenAI Assistant (1 min)
* **Action:** Open the Chatbot UI. Type: "Explain my diagnosis in simple terms."
* **Talking Points:**
  - "Finally, our GenAI assistant is context-aware. It knows the current diagnosis and the IoT data, and can explain the treatment plan in English, Telugu, or Hindi using RAG over our agricultural knowledge base."

## Conclusion
"AgriVision AI bridges the gap between state-of-the-art multimodal AI and practical, farmer-first design. Thank you!"
