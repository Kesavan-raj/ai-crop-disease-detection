# 🌿 Crop Disease Detection

An AI-powered web application that detects crop diseases from leaf images using deep learning.

## 🚀 Features
- Detects diseases across 65 plant classes
- Built with MobileNetV2 (93.57% accuracy)
- Treatment and fertilizer recommendations
- Dark professional UI (black/green theme)

## 🛠️ Tech Stack
- **Backend:** Python, Flask, TensorFlow/Keras
- **Model:** MobileNetV2 (Transfer Learning)
- **Dataset:** PlantVillage + 8 additional fruit disease classes
- **Frontend:** HTML, CSS, JavaScript

## ⚙️ Installation

1. Clone the repository
   git clone https://github.com/Kesavan-raj/ai-crop-disease-detection.git
   cd ai-crop-disease-detection

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the app
   python app.py

5. Open browser and go to
   http://localhost:5000

## 📁 Project Structure
ai-crop-disease-detection/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── result.html
└── static/

## 👨‍💻 Author
Kesavan Raj