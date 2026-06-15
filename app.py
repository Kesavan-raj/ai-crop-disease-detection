import os
import json
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Load class names
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# Load model
print("Loading model...")
try:
    MODEL = tf.keras.models.load_model("models/crop_disease_model.keras")
    print(f"✅ Model loaded! Classes: {len(CLASS_NAMES)}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    MODEL = None

# Disease treatment & fertilizer database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "treatment": "Apply fungicides containing captan or myclobutanil. Remove infected leaves.",
        "fertilizer": "Use balanced NPK fertilizer (10-10-10). Avoid excess nitrogen."
    },
    "Apple___Black_rot": {
        "treatment": "Prune infected branches. Apply copper-based fungicide.",
        "fertilizer": "Apply potassium-rich fertilizer to boost immunity."
    },
    "Apple___Cedar_apple_rust": {
        "treatment": "Apply myclobutanil or mancozeb fungicide in spring.",
        "fertilizer": "Use NPK 12-12-17 with micronutrients."
    },
    "Apple___healthy": {
        "treatment": "No treatment needed. Maintain regular care.",
        "fertilizer": "Apply balanced NPK (10-10-10) every season."
    },
    "Banana___Black_Sigatoka": {
        "treatment": "Apply propiconazole or chlorothalonil fungicide. Remove infected leaves.",
        "fertilizer": "Apply potassium and nitrogen-rich fertilizer."
    },
    "Banana___Bract_Mosaic_Virus": {
        "treatment": "No cure. Remove and destroy infected plants. Control aphid vectors.",
        "fertilizer": "Use balanced fertilizer to keep plants strong."
    },
    "Banana___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 8-10-36 for healthy growth."
    },
    "Banana___Insect_Pest": {
        "treatment": "Use neem oil spray or insecticides like chlorpyrifos.",
        "fertilizer": "Apply nitrogen-rich fertilizer after pest control."
    },
    "Banana___Moko_Disease": {
        "treatment": "No cure. Remove infected plants immediately. Disinfect tools.",
        "fertilizer": "Avoid excess nitrogen. Use balanced NPK."
    },
    "Banana___Panama_Disease": {
        "treatment": "No chemical cure. Use resistant varieties. Improve soil drainage.",
        "fertilizer": "Apply lime to adjust soil pH. Use potassium fertilizer."
    },
    "Banana___Yellow_Sigatoka": {
        "treatment": "Apply mancozeb or propiconazole fungicide.",
        "fertilizer": "Use potassium sulfate fertilizer."
    },
    "Blueberry___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Use acidic fertilizer (ammonium sulfate). pH 4.5-5.5."
    },
    "Cherry_(including_sour)___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply balanced NPK (10-10-10) in spring."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "treatment": "Apply sulfur-based fungicide or potassium bicarbonate.",
        "fertilizer": "Avoid excess nitrogen. Use balanced fertilizer."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "treatment": "Apply strobilurin fungicides. Rotate crops.",
        "fertilizer": "Apply nitrogen fertilizer in split doses."
    },
    "Corn_(maize)___Common_rust_": {
        "treatment": "Apply mancozeb or azoxystrobin fungicide early.",
        "fertilizer": "Use balanced NPK with potassium for resistance."
    },
    "Corn_(maize)___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply urea (nitrogen) and DAP for healthy growth."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "treatment": "Apply propiconazole fungicide. Use resistant hybrids.",
        "fertilizer": "Apply nitrogen in split applications."
    },
    "Cotton___Diseased_Leaf": {
        "treatment": "Apply appropriate fungicide/insecticide based on diagnosis.",
        "fertilizer": "Use NPK 20-10-10 with micronutrients."
    },
    "Cotton___Diseased_Plant": {
        "treatment": "Remove severely infected plants. Apply systemic fungicide.",
        "fertilizer": "Apply potassium to boost plant immunity."
    },
    "Cotton___Healthy_Leaf": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 20-10-10 for healthy cotton growth."
    },
    "Cotton___Healthy_Plant": {
        "treatment": "No treatment needed.",
        "fertilizer": "Use urea and DAP with potassium sulfate."
    },
    "Grape___Black_rot": {
        "treatment": "Apply myclobutanil or captan fungicide. Remove mummified fruit.",
        "fertilizer": "Use balanced NPK with magnesium."
    },
    "Grape___Esca_(Black_Measles)": {
        "treatment": "Prune infected wood. Apply sodium arsenite (where legal).",
        "fertilizer": "Apply potassium and phosphorus fertilizer."
    },
    "Grape___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 10-5-20 for grapes."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "treatment": "Apply copper-based fungicide. Improve air circulation.",
        "fertilizer": "Use balanced NPK with calcium."
    },
    "Mango___Anthracnose": {
        "treatment": "Apply copper oxychloride or mancozeb fungicide.",
        "fertilizer": "Apply potassium and phosphorus before flowering."
    },
    "Mango___Bacterial_Canker": {
        "treatment": "Apply copper-based bactericide. Prune infected parts.",
        "fertilizer": "Avoid excess nitrogen. Use balanced NPK."
    },
    "Mango___Cutting_Weevil": {
        "treatment": "Apply chlorpyrifos or carbaryl insecticide.",
        "fertilizer": "Use NPK with zinc micronutrient."
    },
    "Mango___Die_Back": {
        "treatment": "Prune dead wood. Apply copper fungicide on cut ends.",
        "fertilizer": "Apply balanced NPK with boron."
    },
    "Mango___Gall_Midge": {
        "treatment": "Apply systemic insecticide during bud break.",
        "fertilizer": "Use potassium-rich fertilizer for resistance."
    },
    "Mango___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 10-26-26 during fruiting season."
    },
    "Mango___Powdery_Mildew": {
        "treatment": "Apply sulfur or wettable sulfur fungicide.",
        "fertilizer": "Avoid excess nitrogen. Use potassium fertilizer."
    },
    "Mango___Sooty_Mould": {
        "treatment": "Control scale insects with insecticide. Wash leaves with water.",
        "fertilizer": "Apply balanced NPK fertilizer."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "treatment": "No cure. Remove infected trees. Control psyllid insects.",
        "fertilizer": "Apply micronutrient fertilizer (zinc, manganese, iron)."
    },
    "Peach___Bacterial_spot": {
        "treatment": "Apply copper-based bactericide in spring.",
        "fertilizer": "Use balanced NPK. Avoid excess nitrogen."
    },
    "Peach___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 10-10-10 in early spring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "treatment": "Apply copper hydroxide bactericide. Avoid overhead watering.",
        "fertilizer": "Use calcium-rich fertilizer to strengthen cell walls."
    },
    "Pepper,_bell___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 5-10-10 for pepper plants."
    },
    "Potato___Early_blight": {
        "treatment": "Apply chlorothalonil or mancozeb fungicide.",
        "fertilizer": "Apply nitrogen in split doses. Use potassium fertilizer."
    },
    "Potato___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 12-12-17 for potatoes."
    },
    "Potato___Late_blight": {
        "treatment": "Apply metalaxyl or cymoxanil fungicide urgently.",
        "fertilizer": "Apply potassium to improve disease resistance."
    },
    "Raspberry___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply balanced NPK with magnesium."
    },
    "Soybean___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Use phosphorus and potassium fertilizer. Inoculate with Rhizobium."
    },
    "Squash___Powdery_mildew": {
        "treatment": "Apply potassium bicarbonate or neem oil spray.",
        "fertilizer": "Avoid excess nitrogen. Use balanced NPK."
    },
    "Strawberry___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 10-10-10 with calcium."
    },
    "Strawberry___Leaf_scorch": {
        "treatment": "Apply captan fungicide. Remove infected leaves.",
        "fertilizer": "Use balanced fertilizer with micronutrients."
    },
    "Tomato___Bacterial_spot": {
        "treatment": "Apply copper-based bactericide. Avoid wetting leaves.",
        "fertilizer": "Use calcium nitrate fertilizer."
    },
    "Tomato___Early_blight": {
        "treatment": "Apply chlorothalonil or mancozeb fungicide.",
        "fertilizer": "Apply balanced NPK with calcium and magnesium."
    },
    "Tomato___healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 8-32-16 during fruiting."
    },
    "Tomato___Late_blight": {
        "treatment": "Apply metalaxyl + mancozeb urgently. Remove infected plants.",
        "fertilizer": "Apply potassium fertilizer to boost resistance."
    },
    "Tomato___Leaf_Mold": {
        "treatment": "Apply chlorothalonil fungicide. Improve ventilation.",
        "fertilizer": "Avoid excess nitrogen. Use balanced NPK."
    },
    "Tomato___Septoria_leaf_spot": {
        "treatment": "Apply mancozeb or copper fungicide. Remove lower leaves.",
        "fertilizer": "Use balanced NPK with calcium."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "treatment": "Apply abamectin or neem oil. Spray water on undersides of leaves.",
        "fertilizer": "Use balanced fertilizer. Avoid drought stress."
    },
    "Tomato___Target_Spot": {
        "treatment": "Apply azoxystrobin or chlorothalonil fungicide.",
        "fertilizer": "Apply potassium and calcium fertilizer."
    },
    "Tomato___Tomato_mosaic_virus": {
        "treatment": "No cure. Remove infected plants. Disinfect tools.",
        "fertilizer": "Use balanced NPK to keep healthy plants strong."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "Control whitefly vectors with imidacloprid. Remove infected plants.",
        "fertilizer": "Apply balanced NPK with micronutrients."
    },
    "Dragon_Fruit___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 6-6-6 with iron and magnesium."
    },
    "Dragon_Fruit___Disease": {
        "treatment": "Apply copper-based fungicide. Remove infected stems.",
        "fertilizer": "Use balanced NPK. Avoid overwatering."
    },
    "Onion___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 10-20-10 during bulb formation."
    },
    "Onion___Disease": {
        "treatment": "Apply mancozeb or iprodione fungicide.",
        "fertilizer": "Use balanced NPK. Avoid excess nitrogen."
    },
    "Pineapple___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 6-6-6 with iron sulfate."
    },
    "Pineapple___Disease": {
        "treatment": "Apply copper-based fungicide. Improve drainage.",
        "fertilizer": "Use NPK with zinc and boron micronutrients."
    },
    "Watermelon___Healthy": {
        "treatment": "No treatment needed.",
        "fertilizer": "Apply NPK 5-10-10 during fruiting."
    },
    "Watermelon___Disease": {
        "treatment": "Apply chlorothalonil or mancozeb fungicide.",
        "fertilizer": "Use potassium and calcium fertilizer."
    }
}

DEFAULT_INFO = {
    "treatment": "Consult a local agricultural expert for specific treatment.",
    "fertilizer": "Apply balanced NPK fertilizer as general recommendation."
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template("result.html", error="Model not loaded")
    if "file" not in request.files:
        return render_template("result.html", error="No file uploaded")
    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", error="No file selected")
    try:
        image_bytes = file.read()
        input_arr = preprocess_image(image_bytes)
        predictions = MODEL.predict(input_arr)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        predicted_class = CLASS_NAMES[predicted_index]

        parts = predicted_class.split("___")
        plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        info = DISEASE_INFO.get(predicted_class, DEFAULT_INFO)

        return render_template("result.html",
            plant=plant,
            condition=condition,
            confidence=round(confidence, 2),
            treatment=info["treatment"],
            fertilizer=info["fertilizer"],
            raw_class=predicted_class
        )
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
