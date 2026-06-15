import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

DATASET_PATH = r"C:\Users\Dinesh\Desktop\crop disease project\plantvillage dataset\color"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("📂 Loading dataset...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Save class names
class_names = list(train_data.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print(f"✅ Total classes found: {len(class_names)}")

# Build model
NUM_CLASSES = len(class_names)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Single Phase Training
print("🚀 Training (single phase)...")
model.fit(train_data, validation_data=val_data, epochs=10)

# Save
model.save("crop_disease_model.keras")
print("✅ Model saved: crop_disease_model.keras")
print("✅ class_names.json saved")
print("🎉 Training complete!")
