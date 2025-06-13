from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load model saat aplikasi dijalankan
model = load_model("model.keras")

# Preprocessing helper - sesuaikan dengan model Anda
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # "L" jika grayscale
    image = image.resize((28, 28))  # Sesuaikan dengan ukuran input model Anda
    image_array = np.array(image) / 255.0  # Normalisasi jika diperlukan
    image_array = image_array.reshape(1, 28, 28, 1)  # Tambah batch & channel dimensi
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        input_image = preprocess_image(contents)
        prediction = model.predict(input_image)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
