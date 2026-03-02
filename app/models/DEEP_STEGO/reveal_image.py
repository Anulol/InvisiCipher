import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import imageio
from pathlib import Path
from app.models.DEEP_STEGO.Utils.preprocessing import normalize_batch, denormalize_batch


def reveal_image(stego_image_filepath):
    app_root = Path(__file__).resolve().parents[2]
    model_path = app_root / "models" / "DEEP_STEGO" / "models" / "reveal.h5"
    output_path = app_root / "secret_out.png"

    model = load_model(model_path, compile=False)

    stego_image = Image.open(stego_image_filepath).convert('RGB')

    # Resize the image to 224px*224px
    if stego_image.size != (224, 224):
        stego_image = stego_image.resize((224, 224))
        print("stego_image was resized to 224px * 224px")

    stego_image = np.array(stego_image).reshape(1, 224, 224, 3) / 255.0

    secret_image_out = model.predict([normalize_batch(stego_image)])

    secret_image_out = denormalize_batch(secret_image_out)
    secret_image_out = np.squeeze(secret_image_out) * 255.0
    secret_image_out = np.uint8(secret_image_out)

    imageio.imsave(output_path, secret_image_out)
    print(f"Saved revealed image to {output_path}")

    return




