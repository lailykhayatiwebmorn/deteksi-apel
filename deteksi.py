import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# 1. Konfigurasi halaman
# -------------------------------
st.set_page_config(page_title="Deteksi Apel", layout="centered")
st.title("🍎 Deteksi Buah Apel")
st.markdown("Upload gambar buah untuk mengetahui apakah itu apel atau bukan.")

# -------------------------------
# 2. Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_apel.keras")
    return model

model = load_model()
class_names = ['apel', 'bukan_apel']

# -------------------------------
# 3. Upload gambar
# -------------------------------
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # -------------------------------
    # 4. Preprocessing gambar
    # -------------------------------
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # 5. Prediksi
    # -------------------------------
    prediction = model.predict(img_tensor)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index] * 100
    label = class_names[pred_index]

    # -------------------------------
    # 6. Tampilkan hasil prediksi
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 Hasil Prediksi:")

    if label == 'apel':
        st.success(f"🍎 **Terdeteksi Apel** dengan probabilitas **{confidence:.2f}%**")
    else:
        st.error(f"🚫 **Bukan Apel** (probabilitas: {confidence:.2f}%)")

    # Tampilkan semua probabilitas (opsional)
    st.markdown("### Probabilitas Kelas:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name.capitalize()}: **{prediction[i]*100:.2f}%**")
