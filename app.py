import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="CNN Image Classifier", layout="centered")

# Fungsi memuat model (gunakan cache)
@st.cache_resource
def load_our_model(model_path="cifar10_cnn_model.keras"):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"Model tidak ditemukan atau gagal dimuat: {e}")
        return None

model = load_our_model()

# Fungsi preprocessing gambar
def preprocess_image_for_model(image: Image, target_size=(32, 32)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Header tampilan
st.markdown(
    """
    <h1 style='text-align: center; font-family: "Courier New", monospace; color: #4CAF50;'>
    Aplikasi Klasifikasi Gambar dengan CNN
    </h1><br>
    """,
    unsafe_allow_html=True
)

st.header("Klasifikasi Gambar CIFAR-10")
st.markdown(
    "Unggah gambar **Pesawat, Mobil, Burung, Kucing, Rusa, Anjing, Kodok, Kuda, Kapal, atau Truk** "
    "untuk diklasifikasikan menggunakan model CNN sederhana."
)
st.divider()

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    if model is None:
        st.error("⚠️ Model tidak tersedia. Jalankan training (lihat train_and_evaluate.py) dan pastikan file cifar10_cnn_model.keras ada di folder ini.")
    else:
        # Tombol prediksi
        if st.button("🔍 Prediksi Gambar"):
            with st.spinner("Model sedang menganalisis..."):
                # Preprocess dan prediksi
                input_array = preprocess_image_for_model(image)
                prediction = model.predict(input_array)
                prediction_array = prediction[0]
                pred_label = np.argmax(prediction_array)

                # Kelas CIFAR-10
                class_names = [
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ]

                # Tampilkan hasil utama
                st.success(f"### ✅ Prediksi: **{class_names[pred_label]}**")
                st.write(f"Confidence: `{np.max(prediction_array) * 100:.2f}%`")

                # Semua probabilitas
                st.markdown("---")
                st.subheader("Semua Probabilitas:")
                prob_df = pd.DataFrame({
                    "Kelas": class_names,
                    "Probabilitas": prediction_array
                })
                st.bar_chart(prob_df.set_index("Kelas"))
                st.dataframe(prob_df.sort_values(by="Probabilitas", ascending=False), hide_index=True)

                # 3 prediksi teratas
                st.markdown("---")
                st.subheader("3 Prediksi Teratas:")
                top_n_indices = np.argsort(prediction_array)[-3:][::-1]
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    with col:
                        index = top_n_indices[i]
                        st.metric(
                            label=f"#{i+1}: {class_names[index]}",
                            value=f"{prediction_array[index]*100:.2f}%"
                        )

# Footer
st.markdown(
    """
    <div style='text-align: center; margin-top: 40px;'>
    <h4 style='font-family: "Courier New", monospace; color: #4CAF50;'>
    Made with ❤️ by Group 1
    </h4>
    </div>
    """,
    unsafe_allow_html=True
)
