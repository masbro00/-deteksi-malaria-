import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Diagnosis Malaria AI",
    page_icon="🔬",
    layout="wide"
)

# ==========================================
# FUNGSI MEMUAT MODEL (Biar tidak loading terus)
# ==========================================
@st.cache_resource
def load_model():
    import os
    target_path = '02_Models/model_malaria_final.h5'
    
    # Jika file utuh (.h5) belum ada di server internet, satukan potongan file-nya
    if not os.path.exists(target_path):
        with open(target_path, 'wb') as file_utuh:
            for part_name in ['02_Models/model_part_aa', '02_Models/model_part_ab']:
                with open(part_name, 'rb') as file_potongan:
                    file_utuh.write(file_potongan.read())
                    
    return tf.keras.models.load_model(target_path)

model = load_model()
CLASS_NAMES = ['Parasitized (Terinfeksi)', 'Uninfected (Sehat)']
IMG_SIZE = (224, 224)

# ==========================================
# FUNGSI XAI (GRAD-CAM)
# ==========================================
def buat_gradcam(img_array, model, layer_name='out_relu'):
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):  
            base_model = layer
            break

    grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(layer_name).output, base_model.output]
    )

    rescale_layer = model.get_layer('normalisasi')
    gap_layer     = model.get_layer('global_avg_pool')
    drop_layer    = model.get_layer('dropout')
    dense_layer   = model.get_layer('output')

    with tf.GradientTape() as tape:
        x = rescale_layer(img_array)
        peta_fitur, base_out = grad_model(x)
        tape.watch(peta_fitur)
        x_gap  = gap_layer(base_out)
        x_drop = drop_layer(x_gap, training=False)
        prediksi = dense_layer(x_drop)
        nilai = prediksi[:, 0]

    gradien = tape.gradient(nilai, peta_fitur)
    bobot   = tf.reduce_mean(gradien, axis=(0, 1, 2))
    heatmap = peta_fitur[0] @ bobot[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def overlay_gradcam(img_asli_np, heatmap, alpha=0.4):
    h, w = img_asli_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed    = cv2.addWeighted(img_asli_np, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

# ==========================================
# TAMPILAN USER INTERFACE (UI)
# ==========================================
st.title("🔬 Sistem Cerdas Diagnosis Malaria")
st.markdown("""
Aplikasi ini menggunakan model **Deep Learning (MobileNetV2)** untuk mendeteksi parasit malaria pada citra mikroskopis sel darah merah. 
Dilengkapi dengan fitur **Explainable AI (Grad-CAM)** untuk menunjukkan area fokus infeksi.
""")
st.divider()

# Area Upload File
uploaded_file = st.file_uploader("Unggah gambar sel darah merah (Format: JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan layout 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Input")
        # Buka gambar menggunakan PIL
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
    with st.spinner("AI sedang menganalisis sel darah... ⏳"):
        # Preprocessing gambar
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_batch = tf.expand_dims(img_array, 0)
        
        # Prediksi
        skor = float(model.predict(img_batch, verbose=0)[0][0])
        prediksi_idx = 1 if skor >= 0.5 else 0
        prediksi_nama = CLASS_NAMES[prediksi_idx]
        keyakinan = skor if prediksi_idx == 1 else (1 - skor)
        
        # Buat Grad-CAM
        img_np = img_array.astype("uint8")
        heatmap = buat_gradcam(img_batch, model)
        hasil_xai = overlay_gradcam(img_np, heatmap)
        
    with col2:
        st.subheader("Hasil Analisis AI")
        
        # Tampilkan Status dengan warna menarik
        if prediksi_idx == 1: # Sehat
            st.success(f"**Status:** {prediksi_nama}")
            st.info(f"**Tingkat Keyakinan:** {keyakinan*100:.2f}%")
        else: # Sakit
            st.error(f"**Status:** {prediksi_nama}")
            st.warning(f"**Tingkat Keyakinan:** {keyakinan*100:.2f}%")
            
        st.markdown("**Visualisasi XAI (Grad-CAM):**")
        st.image(hasil_xai, use_column_width=True, caption="Area berwarna terang adalah fokus deteksi AI")