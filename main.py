import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import subprocess
import sys

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('detector_ferrugem.h5')
    return model

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None

st.title("Detector de Ferrugem 🔍🛠️")

if st.button("Iniciar um treinamento..."):
    subprocess.Popen([sys.executable, "training.py"])
    st.info("Treinamento iniciado no background! Por favor verifique seu terminal ou seus logs para acompanhar a produção do modelo. Carregue o modelo com o botão abaixo quando o treinamento for terminado!")
    
if not st.session_state.model_loaded:
    if st.button("Carregar modelo de detecção de ferrugem"):
        with st.spinner("Carregando o modelo..."):
            st.session_state.model = load_model()
        st.session_state.model_loaded = True
        st.success("Modelo carregado com sucesso! Por favor, forneça uma imagem.")

if st.session_state.model_loaded:
    st.write("Arraste ou selecione uma imagem para procurar ferrugem. Por favor, considere que as imagens precisam ser do tipo .jpg.")

    uploaded_file = st.file_uploader("Escolher uma imagem...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagem do Upload', use_container_width=True)
        
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)

        prediction = st.session_state.model.predict(img_array)[0][0]

        threshold = 0.5

        if prediction >= threshold:
            st.success(f"✅ Ferrugem **NÃO detectada** (Confiança: {prediction:.2f})")
        else:
            st.error(f"⚠️ Ferrugem **DETECTADA** (Confiança: {(1 - prediction):.2f})")
