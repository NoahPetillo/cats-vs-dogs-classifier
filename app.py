import streamlit as st
from PIL import Image

from src.inference import CLASS_NAMES, predict_image


st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾", layout="centered")

st.title("Cats vs. Dogs Classifier")
st.write(
    "Upload any cat or dog photo and the fine-tuned CNN (model7) will predict the class "
    "along with confidence scores. The network uses four convolutional blocks, dropout, "
    "and ImageNet normalization to stay stable even on Streamlit Cloud."
)

uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    with st.spinner("Running inference on CPU..."):
        label, confidence, prob_map = predict_image(image)

    st.success(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
    st.write("Class probabilities:")
    cols = st.columns(len(CLASS_NAMES))
    for col, class_name in zip(cols, CLASS_NAMES):
        col.metric(class_name, f"{prob_map[class_name] * 100:.2f}%")
else:
    st.info(
        "Drag and drop an image to begin. If you just want to explore, save any cat/dog image "
        "locally and upload it here."
    )

