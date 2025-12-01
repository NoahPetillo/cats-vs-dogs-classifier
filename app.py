from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import CLASS_NAMES, load_model, predict_image


st.set_page_config(
    page_title="Cats vs Dogs CNN Story",
    page_icon="🐾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "Figures"


@st.cache_data(show_spinner=False)
def load_image(image_name: str):
    image_path = FIGURES_DIR / image_name
    if not image_path.exists():
        st.warning(f"Missing figure: {image_name}")
        return None
    return Image.open(image_path)


@st.cache_resource(show_spinner=False)
def ensure_model_loaded():
    """Warm up the cached model so prediction stays true on Streamlit Cloud."""
    return load_model()


st.title("The Process")
st.write(
    "A walkthrough of how I tuned a convolutional neural network for the Kaggle Cats vs. Dogs "
    "dataset—hyperparameters, mistakes, breakthroughs, and a look at the final learned filters."
)

with st.sidebar:
    st.header("Final Snapshot")
    st.metric("Accuracy", "92.54%")
    st.metric("Final Train Loss", "0.0831")
    st.metric("Final Val Loss", "8.586")
 
    


st.header("Hyperparameters & Loss Journey")
st.markdown(
    """
| Model | Key Changes | Outcome |
| --- | --- | --- |
| model1.pth | lr 0.001, no momentum, ran on CPU |Final train_loss~0.58 |
| model2.pth | lr 0.01, momentum 0.09, switched to MPS |Final train_loss ~0.48 |
| model3.pth | 4 conv blocks, 10 epochs, momentum 0.9 | train_loss routinely 0.07–0.30 |
| Train/Val Split Re-run | 80/20 split, same arch as model3 | Val loss 0.2665, Val acc 0.8930 |
| model4.pth | Batch size 48, 15 epochs | Train loss 0.146, val loss stuck at ~0.3|
| model4.pth (50 epochs) | Extended to 50 epochs | Overfitting by epoch 28, 50 is too much(train 0.047 vs val 0.3059) |
| model4.pth tweaks | Removed 4th conv, dropout 0.3, weight decay 0.0001 | Still overfitting |
| model5.pth | 25 epochs | Train 0.1843, Val 0.3006 |
| model6.pth | Reduced rotation aug to 7.5° | Train 0.0876, Val 0.3924 |
| model7.pth | Restored 4th conv, rotation 10°, 25 epochs | Train 0.0831, Val 0.2234, Accuracy 0.9694 |
| model8.pth | Realized there was a data leakage of train data into test data | Real accuracy: 0.9254
"""
)
st.caption("NOTE: model8 is functionally model 7, I only had to retest accuracy after fixing data leakage, as the training process was okay.")

st.success(
    "Lesson learned: the 50-epoch attemps drove the training loss down relentlessly but "
    "validation loss wobbled upward noisily-clearly overtraining. Settling on 25 epochs with the 4th "
    "convolutional layer, rotation=10°, dropout=0.3, and weight decay=0.0005 struck the right balance."
    "NOTE - most likely this could be fine tuned further, but the fourth convolutional layer paired with"
    "less epochs made the model signifigantly more effecient and accurate"
)


st.header("Loss Curves Across Models")
loss_figures = [
    ("model_3_loss_vs_epoch.png", "Model 3 — early boost from deeper feature extractor"),
    ("model4_loss_vs_epoch.png", "Model 4 — batch size 48, 15 epochs"),
    ("model5_Loss_vs_Epoch.png", "Model 5 — 50 epochs, but around epoch 15 val_loss stopped improving and became noisy"),
    ("model7_Loss_vs_Epoch.png", "Model 7 — final curve. Still a difference but val was much steadier better"),
]

cols = st.columns(2)
for idx, (filename, caption) in enumerate(loss_figures):
    img = load_image(filename)
    col = cols[idx % 2]
    if img:
        col.image(img, caption=caption, use_container_width=True)


st.header("Final Architecture Overview")
structure_img = load_image("cnn_structure_graph.png")
if structure_img:
    center_col = st.columns([1, 2, 1])[1]
    center_col.image(
        structure_img,
        caption="Final CNN structure — four conv blocks feeding dense layers. Sorry it is blurry, this is the way pytorch loads it",
        use_container_width=True,
    )


st.header("What Each Convolution Learned")
st.write(
    "Each block below highlights the aggregated filters (feature maps) that represent what each "
    "convolutional stage is sensitive to. The filters grow from edge detectors up to pet-specific shapes."
)

conv_filters = [
    ("Conv Layer 1 Filters", "conv1_filters.png"),
    ("Conv Layer 2 Filters", "conv2_filters.png"),
    ("Conv Layer 3 Filters", "conv3_filters.png"),
    ("Conv Layer 4 Filters", "conv4_filters.png"),
]

for title, filter_file in conv_filters:
    st.subheader(title)
    filter_img = load_image(filter_file)
    if filter_img:
        st.image(filter_img, caption="Aggregated feature maps", use_container_width=True)


st.caption(
    "Every figure above comes directly from the training logs and visualizations created while "
    "iterating on the cats vs. dogs classifier."
)



st.header("Try the Final Model Yourself")
st.caption("NOTE: Model works best with close up images of the animals face. It is much more likely to fail"
           "If there is a lot of noise or if the face is not visible")
uploaded = st.file_uploader("Upload a cat or dog photo", type=["png", "jpg", "jpeg"])
if uploaded:
    preview = Image.open(uploaded).convert("RGB")
    st.image(preview, caption="Your upload", use_container_width=True)
    with st.spinner("Running inference with model7.pth..."):
        ensure_model_loaded()
        label, confidence, prob_map = predict_image(preview)
    st.success(f"Prediction: **{label}** with {confidence * 100:.2f}% confidence.")
    st.write(
        {
            CLASS_NAMES[0]: f"{prob_map[CLASS_NAMES[0]] * 100:.2f}%",
            CLASS_NAMES[1]: f"{prob_map[CLASS_NAMES[1]] * 100:.2f}%",
        }
    )
else:
    st.info("Drop in a JPG or PNG to see what the trained network thinks.")
