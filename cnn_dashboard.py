from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from cnn_train import CNN


st.set_page_config(
    page_title="Cats vs Dogs CNN Story",
    page_icon="🐾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "Figures"
MODEL_PATH = BASE_DIR / "models" / "model7.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ["Cat", "Dog"]
PRED_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@st.cache_data(show_spinner=False)
def load_image(image_name: str):
    image_path = FIGURES_DIR / image_name
    if not image_path.exists():
        st.warning(f"Missing figure: {image_name}")
        return None
    return Image.open(image_path)


@st.cache_resource(show_spinner=False)
def load_model():
    model = CNN()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(img: Image.Image):
    model = load_model()
    tensor = PRED_TRANSFORMS(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs[pred_idx].item(), probs.cpu().tolist()


st.title("Where This CNN Landed")
st.write(
    "A walkthrough of how I tuned a convolutional neural network for the Kaggle Cats vs. Dogs "
    "dataset—hyperparameters, mistakes, breakthroughs, and a look at the final learned filters."
)

with st.sidebar:
    st.header("Final Snapshot")
    st.metric("Accuracy", "96.94%")
    st.metric("Final Train Loss", "0.0831")
    st.metric("Final Val Loss", "8.586")
 
    


st.header("Hyperparameters & Loss Journey")
st.markdown(
    """
| Model | Key Changes | Outcome |
| --- | --- | --- |
| model1.pth | lr 0.001, no momentum, CPU | Loss ~0.58 |
| model2.pth | lr 0.01, momentum 0.09, switched to MPS | Loss ~0.48 |
| model3.pth | 4 conv blocks, 10 epochs, momentum 0.9 | Loss routinely 0.07–0.30 |
| Train/Val Split Re-run | 80/20 split, same arch as model3 | Val loss 0.2665, Val acc 0.8930 |
| model4.pth | Batch size 48, 15 epochs | Train loss 0.146 |
| model4.pth (50 epochs) | Extended to 50 epochs | Overfitting by epoch 28 (train 0.047 vs val 0.3059) |
| model4.pth tweaks | Removed 4th conv, dropout 0.3, weight decay 0.0001 | Still overfitting |
| model5.pth | 25 epochs | Train 0.1843, Val 0.3006 |
| model6.pth | Reduced rotation aug to 7.5° | Train 0.0876, Val 0.3924 |
| model7.pth | Restored 4th conv, rotation 10°, 25 epochs | Train 0.0831, Val 0.2234, Accuracy 0.9694 |
"""
)
st.caption("Note, these val_losses are averages from the total training process. \nEx. val_loss of 0.2234 at end correlated with val loss of 8.586 when not calculated with each training epoch"
           )

st.success(
    "Lesson learned: the 50-epoch experiment drove the training loss down relentlessly but "
    "validation loss wobbled upward—classic overtraining. Settling on 25 epochs with the 4th "
    "convolutional layer, rotation=10°, dropout=0.3, and weight decay=0.0005 struck the right balance."
)


st.header("Loss Curves Across Experiments")
loss_figures = [
    ("model_3_loss_vs_epoch.png", "Model 3 — early boost from deeper feature extractor"),
    ("model4_loss_vs_epoch.png", "Model 4 — batch size 48, highlights where 50 epochs began to overfit"),
    ("model5_Loss_vs_Epoch.png", "Model 5 — conservative 25 epochs, but still higher validation loss"),
    ("model7_Loss_vs_Epoch.png", "Model 7 — final curve with tighter tracking between train/val"),
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
        caption="Final CNN structure — four conv blocks feeding dense layers.",
        use_container_width=True,
    )


st.header("What Each Convolution Learned")
st.write(
    "Each block below shows the raw learned kernels (left) and the resulting filters/feature maps "
    "aggregated across channels (right). The progression highlights how edges become textures and "
    "eventually pet-specific shapes."
)

conv_sections = [
    ("Conv Layer 1", "Conv1_kernels.png", "conv1_filters.png"),
    ("Conv Layer 2", "conv2_kernels.png", "conv2_filters_averaged.png"),
    ("Conv Layer 3", "conv3_kernels.png", "conv3_filters_averaged.png"),
    ("Conv Layer 4", "conv4_kernels.png", "conv4_filters_averaged.png"),
]

for title, kernel_file, filter_file in conv_sections:
    st.subheader(title)
    kernel_img = load_image(kernel_file)
    filter_img = load_image(filter_file)
    two_cols = st.columns(2)
    if kernel_img:
        two_cols[0].image(kernel_img, caption="Kernels", use_container_width=True)
    if filter_img:
        two_cols[1].image(filter_img, caption="Filters / feature maps", use_container_width=True)


st.caption(
    "Every figure above comes directly from the training logs and visualizations created while "
    "iterating on the cats vs. dogs classifier."
)


st.header("Try the Final Model Yourself")
uploaded = st.file_uploader("Upload a cat or dog photo", type=["png", "jpg", "jpeg"])
if uploaded:
    preview = Image.open(uploaded).convert("RGB")
    st.image(preview, caption="Your upload", use_container_width=True)
    with st.spinner("Running inference with model7.pth..."):
        label, confidence, all_probs = predict_image(preview)
    st.success(f"Prediction: **{label}** with {confidence * 100:.2f}% confidence.")
    st.write(
        {
            CLASS_NAMES[0]: f"{all_probs[0] * 100:.2f}%",
            CLASS_NAMES[1]: f"{all_probs[1] * 100:.2f}%",
        }
    )
else:
    st.info("Drop in a JPG or PNG to see what the trained network thinks.")
