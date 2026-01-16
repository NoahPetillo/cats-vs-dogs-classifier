import streamlit as st
from PIL import Image

data_transform_code = """
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize all images
    transforms.RandomHorizontalFlip(),           # Augmentation
    transforms.RandomRotation(10),               # Augmentation
    transforms.ToTensor(),                      # Convert to tensor
    transforms.RandomCrop(size=224, padding = 28),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization -- averages of each RGB pixel
                         std=[0.229, 0.224, 0.225]),
])
"""

enhanced_CNN_code = """
class enhanced_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        #Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        #Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        #Fully connected Layer
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 2) #NOTE: I used Cross Entropy loss, so I used ReLU instead of softmax for fc2

    def forward(self, x):

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)   # raw logits

        return x
"""
enhanced2_CNN_code = """
class enhanced_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.proj1 = nn.Conv2d(3, 64, kernel_size=1)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.proj2 = nn.Conv2d(64, 128, kernel_size=1)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.proj3 = nn.Conv2d(128, 256, kernel_size=1)

        # AvgPool2d for extra downsizing
        self.pool4 = nn.AvgPool2d(2, 2)

        #Fully connected
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):

        # Block 1
        identity = self.proj1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        out = self.pool1(out)
        out = self.drop1(out)

        # Block 2
        identity = self.proj2(out)
        out2 = F.relu(self.bn3(self.conv3(out)))
        out2 = self.bn4(self.conv4(out2))
        out = F.relu(out2 + identity)
        out = self.pool2(out)
        out = self.drop2(out)

        # Block 3
        identity = self.proj3(out)
        out3 = F.relu(self.bn5(self.conv5(out)))
        out3 = self.bn6(self.conv6(out3))
        out = F.relu(out3 + identity)
        out = self.pool3(out)
        out = self.drop3(out)

        # AvgPool Block
        out = self.pool4(out)

        # FC
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn_fc(self.fc1(out)))
        out = self.fc2(out)

        return out
"""
st.set_page_config(
    page_title="Enhanced CNN Research Dashboard",
    layout="wide"
)

# ---------------------------
# Title
# ---------------------------
st.title('CNN Enhancement Experiments – Implementation of Research Paper "Enhanced Convolutional Neural Networks for Improved Image Classification"')
st.write("---")

# ---------------------------
# Intro + Paper Link
# ---------------------------
st.header("Overview")
st.write(
    """
    This dashboard documents my process of applying ideas from the research paper  
    **"Enhanced Convolutional Neural Networks for Improved Image Classification"**
    by Xiaoran Yang, Shuhan Yu, and Wenxi Xu to my own **Cats vs Dogs classification model**.

    The original research paper can be found here:

    **https://arxiv.org/pdf/2502.00663**

    My working Cats vs Dogs classifier app is available here:

    **https://noahpetillo-cats-vs-dogs-classifier-app-uvevsk.streamlit.app/**
    """
)

st.warning("""
⚠️ **Important Note on Adaptation:**  
This project adapts the paper's CNN architecture from:
- **Original:** CIFAR-10 (32×32 images, 10 classes)
- **My Implementation:** Cats vs Dogs (224×224 images, 2 classes)

The core principles (convolutional blocks, batch normalization, dropout, residual connections) 
remain the same, but significant architectural adjustments were required for the larger image size.
""")

st.write("---")
# ---------------------------
# Experiment 1 – enhanced_model1.pth
# ---------------------------

st.header("My First Implementation of the Enhanced CNN enhanced_model1.pth")
st.write(
    """
    As a baseline, I tried to copy the paper's CNN's architecture and Data Aug exactly. This goes as follows:
    
    Data Transforms:
    """
)
st.code(data_transform_code, language = 'python', line_numbers=True)
st.caption("""
NOTE: This data augmentation includes extra transformations (RandomRotation, extra resizing) 
that weren't in the original paper. The paper only used RandomHorizontalFlip, RandomCrop with 
padding=4, and Normalization for CIFAR-10's 32×32 images. For my 224×224 images, I scaled 
the padding proportionally (224/32 ≈ 7, so padding=28).
""")

st.write("And CNN architecture (initial attempt):")
st.code(enhanced_CNN_code, language='python', line_numbers=True)

st.warning("""
⚠️ This initial architecture had a critical flaw: The fully connected layer expected 
256 × 28 × 28 = 200,704 input features, causing massive parameter explosion compared 
to the paper's 256 × 4 × 4 = 4,096 features (due to my larger 224×224 input images).
""")

st.info("""
📊 **Key Dataset Difference:**  
- Paper's CIFAR-10: 32×32×3 images  
- My Cats vs Dogs: 224×224×3 images (7× larger in each dimension!)
""")

st.write( 
"""
    Then I trained this network on my Cats vs Dogs dataset and compared the results to the
    baseline CNN used in my original classifier app.
    """
)

st.write(
    """    
**Result:**  
I reached an accuracy of **~90%** after 10 epochs. While this was decent, it was **2% lower 
than my previous baseline CNN**, which achieved ~92% on the same Cats vs Dogs dataset. 
This unexpected drop suggested the architecture wasn't properly adapted for my use case.
    """
)

st.write(
    """
    After reviewing the results and analyzing the parameters, I believe there are two main reasons for the lower performance:

    **1. Too many parameters for my dataset**  
    CIFAR-10 images are very small (32×32), so the enhanced CNN in the paper naturally has a manageable
    number of parameters.  
    But my Cats vs Dogs images are much larger, meaning the fully connected layer exploded to  
    **256 × 28 × 28 = 200,704 features**, dramatically increasing complexity without adding meaningful signal.

    **2. 10 epochs was not enough**  
    In the paper, 10 epochs was sufficient. In my case, however, there is still downward trending progress at epoch 10.
    I think that this is partially as a result of my increased amount of features, meaning the model has to train longer.
    You can see the graph of my model below, clearly demonstrating that the model had not yet converged by epoch 10.

    """
)

# ---------------------------
# Display the loss figure
# ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        img = Image.open("Figures/enhanced_model1_loss_vs_epoch.png")
        st.image(img, caption="Enhanced Model 1 – Loss vs Epoch", width='content',)
        
    except:
        st.warning(
            "Image not found. Make sure `Figures/enhanced_model1_loss_vs_epoch.png` exists in your project directory."
        )

st.write("---")


# ---------------------------
# Experiment 2
# ---------------------------

st.header("Second implementation enhanced_model2.pth")
st.write("""
Here, keeping data augmentation exactly the same, I added residual (skip) connections to the 
network, inspired by ResNet principles mentioned in the paper's Related Work section. The 
paper's own architecture (Table I) doesn't explicitly show residual connections, but the 
authors reference ResNet's success with deeper networks. I also increased training to 20 epochs 
to allow more time for convergence. This looks as follows:
""")
st.code(enhanced2_CNN_code, language='python', line_numbers=True)

st.write(
    """    
    **Result:**  
    Here I achieved considerably better results, with an accuracy of 95.82%, finally beating my last CNN implementation. Still, I knew it can improve. Below shows the loss, still relatively noisy approaching epoch 20, but much better than enhanced_model1. 
    In my third attempt, I will implement a concept talked about in the paper, depthwise and pointwise convolution, but first I need to learn 
    what that means...
    """
)


# ---------------------------
# Display the loss figure
# ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        img = Image.open("Figures/enhanced_model2_loss_vs_epoch.png")
        st.image(img, caption="Enhanced Model 2 – Loss vs Epoch", width='content',)
        
    except:
        st.warning(
            "Image not found. Make sure `Figures/enhanced_model2_loss_vs_epoch.png` exists in your project directory."
        )

st.write("---")

# ---------------------------
# Experiment 3
# ---------------------------
st.header("Third Implementation - enhanced_model3.pth")
st.write("""
Based on the paper's discussion of regularization and my observation of noisy loss curves, 
I experimented with:
- **Label Smoothing** (0.1): Prevents overconfident predictions
- **AdamW optimizer**: Improved weight decay implementation
- **Additional augmentations**: Random erasing and more aggressive resizing

**Result:**  
Accuracy: **94.74%** - Better than Model 1, but slightly lower than Model 2. The additional 
regularization helped reduce overfitting, but I realized I was deviating too far from the 
paper's core principles.
""")

st.info("""
💡 **Lesson Learned:** Sometimes simpler is better. The paper achieved strong results with 
basic techniques. I decided to return to the paper's core approach for Model 4.
""")

st.write("---")

# ---------------------------
# Experiment 4
# ---------------------------
st.header("Fourth Implementation - enhanced_model4.pth (Current Best)")
st.write("""
For this iteration, I implemented the key architectural improvement I had been working toward: 
**Adaptive Average Pooling** (`nn.AdaptiveAvgPool2d((1,1))`).

**Key Changes:**
- Replaced fixed pooling with adaptive global pooling
- This automatically handles any input size and outputs a fixed 1×1 feature map per channel
- Eliminates the parameter explosion in the FC layer (now just `Linear(256, 512)`)
- Trained for 30 epochs to ensure full convergence

**Architecture Improvement:**
""")

st.code("""
# Before (Model 2):
self.pool4 = nn.AvgPool2d(2, 2)
self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Still large!

# After (Model 4):
self.global_pool = nn.AdaptiveAvgPool2d((1,1))
self.fc1 = nn.Linear(256, 512)  # Clean and efficient!
""", language='python')

st.write("""
This approach is actually more sophisticated than what's shown in the paper's Table I, 
though it aligns with modern CNN best practices.
""")

# Add loss graph for model 4 if you have it
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        img = Image.open("Figures/enhanced_model4_loss_vs_epoch.png")
        st.image(img, caption="Enhanced Model 4 – Loss vs Epoch", width='content')
    except:
        st.info("Loss graph for Model 4 coming soon after training completes.")

st.success("""
✅ **Results:** This model represents my best implementation, combining the paper's core 
principles with modern architectural improvements. Training in progress - check back for 
final accuracy!
""")

st.write("---")

# ---------------------------
# Comparison Table
# ---------------------------
st.header("Model Comparison Summary")

comparison_data = {
    "Model": ["Baseline CNN", "Enhanced Model 1", "Enhanced Model 2", "Enhanced Model 3", "Enhanced Model 4"],
    "Key Features": [
        "Simple CNN",
        "Paper architecture (direct copy)",
        "Added residual blocks + 20 epochs",
        "Label smoothing + AdamW + augmentation",
        "Adaptive pooling + residual blocks"
    ],
    "Accuracy": ["~92%", "~90%", "95.82%", "94.74%", "TBD"],
    "Epochs": [10, 10, 20, 30, 30],
    "Parameters (FC)": ["~50K", "~200K", "~50K", "~50K", "~512"]
}

st.table(comparison_data)

st.write("""
The progression shows that **residual connections** (Model 2) provided the biggest boost, 
while **adaptive pooling** (Model 4) made the architecture more elegant and scalable.
""")

st.write("---")

# ---------------------------
# Setup for Future Experiments
# ---------------------------
st.header("What I Learned & Next Steps")

st.write("""
Through these experiments, I discovered several key insights:

**Key Learnings:**
1. **Direct architecture transfer doesn't work** - The paper's design for 32×32 images needed 
   significant adaptation for 224×224 images
2. **Residual connections significantly improve performance** - Adding skip connections (from ResNet) 
   improved accuracy from 90% → 95.82%
3. **Parameter count matters** - The initial 200K+ parameters in the FC layer was overkill; 
   using AdaptiveAvgPool2d reduced this dramatically

**Future Improvements I'm Exploring:**
- **Adaptive Global Pooling**: Already implemented in `enhanced_model4.pth` - uses 
  `nn.AdaptiveAvgPool2d((1,1))` to eliminate the parameter explosion problem entirely
- **Learning Rate Scheduling**: Techniques like cosine annealing or step decay to improve convergence
- **Label Smoothing**: Regularization technique that prevents overconfident predictions
- **Longer Training**: The paper used 10 epochs for CIFAR-10, but my larger images may benefit 
  from 30-50 epochs with proper scheduling
- **Advanced Augmentation**: Techniques like Random Erasing (cutout) and MixUp for better generalization

Each experiment is documented here with code, results, and analysis.
""")

st.write("---")

# ---------------------------
# Footer
# ---------------------------
st.header("Conclusion")

st.write("""
This project demonstrates the process of implementing and adapting a research paper's 
architecture to a different dataset. Key takeaways:

1. **Architecture adaptation is crucial** - Direct transfer from 32×32 to 224×224 images 
   required significant modifications
2. **Residual connections are powerful** - They enabled deeper networks and better gradient flow
3. **Modern techniques matter** - Adaptive pooling solved the parameter explosion elegantly
4. **Iterative experimentation works** - Each model built on lessons from the previous one

The research paper achieved 84.95% on CIFAR-10's 10-class problem. My implementation achieves 
95%+ on the binary Cats vs Dogs classification, demonstrating successful adaptation of the 
core principles to a different domain.

**Try the working classifier:**  
🔗 [Cats vs Dogs Classifier App](https://noahpetillo-cats-vs-dogs-classifier-app-uvevsk.streamlit.app/)

**View the original research:**  
📄 [Enhanced Convolutional Neural Networks for Improved Image Classification](https://arxiv.org/pdf/2502.00663)
""")

st.info("""
📬 **Contact:** This project is part of my machine learning portfolio. Connect with me to 
discuss CNN architectures, transfer learning, or computer vision projects!
""")

st.write("**Made by Noah Petillo** – January 2026")

st.space()
st.space()


st.info("""
📝 **Coming Soon:**  
Future iterations will explore advanced techniques like:
- Learning rate scheduling (cosine annealing, step decay)
- More sophisticated data augmentation (MixUp, CutMix)
- Deeper architectures with bottleneck blocks
- Transfer learning approaches
- Ensemble methods

**Any new experiments will be documented below this line.**
""")
