import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

st.set_page_config(
    page_title="Emotion Detection using FER2013",
    
    layout="centered",
)

st.markdown(
    """
    <style>
        .reportview-container {
            background: #F5F5F5;  /* Light grey background */
        }
        .sidebar .sidebar-content {
            background: #2C3E50;  /* Dark sidebar background */
            color: white;  /* Text color in sidebar */
        }
        h1 {
            color: #E74C3C;  /* Red text for the main title */
        }
        .stTextInput > div > div > input {
            background-color: #E6E6FA;  /* Light lavender for input fields */
            color: #2C3E50;  /* Dark text for input */
        }
    </style>
    """,
    unsafe_allow_html=True
)

class DeepEmotionCNN1(nn.Module):
    def __init__(self):
        super(DeepEmotionCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512*8*8,512)
        self.fc2 = nn.Linear(512,7)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1,512*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model():
    model = DeepEmotionCNN1()
    model.load_state_dict(torch.load('D:/Data science projects/Emotion detection/models/model_DE_CNN2.pth'))
    model.eval()
    return model

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image):
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128,128)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

st.title("Emotion Detection using FER2013")
    
model = load_model()
    
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.type not in ["image/jpeg", "image/png", "image/jpg"]:
        st.error("Unsupported file type. Please upload a JPG, JPEG, or PNG file.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

    processed_image = preprocess_image(image)
    
    with torch.no_grad():  
        output = model(processed_image)
        _, predicted = torch.max(output, 1)
        emotion = EMOTION_LABELS[predicted.item()]

    st.write(f'Predicted Emotion: **{emotion}**')