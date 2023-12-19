import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Load the pretrained DenseNet model
model = models.densenet121()
model.eval()

# Transform the image before passing it to the model
def transform_image(image):
    # Convert the image to RGB
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(image):
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        return predicted_idx.item()

def main():
    st.title("DenseNet Image Classification App")

    # Load an image
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if image_file:
        # Display the selected image
        image = Image.open(image_file)
        st.image(image, caption="Selected image", use_column_width=True)

        # Processing and prediction
        if st.button("Classify image"):
            # Transform the image
            image_tensor = transform_image(image)

            # Prediction using the model
            prediction = predict(image_tensor)

            # Display the result
            st.write("Predicted class (according to ImageNet class index):", prediction)

if __name__ == "__main__":
    main()
