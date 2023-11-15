import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Загрузка предварительно обученной модели DenseNet
model = models.densenet121(pretrained=True)
model.eval()

# Преобразование изображения перед передачей его модели
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

    # Загрузка изображения
    image_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    if image_file:
        # Отображение выбранного изображения
        image = Image.open(image_file)
        st.image(image, caption="Выбранное изображение", use_column_width=True)

        # Обработка и предсказание
        if st.button("Классифицировать изображение"):
            # Преобразование изображения
            image_tensor = transform_image(image)

            # Предсказание с использованием модели
            prediction = predict(image_tensor)

            # Отображение результата
            st.write("Предсказанный класс (согласно индексу класса ImageNet):", prediction)

if __name__ == "__main__":
    main()
