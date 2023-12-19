# Image Classification App

This is a simple web application for image classification using the DenseNet model. The app is built with Streamlit and PyTorch.

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:NadezhdinM/streamlit_yolov5.git
   cd streamlit_yolov5

2. Then install:
    ```bash
    pip install -r requirements.txt

Usage
----
1. Run the Streamlit app:

   ```bash
   streamlit run main.py
   
2. Open your web browser and go to the provided local address (usually http://localhost:8501).
3. Upload an image using the provided button and click "Classify Image" to see the predicted class.

Notes
---

1. This app uses the DenseNet model for image classification. Make sure you have a compatible version of PyTorch installed.
2. The app is configured to resize uploaded images to 224x224 pixels, as required by the model.
3. Feel free to customize the app or use a different pre-trained model according to your requirements.

Test
---

1. For start test using the next command:
   ```bash
   pytest

Contributors
---

1. Maksim Nadezhdin
