# test_api.py
import pytest
from PIL import Image
from main import transform_image, predict

# Mock image for testing
@pytest.fixture
def mock_image():
    return Image.new('RGB', (224, 224))

def test_transform_image(mock_image):
    # Test the transform_image function
    transformed_image = transform_image(mock_image)
    assert transformed_image.shape == (1, 3, 224, 224)  # Assuming the shape returned by transform_image

def test_predict(mock_image):
    # Test the predict function
    transformed_image = transform_image(mock_image)
    prediction = predict(transformed_image)
    assert isinstance(prediction, int)  # Assuming predict returns an integer
