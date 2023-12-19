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
    assert transformed_image.max() <= 1.0  # Check if values are normalized

def test_predict(mock_image):
    # Test the predict function
    transformed_image = transform_image(mock_image)
    prediction = predict(transformed_image)
    assert isinstance(prediction, int)  # Assuming predict returns an integer

# Additional test cases for edge scenarios

def test_transform_image_invalid_input():
    # Test transform_image with invalid input (non-image)
    with pytest.raises(AttributeError):
        transform_image("invalid_input")

def test_predict_invalid_input():
    # Test predict with invalid input (non-tensor)
    with pytest.raises(TypeError):
        predict("invalid_input")
