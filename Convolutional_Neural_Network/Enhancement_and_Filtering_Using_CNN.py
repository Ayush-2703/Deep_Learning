import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload the image
uploaded = files.upload()

# Assuming the file is uploaded, get its name from the uploaded dictionary
image_path = list(uploaded.keys())[0]

# Load the image
def load_image(image_path):
    # Open the image with Pillow
    image = Image.open(image_path)
    return image

# Function for blurring the image
def apply_blur(image, radius=5):
    return image.filter(ImageFilter.GaussianBlur(radius))

# Function for sharpening the image
def apply_sharpen(image):
    return image.filter(ImageFilter.SHARPEN)

# Volvo8 sharpening filter (Specific sharpening filter using a custom kernel)
def volvo8_sharpen(image):
    image = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])  # A basic sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened)

# General sharpen (using a higher strength kernel)
def general_sharpen(image):
    image = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])  # Basic sharpen kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened)

# Function to display the images
def display_images(original, blurred, sharpened, volvo8_sharpened, general_sharpened):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(blurred)
    plt.title("Blurred Image")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(sharpened)
    plt.title("Sharpened Image")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(volvo8_sharpened)
    plt.title("Volvo8 Sharpened Image")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(general_sharpened)
    plt.title("General Sharpened Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Load the image
image = load_image(image_path)

# Apply blurring and sharpening filters
blurred_image = apply_blur(image)
sharpened_image = apply_sharpen(image)
volvo8_sharpened_image = volvo8_sharpen(image)
general_sharpened_image = general_sharpen(image)

# Display all the images
display_images(image, blurred_image, sharpened_image, volvo8_sharpened_image, general_sharpened_image)
