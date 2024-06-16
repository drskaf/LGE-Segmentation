import os
import numpy as np
import pydicom
from skimage.transform import resize
from skimage.color import rgb2hsv, rgba2rgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import *
import tensorflow as tf


# Load the trained model
# Specify the path to your saved model
model_path = 'Models/unet_model_classweigth.h5'
model = load_model(model_path)

# Load and preprocess the new data
new_data_dicom_root_dir = 'Test'
images = load_legacy_lge_images(new_data_dicom_root_dir, im_size=256)
all_images = np.array([img for sublist in images for img in sublist])
print(f"All images shape: {all_images.shape}")

# Make predictions on the new data
predictions = model.predict(all_images[:50])

# Function to plot the results
def plot_results(images, predictions, num_samples=50):
    for i in range(num_samples):
        plt.figure(figsize=(12, 6))

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title('Original Image')

        # Plot predicted mask
        plt.subplot(1, 2, 2)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')

        plt.show()


# Plot the results for a few samples
plot_results(all_images, predictions, num_samples=50)