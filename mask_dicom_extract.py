import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from skimage import io as skio
from skimage.color import rgb2hsv
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt

def read_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image = apply_modality_lut(dicom.pixel_array, dicom)
    return image


def enhance_color_contrast(image):
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Color(img)
    enhanced_img = enhancer.enhance(2)  # Increase the color contrast
    return np.array(enhanced_img)


def create_mask_from_yellow_hsv(image):
    # Convert to HSV
    hsv_image = rgb2hsv(image)

    # Define yellow color range in HSV based on visualization
    yellow_lower = np.array([0.1, 0.4, 0.4])  # Adjust these values
    yellow_upper = np.array([0.3, 1.0, 1.0])  # Adjust these values

    # Create a mask for yellow regions
    mask = np.all((hsv_image >= yellow_lower) & (hsv_image <= yellow_upper), axis=-1)
    return mask

def read_and_create_mask(dicom_path):
    image = read_dicom_image(dicom_path)
    enhanced_image = enhance_color_contrast(image)
    mask = create_mask_from_yellow_hsv(enhanced_image)
    return mask

def save_dicom_mask(original_dicom_path, mask, output_dir):
    dicom = pydicom.dcmread(original_dicom_path)
    mask = (mask * 255).astype(np.uint8)
    dicom.PixelData = mask.tobytes()
    dicom.Rows, dicom.Columns = mask.shape
    dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #mask_path = os.path.join(output_dir, os.path.basename(original_dicom_path))
    #dicom.save_as(mask_path)


def process_and_plot(image_dicom_root_dir, mask_dicom_root_dir):
    for case_id in os.listdir(image_dicom_root_dir):
        image_case_dir = os.path.join(image_dicom_root_dir, case_id)
        mask_case_dir = os.path.join(mask_dicom_root_dir, case_id)

        if os.path.isdir(image_case_dir) and os.path.isdir(mask_case_dir):
            image_subdirs = [d for d in os.listdir(image_case_dir) if os.path.isdir(os.path.join(image_case_dir, d))]
            mask_subdirs = [d for d in os.listdir(mask_case_dir) if os.path.isdir(os.path.join(mask_case_dir, d))]

            if len(image_subdirs) == 1 and len(mask_subdirs) == 1:
                image_subdir = os.path.join(image_case_dir, image_subdirs[0])
                mask_subdir = os.path.join(mask_case_dir, mask_subdirs[0])

                image_files = [f for f in os.listdir(image_subdir) if f.endswith('.dcm')]
                mask_files = [f for f in os.listdir(mask_subdir) if f.endswith('.dcm')]

                for image_file, mask_file in zip(sorted(image_files), sorted(mask_files)):
                    image_dicom_path = os.path.join(image_subdir, image_file)
                    mask_dicom_path = os.path.join(mask_subdir, mask_file)

                    print(f"Processing: {image_dicom_path}")
                    print(f"Using Mask: {mask_dicom_path}")

                    original_image = read_dicom_image(image_dicom_path)
                    mask = read_and_create_mask(mask_dicom_path)

                    # Display the original image and mask for verification
                    if original_image is not None and mask is not None:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.title('Original Image')
                        plt.imshow(original_image, cmap='gray')
                        plt.subplot(1, 2, 2)
                        plt.title('Segmentation Mask')
                        plt.imshow(mask, cmap='gray')
                        plt.show()
                    else:
                        print("Error: Original image or mask is None")


# Root directories containing DICOM files and masks
dicom_root_dir = 'Image_DICOMs'
mask_root_dir = 'Mask_DICOMs'
process_and_plot(dicom_root_dir, mask_root_dir)