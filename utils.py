import pandas as pd
import numpy as np
import pydicom
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
from PIL import Image, ImageEnhance
from pydicom.pixel_data_handlers.util import apply_modality_lut
import re

# Loading data and creating masks
def load_dicom_image(dicom_path):
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
    image = load_dicom_image(dicom_path)
    enhanced_image = enhance_color_contrast(image)
    mask = create_mask_from_yellow_hsv(enhanced_image)
    return mask

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def load_dicom_images_and_generate_masks(image_dicom_root_dir, mask_dicom_root_dir, img_size):
    images = []
    masks = []

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

                    # Load and resize the DICOM images
                    image = load_dicom_image(image_dicom_path)
                    image = resize(image, IMG_SIZE)

                    # Convert to grayscale
                    if image.ndim == 3:
                        image = rgb2gray(image)

                    # Normalize image
                    image = normalize_image(image)

                    # Create masks from segmented DICOMs
                    mask = read_and_create_mask(mask_dicom_path)
                    mask = resize(mask, IMG_SIZE)

                    images.append(image)
                    masks.append(mask)

    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    return images, masks

def load_legacy_lge_images(directory, im_size):
    Images = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for dir in dirs:
            folder_strip = dir.rstrip('_')
            dir_path = os.path.join(directory, dir)
            files = sorted(os.listdir(dir_path))

            # Loop over cases with single dicoms
            if len(files) > 5:
                bun_sa = []
                for file in files:
                    if not file.startswith('.'):
                        dicom = pydicom.read_file(os.path.join(dir_path, file))
                        dicom_series = dicom.SeriesDescription
                        dicom_series = dicom_series.split('_')
                        img = dicom.pixel_array
                        img = normalize_image(img)
                        img = resize(img, (im_size, im_size))
                        img = np.expand_dims(img, axis=-1)
                        mat_2c = findWholeWord('2ch.*')(str(dicom_series))
                        mat_3c = findWholeWord('3ch.*')(str(dicom_series))
                        mat_4c = findWholeWord('4ch.*')(str(dicom_series))
                        sa = []

                        if not (mat_2c or mat_3c or mat_4c):
                            bun_sa.append(img)

                l = len(bun_sa) // 3
                imgList_sa = (bun_sa[l:l+10] if len(bun_sa) > 25 else bun_sa[1:11])
                imgList = imgList_sa
                imgStack = np.stack(imgList, axis=0)
                Images.append(imgStack)

            # Loop over cases with stacked dicoms
            else:
                bun_sa = []
                for file in files:
                    if not file.startswith('.'):
                        dicom = pydicom.read_file(os.path.join(dir_path, file))
                        dicom_series = dicom.SeriesDescription
                        dicom_series = dicom_series.split('_')
                        img = dicom.pixel_array
                        img = normalize_image(img)
                        mat_2c = findWholeWord('2ch.*')(str(dicom_series))
                        mat_3c = findWholeWord('3ch.*')(str(dicom_series))
                        mat_4c = findWholeWord('4ch.*')(str(dicom_series))

                        if not (mat_2c or mat_3c or mat_4c):
                            images = range(len(img[:, ]))
                            l = len(images) // 3
                            if len(images) > 25:
                                img = img[l:l+10]
                                for i in img[:]:
                                    img = resize(i, (im_size, im_size))
                                    img = np.expand_dims(img, axis=-1)
                                    bun_sa.append(img)
                            else:
                                img = img[1:11]
                                for i in img[:]:
                                    img = resize(i, (im_size, im_size))
                                    img = np.expand_dims(img, axis=-1)
                                    bun_sa.append(img)

                imgList = bun_sa
                imgStack = np.stack(imgList, axis=0)
                Images.append(imgStack)

    return Images






