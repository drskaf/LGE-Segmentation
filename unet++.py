import os
import numpy as np
import pydicom
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageEnhance
from pydicom.pixel_data_handlers.util import apply_modality_lut
from tensorflow.keras import backend as K
from utils import *


##################################################################################
# Load and preprocess data
##################################################################################

# Load data
# Set image size
IMG_SIZE = (256, 256)
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

# Load the images and generate masks
image_dicom_root_dir = 'Image_DICOMS'
mask_dicom_root_dir = 'Mask_DICOMS'
images, masks = load_dicom_images_and_generate_masks(image_dicom_root_dir, mask_dicom_root_dir, IMG_SIZE)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

print(f"Training images shape: {X_train.shape}")
print(f"Training masks shape: {y_train.shape}")
print(f"Testing images shape: {X_test.shape}")
print(f"Testing masks shape: {y_test.shape}")


##################################################################################
# Define segmentation model
##################################################################################

def unet_plus_plus(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Nested connections
    up4_5 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv4_5 = Conv2D(512, 3, activation='relu', padding='same')(up4_5)
    conv4_5 = Conv2D(512, 3, activation='relu', padding='same')(conv4_5)

    up3_4 = concatenate([UpSampling2D(size=(2, 2))(conv4_5), conv3], axis=-1)
    conv3_4 = Conv2D(256, 3, activation='relu', padding='same')(up3_4)
    conv3_4 = Conv2D(256, 3, activation='relu', padding='same')(conv3_4)

    up2_3 = concatenate([UpSampling2D(size=(2, 2))(conv3_4), conv2], axis=-1)
    conv2_3 = Conv2D(128, 3, activation='relu', padding='same')(up2_3)
    conv2_3 = Conv2D(128, 3, activation='relu', padding='same')(conv2_3)

    up1_2 = concatenate([UpSampling2D(size=(2, 2))(conv2_3), conv1], axis=-1)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(up1_2)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(conv1_2)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv1_2)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

print("[INFO] compiling model ...")
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model = unet_plus_plus()

# Compile the model with weighted binary crossentropy loss
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

##################################################################################
# Train and save model
##################################################################################

# Create a logs directory for TensorBoard
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Define the data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_gen_args = dict(rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the flow methods
seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train, batch_size=32, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=32, seed=seed)

# Combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# Train the model
class_weight = {0: 0.5466,
                1: 5.2927}
checkpoint = ModelCheckpoint('Models/unet++_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
print("[INFO] training model ...")
history = model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping, tensorboard_callback])
