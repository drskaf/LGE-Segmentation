import os
import numpy as np
import pydicom
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
import datetime
import tensorflow as tf
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

def unet_model(input_size=(256, 256, 1)):
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

    # Decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

print("[INFO] compiling model ...")
learning_rate = 0.001
optimizer = Adam()
model = unet_model()

# Address class imbalance with metrics focusing on positive pixels
def dice_coefficient(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)

def weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        return K.mean(zero_weight * (1 - y_true) * K.binary_crossentropy(y_true, y_pred) +
                      one_weight * y_true * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

# Adjust the weights as needed
zero_weight = 0.1
one_weight = 0.9

# Compile the model with weighted binary crossentropy loss
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
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
checkpoint = ModelCheckpoint('Models/unet_model_classweigth.h5', monitor='val_prc', mode='max', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_prc', mode='max', patience=5)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
print("[INFO] training model ...")
history = model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=50, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping, tensorboard_callback])
