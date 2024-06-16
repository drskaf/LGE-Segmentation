import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score, matthews_corrcoef, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns


# Define additional metrics
def specificity_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_negatives = K.sum(K.round(K.clip(1 - y_true) * K.round(K.clip(1 - y_pred, 0, 1))))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity


def f1_score_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def accuracy_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    correct_predictions = K.sum(K.cast(K.equal(y_true, y_pred), tf.float32))
    return correct_predictions / tf.size(y_true, out_type=tf.float32)


# Define function to calculate and print metrics
def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > 0.5).astype(int)

    iou = MeanIoU(num_classes=2)(y_true, y_pred).numpy()
    dice = f1_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    specificity = specificity_metric(y_true, y_pred).numpy()
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    roc_auc = roc_auc_score(y_true_flat, y_pred_flat)
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    jaccard = jaccard_score(y_true_flat, y_pred_flat)

    print(f'IOU: {iou}')
    print(f'Dice Index: {dice}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Specificity: {specificity}')
    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Matthews Correlation Coefficient: {mcc}')
    print(f'Jaccard Index: {jaccard}')

    return iou, dice, precision, recall, f1, specificity, accuracy, roc_auc, mcc, jaccard


# Define function to plot ROC curve
def plot_roc_curve(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat)
    roc_auc = roc_auc_score(y_true_flat, y_pred_flat)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


# Define function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > 0.5).astype(int)
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Assuming `model` is already trained and saved as 'unet_model.h5'
model = load_model('unet_model.h5', compile=False)

# Load your new test data (example)
# Replace `X_new` and `y_new` with your actual test data
X_new = np.load('path_to_new_images.npy')
y_new = np.load('path_to_new_masks.npy')

# Get predictions
y_pred = model.predict(X_new)

# Calculate metrics and plot results
iou, dice, precision, recall, f1, specificity, accuracy, roc_auc, mcc, jaccard = calculate_metrics(y_new, y_pred)
plot_roc_curve(y_new, y_pred)
plot_confusion_matrix(y_new, y_pred)

