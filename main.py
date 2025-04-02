import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.filters import frangi
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from tabulate import tabulate


def remove_border(color_img: np.ndarray, image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    return cv2.bitwise_and(image, mask)

# Load the image
img = cv2.imread("healthy/05_h.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

RESULTS = []

# Figure 1: Original image and channels
fig1 = plt.figure(figsize=(15, 8))
fig1.suptitle('Original Image and Channel Visualizations')

# Original image
ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_title('Original image')
ax1.imshow(img)

# Channel visualizations
ax2 = fig1.add_subplot(2, 3, 2)
ax2.set_title('Red channel')
ax2.imshow(img[:, :, 0], cmap='gray')

ax3 = fig1.add_subplot(2, 3, 3)
ax3.set_title('Green channel')
ax3.imshow(img[:, :, 1], cmap='gray')

ax4 = fig1.add_subplot(2, 3, 4)
ax4.set_title('Blue channel')
ax4.imshow(img[:, :, 2], cmap='gray')

ax5 = fig1.add_subplot(2, 3, 5)
ax5.set_title('Gray channel')
ax5.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap='gray')

fig1.tight_layout()

# CLAHE normalization
img_gray = img[:, :, 1]
img_gray = np.clip(img_gray, 10, 245)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_gray_norm = clahe.apply(img_gray)

# Figure 2: CLAHE comparison
fig2 = plt.figure(figsize=(10, 5))
fig2.suptitle('CLAHE Normalization')

ax6 = fig2.add_subplot(1, 2, 1)
ax6.set_title('Original grayscale')
ax6.imshow(img_gray, cmap='gray')

ax7 = fig2.add_subplot(1, 2, 2)
ax7.set_title('CLAHE normalized')
ax7.imshow(img_gray_norm, cmap='gray')

fig2.tight_layout()

# Figure 3: Frangi filter
fig3 = plt.figure(figsize=(12, 6))
fig3.suptitle('Frangi Filtering')

img_frangi = frangi(img_gray_norm, sigmas=np.arange(1, 5, 0.5), black_ridges=True)
img_frangi = img_frangi / img_frangi.max()
img_frangi = np.where(img_frangi > 0.065, 1, 0)

# Apply border removal to Frangi filtered image
img_frangi_uint8 = (img_frangi * 255).astype(np.uint8)
img_frangi = remove_border(img, img_frangi_uint8)
# Convert back to binary
img_frangi = np.where(img_frangi > 0, 1, 0)

ax10 = fig3.add_subplot(1, 2, 1)
ax10.set_title('Expert mask')
ax10.imshow(cv2.imread('healthy_manualsegm/05_h.tif'), cmap='gray')

ax11 = fig3.add_subplot(1, 2, 2)
ax11.set_title('Frangi filter ')
ax11.imshow(img_frangi, cmap='gray')

fig3.tight_layout()
RESULTS.append(img_frangi)

# Figure 4: Morphological operations
KERNEL_SIZES = [5, 7, 15, 23]
images = []
images.append(np.copy(img_gray))

for i in range(len(KERNEL_SIZES)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZES[i], KERNEL_SIZES[i]))
    new_img = cv2.morphologyEx(images[-1], cv2.MORPH_OPEN, kernel)
    images.append(np.copy(new_img))
    new_img = cv2.morphologyEx(images[-1], cv2.MORPH_CLOSE, kernel)
    images.append(np.copy(new_img))

images = images[1:]

# Figure 5: Background removal
fig5 = plt.figure(figsize=(15, 5))
fig5.suptitle('Background Removal')

background = images[-1]
img_no_background = cv2.subtract(background, img_gray)
img_no_background_blur = cv2.GaussianBlur(img_no_background, (5, 5), 0)

ax14 = fig5.add_subplot(1, 3, 1)
ax14.set_title('Original for bg removal')
ax14.imshow(img_gray, cmap='gray')

ax15 = fig5.add_subplot(1, 3, 2)
ax15.set_title('Without background')
ax15.imshow(img_no_background, cmap='gray')

ax16 = fig5.add_subplot(1, 3, 3)
ax16.set_title('Without bg + blur')
ax16.imshow(img_no_background_blur, cmap='gray')

fig5.tight_layout()

# Figure 6: Normalization after background removal
fig6 = plt.figure(figsize=(15, 5))
fig6.suptitle('Normalization After Background Removal')

#Limit pixel values to 0-20
img_normalized_clip = np.clip(img_no_background_blur, 0, 20)
img_normalized = (img_normalized_clip / img_normalized_clip.max()) * 255
img_clahe = clahe.apply(img_no_background_blur)

ax17 = fig6.add_subplot(1, 3, 1)
ax17.set_title('After bg removal')
ax17.imshow(img_no_background_blur, cmap='gray')

ax18 = fig6.add_subplot(1, 3, 2)
ax18.set_title('Normalized image')
ax18.imshow(img_normalized, cmap='gray')

ax19 = fig6.add_subplot(1, 3, 3)
ax19.set_title('CLAHE after bg removal')
ax19.imshow(img_clahe, cmap='gray')

fig6.tight_layout()

# New comparison figure
fig7 = plt.figure(figsize=(15, 5))
fig7.suptitle('Expert vs Generated Comparison')

# Process images for comparison
img_gen = img_normalized.copy()
img_gen_norm = img_gen / img_gen.max()
img_gen_binary = np.where(img_gen_norm > 0.75, 1, 0).astype(np.uint8) * 255
img_expert = cv2.cvtColor(cv2.imread('healthy_manualsegm/05_h.tif'), cv2.COLOR_BGR2GRAY)

ax20 = fig7.add_subplot(1, 3, 1)
ax20.set_title('Expert Mask')
ax20.imshow(img_expert, cmap='gray')

ax21 = fig7.add_subplot(1, 3, 2)
ax21.set_title('Normalized')
ax21.imshow(img_normalized, cmap='gray')

fig7.tight_layout()

RESULTS.append(img_normalized)
RESULTS.append(img_clahe)

# Results evaluation - each method gets its own figure
img_true = cv2.imread('healthy_manualsegm/05_h.tif')
img_true_greyscale = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)

y_true = img_true_greyscale.ravel()
y_true = y_true / y_true.max()

# Update headers to include all metrics
headers = ["Method", "Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy", "G-Mean"]
data = []

# Create separate figures for each result
method_names = ['Frangi', 'Normalized']

for idx, (img_pred, method_name) in enumerate(zip(RESULTS, method_names)):
    fig_result = plt.figure(figsize=(12, 6))
    fig_result.suptitle(f'Results Evaluation: {method_name}')

    y_pred = img_pred.ravel()
    y_pred = y_pred / y_pred.max()

    y_true_binary = np.where(y_true > 0.75, 1, 0)
    y_pred_binary = np.where(y_pred > 0.75, 1, 0)

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    ax_img = fig_result.add_subplot(1, 2, 1)
    ax_img.imshow(img_pred, cmap='gray')
    ax_img.set_title(f"{method_name}")

    ax_cm = fig_result.add_subplot(1, 2, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap='Reds', colorbar=True)
    ax_cm.set_title(f"Confusion Matrix")

    # Calculate all metrics correctly
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    sensitivity = recall_score(y_true_binary, y_pred_binary)  # Same as recall
    specificity = tn / (tn + fp)  # True negative rate

    # Measures for imbalanced data
    balanced_accuracy = (sensitivity + specificity) / 2  # Arithmetic mean
    g_mean = math.sqrt(sensitivity * specificity)  # Geometric mean

    data.append([
        method_name,
        f"{accuracy:.4f}",
        f"{sensitivity:.4f}",
        f"{specificity:.4f}",
        f"{balanced_accuracy:.4f}",
        f"{g_mean:.4f}"
    ])

    # Add text annotation with metrics to the figure
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Sensitivity: {sensitivity:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"Balanced Accuracy: {balanced_accuracy:.4f}\n"
        f"G-Mean: {g_mean:.4f}"
    )

    fig_result.text(0.02, 0.02, metrics_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

    fig_result.tight_layout()

print(tabulate(data, headers=headers, tablefmt="pipe"))

# Show all figures
plt.show()
