import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
from tabulate import tabulate
import joblib

def statistics_extraction(crop):
    feat = []

    # Basic statistics
    var = np.var(crop)
    feat.append(var)
    mean = np.mean(crop)
    feat.append(mean)
    std = np.std(crop)
    feat.append(std)

    # Min and max values
    min_val = np.min(crop)
    max_val = np.max(crop)
    feat.append(min_val)
    feat.append(max_val)

    # Range
    feat.append(max_val - min_val)

    # Moments
    moments = cv2.moments(crop)
    hu = cv2.HuMoments(moments).flatten()
    feat.extend(hu)

    # Add central moments
    feat.append(moments['mu20'])
    feat.append(moments['mu11'])
    feat.append(moments['mu02'])
    feat.append(moments['mu30'])
    feat.append(moments['mu21'])
    feat.append(moments['mu12'])
    feat.append(moments['mu03'])

    return np.array(feat, dtype=np.float32)


def clip_extraction(clip):
    # Get the center pixel value (for 5x5 clip, center is at [2,2])
    if clip[2, 2] > 128:  # Threshold for binary decision
        return np.float32(1)  # Vessel
    else:
        return np.float32(0)  # Background


def build_rounded_mask(height, width):
    center = (int(width / 2), int(height / 2))
    radius = min(center[0], center[1], width - center[0], height - center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def handle_inside_circle(height, width, rounded_mask, data, process_function, output_list):
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            if rounded_mask[y, x]:
                crop = data[y - 2:y + 3, x - 2:x + 3]
                output_list.append(process_function(crop))


def load_images(start=1, end=10):
    images = []
    masks = []

    for i in range(start, end + 1):
        img_path = f"healthy/{i:02d}_h.jpg"
        mask_path = f"healthy_manualsegm/{i:02d}_h.tif"

        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)

    return images, masks

# Path to the saved model
MODEL_PATH = "vessel_segmentation_model.joblib"

# Check if the model already exists
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    best_model = joblib.load(MODEL_PATH)
else:
    #------------------------------------------------------ Training model -----------------------------------------------------------------
    print("Training new model...")
    # Load first 10 images for training
    train_images, train_masks = load_images(1, 10)

    # Prepare data for training
    X = []
    y = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Extract labels from training masks
    for mask in train_masks:
        height, width = mask.shape
        rounded_mask = build_rounded_mask(height, width)
        handle_inside_circle(height, width, rounded_mask, mask, clip_extraction, y)

    # Extract feat from training images
    for image in train_images:
        # Use green channel and apply preprocessing
        image_green = image[:, :, 1]
        image_green = np.clip(image_green, 10, 245)
        image_green = cv2.GaussianBlur(image_green, (5, 5), 0)
        image_green = clahe.apply(image_green)

        height, width = image_green.shape
        rounded_mask = build_rounded_mask(height, width)
        handle_inside_circle(height, width, rounded_mask, image_green, statistics_extraction, X)

    # Get feature dimension
    f = len(X[0])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training set size: ", len(X_train))
    print('X_train shape: ', np.array(X_train).shape)
    print('y_train shape: ', np.array(y_train).shape)

    print('y_train label 1 (vessels): ', np.count_nonzero(y_train))
    print('y_train label 0 (background): ', len(y_train) - np.count_nonzero(y_train))

    # Apply random undersampling to balance classes
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    print("Training set size after sampling: ", len(X_train))
    print('y_train label 1 (vessels): ', np.count_nonzero(y_train))
    print('y_train label 0 (background): ', len(y_train) - np.count_nonzero(y_train))

    # Reduce dataset size if necessary
    if len(X_train) > 30000:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=30000, random_state=42)
        print("Training set size after reduction: ", len(X_train))
        print('y_train label 1 (vessels): ', np.count_nonzero(y_train))
        print('y_train label 0 (background): ', len(y_train) - np.count_nonzero(y_train))

    # Build pipeline with Random Forest classifier
    rfc = RandomForestClassifier()
    pipeline = Pipeline([('rfc', rfc)])

    # Define parameter grid for search
    param_grid = {
        'rfc__n_estimators': [10, 110],
        'rfc__max_depth': [None, 30],
        'rfc__min_samples_leaf': [1, 3],
        'rfc__min_samples_split': [2, 6]
    }

    # Perform grid search to find optimal parameters
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train)

    # Print results
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print('Training accuracy:', grid_search.score(X_train, y_train))
    print('Validation accuracy:', grid_search.score(X_val, y_val))

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
#--------------------------------------------------------------------------------------------------------------------------------------

# Now use the model on images 11-15
test_images, test_masks = load_images(11, 15)
n = len(test_images)

# Prepare plots
fig, ax = plt.subplots(3, n)
fig.set_size_inches(18.5, 10.5)
fig.suptitle('Image Processing Results')

PREDICTED = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for k, new_img, mask in zip(range(n), test_images, test_masks):
    # Display original image
    ax[0, k].imshow(new_img)
    ax[0, k].set_title(f'Image {k + 11}')
    ax[0, k].axis('off')

    # Display expert mask
    ax[1, k].imshow(mask, cmap='gray')
    ax[1, k].set_title(f'Expert Mask {k + 11}')
    ax[1, k].axis('off')

    # Process image
    new_img_green = new_img[:, :, 1]
    new_img_green = np.clip(new_img_green, 10, 245)
    new_img_green = cv2.GaussianBlur(new_img_green, (5, 5), 0)
    new_img_green = clahe.apply(new_img_green)

    # Extract feat for each pixel
    f = len(statistics_extraction(new_img_green[2:7, 2:7]))  # Get feature dimension
    new_img_features = np.zeros((new_img_green.shape[0], new_img_green.shape[1], f), dtype=np.float32)

    for i in range(2, new_img_green.shape[0] - 2):
        for j in range(2, new_img_green.shape[1] - 2):
            window = new_img_green[i - 2:i + 3, j - 2:j + 3]
            new_img_features[i, j] = statistics_extraction(window)

    # Predict using trained model
    y_pred = best_model.predict(new_img_features.reshape(-1, f)).reshape(new_img_green.shape[0],
                                                                         new_img_green.shape[1])
    PREDICTED.append(y_pred)

    # Display prediction
    ax[2, k].imshow(y_pred, cmap='gray')
    ax[2, k].set_title(f'Prediction {k + 11}')
    ax[2, k].axis('off')

plt.tight_layout()

# Evaluation of results
fig, axs = plt.subplots(2, n)
fig.set_size_inches(25.5, 9)
fig.suptitle('Evaluation Results')

headers = ["Method", "Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy", "G-mean"]
test_data = []

for i, expert_mask, predicted in zip(range(n), test_masks, PREDICTED):
    # Display prediction
    axs[0, i].imshow(predicted, cmap='gray')
    axs[0, i].set_title(f'Prediction {i + 11}')
    axs[0, i].axis('off')

    # Prepare data for evaluation
    y_true = expert_mask.ravel()
    y_true_binary = np.where(y_true > 128, 1, 0)

    y_pred = predicted.ravel()
    y_pred_binary = y_pred  # Already binary from classifier

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axs[1, i], cmap='Reds', colorbar=True)
    axs[1, i].set_title(f'Confusion Matrix {i + 11}')

    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    sensitivity = recall_score(y_true_binary, y_pred_binary)  # Same as recall
    specificity = tn / (tn + fp)  # True negative rate

    # Measures for imbalanced data
    balanced_accuracy = (sensitivity + specificity) / 2  # Arithmetic mean
    g_mean = math.sqrt(sensitivity * specificity)  # Geometric mean

    metrics_data = [
        f"Image {i + 11}",
        f"{accuracy:.4f}",
        f"{sensitivity:.4f}",
        f"{specificity:.4f}",
        f"{balanced_accuracy:.4f}",
        f"{g_mean:.4f}"
    ]

    test_data.append(metrics_data)

# Calculate average metrics for the test set
test_data.append([
    "Test Average",
    f"{np.mean([float(x[1]) for x in test_data]):.4f}",
    f"{np.mean([float(x[2]) for x in test_data]):.4f}",
    f"{np.mean([float(x[3]) for x in test_data]):.4f}",
    f"{np.mean([float(x[4]) for x in test_data]):.4f}",
    f"{np.mean([float(x[5]) for x in test_data]):.4f}"
])

plt.tight_layout()

print("\nTest set evaluation metrics:")
print(tabulate(test_data, headers=headers, tablefmt="pipe"))

plt.show()
