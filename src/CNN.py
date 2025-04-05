import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_1d_hsi_classifier(input_shape, num_classes):
    model = Sequential()

    # Warstwa konwolucyjna 1
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(input_shape,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Warstwa konwolucyjna 2
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Spłaszczenie
    model.add(Flatten())

    # Warstwa w pełni połączona 1
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Warstwa w pełni połączona 2 (warstwa wyjściowa)
    model.add(Dense(num_classes, activation='softmax'))

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model






# Load data
data = sio.loadmat('data/data/Indian_pines_NS_data.mat')
gt_data = sio.loadmat('data/gt/Indian_pines_NS_gt.mat')

# Extract hyperspectral image and ground truth
hyper_image = data[list(data.keys())[-1]]  # Assuming last key contains data
ground_truth = gt_data[list(gt_data.keys())[-1]]  # Assuming last key contains ground truth
h, w, p = hyper_image.shape
print(hyper_image.shape)

reshaped_image = hyper_image.reshape(hyper_image.shape[0] * hyper_image.shape[1], hyper_image.shape[2])
reshaped_ground_truth = ground_truth.reshape(ground_truth.shape[0] * ground_truth.shape[1])


# Filter classes with more than 100 samples




class_counts = np.bincount(reshaped_ground_truth)
valid_classes = np.where(class_counts > 5000)[0]

mask = np.isin(reshaped_ground_truth, valid_classes)
reshaped_image = reshaped_image[mask]
reshaped_ground_truth = reshaped_ground_truth[mask]

# Remap labels to be sequential
unique_labels, remapped_labels = np.unique(reshaped_ground_truth, return_inverse=True)
num_classes = len(unique_labels)
print(num_classes)
y_one_hot = to_categorical(remapped_labels, num_classes=num_classes)



# Filter out pixels with no class (assuming class 0 is the "no class" label)
valid_indices = reshaped_ground_truth > 0
background_indices = reshaped_ground_truth == 0

# Downsample each class to a maximum of 5000 samples
filtered_image = []
filtered_labels = []

for class_label in range(num_classes):
    class_indices = np.where(remapped_labels == class_label)[0]
    if len(class_indices) > 5000:
        sampled_indices = resample(
            class_indices,
            n_samples=5000,
            random_state=42,
            replace=False
        )
    else:
        sampled_indices = class_indices

    filtered_image.append(reshaped_image[sampled_indices])
    filtered_labels.append(y_one_hot[sampled_indices])

# Combine all classes
filtered_image = np.vstack(filtered_image)
filtered_labels = np.vstack(filtered_labels)

# Combine valid samples and limited background samples
# filtered_image = np.vstack((reshaped_image[valid_indices], background_samples))
# filtered_labels = np.vstack((y_one_hot[valid_indices], background_labels))








# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_image, filtered_labels, test_size=0.3, random_state=42)

print(y_test.shape)
print(X_train.shape)

scaler = StandardScaler()
data_normalized = scaler.fit_transform(X_train)

# PCA jako krok MNF
pca = PCA(n_components=2)
data_normalized = pca.fit_transform(data_normalized)
X_test = pca.transform(X_test)



scaler_post_PCA = StandardScaler()
X_train = scaler_post_PCA.fit_transform(data_normalized)
X_test = scaler_post_PCA.transform(X_test)


# # Build deeper CNN model with adjusted pooling layers
# model = Sequential([
#     Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     MaxPooling1D(2),
#     Conv1D(128, 3, activation='relu'),
#     MaxPooling1D(2),
#     Conv1D(256, 3, activation='relu'),
#     Flatten(),  # Removed one pooling layer to prevent negative output size
#     Dense(256, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(num_classes, activation='softmax')
#     ])
model = build_1d_hsi_classifier(X_train.shape[1], num_classes)
# Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_test, axis=-1)
# Print classification report
print(classification_report(y_true_classes, y_pred_classes))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot training and validation accuracy
# history = model.history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




# Predict the entire image


# all_X = scaler.transform(reshaped_image)
# all_X = pca.transform(all_X)
# all_X = scaler_post_PCA.transform(all_X)


# predicted_labels = model.predict(all_X)
# predicted_labels_image = np.argmax(predicted_labels, axis=-1).reshape(h, w)

# # Visualize ground truth and predicted labels
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Ground truth map
# axes[0].imshow(ground_truth, cmap='jet')
# axes[0].set_title('Ground Truth')
# axes[0].axis('off')

# # Predicted map
# axes[1].imshow(predicted_labels_image, cmap='jet')
# axes[1].set_title('Predicted Map')
# axes[1].axis('off')

# plt.show()
