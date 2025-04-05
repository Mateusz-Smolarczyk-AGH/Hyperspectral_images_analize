import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
# Removed unused import
# Removed unused import

# Load data
data = sio.loadmat('data/data/Indian_pines_data.mat')
gt_data = sio.loadmat('data/gt/Indian_pines_gt.mat')

# Assign data to variables
hyper_image = data[list(data.keys())[-1]]  # Assuming the last key contains the data
ground_truth = gt_data[list(gt_data.keys())[-1]]  # Assuming the last key contains ground truth
h, w, p = hyper_image.shape

# Reshape data for CNN
X = hyper_image.reshape(h, w, p)  # (height, width, bands)
y = ground_truth.ravel()  # Flatten ground truth

# Filter out background (class 0)
mask = y > 0
X = X[mask.reshape(h, w)].reshape(-1, p)
y = y[mask]

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data into training, validation, and testing sets based on image rows
split_ratio = 0.7  # 70% for training, 30% for testing
split_index = int(h * split_ratio)
# Training set (bottom
# Training set (bottom part of the image)
X_train = X[:split_index, :, :].reshape(-1, p)
y_train = y[:split_index, :].reshape(-1, y.shape[1])

# Testing set (top part of the image)
X_test = X[split_index:, :, :].reshape(-1, p)
y_test = y[split_index:, :].reshape(-1, y.shape[1])

# Reshape for CNN input
X_train = X_train.reshape(-1, 1, 1, p)
X_test = X_test.reshape(-1, 1, 1, p)

# Build CNN model
model = Sequential([
    Conv2D(64, (1, 1), activation='relu', input_shape=(1, 1, p)),
    MaxPooling2D((1, 1)),
    Conv2D(128, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Predict for the entire image
X_full = hyper_image.reshape(-1, 1, 1, p)
predicted_labels = model.predict(X_full)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Reshape predictions to original image dimensions
predicted_labels_image = predicted_labels.reshape(h, w)

# Visualize ground truth and predicted labels
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Ground truth map
axes[0].imshow(ground_truth, cmap='jet')
axes[0].set_title('Ground Truth')
axes[0].axis('off')

# Predicted map
axes[1].imshow(predicted_labels_image, cmap='jet')
axes[1].set_title('Predicted Map')
axes[1].axis('off')

plt.show()

