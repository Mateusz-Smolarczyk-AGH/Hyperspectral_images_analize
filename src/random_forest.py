import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('data/data/Indian_pines_data.mat')
gt_data = sio.loadmat('data/gt/Indian_pines_gt.mat')
# Przypisanie danych do zmiennych
hyper_image = data[list(data.keys())[-1]]  # Zakładam, że ostatni klucz to dane
ground_truth = gt_data[list(gt_data.keys())[-1]]  # Zakładam, że ostatni klucz to ground truth
h, w, p = hyper_image.shape

X = hyper_image.reshape(-1, p)  # (liczba_pikseli, liczba_pasm)
y = ground_truth.ravel()  # (liczba_pikseli,)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)


# Przewidywanie klasyfikacji dla całego obrazu
predicted_labels = rf_classifier.predict(X)

# Przekształcenie wyników do oryginalnych wymiarów obrazu
predicted_labels_image = predicted_labels.reshape(h, w)

# Wizualizacja rzeczywistej mapy klas i przewidywanej mapy
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Mapa rzeczywista (ground truth)
axes[0].imshow(ground_truth, cmap='jet')
axes[0].set_title('Rzeczywista mapa klas (Ground Truth)')
axes[0].axis('off')

# Mapa przewidywana
axes[1].imshow(predicted_labels_image, cmap='jet')
axes[1].set_title('Przewidywana mapa klas')
axes[1].axis('off')

plt.show()

