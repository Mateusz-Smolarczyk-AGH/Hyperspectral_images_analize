from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier

def load_clr(filepath):
    class_colors = {}
    class_labels = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            r, g, b = map(int, parts[1:4])
            name = parts[4].split(':')[1].replace('~', '')
            class_colors[class_id] = (r/255, g/255, b/255)  # Normalizujemy RGB do 0-1
            class_labels[class_id] = name
    return class_colors, class_labels

def load_data(dataset, keep_background=False):
    data = sio.loadmat('data/data/' + dataset +'_data.mat')
    gt_data = sio.loadmat('data/gt/' + dataset + '_gt.mat')

    hyper_image = data[list(data.keys())[-1]]  # Zakładam, że ostatni klucz to dane
    ground_truth = gt_data[list(gt_data.keys())[-1]]  # Zakładam, że ostatni klucz to ground truth
    h, w, p = hyper_image.shape
    X = hyper_image.reshape(-1, p)  # (liczba_pikseli, liczba_pasm)
    y = ground_truth.ravel()  # (liczba_pikseli,)
    if keep_background:
        return X, y, h, w, ground_truth, np.ones_like(y, dtype=int)

    mask = (y != 0).astype(int)  # maska z 0 i 1
    X_filtered = X[mask == 1]  # zachowujemy tylko elementy, gdzie y != 0
    y_filtered = y[mask == 1]
    return X_filtered, y_filtered, h, w, ground_truth, mask

def evaluation_custom(y_test, y_pred, class_colors, class_labels):
    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Wyświetlenie wyników
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_rep)

    # Wyświetlenie macierzy pomyłek
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
 

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[class_labels[i] for i in sorted(class_labels)],
                yticklabels=[class_labels[i] for i in sorted(class_labels)],
                ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Znormalizowana Macierz Pomyłek')


    patches = [mpatches.Patch(color=class_colors[i], label=class_labels[i]) for i in sorted(class_labels)]
    # Legenda jako osobny blok z boku
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.35, 0.5), title='Klasy')

    plt.tight_layout()

def evaluation(y_test, y_pred, predicted_labels, y):
    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y, predicted_labels)

    # Wyświetlenie wyników
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_rep)

    # Wyświetlenie macierzy pomyłek
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
 

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Znormalizowana Macierz Pomyłek')

    plt.tight_layout()

def generate_image_custom(class_colors, ground_truth, predicted_labels_image, class_labels):
    # Rysujemy obrazki z legendą
    clr_colors_list = [class_colors[i] for i in sorted(class_colors)]

    # Tworzymy niestandardową colormapę
    custom_cmap = ListedColormap(clr_colors_list)

    # Rysowanie z poprawną colormapą
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    axes[0].imshow(ground_truth, cmap=custom_cmap)
    axes[0].set_title('Rzeczywista mapa klas (Ground Truth)')
    axes[0].axis('off')

    axes[1].imshow(predicted_labels_image, cmap=custom_cmap)
    axes[1].set_title('Przewidywana mapa klas')
    axes[1].axis('off')

    # Dodanie legendy jak wcześniej
    patches = [mpatches.Patch(color=class_colors[i], label=class_labels[i]) for i in sorted(class_labels)]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

def generate_image(ground_truth, predicted_labels_image):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(ground_truth)
    axes[0].set_title('Rzeczywista mapa klas (Ground Truth)')
    axes[0].axis('off')

    axes[1].imshow(predicted_labels_image)
    axes[1].set_title('Przewidywana mapa klas')
    axes[1].axis('off')

    plt.tight_layout()

def main(test_size, data_path, gt_names=None, keep_background=False):
    X, y, h, w, ground_truth, mask = load_data(data_path, keep_background)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # pca = PCA(n_components=0.95)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # X = pca.transform(X)
    # print(f"pca: {pca.n_components_}")

    best_rf = RandomForestClassifier(
        ccp_alpha=0.000102,
        max_depth=None,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200,
        class_weight='balanced',  # jeśli używałeś wcześniej
        random_state=42
    )
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)

    # Przewidywanie klasyfikacji dla całego obrazu
    predicted_labels = best_rf.predict(X)
    y_reconstructed = np.zeros_like(mask)        # utwórz nową tablicę zer
    y_reconstructed[mask == 1] = predicted_labels 
    # Przekształcenie wyników do oryginalnych wymiarów obrazu
    predicted_labels_image = y_reconstructed.reshape(h, w)
    if gt_names is not None:
        class_colors, class_labels = load_clr(gt_names)
        generate_image_custom(class_colors, ground_truth, predicted_labels_image, class_labels)
        if not keep_background:
            del class_labels[0]
            del class_colors[0]

        evaluation_custom(y_test, y_pred, class_colors, class_labels)

        plt.figure(figsize=(12, 6))
        class_colors[0] = np.array([0.8, 0.8, 0.8])

        for label in class_labels:
            X_class = X[y == label]
            mean_spectrum = X_class.mean(axis=0)
            std_spectrum = X_class.std(axis=0)

            color = np.array(class_colors[label])
            nazwa = class_labels[label]

            plt.plot(mean_spectrum, color=color, label=f'{nazwa} (klasa {label})')
            plt.fill_between(
                np.arange(X.shape[1]),
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                color=color,
                alpha=0.3
            )

        plt.xlabel('Numer kanału')
        plt.ylabel('Wartość piksela')
        plt.title('Średnie spektrum z odchyleniem standardowym dla każdej klasy')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))

        importances = best_rf.feature_importances_
        plt.plot(importances)
        plt.show()
    else:
        generate_image(ground_truth, predicted_labels_image)

        evaluation(y_test, y_pred, predicted_labels, y)

        plt.figure(figsize=(12, 6))
        
        for label in range(1, 10):
            X_class = X[y == label]
            mean_spectrum = X_class.mean(axis=0)
            std_spectrum = X_class.std(axis=0)
            

            plt.plot(mean_spectrum)
            plt.fill_between(
                np.arange(X.shape[1]),
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.3
            )

        plt.xlabel('Numer kanału')
        plt.ylabel('Wartość piksela')
        plt.title('Średnie spektrum z odchyleniem standardowym dla każdej klasy')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
#main("Pavia", r'data\gt\19920612_AVIRIS_IndianPine_Site3_gr.clr')
main(0.7, "Pavia", "data//gt//Pavia_names.clr", keep_background=True)
