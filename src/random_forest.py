from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
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

def load_data(dataset):
    data = sio.loadmat('data/data/' + dataset +'_data.mat')
    gt_data = sio.loadmat('data/gt/' + dataset + '_gt.mat')

    hyper_image = data[list(data.keys())[-1]]  # Zakładam, że ostatni klucz to dane
    ground_truth = gt_data[list(gt_data.keys())[-1]]  # Zakładam, że ostatni klucz to ground truth
    h, w, p = hyper_image.shape
    X = hyper_image.reshape(-1, p)  # (liczba_pikseli, liczba_pasm)
    y = ground_truth.ravel()  # (liczba_pikseli,)
    return X, y, h, w, ground_truth

def evaluation(y_test, y_pred, predicted_labels, y, class_colors, class_labels):
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

def generate_image(class_colors, ground_truth, predicted_labels_image, class_labels):
    # Rysujemy obrazki z legendą
    clr_colors_list = [class_colors[i] for i in sorted(class_colors)]

    # Tworzymy niestandardową colormapę
    custom_cmap = ListedColormap(clr_colors_list)

    # Rysowanie z poprawną colormapą
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

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
    plt.show()


def main():
    X, y, h, w, ground_truth = load_data("Indian_pines")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    # p = 20

    # pca = PCA(n_components=p)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # X = pca.transform(X)

    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    # rf_classifier = DecisionTreeClassifier(class_weight='balanced', random_state=42)

    # rf_classifier = RandomForestClassifier(
    # class_weight='balanced',
    # n_estimators=100,
    # max_depth=5,              # każde drzewo ma ograniczoną głębokość
    # min_samples_leaf=5,       # liście nie mogą być zbyt małe
    # ccp_alpha=0.01,           # pruning wewnątrz drzew w lesie
    # random_state=42
    # )
#     rf_classifier = DecisionTreeClassifier(
#     class_weight='balanced',
#     # max_depth=5,              # maksymalna głębokość drzewa
#     # min_samples_split=10,     # minimalna liczba próbek do podziału
#     # min_samples_leaf=5,       # minimalna liczba próbek w liściu
#     # ccp_alpha=0.005,           # pruning przez koszt złożoności 
#     random_state=42
# )

    # path = rf_classifier.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas = path.ccp_alphas
    # print(len(ccp_alphas))

    # Dla każdej wartości alpha trenujemy osobne drzewo
    # trees = [DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train) for alpha in ccp_alphas]
    print("Parametry lasu:")
    print(rf_classifier.get_params())
    # Fit the classifier to the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Przewidywanie klasyfikacji dla całego obrazu
    predicted_labels = rf_classifier.predict(X)

    # Przekształcenie wyników do oryginalnych wymiarów obrazu
    predicted_labels_image = predicted_labels.reshape(h, w)

    clr_path = r'data\gt\19920612_AVIRIS_IndianPine_Site3_gr.clr'
    class_colors, class_labels = load_clr(clr_path)
    evaluation(y_test, y_pred, predicted_labels, y, class_colors, class_labels)
    generate_image(class_colors, ground_truth, predicted_labels_image, class_labels)
main()