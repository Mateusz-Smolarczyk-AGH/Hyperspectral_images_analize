import scipy.io as sio
import matplotlib.pyplot as plt
from spectral import kmeans
from sklearn.decomposition import PCA
import numpy as np

# Wczytywanie danych
data = sio.loadmat('data/data/Lublin2_wycinek.mat')
hyper_image = data[list(data.keys())[-1]]  # Zakładam, że ostatni klucz to dane
h, w, p = hyper_image.shape

# Przekształcenie obrazu hiperspektralnego w wektory cech
X = hyper_image.reshape(-1, p)  # (pixels, bands)

# Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
n_components = X_pca.shape[1]

print(f"Zredukowana liczba komponentów PCA: {n_components}")

# Przekształcenie z powrotem do kształtu obrazu 3D
image_pca = X_pca.reshape(h, w, n_components)

# K-means na PCA-zredukowanym obrazie
n_clusters = 20
class_map, centers = kmeans(image_pca, nclusters=n_clusters, max_iterations=50)
end_clusters = len(np.unique(class_map))
# Wizualizacja mapy klas
plt.figure(figsize=(6, 6))
plt.imshow(class_map, cmap='jet')
plt.title(f'Przewidywana mapa klas\n (PCA={n_components}, klastry startowe={n_clusters},\n klastry końcowe={end_clusters})')
plt.axis('off')
plt.savefig(f'pca_{n_components}_sc_{n_clusters}_ec_{end_clusters}.png')
plt.show()