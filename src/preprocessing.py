import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import eigh

class MNFTransform:
    def __init__(self, num_components=None):
        self.num_components = num_components
        self.whitening_matrix = None
        self.pca = None

    def estimate_noise_covariance(self, image):
        """
        Szacowanie macierzy kowariancji szumu bez pętli for.
        Zamiast iterować po pikselach, stosujemy przesunięcia macierzy (NumPy slicing).
        """
        shifted_image = image[:, 1:, 1:] - image[:, :-1, :-1]  # Przesunięcie o 1 piksel (różnica sąsiednich pikseli)
        noise = shifted_image.reshape(image.shape[0], -1)  # Przekształcenie do 2D (bands x pixels)
        noise_cov = np.cov(noise)  # Macierz kowariancji szumu
        return noise_cov

    def fit(self, image):
        """
        Trenuje transformację MNF na obrazie referencyjnym.
        """
        bands, rows, cols = image.shape
        image_reshaped = image.reshape(bands, -1)  # Konwersja do 2D (pasma x piksele)

        # 1️⃣ Szacowanie macierzy kowariancji szumu
        noise_cov = self.estimate_noise_covariance(image)

        # 2️⃣ Whitening danych (usuwanie wpływu szumu)
        w, v = eigh(noise_cov)  # Wartości i wektory własne
        self.whitening_matrix = np.linalg.inv(np.sqrt(np.diag(w))) @ v.T  # Macierz whitening

        whitened_data = self.whitening_matrix @ image_reshaped  # Przekształcenie danych

        # 3️⃣ PCA na oczyszczonych danych
        self.pca = PCA(n_components=self.num_components)
        self.pca.fit(whitened_data.T)  # Dopasowanie PCA

    def transform(self, image):
        """
        Wykonuje transformację MNF na nowym obrazie (inferencja).
        Wymaga wcześniejszego dopasowania modelu (fit).
        """
        if self.whitening_matrix is None or self.pca is None:
            raise ValueError("Model MNF nie został jeszcze wytrenowany. Najpierw użyj .fit()")

        bands, rows, cols = image.shape
        image_reshaped = image.reshape(bands, -1)  # Przekształcenie do 2D

        # 1️⃣ Whitening danych
        whitened_data = self.whitening_matrix @ image_reshaped  

        # 2️⃣ Transformacja PCA (MNF)
        mnf_data = self.pca.transform(whitened_data.T).T  

        # 3️⃣ Przekształcenie do oryginalnego kształtu
        mnf_image = mnf_data.reshape(-1, rows, cols)
        return mnf_image
    

class PixelPurityIndex:
    def __init__(self, num_projections=5000, num_endmembers=5):
        """
        Pixel Purity Index (PPI) do ekstrakcji endmemberów.

        Parametry:
        - num_projections: liczba losowych rzutowań
        - num_endmembers: liczba wybranych endmemberów
        """
        self.num_projections = num_projections
        self.num_endmembers = num_endmembers

    def fit(self, hyperspectral_image):
        """
        Oblicza indeks PPI oraz zwraca endmembery.

        Parametry:
        - hyperspectral_image: numpy array (bands, rows, cols)
        
        Zwraca:
        - ppi_image: Mapa PPI (rows, cols)
        - endmembers: Wybrane endmembery (bands, num_endmembers)
        """
        bands, rows, cols = hyperspectral_image.shape
        num_pixels = rows * cols

        # 1️⃣ Przekształcenie obrazu do 2D (bands, num_pixels)
        image_reshaped = hyperspectral_image.reshape(bands, num_pixels)

        # 2️⃣ Inicjalizacja licznika PPI (dla każdego piksela)
        ppi_counts = np.zeros(num_pixels, dtype=int)

        # 3️⃣ Generowanie losowych wektorów projekcji (bands, num_projections)
        random_vectors = np.random.randn(bands, self.num_projections)

        # 4️⃣ Rzutowanie pikseli na losowe wektory (produkty skalarne)
        projections = np.dot(random_vectors.T, image_reshaped)  # (num_projections, num_pixels)

        # 5️⃣ Identyfikacja pikseli o wartościach MIN i MAX dla każdej projekcji
        min_indices = np.argmin(projections, axis=1)
        max_indices = np.argmax(projections, axis=1)

        # 6️⃣ Zliczanie, ile razy dany piksel znalazł się na krańcu rzutowania
        np.add.at(ppi_counts, min_indices, 1)
        np.add.at(ppi_counts, max_indices, 1)

        # 7️⃣ Przekształcenie wyników do obrazu 2D
        ppi_image = ppi_counts.reshape(rows, cols)

        # 8️⃣ Wybór pikseli o najwyższym PPI jako endmembery
        top_indices = np.argsort(ppi_counts)[-self.num_endmembers:]  # Indeksy top endmemberów
        endmembers = image_reshaped[:, top_indices]  # Pobranie spektralnych wartości tych pikseli
        
        return ppi_image, endmembers


        


class SpectralAngleMapper:
    def __init__(self, endmembers):
        """
        endmembers: numpy array o wymiarach (bands, num_endmembers)
        """
        self.endmembers = endmembers / np.linalg.norm(endmembers, axis=0, keepdims=True)  # Normalizacja

    def fit(self, hyperspectral_image):
        """
        Oblicza mapę kątów SAM dla całego obrazu hiperspektralnego.
        
        hyperspectral_image: numpy array o wymiarach (bands, rows, cols)
        """
        bands, rows, cols = hyperspectral_image.shape
        num_pixels = rows * cols
        num_endmembers = self.endmembers.shape[1]

        # Przekształcenie do postaci 2D: (bands, num_pixels)
        image_reshaped = hyperspectral_image.reshape(bands, num_pixels)
        
        # Normalizacja widm pikseli
        image_normalized = image_reshaped / np.linalg.norm(image_reshaped, axis=0, keepdims=True)

        # Obliczenie cos(θ) dla każdego piksela i endmembera (macierzowy iloczyn skalarny)
        cos_theta = np.dot(self.endmembers.T, image_normalized)  # (num_endmembers, num_pixels)
        cos_theta = np.clip(cos_theta, -1, 1)  # Uniknięcie błędów numerycznych

        # Obliczenie kąta θ (w radianach)
        sam_angles = np.arccos(cos_theta)  # (num_endmembers, num_pixels)

        # Przypisanie piksela do endmembera z najmniejszym kątem
        classification_map = np.argmin(sam_angles, axis=0)

        # Rekonstruowanie obrazu 2D
        self.sam_image = classification_map.reshape(rows, cols)

    def get_sam_image(self):
        """Zwraca obraz klasyfikacji SAM."""
        if hasattr(self, "sam_image"):
            return self.sam_image
        else:
            raise ValueError("Najpierw uruchom `fit(hyperspectral_image)`.")