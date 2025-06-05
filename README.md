# Analiza obrazów hiperspektralnych z wykorzystaniem lasów losowych

Projekt dotyczy segmentacji obrazów hiperspektralnych przy użyciu algorytmu lasu losowego (*Random Forest*). Do eksperymentów wykorzystano popularne bazy danych: **Pavia Centre**, **Pavia University** **Indian Pines** oraz **Salinas**.

## Opis metody

Każdy obraz został potraktowany jako zbiór próbek — pojedynczych pikseli, z których każdy opisany był wektorem cech o długości odpowiadającej liczbie kanałów spektralnych. Następnie dane zostały podzielone na zbiór uczący (30%) oraz testowy (70%), z pominięciem pikseli tła.

Do klasyfikacji wykorzystano klasyfikator **RandomForestClassifier** z biblioteki `scikit-learn`. Dobór optymalnych hiperparametrów przeprowadzono z użyciem **GridSearchCV**. Wstępnie analizowano również zastosowanie PCA do redukcji wymiarowości, jednak redukcja pogarszała skuteczność modelu i ostatecznie z niej zrezygnowano.

## Wyniki

Dla obu baz danych osiągnięto wysoką dokładność klasyfikacji:

- **Pavia Centre**: Accuracy = **98.42%**
- **Pavia University**: Accuracy = **91.53%**
- **Indian Pines**: Accuracy = **82,2%**
- **Salinas**: Accuracy = **92,44%**
Macierze pomyłek (confusion matrices) dla każdego przypadku znajdują się w folderze [`/results`](./results) i prezentują szczegółowe wyniki dla zbioru testowego.

Dodatkowo, po wytrenowaniu modelu dokonano predykcji całego obrazu. Dzięki pominięciu klasy tła w procesie uczenia, każdy piksel został przypisany do jednej z rzeczywistych klas, co skutkuje spójną i kompletną segmentacją obrazu. Taki zabieg znacząco poprawia jakość końcowego wyniku w porównaniu do klasyfikacji uwzględniającej tło.

## Struktura repozytorium

- `/src/random_forest.py` – główny skrypt odpowiedzialny za przygotowanie danych, trenowanie modelu oraz ewaluację.
- `/results/` – folder z wynikami klasyfikacji: macierze pomyłek, mapy predykcji, dokładności.
- `/data/` – dane wejściowe (obrazy hiperspektralne oraz maski klas). Dodano tylko Pavia University, gdyż Pavia Centre jest zbyt duże. Można je pobrać tu: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

