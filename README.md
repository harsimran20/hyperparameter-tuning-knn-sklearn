# Hyperparameter-Tuning-KNN-Sklearn

# KNN Hyperparameter Tuning using GridSearchCV

## 📌 Overview

This project demonstrates hyperparameter tuning of the K-Nearest Neighbors (KNN) algorithm using Grid Search with cross-validation. The goal is to identify the optimal number of neighbors (`n_neighbors`) to improve model performance.

The implementation uses the scikit-learn framework for model training, evaluation, and tuning.

---

## ⚙️ Features

* KNN classification model implementation
* Hyperparameter tuning using GridSearchCV
* 5-fold cross-validation
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
* Tabular visualization of cross-validation results

---

## 🛠️ Technologies Used

* Python
* scikit-learn
* Pandas
* NumPy (optional for preprocessing)

---

## 📂 Project Structure

```
├── knn_gridsearch.py        # Main implementation script
├── README.md               # Project documentation
```

---

## 🚀 How It Works

1. Initialize the KNN classifier.
2. Define a parameter grid for `n_neighbors`.
3. Apply GridSearchCV with cross-validation.
4. Train the model on scaled training data.
5. Evaluate performance on test data.
6. Analyze cross-validation results.

---

## 🧪 Code Snippet

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

param_grid = {"n_neighbors": [3, 5, 7, 9]}

grid = GridSearchCV(classifier, param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
```

---

## 📊 Evaluation Metrics

* **Accuracy**: Measures overall correctness
* **Precision**: Measures correctness of positive predictions
* **Recall**: Measures model's ability to detect positives

---

## ⚠️ Important Notes

* Ensure data is scaled before using KNN.
* Use `average='weighted'` for multiclass classification in precision and recall.
* Use `cv_results_` (not `cv_results`) to extract GridSearch results.

---

## 📈 Example Output

```
Accuracy Score: 92.5 %
Precision Score: 91.8 %
Recall Score: 92.1 %
```

---

## 🧠 Future Improvements

* Add visualization of performance vs. `n_neighbors`
* Extend to other hyperparameters (e.g., distance metrics)
* Compare with other algorithms (SVM, Random Forest)

---
