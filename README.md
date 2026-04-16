# ML Internship Projects – Codveda

A collection of machine learning projects completed during my internship at Codveda. The work is split into three task groups, each with three difficulty levels: beginner, intermediate, and advanced.

All notebooks are self-contained — no local file paths, no external CSV downloads needed. Just clone and run.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Task 1 – Data Preprocessing & Classification](#task-1--data-preprocessing--classification)
- [Task 2 – Regression & Classification Models](#task-2--regression--classification-models)
- [Task 3 – KNN & Clustering](#task-3--knn--clustering)
- [Setup & Installation](#setup--installation)
- [Tech Stack](#tech-stack)

---


---

## Task 1 – Data Preprocessing & Classification

### Level 1 — Data Preprocessing (`Task1_1.ipynb`)

The foundation of any ML pipeline. Raw data almost never comes in a model-ready state, so this notebook covers the full preprocessing workflow on the Iris dataset.

**What it does:**
- Loads the Iris dataset via seaborn (no CSV needed)
- Checks for missing values
- Encodes the target label (`species`) with `LabelEncoder` and prints the class-to-integer mapping
- Normalises all features to [0, 1] using `MinMaxScaler`
- Splits into 80/20 train/test sets with a fixed random seed

**Key concept:** Fitting the scaler on training data only, then applying it to the test set — the correct way to prevent data leakage.

---

### Level 3 — Random Forest Classifier (`Task1_3.ipynb`)

Takes the same Iris dataset further with an ensemble model and systematic hyperparameter search.

**What it does:**
- Runs `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split`
- Reports cross-validation accuracy with standard deviation
- Plots feature importances (which measurements drove the most decisions)
- Renders a confusion matrix heatmap

**Key concept:** Grid search with 5-fold CV finds the best hyperparameter combination without touching the test set, keeping the final evaluation honest.

**Results:** Consistently achieves ~97% test accuracy on Iris.

---

## Task 2 – Regression & Classification Models

### Level 1 — Simple Linear Regression (`task2_1.ipynb`)

**Dataset:** California Housing (sklearn built-in, ~20,000 samples)

Demonstrates how a single feature — median household income — can predict house values, and explains both the coefficient and the model's limitations.

**What it does:**
- Selects `MedInc` as the predictor (highest correlation with target)
- Trains `LinearRegression` and prints a plain-English interpretation of the slope
- Evaluates with R² and RMSE
- Plots actual data points against the fitted regression line

**Results:** R² ≈ 0.47 — income explains about half the variance in house prices; a solid result for a single-feature model.

---

### Level 2 — Decision Tree Classifier (`task2_2.ipynb`)

**Dataset:** Iris

Decision Trees are the most interpretable model in supervised learning. This notebook focuses on pruning to avoid overfitting and visualising the actual decision rules.

**What it does:**
- Trains a `DecisionTreeClassifier` with `max_depth=3` and entropy criterion
- Evaluates precision, recall, and F1 per class
- Renders the full tree structure — every split, threshold, and leaf label visible

**Key concept:** `max_depth=3` limits the tree to 8 possible leaves, which prevents memorisation of the training data while keeping the rules readable.

---

### Level 3 — Support Vector Machine (`task2_3.ipynb`)

**Dataset:** Breast Cancer Wisconsin (sklearn built-in, 569 samples, binary classification)

SVMs are particularly well-suited to medical diagnosis because they maximise the margin between classes — that margin acts as a safety buffer for borderline cases.

**What it does:**
- Restricts to 2 features (Mean Radius, Mean Texture) to enable 2D visualisation
- Scales features with `StandardScaler` (essential for SVMs)
- Compares Linear and RBF kernels on accuracy and AUC-ROC
- Plots the RBF decision boundary across the full feature space

**Results:** RBF kernel typically reaches ~90% accuracy and AUC > 0.93 on this 2-feature subset.

---

## Task 3 – KNN & Clustering

### Level 1 — K-Nearest Neighbours (`task3_1.ipynb`)

**Dataset:** Wine (sklearn built-in, 178 samples, 13 chemical features, 3 classes)

KNN is one of the most intuitive algorithms — a sample is classified by the majority vote of its K closest neighbours. The catch is that distance calculations are meaningless without scaling.

**What it does:**
- Scales features with `StandardScaler` before any distance calculations
- Tests K = 1, 3, 5, 7, 9 and plots accuracy vs K
- Selects the best K and prints a full classification report
- Renders a confusion matrix with class names

**Key concept:** Without scaling, the Proline feature (range 400–1700) would completely dominate Euclidean distances over features like Flavanoids (range 0.3–5.1).

---

### Level 2 — K-Means Clustering (`task3_2.ipynb`)

**Dataset:** Simulated retail customer data (200 samples, 2 features: annual income + spending score)

Unlike the other notebooks, this one is fully unsupervised — there are no labels. The goal is to discover natural groupings in customer behaviour.

**What it does:**
- Simulates a realistic mall-customer dataset with reproducible random seed
- Scales features before clustering
- Runs the Elbow Method (WCSS for K=1 to 10) to determine optimal K
- Fits K-Means with K=5 and plots segments with centroid markers
- Adds an interpretation cell that describes what each segment likely represents (e.g. high income + high spending = "Premium" customers)

**Key concept:** K-Means minimises within-cluster variance. Without scaling, income (larger range) would pull all centroids toward income differences while ignoring spending score variation.

---

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/Adejare-Workshop/Codveda-Technologies-Internship.git 
cd Codveda-Technologies-Internship

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch Jupyter
jupyter notebook
```

All datasets are loaded directly from `sklearn.datasets` or `seaborn` — no external files to download.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| pandas | latest | Data loading and manipulation |
| NumPy | latest | Numerical operations |
| scikit-learn | latest | All ML models and preprocessing |
| matplotlib | latest | Plots and visualisations |
| seaborn | latest | Statistical charts and dataset loading |
| Jupyter | latest | Interactive notebook environment |

---

## Author

**Adejare** — Freelance ML Engineer  
[GitHub](https://github.com/adejare-dev) · [LinkedIn](https://linkedin.com/in/adejare)
[portfolio](https://my-work-website.pages.dev/)
