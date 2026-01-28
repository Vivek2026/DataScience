
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier()
rf_random_search = RandomizedSearchCV(rf, rf_params, scoring='f1', cv=5, random_state=42)
rf_random_search.fit(X_train, y_train)
evaluate_model("Random Forest (Tuned)", rf_random_search.best_estimator_, X_test, y_test)


svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC()
svm_grid_search = GridSearchCV(svm, svm_params, scoring='f1', cv=5)
svm_grid_search.fit(X_train, y_train)
evaluate_model("SVM (Tuned)", svm_grid_search.best_estimator_, X_test, y_test)


results = {}
for name, model in [
    ("Logistic Regression", models["Logistic Regression"]),
    ("Random Forest (Tuned)", rf_random_search.best_estimator_),
    ("SVM (Tuned)", svm_grid_search.best_estimator_)
]:
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results[name] = f1

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with F1-Score = {results[best_model]:.4f}")

