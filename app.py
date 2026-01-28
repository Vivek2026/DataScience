# app.py
import os, warnings, streamlit as st
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import joblib, shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

MODEL_PATH = "iris_model.pkl"

# -------------------------------------------------
# 1. Train or load the model
# -------------------------------------------------
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    return clf

model = get_model()
iris = load_iris()
feature_names = [s.replace(" (cm)", "") for s in iris.feature_names]
target_names = iris.target_names

# -------------------------------------------------
# 2. Sidebar – user inputs
# -------------------------------------------------
st.sidebar.title("🔧 Iris Flower Parameters")

def user_input():
    data = {
        "sepal length": st.sidebar.slider("Sepal length (cm)", 4.3, 7.9, 5.8),
        "sepal width":  st.sidebar.slider("Sepal width (cm)", 2.0, 4.4, 3.0),
        "petal length": st.sidebar.slider("Petal length (cm)", 1.0, 6.9, 2.2),
        "petal width":  st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.3),
    }
    return pd.DataFrame([data])

X_in = user_input()

# -------------------------------------------------
# 3. Predict
# -------------------------------------------------
proba = model.predict_proba(X_in)[0]
pred_class = int(model.predict(X_in)[0])

st.title("🌸 Iris Classifier")
st.write("### Current input")
st.dataframe(X_in.style.format("{:.2f}"))

col1, col2 = st.columns(2)
col1.metric("Predicted species", target_names[pred_class])
col2.metric("Confidence", f"{proba[pred_class]:.1%}")

# -------------------------------------------------
# 4. Probability bar chart
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 2))
sns.barplot(x=proba, y=target_names, ax=ax, palette="viridis")
ax.set_xlim(0, 1)
ax.set_xlabel("Probability")
st.pyplot(fig)

# -------------------------------------------------
# 5. SHAP explanation
# -------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_in)

st.write("### 🧠 Local explanation (SHAP waterfall)")
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[pred_class][0],
        base_values=explainer.expected_value[pred_class],
        data=X_in.iloc[0],
        feature_names=feature_names,
    ),
    max_display=10,
    show=False,
)
st.pyplot(plt.gcf())

with st.expander("📊 Training data preview"):
    df_preview = pd.DataFrame(iris.data, columns=feature_names)
    df_preview["species"] = iris.target_names[iris.target]
    st.dataframe(df_preview.head())