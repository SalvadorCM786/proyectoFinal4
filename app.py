import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import gdown

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Predicción de enfermedad cardíaca", layout="wide")

FILE_ID = "1T-kE0Ho5mEah1WBwZFeJpAIeyE0Di5cH"

FEATURE_LABELS = {
    "age": "Edad",
    "sex": "Sexo (1 = hombre, 0 = mujer)",
    "cp": "Tipo de dolor de pecho (0-3)",
    "trestbps": "Presión arterial en reposo (mm Hg)",
    "chol": "Colesterol (mg/dl)",
    "fbs": "Glucosa en ayunas > 120 mg/dl (1 = sí, 0 = no)",
    "restecg": "Resultado electrocardiograma en reposo (0-2)",
    "thalach": "Frecuencia cardíaca máxima alcanzada",
    "exang": "Angina inducida por ejercicio (1 = sí, 0 = no)",
    "oldpeak": "Depresión del ST inducida por ejercicio",
    "slope": "Pendiente del segmento ST (0-2)",
    "ca": "Número de vasos principales coloreados (0-4)",
    "thal": "Thal (0-3)",
}


# mismo dataset y limpieza que en el notebook (quita los 723 duplicados)
# uso gdown en vez de pd.read_csv directo porque Drive a veces regresa una
# página de confirmación (HTML) en lugar del CSV, y eso tronaba en Streamlit Cloud
@st.cache_data
def load_data():
    local_path = "heart.csv"
    gdown.download(id=FILE_ID, output=local_path, quiet=True)
    df = pd.read_csv(local_path)
    df = df.drop_duplicates()
    return df


# entrena los 4 modelos una sola vez y deja todo listo para las pestañas
@st.cache_resource
def train_models(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
        })
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Modelo"]
    best_model = trained[best_name]
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best, output_dict=True)

    return {
        "X": X,
        "scaler": scaler,
        "results_df": results_df,
        "best_name": best_name,
        "best_model": best_model,
        "confusion_matrix": cm,
        "report": report,
    }


def main():
    st.image("assets/banner.jpg", use_container_width=True)
    st.title("Predicción de enfermedad cardíaca")
    st.caption("App basada en el proyecto del módulo 3: carga y limpieza del dataset, comparación de 4 modelos y predicción para un paciente nuevo.")

    df = load_data()
    state = train_models(df)

    tab1, tab2, tab3 = st.tabs(["Dataset", "Comparación de modelos", "Predicción"])

    with tab1:
        st.subheader("Dataset (ya sin duplicados)")
        col1, col2 = st.columns(2)
        col1.metric("Registros", df.shape[0])
        col2.metric("Variables", df.shape[1] - 1)

        st.dataframe(df.head(20), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="target", ax=ax)
        ax.set_title("Distribución de la variable objetivo")
        ax.set_xlabel("Clase (0 = no, 1 = sí)")
        st.pyplot(fig)

    with tab2:
        st.subheader("Comparación de los 4 modelos")
        st.dataframe(
            state["results_df"].style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1-score": "{:.4f}"
            }),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=state["results_df"], x="Modelo", y="Accuracy", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy por modelo")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)

        st.success(f"Mejor modelo: **{state['best_name']}** "
                   f"(accuracy = {state['results_df'].iloc[0]['Accuracy']:.4f})")

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(state["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title(f"Matriz de confusión - {state['best_name']}")
        ax2.set_xlabel("Predicción")
        ax2.set_ylabel("Valor real")
        st.pyplot(fig2)

    with tab3:
        st.subheader(f"Predecir con el mejor modelo ({state['best_name']})")
        st.write("Llena los datos del paciente para obtener una predicción.")

        X = state["X"]
        input_values = {}
        cols = st.columns(3)
        for i, col_name in enumerate(X.columns):
            label = FEATURE_LABELS.get(col_name, col_name)
            col = cols[i % 3]
            min_v, max_v, mean_v = float(X[col_name].min()), float(X[col_name].max()), float(X[col_name].mean())
            if X[col_name].nunique() <= 5:
                options = sorted(X[col_name].unique().tolist())
                input_values[col_name] = col.selectbox(label, options, index=options.index(int(round(mean_v))) if int(round(mean_v)) in options else 0)
            else:
                input_values[col_name] = col.number_input(label, min_value=min_v, max_value=max_v, value=mean_v)

        if st.button("Predecir", type="primary"):
            patient_df = pd.DataFrame([input_values])[X.columns]
            patient_scaled = state["scaler"].transform(patient_df)
            pred = state["best_model"].predict(patient_scaled)[0]
            proba = None
            if hasattr(state["best_model"], "predict_proba"):
                proba = state["best_model"].predict_proba(patient_scaled)[0][1]

            if pred == 1:
                st.error(f"El modelo predice **enfermedad cardíaca presente**"
                         + (f" (probabilidad: {proba:.1%})" if proba is not None else ""))
            else:
                st.success(f"El modelo predice **sin enfermedad cardíaca**"
                           + (f" (probabilidad de estar enfermo: {proba:.1%})" if proba is not None else ""))

            st.caption("Esto es solo un ejercicio del proyecto de ML, no un diagnóstico médico.")


if __name__ == "__main__":
    main()