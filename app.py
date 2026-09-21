import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Modelo de predicción de Enfermedad Cardíaca ''')
st.image("enfermedad.jpg", caption="Las enfermedades cardiacas son una de las principales causas de muerte.")

st.header('Datos del paciente')

def user_input_features():
    age = st.number_input('Edad:', min_value=20, max_value=100, value=50, step=1)
    sex = st.number_input('Sexo (1=Hombre, 0=Mujer):', min_value=0, max_value=1, value=1, step=1)
    cp = st.number_input('Tipo de dolor en el pecho (0-3):', min_value=0, max_value=3, value=0, step=1)
    trestbps = st.number_input('Presión arterial en reposo:', min_value=80, max_value=200, value=120, step=1)
    chol = st.number_input('Colesterol sérico:', min_value=100, max_value=600, value=200, step=1)
    fbs = st.number_input('Glucosa en ayunas > 120 mg/dl (1=Sí,0=No):', min_value=0, max_value=1, value=0, step=1)
    restecg = st.number_input('Resultados ECG en reposo (0-2):', min_value=0, max_value=2, value=1, step=1)
    thalach = st.number_input('Frecuencia cardíaca máxima alcanzada:', min_value=70, max_value=220, value=150, step=1)
    exang = st.number_input('Angina inducida por ejercicio (1=Sí,0=No):', min_value=0, max_value=1, value=0, step=1)
    oldpeak = st.number_input('Depresión ST inducida por ejercicio:', min_value=0.0, max_value=6.5, value=1.0, step=0.1)
    slope = st.number_input('Pendiente del segmento ST (0-2):', min_value=0, max_value=2, value=1, step=1)
    ca = st.number_input('Número de vasos coloreados (0-3):', min_value=0, max_value=3, value=0, step=1)
    thal = st.number_input('Talasemia (1=Normal,2=Defecto fijo,3=Defecto reversible):', min_value=0, max_value=3, value=2, step=1)

    user_input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

# Cargar dataset
heart = pd.read_csv('heart2.csv')
X = heart.drop(columns='target')
Y = heart['target']

# Entrenar modelo Árbol de Decisión
classifier = DecisionTreeClassifier(
    max_depth=6, criterion='entropy', min_samples_leaf=10, random_state=0
)
classifier.fit(X, Y)

# Predicción
prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction[0] == 0:
    st.write('No presenta enfermedad cardíaca')
elif prediction[0] == 1:
    st.write('Presenta enfermedad cardíaca')
else:
    st.write('Sin predicción')