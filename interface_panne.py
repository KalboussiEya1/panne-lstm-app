import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from tensorflow.keras.models import load_model
import joblib

# Config Streamlit
st.set_page_config(page_title="Prédiction de panne (LSTM pour la maintenance prédictive)", layout="centered")

st.title("Interface de prédiction de panne (LSTM pour la maintenance prédictive)")


# Upload scaler.pkl optionnel
uploaded_scaler_file = st.file_uploader(
    " Optionnel : Charger un scaler (.pkl)",
    type=["pkl"]
)

@st.cache_resource
def load_default_scaler():
    return joblib.load("scaler.pkl")

if uploaded_scaler_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(uploaded_scaler_file.read())
        scaler_path = tmp.name
    scaler = joblib.load(scaler_path)
    st.success(" Scaler chargé avec succès.")
else:
    scaler = load_default_scaler()

# Upload modèle optionnel
uploaded_model_file = st.file_uploader(
    "Optionnel : Charger un modèle (.h5 ou .keras)",
    type=["h5", "keras"]
)

@st.cache_resource
def load_default_model():
    return load_model("mon_model_seq3_horizon2.keras", compile=False)

if uploaded_model_file:
    suffix = ".keras" if uploaded_model_file.name.endswith(".keras") else ".h5"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_model_file.read())
        model_path = tmp.name
    try:
        model = load_model(model_path, compile=False)
        st.success(" Modèle chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        model = load_default_model()
else:
    model = load_default_model()

features = [
    'Température', 'Vibration', 'Pression',
    'Bruit', 'Intensité électrique', 'Temps de fonctionnement'
]

input_data = None

# Titre section données
st.header("Entrer les données capteurs (3 lignes)")

# Upload CSV exemple entre header et saisie manuelle
uploaded_csv_example = st.file_uploader(
    "Charger un fichier CSV exemple (3 lignes avec colonne 'Panne')",
    type=["csv"],
    key="csv_example"
)

if uploaded_csv_example:
    try:
        input_data = pd.read_csv(uploaded_csv_example)
        expected_cols = features + ['Panne']
        if input_data.shape[0] != 3 or not all(col in input_data.columns for col in expected_cols):
            st.error("Le fichier doit contenir exactement 3 lignes et toutes les colonnes (y compris 'Panne').")
            input_data = None
        else:
            st.success("Données exemple chargées avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV exemple : {e}")
        input_data = None

st.markdown("Saisir manuellement les valeurs :")

# Saisie manuelle si pas de CSV uploadé
if input_data is None:
    data = []
    panne_vals = []
    for i in range(3):
        st.markdown(f"**Ligne {i+1}**")
        row = []
        for feat in features:
            val = st.number_input(f"{feat} (ligne {i+1})", format="%.2f", key=f"{feat}_{i}")
            row.append(val)
        panne_val = st.number_input(f"Panne (ligne {i+1})", min_value=0, max_value=1, step=1, key=f"Panne_{i}")
        panne_vals.append(panne_val)
        data.append(row)
    input_data = pd.DataFrame(data, columns=features)
    input_data['Panne'] = panne_vals

# Afficher le résultat de la prédiction, même si vide au départ
st.subheader(" Résultat de la prédiction")

# Condition pour savoir si on a des données non nulles (pour commencer à prédire)
def data_entered(df, feats):
    return (df[feats] != 0).any().any()

if input_data is not None and data_entered(input_data, features):
    try:
        X_scaled = scaler.transform(input_data[features])
        X_seq = np.expand_dims(X_scaled, axis=0)  # (1, 3, n_features)
        y_pred_proba = model.predict(X_seq).flatten()[0]
        st.write(f"**Probabilité de panne :** {y_pred_proba:.2f}")
        st.info("Analyse terminée. Consultez la probabilité ci-dessus.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.write("**Probabilité de panne :** _--_")
    st.info("Commencez la saisie ou chargez un fichier CSV pour voir la prédiction.")
