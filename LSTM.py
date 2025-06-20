import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K

# 1️⃣ Charger les données
df = pd.read_csv("C:/Users/kalbouss1-admin/Downloads/donnees_panne_motifs_realistes.csv")

# 2️⃣ Normaliser les features
features = ['Température', 'Vibration', 'Pression', 'Bruit', 'Intensité électrique', 'Temps de fonctionnement']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
y = df['Panne'].values

# ⚡️ LOG proportion brute
print(f"✅ Proportion de pannes dans le y original : {np.mean(y):.4f}")

# 3️⃣ Créer les séquences avec horizon
def create_sequences_with_horizon(X, y, sequence_length=3, horizon=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length - horizon + 1):
        X_seq.append(X[i:i+sequence_length])
        future_window = y[i+sequence_length : i+sequence_length+horizon]
        y_seq.append(1 if np.any(future_window) else 0)
    return np.array(X_seq), np.array(y_seq)

sequence_length = 3   # taille historique
horizon = 2           # horizon futur (>0 comme tu le veux)

X_seq, y_seq = create_sequences_with_horizon(X_scaled, y, sequence_length, horizon)

# ⚡️ LOG après séquençage
print(f"✅ X_seq shape : {X_seq.shape}")
print(f"✅ y_seq shape : {y_seq.shape}")
print(f"✅ Proportion après séquençage : {np.mean(y_seq):.4f}")

# 4️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# ⚡️ Vérification des classes
from collections import Counter
counter = Counter(y_train)
print(f"✅ Classes après split : {counter}")

# 5️⃣ Oversampling pour équilibrer les séquences avec un ratio contrôlé

# Trouver indices majorité / minorité
majority_idx = np.where(y_train == 0)[0]
minority_idx = np.where(y_train == 1)[0]

print(f"Proportion avant oversampling : {len(minority_idx) / len(y_train):.4f}")

# 👇 Choisis un ratio cible plus réaliste (ex. 70% panne)
desired_ratio = 0.8  # Ajuste si besoin : 0.6, 0.7, 0.8 ...

n_majority = len(majority_idx)
n_minority_target = int(n_majority * desired_ratio / (1 - desired_ratio))

print(f"Nombre original minorité : {len(minority_idx)}")
print(f"Nombre majorité : {n_majority}")
print(f"Nombre minorité cible : {n_minority_target}")

# Resample la minorité pour atteindre ce nombre cible
minority_oversampled_idx = resample(
    minority_idx,
    replace=True,
    n_samples=n_minority_target,
    random_state=42
)

# Combiner pour former le nouvel ensemble équilibré
balanced_idx = np.concatenate([majority_idx, minority_oversampled_idx])
X_train_balanced = X_train[balanced_idx]
y_train_balanced = y_train[balanced_idx]


print(f"✅ X_train_balanced shape : {X_train_balanced.shape}")
print(f"✅ y_train_balanced distribution : {Counter(y_train_balanced)}")

# 6️⃣ Définir la perte pondérée (optionnel, pas obligatoire si classes équilibrées)
def weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        return K.mean(weight_vector * bce)
    return loss

# 7️⃣ Construire le modèle LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# 💡 Adapter les poids si nécessaire
model.compile(
    loss=weighted_binary_crossentropy(zero_weight=1.0, one_weight=1.0),  # pondération neutre
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 8️⃣ Entraîner
history = model.fit(
    X_train_balanced, y_train_balanced,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 9️⃣ Évaluer
y_pred_proba = model.predict(X_test).flatten()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"✅ Seuil optimal : {optimal_threshold:.2f}")

y_pred = (y_pred_proba > optimal_threshold).astype(int)
print(f"✅ Accuracy finale : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred))

model.save("mon_model_seq3_horizon2.keras", include_optimizer=False)

# Sauvegarder :
joblib.dump(scaler, "scaler.pkl")