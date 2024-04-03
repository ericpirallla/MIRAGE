import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Generazione e preparazione dei dati
np.random.seed(42)
X = np.random.rand(1000, 5)  # Genera dati fittizi
y = np.random.randint(2, size=1000)  # Target binario

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Costruzione del modello di apprendimento profondo
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(X_train_scaled.shape[1])

# Addestramento del modello
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Valutazione del modello
val_loss, val_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Accuracy sul test set: {val_accuracy:.4f}")

# Previsioni e uso per decisioni di trading
predictions = model.predict(X_test_scaled)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Esempio: stampa le prime 10 previsioni
for i in range(10):
    print(f"Previsione: {predicted_labels[i]}, Reale: {y_test[i]}")
