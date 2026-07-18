import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras import ops
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

#################################
### 1. Load data from DeepNet ###
#################################

array_geom = np.load("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/array_geom.npy")
array_perf = np.load("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/array_perf.npy")

################################
########## 2. Model ############
################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

scaler_X = StandardScaler()    # Media 0, std 1
scaler_y = MinMaxScaler()      # [0, 1]

X_scaled = scaler_X.fit_transform(array_perf)
y_scaled = scaler_y.fit_transform(array_geom)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)    # 70% di training data
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),     
    layers.Dense(46, activation="linear")
])

##################################
### 3. Training and prediction ###
##################################

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

callbacks = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),   # Stop se loss non diminuisce dopo 20 epoche, restituisce gli ultimi pesi/bias 
             keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)]              # Riduce il learning rate del 50% (i passi dell'ottimizzatore), dopo 10 epoche in cui loss non diminuisce

history = model.fit(         # Training modello
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=200,
    shuffle=True,
    callbacks=callbacks
)

train_loss = history.history["loss"]           # MSE su training
val_loss = history.history["val_loss"]         # MSE su validation
train_mae = history.history["mae"]             # MAE su training
val_mae = history.history["val_mae"]           # MAE su validation

#############################
### Plot MSE/MAE training ###
#############################

plt.figure(figsize=(10,4))
plt.plot(train_loss, label="Training MSE")
plt.plot(val_loss, label="Validation MSE")
plt.xlabel("Epoche")
plt.ylabel("MSE")
plt.title("Andamento MSE training/validation")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Plots/Training_MSE.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(train_mae, label="Training MAE")
plt.plot(val_mae, label="Validation MAE")
plt.xlabel("Epoche")
plt.ylabel("MAE")
plt.title("Andamento MAE training/validation")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Plots/Training_MAE.png")
plt.close()

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss}, Test MAE: {test_mae}")

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

#######################
### MSE/MAE testing ###
#######################

y_true = scaler_y.inverse_transform(y_test)

sample_mse = np.mean((y_test - y_pred_scaled)**2, axis=1)
sample_mae = np.mean(np.abs(y_test - y_pred_scaled), axis=1)

global_mse = np.mean(sample_mse)
global_mae = np.mean(sample_mae)

plt.figure(figsize=(10,4))
plt.plot(sample_mse, label="MSE per campione")
plt.axhline(global_mse, color="red", linestyle="--", label=f"MSE globale: {global_mse:.4f}")
plt.xlabel("Indice campione test")
plt.ylabel("MSE")
plt.title("Andamento MSE per campione (Test Set)")
plt.legend()
plt.grid()
plt.savefig("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Plots/Testing_MSE.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(sample_mae, label="MAE per campione")
plt.axhline(global_mae, color="red", linestyle="--", label=f"MAE globale: {global_mae:.4f}")
plt.xlabel("Indice campione test")
plt.ylabel("MAE")
plt.title("Andamento della MAE per campione (Test Set)")
plt.legend()
plt.grid()
plt.savefig("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Plots/Testing_MAE.png")
plt.close()

####################################
### 4. Plot, confronto risultati ###
####################################

import matplotlib.pyplot as plt
plot_output = "/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Plots"

for i in range (46):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()],
            color="red", linestyle="--") 
    plt.xlabel("Valori reali")
    plt.ylabel("Valori predetti")
    plt.title("Reale vs Predetto")
    plt.savefig(f"{plot_output}/scatter_output_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.close()

### Viusalizzazione errori su excel ###

error_path = "/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Error"

errors = abs((y_pred)-(y_true))                            # Calcolo errore di ogni predizione
relative_errors = (errors/y_true)*100    # Calcolo errore relativo di ogni predizione (percentuale)

df_errors = pd.DataFrame(errors)
df_relative_errors = pd.DataFrame(relative_errors)
percorso = os.path.join(error_path, "errors.xlsx")
df_errors.to_excel(percorso, index=False, header=False)
percorso_relative = os.path.join(error_path, "relative_errors.xlsx")
df_relative_errors.to_excel(percorso_relative, index=False, header=False)

pred_path = "/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Predictions"
df_pred = pd.DataFrame(y_pred)
percorso_pred = os.path.join(pred_path, "predictions.xlsx")
df_pred.to_excel(percorso_pred, "predictions.xlsx")

######################
### 5. Salvataggio ###
######################

model.save("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/Sequential_model.keras")

np.save("/Users/miche/Desktop/Tesi/Pyhton/Sensitivity_analysis/3hidden_layer/512+256+128/predictions.npy", y_pred)

import joblib

joblib.dump(scaler_X, "scaler_X.gz")
joblib.dump(scaler_y, "scaler_y.gz")
