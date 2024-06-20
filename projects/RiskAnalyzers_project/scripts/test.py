from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import joblib

train_data = pd.read_parquet("./data/train_data/train_data.parquet")
test_data = pd.read_parquet("./data/test_data/test_data.parquet")

X_train = train_data.drop(columns='default')
y_train = train_data['default']

X_test = test_data.drop(columns='default')
y_test = test_data['default']

# Aplicar MaxAbsScaler a las características de entrenamiento
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Aplicar el mismo scaler a las características de prueba
X_test_scaled = scaler.transform(X_test)

model = LGBMClassifier(
    boosting_type='gbdt',
    class_weight=None,
    colsample_bytree=1.0,
    importance_type='split',
    learning_rate=0.1,
    max_depth=-1,
    min_child_samples=20,
    min_child_weight=0.001,
    min_split_gain=0.0,
    n_estimators=100,
    n_jobs=-1,
    num_leaves=31,
    objective=None,
    random_state=None,
    reg_alpha=0.0,
    reg_lambda=0.0,
    silent=True,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=0,
    verbose=-10
)

# Entrenar el modelo
model.fit(X_train_scaled, y_train)


# Realizar las predicciones en el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Convertir las predicciones a un DataFrame de Pandas
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])

# Crear un DataFrame con las características originales y las predicciones
results_df = pd.concat([X_test, y_test, predictions_df], axis=1)
results_df.rename(columns={'default': 'Actual'}, inplace=True)
# Guardar el DataFrame con las predicciones como un archivo CSV
results_df.to_csv('./results/predictions.csv', index=False)
print(f'Archivo "predictions.csv" generado con éxito.')
# Guardar el modelo entrenado en un archivo
joblib.dump(model, './data/models/local_model.joblib')
print('Modelo guardado en trained_model.joblib')
joblib.dump(scaler, './data/models/local_model_scaler.joblib')
print('Scaler guardado en scaler.joblib')
