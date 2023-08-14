import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    municipio: int
    tipo: int

@app.on_event("startup")
def load_model():
    global model
    global encoder
    model_filename = 'models/modelo.pkl'
    model = joblib.load(model_filename)
    
    # Ajustar el encoder con los datos de entrenamiento adecuados
    data = pd.read_csv('data/tiempos.csv')
    X = data.drop(['tiempo1', 'tiempo2', 'tiempo3', 'tiempo4', 'tiempo5', 'tiempo6'], axis=1)
    encoder = OneHotEncoder()
    encoder.fit(X[['Municipio', 'Tipo']])

@app.post("/predict/")
async def predict(pinput: PredictionInput):
    municipio = pinput.municipio
    tipo = pinput.tipo
    
    manual_data = pd.DataFrame({
        'Municipio': [municipio],
        'Tipo': [tipo]
    })
    
    manual_data['Municipio'] = manual_data['Municipio'].astype(int)
    manual_data['Tipo'] = manual_data['Tipo'].astype(int)
    
    nuevos_datos_encoded = encoder.transform(manual_data)
    
    nuevas_predicciones = model.predict(nuevos_datos_encoded)
    
    prediction_results = {
        f"Tiempo {i+1}": tiempo_predicho.item() for i, tiempo_predicho in enumerate(nuevas_predicciones[0])
    }
    
    tiempo_total_manual = sum(nuevas_predicciones[0])
    prediction_results["Tiempo Total"] = tiempo_total_manual.item()
    
    return prediction_results
