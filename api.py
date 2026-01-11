from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Student Depression API")

# Carregar modelo
modelo = joblib.load('modelo_salvo/random_forest_model.pkl')
scaler = joblib.load('modelo_salvo/scaler.pkl')
colunas_info = joblib.load('modelo_salvo/colunas_modelo.pkl')

class StudentInput(BaseModel):
    Age: float
    Gender_encoded: int
    CGPA: float
    Academic_Pressure: float
    Study_Satisfaction: float
    Work_Study_Hours: float
    Sleep_Duration_encoded: int
    Dietary_Habits_encoded: int
    Have_you_ever_had_suicidal_thoughts_encoded: int
    Financial_Stress_encoded: float
    Family_History_of_Mental_Illness_encoded: int

@app.post("/predict")
def predict_depression(student: StudentInput):
    try:
        # Criar dicionario
        dados = {
            'Age': student.Age,
            'Gender_encoded': student.Gender_encoded,
            'CGPA': student.CGPA,
            'Academic Pressure': student.Academic_Pressure,
            'Study Satisfaction': student.Study_Satisfaction,
            'Work/Study Hours': student.Work_Study_Hours,
            'Sleep Duration_encoded': student.Sleep_Duration_encoded,
            'Dietary Habits_encoded': student.Dietary_Habits_encoded,
            'Have you ever had suicidal thoughts ?_encoded': student.Have_you_ever_had_suicidal_thoughts_encoded,
            'Financial Stress_encoded': student.Financial_Stress_encoded,
            'Family History of Mental Illness_encoded': student.Family_History_of_Mental_Illness_encoded
        }
        
        # Criar DataFrame
        df = pd.DataFrame([dados])
        
        # Adicionar colunas faltantes
        for col in colunas_info['features']:
            if col not in df.columns:
                df[col] = 0
        
        # Ordenar colunas
        df = df[colunas_info['features']]
        
        # Normalizar
        df[colunas_info['numerical_features']] = scaler.transform(
            df[colunas_info['numerical_features']]
        )
        
        # Prever
        probabilidade = modelo.predict_proba(df)[0][1]
        predicao = modelo.predict(df)[0]
        
        return {
            "depressao": bool(predicao),
            "probabilidade": float(probabilidade),
            "percentual": float(probabilidade * 100)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"api": "online"}