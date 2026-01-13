# app.py - Interface minima

import streamlit as st
import requests

st.set_page_config(page_title="Previsao Depressao")

st.title("Previsao de Depressao Estudantil")
st.markdown("Analise o risco de depressao")

# Formulario
with st.form("form_depressao"):
    idade = st.number_input("Idade", 18, 59, 21)
    
    genero = st.radio("Genero", ["Feminino", "Masculino"], horizontal=True)
    genero_encoded = 0 if genero == "Feminino" else 1
    
    cgpa = st.slider("CGPA (0-10)", 0.0, 10.0, 7.5, 0.1)
    pressao = st.slider("Pressao Academica (1-10)", 1, 10, 6)
    satisfacao = st.slider("Satisfacao Estudos (1-10)", 1, 10, 5)
    horas = st.slider("Horas Estudo/Dia", 0, 16, 8)
    
    sono = st.selectbox("Sono", 
        ["Menos de 5 horas", "5-6 horas", "7-8 horas", "Mais de 8 horas"])
    sono_encoded = ["Menos de 5 horas", "5-6 horas", "7-8 horas", "Mais de 8 horas"].index(sono) + 1
    
    alimentacao = st.selectbox("Alimentacao",
        ["Nao saudavel", "Outros", "Moderado", "Saudavel"])
    alimentacao_encoded = ["Nao saudavel", "Outros", "Moderado", "Saudavel"].index(alimentacao) + 1
    
    pensamentos = st.radio("Pensamentos Suicidas?", ["Nao", "Sim"], horizontal=True)
    pensamentos_encoded = 0 if pensamentos == "Nao" else 1
    
    estresse = st.slider("Estresse Financeiro (1-5)", 1, 5, 3)
    
    historico = st.radio("Historico Familiar?", ["Nao", "Sim"], horizontal=True)
    historico_encoded = 0 if historico == "Nao" else 1
    
    submit = st.form_submit_button("Analisar")

# Processar
if submit:
    dados = {
        "Age": float(idade),
        "Gender_encoded": genero_encoded,
        "CGPA": float(cgpa),
        "Academic_Pressure": float(pressao),
        "Study_Satisfaction": float(satisfacao),
        "Work_Study_Hours": float(horas),
        "Sleep_Duration_encoded": sono_encoded,
        "Dietary_Habits_encoded": alimentacao_encoded,
        "Have_you_ever_had_suicidal_thoughts_encoded": pensamentos_encoded,
        "Financial_Stress_encoded": float(estresse),
        "Family_History_of_Mental_Illness_encoded": historico_encoded
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=dados)
        
        if response.status_code == 200:
            resultado = response.json()
            
            st.markdown("---")
            st.header("Resultado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Probabilidade",
                    f"{resultado['percentual']:.1f}%"
                )
            
            with col2:
                if resultado['depressao']:
                    st.error("COM DEPRESSAO")
                else:
                    st.success("SEM DEPRESSAO")
            
            # Interpretacao simples
            st.markdown("---")
            st.subheader("Interpretacao:")
            
            percentual = resultado['percentual']
            
            if percentual < 30:
                st.success("Baixo risco de depressao")
            elif percentual < 60:
                st.warning("Risco moderado de depressao")
            else:
                st.error("Alto risco de depressao")
        
        else:
            st.error(f"Erro: {response.status_code}")
            
    except:
        st.error("API nao encontrada. Execute: uvicorn api:app --reload")

# Info do modelo
st.markdown("---")
st.markdown("**Modelo:** Random Forest | **Acuracia:** 84.1%")