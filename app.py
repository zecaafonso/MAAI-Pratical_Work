import streamlit as st
import requests

st.set_page_config(page_title="Previsao Depressao Estudantil")

st.title("Previsao de Depressao Estudantil")
st.markdown("Analise o risco de depressao com base nos dados do estudante")

# Formulario
with st.form("form_depressao"):
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade", 17, 30, 21)
        genero = st.selectbox("Genero", ["Feminino (0)", "Masculino (1)"])
        cgpa = st.slider("CGPA (0-10)", 0.0, 10.0, 7.5, 0.1)
        pressao = st.slider("Pressao Academica (1-10)", 1, 10, 6)
        satisfacao = st.slider("Satisfacao Estudos (1-10)", 1, 10, 5)
    
    with col2:
        horas = st.slider("Horas Estudo/Dia", 0, 16, 8)
        sono = st.selectbox("Sono", 
            ["Menos 5h (1)", "5-6h (2)", "7-8h (3)", "Mais 8h (4)"])
        alimentacao = st.selectbox("Alimentacao",
            ["Nao saudavel (1)", "Outros (2)", "Moderado (3)", "Saudavel (4)"])
        pensamentos = st.selectbox("Pensamentos Suicidas", ["Nao (0)", "Sim (1)"])
        estresse = st.slider("Estresse Financeiro (1-5)", 1, 5, 3)
        historico = st.selectbox("Historico Familiar", ["Nao (0)", "Sim (1)"])
    
    submit = st.form_submit_button("Analisar Risco")

# Processar quando enviar
if submit:
    # Converter valores
    genero_encoded = 0 if "Feminino" in genero else 1
    sono_encoded = int(sono.split("(")[1].replace(")", ""))
    alimentacao_encoded = int(alimentacao.split("(")[1].replace(")", ""))
    pensamentos_encoded = 0 if "Nao" in pensamentos else 1
    historico_encoded = 0 if "Nao" in historico else 1
    
    # Preparar dados para API
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
    
    # Enviar para API
    try:
        response = requests.post("http://localhost:8000/predict", json=dados)
        
        if response.status_code == 200:
            resultado = response.json()
            
            # Mostrar resultados
            st.markdown("---")
            st.header("Resultado da Analise")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric(
                    "Probabilidade de Depressao",
                    f"{resultado['percentual']:.1f}%"
                )
                
                if resultado['depressao']:
                    st.error("COM DEPRESSAO")
                else:
                    st.success("SEM DEPRESSAO")
            
            with col_res2:
                st.metric("Nivel de Risco", resultado['risco'])
                st.info(resultado['mensagem'])
            
            # Explicacao
            st.markdown("---")
            st.subheader("Interpretacao:")
            
            if resultado['percentual'] < 30:
                st.success("""
                **Baixo Risco:** A probabilidade e baixa. Continue com habitos saudaveis 
                e mantenha equilibrio entre estudo e vida pessoal.
                """)
            elif resultado['percentual'] < 60:
                st.warning("""
                **Risco Moderado:** Atencao necessaria. Considere reduzir estresse, 
                melhorar qualidade do sono e buscar apoio se necessario.
                """)
            else:
                st.error("""
                **Alto Risco:** Recomenda-se avaliacao com profissional de saude mental. 
                Procure apoio psicologico e converse com alguem de confianca.
                """)
        
        else:
            st.error(f"Erro na API: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("""
        Nao foi possivel conectar a API.
        
        Para resolver:
        1. Abra um terminal
        2. Execute: uvicorn api:app --reload
        3. Tente novamente
        """)

# Informacoes adicionais
with st.expander("Sobre o Modelo"):
    st.markdown("""
    **Modelo Utilizado:** Random Forest Classifier
    **Acuracia:** 84.1%
    **AUC-ROC:** 91.6%
    **F1-Score:** 86.5%
    
    **Fatores Analisados:**
    - Dados demograficos (idade, genero)
    - Desempenho academico (CGPA, pressao, satisfacao)
    - Habitos de vida (sono, alimentacao, horas estudo)
    - Fatores psicologicos (pensamentos suicidas, estresse)
    - Historico familiar
    
    **Aviso:** Esta e uma ferramenta de triagem, nao substitui avaliacao profissional.
    """)