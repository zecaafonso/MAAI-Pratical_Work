import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix,classification_report, roc_curve



#Carregar dados
df=pd.read_csv("student_depression_dataset.csv")

# Informações gerais do dataset
print("DATASET ORGINAL")
print(f"Dimensoes: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
   
#features disponiveis no dataset
features =['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
            'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
              'Financial Stress', 'Family History of Mental Illness', 'Depression']
print(f"features disponiveis:{features}")

#Tipos de dados
print("\nTipos de dados:")
tipos_dados = pd.DataFrame({
    'Coluna': df.columns,
    'Tipo': df.dtypes.values,
    'Valores Únicos': [df[col].nunique() for col in df.columns]
}) 
print(df.describe())

#valores nulos e duplicados
print(f"Total Valores nulos: {df.isnull().sum().sum()}")
print(f"Valores nulos:\n{df.isnull().sum()}")
print(f"\nLinhas duplicadas: {df.duplicated().sum()}")
print(f"Total de linhas no dataset: {df.shape[0]}") 
  
# 3.4. Análise da variável alvo (cnt)
print("\nANÁLISE DA VARIÁVEL ALVO (Depression):")
print(f"Média: {df['Depression'].mean():.1f} aluguéis/dia")
print(f"Desvio padrão: {df['Depression'].std():.1f}")
print(f"Mínimo: {df['Depression'].min()}")
print(f"Máximo: {df['Depression'].max()}")
print(f"Mediana: {df['Depression'].median()}")


corr_cols = ['Age','Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
            'Job Satisfaction','Work/Study Hours', 'Depression']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, cbar_kws={'label': 'Correlação'})
plt.title('Matriz de Correlação (Variáveis Numéricas)')
plt.tight_layout()

df_copy=df.copy()
print(f"Features atuais: {list(df_copy.columns)}")
print(f"Valores nulos: {df_copy.isnull().sum().sum()}")
print(f"Linhas duplicadas: {df_copy.duplicated().sum()}")
print(f"Total de linhas no dataset: {df.shape[0]}") 

#features numericas
numericas =df_copy.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nfeatures numericas:{numericas}")
df_copy[numericas] = df_copy[numericas].fillna(df_copy[numericas].median())
print(f"valores nulos preenchidos com mediana")
#features categoricas
categoricas=df_copy.select_dtypes(include=['object']).columns.tolist()
print(f"\nfeatures categoricas:{categoricas}")
df_copy[categoricas] = df_copy[categoricas].fillna(df_copy[categoricas].mode().iloc[0])
print(f"valores nulos preenchidos com moda")

# verificar nulos apos novas features
print(f"Total de valores nulos depois da limpeza: {df_copy.isnull().sum().sum()}")


# Remover colunas redundantes/desnecessárias
cols_to_drop = ['id',#nao vai ser necessario para a previsao 
                'City',# para ser menos limitante mais generalizado 
                'Profession',# para ser menos limitante mais generalizado 
                'Degree',# para ser menos limitante mais generalizado 
                'Work Pressure',
                'Job Satisfaction'
                ]
df_copy = df_copy.drop(columns=[c for c in cols_to_drop if c in df_copy.columns])
print(f"\nColunas removidas: {cols_to_drop}")
print(f"Features finais: {list(df_copy.columns)}")


print(f"\nPrimeiras 5 linhas do dataset:")  
print(f"\nDataset após pré-processamento: {df_copy.shape}")
print(f"Colunas finais ({len(df_copy.columns)}):")
for i, col in enumerate(df_copy.columns, 1):
    if col =='Depression':
     print(f"{i}. {col}(target)")
    else:
       print(f"{i}. {col}")

       # Verificar valores únicos
print(f"Valores únicos em 'Depression': {df_copy['Depression'].unique()}")
print(f"Tipo de dados: {df_copy['Depression'].dtype}")

# Distribuição
depression_dist = df_copy['Depression'].value_counts(normalize=True) * 100
print(f"\nDistribuição de 'Depression':")
print(depression_dist)

# Visualizar
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df_copy, x='Depression')
plt.title('Distribuição da Depressão (Target)', fontsize=14, fontweight='bold')
plt.xlabel('Depressão (0=Não, 1=Sim)')
plt.ylabel('Número de Estudantes')

# Adicionar porcentagens
total = len(df_copy)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y_height = p.get_height()
    ax.annotate(percentage, (x, y_height + total*0.01), ha='center')

plt.tight_layout()
plt.show()


# Identificar categóricas
categorical_features = df_copy.select_dtypes(include=['object']).columns.tolist()
numerical_features = df_copy.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nVariáveis Categóricas ({len(categorical_features)}):")
print(f"  {categorical_features}")
print(f"\nVariáveis Numéricas ({len(numerical_features)}):")
print(f"  {numerical_features}")


# Criar cópia para não afetar original
df_encoded = df_copy.copy()



#VARIÁVEIS BINÁRIAS - LabelEncoder
print("\n" + "="*40)
print("VARIÁVEIS BINÁRIAS")
print("="*40)
binary_cols = ['Gender', 'Have you ever had suicidal thoughts ?', 
               'Family History of Mental Illness']

for col in binary_cols:
    if col in df_encoded.columns:
        le = LabelEncoder()
        encoded_col = f"{col}_encoded"
        df_encoded[encoded_col] = le.fit_transform(df_encoded[col].astype(str))
       
        
        print(f"\n{col}:")
        print(f"   Original: {df_encoded[col].unique()}")
        print(f"   Encoded:  {df_encoded[encoded_col].unique()}")

#VARIÁVEIS ORDINAIS -Mapeamento manual
print("\n" + "="*40)
print("VARIÁVEIS ORDINAIS")
print("="*40)

#Sleep Duration
if 'Sleep Duration' in df_encoded.columns:
    print(f"\nSleep Duration - Ordem lógica:")
    # Primeiro: limpar as aspas extras
    df_encoded['Sleep Duration'] = df_encoded['Sleep Duration'].str.replace("'", "")
    
    # Mapeamento(ordem crescente de horas)
    sleep_mapping = {
        'Less than 5 hours': 1,    # Muito pouco
        '5-6 hours': 2,            # Pouco
        '7-8 hours': 3,            # Normal/Recomendado
        'More than 8 hours': 4,    # Muito
        'Others': 2.5              # Intermediário (como pode ser mais ou menos)
    }
    
    df_encoded['Sleep Duration_encoded'] = df_encoded['Sleep Duration'].map(sleep_mapping)
    print(f"   Mapeamento aplicado: {sleep_mapping}")
    print(f"   Valores únicos após: {sorted(df_encoded['Sleep Duration_encoded'].unique())}")

# Dietary Habits
if 'Dietary Habits' in df_encoded.columns:
    print(f"\nDietary Habits - Ordem de saúde:")
    
    dietary_mapping = {
        'Unhealthy': 1,     # Menos saudável
        'Others': 2,        # Neutro/Desconhecido
        'Moderate': 3,      # Moderado
        'Healthy': 4        # Mais saudável
    }
    
    df_encoded['Dietary Habits_encoded'] = df_encoded['Dietary Habits'].map(dietary_mapping) 
    print(f"   Mapeamento aplicado: {dietary_mapping}")
    print(f"   Valores únicos após: {sorted(df_encoded['Dietary Habits_encoded'].unique())}")

#Financial Stress
if 'Financial Stress' in df_encoded.columns:
    print(f"\nFinancial Stress - Escala 1-5:")
    
    # Converter para string para tratamento uniforme
    df_encoded['Financial Stress'] = df_encoded['Financial Stress'].astype(str)
    
    financial_mapping = {
        '1.0': 1, '1': 1,   # Baixo stress
        '2.0': 2, '2': 2,
        '3.0': 3, '3': 3,   # Moderado
        '4.0': 4, '4': 4,
        '5.0': 5, '5': 5,   # Alto stress
        '?': 3,   # Tratar como moderado
    }
    
    df_encoded['Financial Stress_encoded'] = df_encoded['Financial Stress'].map(financial_mapping)
    print(f"   Mapeamento aplicado: {financial_mapping}")
    print(f"   Valores únicos após: {sorted(df_encoded['Financial Stress_encoded'].unique())}")


#REMOVER COLUNAS ORIGINAIS CATEGÓRICAS
print("\n" + "="*60)
print("LIMPEZA E ORGANIZAÇÃO FINAL")
print("="*60)

# Listar todas as colunas codificadas criadas
encoded_columns = [col for col in df_encoded.columns if '_encoded' in col or '_freq' in col or '_target' in col]
print(f"\nColunas codificadas criadas ({len(encoded_columns)}):")
for col in encoded_columns:
    print(f"   • {col}")

#remover colunas originais categóricas
cols_to_drop = [col for col in categorical_features if col in df_encoded.columns]
df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')

print(f"\nColunas categóricas originais removidas: {cols_to_drop}")

# Verificar resultado final
print(f"\nDataset após encoding:")
print(f"Shape: {df_encoded.shape}")
print(f"Colunas: {list(df_encoded.columns)}")
print(f"Tipos de dados:")
print(df_encoded.dtypes.value_counts())

# Remover coluna target das features
X = df_encoded.drop('Depression', axis=1)
#variável target
y = df_encoded['Depression'].copy()

print(f"Shape final:")
print(f"  X (features): {X.shape}")
print(f"  y (target): {y.shape}")

# Verificar tipos de dados em X
print(f"\nTipos de dados em X:")
print(X.dtypes.value_counts())


# Dividir mantendo proporção das classes (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Divisão estratificada:")
print(f"Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X):.1%})")
print(f"Teste:  {X_test.shape[0]} amostras ({X_test.shape[0]/len(X):.1%})")

print(f"\n Distribuição no Treino:")
train_dist = y_train.value_counts(normalize=True) * 100
print(f"Classe 0: {train_dist[0]:.1f}%")
print(f"Classe 1: {train_dist[1]:.1f}%")

print(f"\nDistribuição no Teste:")
test_dist = y_test.value_counts(normalize=True) * 100
print(f"Classe 0: {test_dist[0]:.1f}%")
print(f"Classe 1: {test_dist[1]:.1f}%")


print("\n" + "="*60)
print("NORMALIZAÇÃO DAS FEATURES")
print("="*60)
# Identificar colunas numéricas
num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

# Aplicar apenas nas colunas numéricas
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print(f"{len(num_cols)} features numéricas normalizadas:")
for col in num_cols:
  print(f"{col}")


print("\n" + "="*60)
print("TREINAMENTO DO RANDOM FOREST")
print("="*60)

# Criar modelo com configuração inicial
rf_model = RandomForestClassifier(
    n_estimators=100,           # Número de árvores
    max_depth=10,             # Profundidade máxima
    min_samples_split=10,        # Mínimo de amostras para dividir
    min_samples_leaf=5,         # Mínimo de amostras nas folhas
    max_features='sqrt',
    random_state=42,            # Reprodutibilidade
    n_jobs=-1,                  # Usar todos os processadores
    class_weight='balanced'     # Lidar com desbalanceamento (se houver)
)

print(" Treinando Random Forest...")
rf_model.fit(X_train_scaled, y_train)

print(f" Modelo treinado com {rf_model.n_estimators} árvores")
print(f" Features importantes consideradas: {rf_model.max_features}")


print("\n" + "="*60)
print("AVALIAÇÃO DO MODELO")
print("="*60)
# Previsões
y_train_pred = rf_model.predict(X_train_scaled)
y_train_proba = rf_model.predict_proba(X_train_scaled)[:, 1]

y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("PERFORMANCE NO TREINO:")
print(f"  Acurácia:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"   Precisão:  {precision_score(y_train, y_train_pred):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_train, y_train_pred):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_train, y_train_proba):.4f}")

print("\nPERFORMANCE NO TESTE:")
print(f"  Acurácia:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precisão:  {precision_score(y_test, y_test_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_test_proba):.4f}")


#Visualização dos resultados

print("\n" + "="*60)
print("VISUALIZAÇÃO DOS RESULTADOS")
print("="*60)

#Matriz de Confusão
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sem Depressão', 'Com Depressão'],
            yticklabels=['Sem Depressão', 'Com Depressão'])
plt.title('Matriz de Confusão - Random Forest', fontsize=16, fontweight='bold')
plt.ylabel('Real', fontsize=12)
plt.xlabel('Previsto', fontsize=12)
plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMatriz de Confusão:")
print(f"  Verdadeiros Negativos (VN): {cm[0, 0]} - Previu não depressão e era não depressão")
print(f"  Falsos Positivos (FP):     {cm[0, 1]} - Previu depressão mas era não depressão")
print(f"  Falsos Negativos (FN):     {cm[1, 0]} - Previu não depressão mas era depressão")
print(f"  Verdadeiros Positivos (VP): {cm[1, 1]} - Previu depressão e era depressão")

#Relatório de Classificação
print("\nRELATÓRIO DE CLASSIFICAÇÃO DETALHADO:")
print(classification_report(y_test, y_test_pred,
                           target_names=['Sem Depressão', 'Com Depressão']))

#Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
plt.title('Curva ROC - Random Forest para Depressão Estudantil', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curva_roc.png', dpi=300, bbox_inches='tight')
plt.show()



print("\n" + "="*60)
print("ANÁLISE DAS VARIÁVEIS MAIS IMPORTANTES")
print("="*60)

# Obter importâncias
feature_importance = pd.DataFrame({
    'Variável': X.columns,
    'Importância': rf_model.feature_importances_
}).sort_values('Importância', ascending=False)

# Visualizar top 15
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)
bars = plt.barh(range(len(top_features)), top_features['Importância'])
plt.yticks(range(len(top_features)), top_features['Variável'])
plt.xlabel('Importância', fontsize=12)
plt.title('Top 15 Variáveis para Previsão de Depressão', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()  # Maior importância no topo

# Adicionar valores nas barras
for i, (bar, importance) in enumerate(zip(bars, top_features['Importância'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{importance:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('importancia_variaveis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RESUMO FINAL E CONCLUSÕES")
print("="*60)

# Calcular métricas finais
acuracia = accuracy_score(y_test, y_test_pred)
precisao = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc_score = roc_auc_score(y_test, y_test_proba)

print(f"""
 RESULTADO DO MODELO DE PREVISÃO DE DEPRESSÃO

 DADOS UTILIZADOS:
   - Total de estudantes: {df.shape[0]}
   - Com depressão: {sum(y == 1)} ({sum(y == 1)/len(y):.1%})
   - Sem depressão: {sum(y == 0)} ({sum(y == 0)/len(y):.1%})
   - Variáveis preditoras: {X.shape[1]}

 MODELO RANDOM FOREST:
   - Número de árvores: {rf_model.n_estimators}
   - Variáveis por split: {rf_model.max_features}
   - Balanceamento: {'Sim' if rf_model.class_weight else 'Não'}

 DESEMPENHO NO TESTE:
   - Acurácia:  {acuracia:.2%}
   - Precisão:  {precisao:.2%} (dos previstos como depressão, quantos realmente têm)
   - Recall:    {recall:.2%} (dos que têm depressão, quantos foram identificados)
   - F1-Score:  {f1:.2%} (média harmônica entre precisão e recall)
   - AUC-ROC:   {auc_score:.2%} (capacidade de discriminar entre classes)

 FATORES MAIS IMPORTANTES:
   1. {feature_importance.iloc[0]['Variável']} ({feature_importance.iloc[0]['Importância']:.3%})
   2. {feature_importance.iloc[1]['Variável']} ({feature_importance.iloc[1]['Importância']:.3%})
   3. {feature_importance.iloc[2]['Variável']} ({feature_importance.iloc[2]['Importância']:.3%})

 IMPLICAÇÕES PRÁTICAS:
   - O modelo pode identificar {recall:.1%} dos estudantes com depressão
   - {precisao:.1%} das previsões positivas são corretas
   - Focando nos fatores mais importantes, podemos criar programas preventivos
""")
