import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Predição Tabela FIPE", layout="wide")

st.title("🚗 Sistema de Predição de Preços - Tabela FIPE")
st.markdown("Este sistema utiliza **Machine Learning (Random Forest)** com a base de dados **COMPLETA**.") #nao foi feito recorde, utilizamos todos os dados da tabela


@st.cache_data
def carregar_dados_completo_v2():
    try:
        
        try:
            df = pd.read_csv("tabela-fipe-historico-precos.csv") #nome do csv (pegamnos do kaggle)
        except:
            df = pd.read_csv("tabela-fipe-historico-precos.csv", sep=';')

        
        if 'anoModelo' in df.columns:
            df = df.rename(columns={'anoModelo': 'ano', 'valor': 'preco'}) #trocamos os nomes das colunas pra enxergar melhor as colunas
        
        
        if 'anoReferencia' in df.columns:
            ano_max = df['anoReferencia'].max()
            df = df[df['anoReferencia'] == ano_max]
        
        
        if 'preco' in df.columns:
            df = df[df['preco'] > 0]
            
        
        return df
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return pd.DataFrame()


@st.cache_resource
def treinar_modelo_final(_df):
    target = 'preco'
    features = ['marca', 'modelo', 'ano'] 
    
  
    if not all(col in _df.columns for col in features):
        return None, None, None, None

    X = _df[features]
    y = _df[target]

    categorical_features = ['marca', 'modelo']
    numerical_features = ['ano']

   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features), #se for numero okay pode passar 
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) #marca e modelo sao vetores binarios pro codigo nao quebrar e conseguir identificar a diferente entre modelos, marcas e anos
        ])

    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) #o RF usa 100 arvores de decisao e usa todos os processadores (n_jobs = -1) p treinar
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # usamos 20% para teste e 80% pro treinamento
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, y_train


if st.sidebar.button("⚠️ Limpar Memória (Cache)"): #por algum motivo estava dando um erro por causa do @st.cache_resource, entao add um botao caso ele cotinuasse guardando informação antiga
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

df = carregar_dados_completo_v2()

if not df.empty:
   
    st.toast(f"Base carregada com sucesso: {len(df)} veículos encontrados! :)", icon="✅")

    st.sidebar.header("Filtros")
    if st.sidebar.checkbox("Mostrar Dados Brutos"):
        st.dataframe(df.head())

   
    with st.spinner(f'Treinando IA com todos os {len(df)} veículos... Pode levar alguns minutos!! Não é culpa sua, mas pode ser do seu PC...'):
        model_pipeline, X_test, y_test, y_train = treinar_modelo_final(df)
        
    if model_pipeline is not None:
        y_pred = model_pipeline.predict(X_test)

       
        st.subheader("📊 Performance do Modelo (Base Completa)")
        c1, c2, c3 = st.columns(3)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        c1.metric("Precisão (R²)", f"{r2:.2f}")
        
        c2.metric("Erro Médio", f"R$ {mae:,.2f}")
        
        c3.info(f"Treinado com {len(df)} veículos")

        
        st.subheader("📈 Análise Visual")
        tab1, tab2 = st.tabs(["Preço Real vs Previsão", "Distribuição"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax1)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel("Preço Real")
            ax1.set_ylabel("Preço Previsto")
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(df['preco'], kde=True, ax=ax2, color='green')
            ax2.set_title("Distribuição de Preços")
            st.pyplot(fig2)

      
        st.markdown("---")
        st.header("🙃 Simulador de Preço 🙃")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            marcas = sorted(df['marca'].unique())
            marca_sel = st.selectbox("Marca", marcas)
        
        with col2:
            modelos = sorted(df[df['marca'] == marca_sel]['modelo'].unique())
            modelo_sel = st.selectbox("Modelo", modelos)
            
        with col3:
            ano_min = int(df['ano'].min())
            ano_max = int(df['ano'].max())
            ano_sel = st.number_input("Ano", min_value=ano_min, max_value=ano_max, value=2020)

        if st.button("Calcular Preço"):
            entrada = pd.DataFrame({'marca': [marca_sel], 'modelo': [modelo_sel], 'ano': [ano_sel]})
            preco_est = model_pipeline.predict(entrada)[0]
            st.success(f"Valor Estimado: **R$ {preco_est:,.2f}**")
            st.caption("Baseado nos dados históricos da Tabela FIPE.")
    else:
        st.error("Erro nas colunas do CSV.")

else:
    st.warning("Aguardando arquivo 'tabela-fipe-historico-precos.csv'...")