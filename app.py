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
st.markdown("Este sistema utiliza **Machine Learning (Random Forest)** para estimar o preço de veículos.")


@st.cache_data
def carregar_dados_sem_duplicatas():
    try:
        try:
            df = pd.read_csv("tabela-fipe-historico-precos.csv")
        except:
            df = pd.read_csv("tabela-fipe-historico-precos.csv", sep=';')

        if 'anoModelo' in df.columns:
            df = df.rename(columns={'anoModelo': 'ano', 'valor': 'preco'}) #renomeando para padronizar
        
        if 'anoReferencia' in df.columns:
            ano_max = df['anoReferencia'].max()
            df = df[df['anoReferencia'] == ano_max]
        
        if 'preco' in df.columns:
            df = df[df['preco'] > 0]
            
       
        df = df.drop_duplicates() #removendo linhas duplicadas
            
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
            ('num', 'passthrough', numerical_features), #se for numero, deixa passar
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) #categorias pra que o cod consiga entender os modelos e as marcas
        ])

    
    n_estimators = 100 if len(_df) > 10000 else 50 #se a base for muito grande, aumentamos as árvores, senão mantemos 50 para ser + rápido
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)) #usando todo o processador aqui
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, y_train

st.sidebar.header("Configurações do Modelo")

usar_amostra = st.sidebar.checkbox("Usar Amostra Reduzida (5.000 veículos)", value=True, help="Marque para treinar mais rápido e simular maior variância de erro.") #p alternar a qntde de dados

if st.sidebar.button("⚠️ Recarregar Cache"): #tava dando muito problema por conta do @st.cache_resource (ficava cte guardando informacoes de implementacoes anteriores), entao as vezes era necessario reiniciar tudo ent criamos esse botaozinho pra ser mais rapido
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

df_full = carregar_dados_sem_duplicatas()

if not df_full.empty:
    if usar_amostra:
        if len(df_full) > 5000:
            df_atual = df_full.sample(5000, random_state=42)
            st.toast("Modo Amostra Ativo: Usando 5.000 veículos.", icon="⚡")
        else:
            df_atual = df_full
    else:
        df_atual = df_full
        st.toast(f"Modo Completo Ativo: Usando {len(df_full)} veículos.", icon="📚")

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros de Visualização")
    if st.sidebar.checkbox("Mostrar Dados Brutos"):
        st.dataframe(df_atual.head())

    #treinando
    with st.spinner(f'Treinando IA com {len(df_atual)} veículos...'):
        model_pipeline, X_test, y_test, y_train = treinar_modelo_final(df_atual)
        
    if model_pipeline is not None:
        y_pred = model_pipeline.predict(X_test)

        st.subheader("📊 Performance do Modelo")
        c1, c2, c3 = st.columns(3)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        c1.metric("Precisão (R²)", f"{r2:.4f}")
        
        c2.metric("Erro Médio (MAE)", f"R$ {mae:,.2f}")
        
        c3.info(f"Veículos usados no treino: {len(df_atual)}")

        print(f"Treino com {len(df_atual)} linhas. R2: {r2}") #usei pra ver se estava funcionando os dois cenarios no terminal

        st.subheader("📈 Análise Visual")
        tab1, tab2 = st.tabs(["Preço Real vs Predito", "Distribuição"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax1)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel("Preço Real")
            ax1.set_ylabel("Preço Previsto")
            ax1.set_title("Real x Previsão")
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(df_atual['preco'], kde=True, ax=ax2, color='green')
            ax2.set_title(f"Distribuição de Preços ({len(df_atual)} veículos)")
            st.pyplot(fig2)

      
        st.markdown("---")
        st.header("🙃 Simulador de Preço 🙃")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            marcas = sorted(df_full['marca'].unique())
            marca_sel = st.selectbox("Marca", marcas)
        
        with col2:
            modelos = sorted(df_full[df_full['marca'] == marca_sel]['modelo'].unique())
            modelo_sel = st.selectbox("Modelo", modelos)
            
        with col3:
            ano_min = int(df_full['ano'].min())
            ano_max = int(df_full['ano'].max())
            ano_sel = st.number_input("Ano", min_value=ano_min, max_value=ano_max, value=2020)

        if st.button("Calcular Preço"):
            entrada = pd.DataFrame({'marca': [marca_sel], 'modelo': [modelo_sel], 'ano': [ano_sel]})  #caso o modelo treinado com amostra não conheça a marca escolhida
           
            try:
                preco_est = model_pipeline.predict(entrada)[0]
                st.success(f"Valor Estimado: **R$ {preco_est:,.2f}**")
            except Exception as e:
                st.warning("O modelo reduzido não treinou com este carro específico :(")
                st.error(f"Erro técnico: {e}")
    else:
        st.error("Erro nas colunas do CSV.")

else:
    st.warning("Aguardando arquivo 'tabela-fipe-historico-precos.csv'...")