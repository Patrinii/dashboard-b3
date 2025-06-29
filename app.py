# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modelo import carregar_dados, preparar_dados, treinar_modelo, simular_retorno, avaliar_modelo

st.set_page_config(page_title="Dashboard B3", layout="wide")

# Sidebar
st.sidebar.title("Configurações")
ano_treino = st.sidebar.selectbox("Ano de Treinamento:", [2023, 2024, "2023 e 2024"])
exibir_graficos = st.sidebar.multiselect("Gráficos", ["Fechamento", "Distribuição de Alvo", "Lucro Acumulado", "Volume x Preço"])

# Dados
dados = carregar_dados()
dados = dados.dropna()
dados = preparar_dados(dados)

# Separar Treino/Teste
if ano_treino == "2023 e 2024":
    treino = dados[dados['Ano'].isin([2023, 2024])]
else:
    treino = dados[dados['Ano'] == ano_treino]

teste = dados[dados['Ano'] == 2025]

# Modelo
modelo, X_teste, y_teste, teste_completo = treinar_modelo(treino, teste)
y_pred = avaliar_modelo(modelo, X_teste, y_teste)
teste_completo = simular_retorno(teste_completo, y_pred)

# Dashboard
st.title("📈 Painel de Análise - PETR4.SA")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Acurácia", f"{modelo.score(X_teste, y_teste)*100:.2f}%")
with col2:
    st.metric("Retorno Total", f"R$ {teste_completo['Lucro'].sum():.2f}")
with col3:
    st.metric("Operações com Lucro", f"{(teste_completo['Lucro'] > 0).sum()}")

# Gráficos
if "Fechamento" in exibir_graficos:
    st.subheader("Gráfico de Fechamento")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados['Date'], dados['Close'])
    st.pyplot(fig)

if "Distribuição de Alvo" in exibir_graficos:
    st.subheader("Distribuição da Variável Alvo")
    fig, ax = plt.subplots()
    sns.countplot(x='Target', data=dados, ax=ax)
    st.pyplot(fig)

if "Lucro Acumulado" in exibir_graficos:
    st.subheader("Lucro Acumulado")
    dados_plot = teste_completo.copy()
    dados_plot['Lucro_Acumulado'] = dados_plot['Lucro'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados_plot['Date'], dados_plot['Lucro_Acumulado'], color='green')
    st.pyplot(fig)

if "Volume x Preço" in exibir_graficos:
    st.subheader("Dispersão Volume x Preço")
    fig, ax = plt.subplots()
    ax.scatter(dados['Volume'], dados['Close'], alpha=0.5)
    st.pyplot(fig)
