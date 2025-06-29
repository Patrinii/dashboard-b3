import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modelo import instalar_e_coletar, criar_target, separar_dados, treinar_modelo, avaliar_modelo, simular_retorno

# Título
st.set_page_config(layout="wide")
st.title("Dashboard de Previsão de Ações - PETR4.SA")

# Filtro por ano
st.sidebar.header("Filtros")
ano_inicio = st.sidebar.selectbox("Ano de Início", [2023, 2024])
ano_fim = st.sidebar.selectbox("Ano de Fim", [2023, 2024, 2025])
if ano_fim < ano_inicio:
    st.sidebar.warning("O ano final deve ser maior ou igual ao ano inicial.")

# Baixar e preparar dados
with st.spinner("Carregando dados..."):
    dados = instalar_e_coletar()
    dados = dados[f"{ano_inicio}-01-01":f"{ano_fim}-12-31"]
    dados = criar_target(dados)

# Exibir amostra
st.subheader("📄 Amostra dos Dados")
st.dataframe(dados.tail(), use_container_width=True)

# Separar dados
X_treino, X_teste, y_treino, y_teste = separar_dados(dados)

# Treinar modelo
modelo, scaler, X_teste_scaled = treinar_modelo(X_treino, y_treino, X_teste)

# Avaliar
resultados = avaliar_modelo(modelo, X_teste_scaled, y_teste)

# Exibir métricas
st.subheader("Métricas do Modelo")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Acurácia", f"{resultados['acuracia']:.2f}")
col2.metric("Precisão", f"{resultados['precisao']:.2f}")
col3.metric("Recall", f"{resultados['recall']:.2f}")
col4.metric("F1-Score", f"{resultados['f1_score']:.2f}")
col5.metric("Especificidade", f"{resultados['especificidade']:.2f}")

# Exibir matriz de confusão
st.subheader("Matriz de Confusão")
fig, ax = plt.subplots()
sns.heatmap(resultados["matriz"], annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)

# Simular retorno
df_simulado = simular_retorno(dados.iloc[-len(resultados["y_pred"]):], resultados["y_pred"])

# Exibir resultados financeiros
st.subheader("Simulação de Retorno com Base na Estratégia")
retorno_total = df_simulado["Ganho"].sum()
retorno_medio = df_simulado["Ganho"].mean()
total_ganhos = df_simulado[df_simulado["Ganho"] > 0]["Ganho"].sum()
total_perdas = df_simulado[df_simulado["Ganho"] < 0]["Ganho"].sum()

st.markdown(f"""
- **Retorno total simulado:** R$ {retorno_total:.2f}  
- **Retorno médio por operação:** R$ {retorno_medio:.2f}  
- **Total de ganhos:** R$ {total_ganhos:.2f}  
- **Total de perdas:** R$ {total_perdas:.2f}  
- **Retorno final (ganhos - perdas):** R$ {retorno_total:.2f}
""")

# Gráfico de fechamento
st.subheader("Gráfico de Fechamento")
fig2, ax2 = plt.subplots()
ax2.plot(dados.index, dados["Close"])
ax2.set_title("Preço de Fechamento - PETR4.SA")
ax2.set_xlabel("Data")
ax2.set_ylabel("Preço (R$)")
st.pyplot(fig2)
