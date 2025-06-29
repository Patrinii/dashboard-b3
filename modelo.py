import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def instalar_e_coletar(ticker='PETR4.SA', inicio='2023-01-01', fim='2025-12-31'):
    dados = yf.download(ticker, start=inicio, end=fim, interval='1d')
    dados.dropna(inplace=True)
    return dados

def criar_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def separar_dados(dados):
    X = dados[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = dados['Target']
    return train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

def treinar_modelo(X_treino, y_treino, X_teste):
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)
    
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_treino_scaled, y_treino)
    
    return modelo, scaler, X_teste_scaled

def avaliar_modelo(modelo, X_teste_scaled, y_teste):
    y_pred = modelo.predict(X_teste_scaled)
    matriz = confusion_matrix(y_teste, y_pred)
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred, zero_division=0)
    recall = recall_score(y_teste, y_pred, zero_division=0)
    f1 = f1_score(y_teste, y_pred, zero_division=0)

    tn, fp, fn, tp = matriz.ravel()
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "matriz": matriz,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1_score": f1,
        "especificidade": especificidade,
        "y_pred": y_pred
    }

def simular_retorno(df_teste, y_pred):
    df_teste = df_teste.copy()
    df_teste['Target_Previsto'] = y_pred
    df_teste['Retorno'] = df_teste['Close'].pct_change().shift(-1)
    df_teste['Ganho'] = df_teste['Retorno'] * df_teste['Target_Previsto']
    return df_teste
