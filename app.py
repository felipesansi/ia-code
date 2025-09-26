import pandas as pd
from flask import Flask, request, jsonify
import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("code_classification_dirty.csv")

# ------------------ limpeza ------------------
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df['Code'] = df['Code'].str.lower()
df['Language'] = df['Language'].str.lower()
df = df.drop(columns=["Id"])
# ------------------ fim da limpeza ------------------

# ✅ cria o LabelEncoder e guarda ele
le = LabelEncoder()
df['Language'] = le.fit_transform(df['Language'])

# Vetorização
vetorizarCode = TfidfVectorizer()
X = vetorizarCode.fit_transform(df['Code'])
y = df['Language']

# Split
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
modelo_arvore = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treino
modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# Avaliação
print("Acurácia Random Forest:", modelo_arvore.score(x_teste, y_teste))
print("Acurácia KNN:", modelo_knn.score(x_teste, y_teste))

# Previsão com input do usuário
codigo_usuario = input("Digite o código que deseja classificar: ")
codigo_usuario = vetorizarCode.transform([codigo_usuario])

linguagem_predita_arvore = modelo_arvore.predict(codigo_usuario)
linguagem_predita_knn = modelo_knn.predict(codigo_usuario)

#  Converte de volta para nomes
print("Random Forest:", le.inverse_transform(linguagem_predita_arvore)[0])
print("KNN:", le.inverse_transform(linguagem_predita_knn)[0])
