import pandas as pd
from flask import Flask, request, jsonify
import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
df = pd .read_csv("code_classification_dirty.csv");


# Limpando o dados
df = df.dropna() #remove valores nulos
df = df.drop_duplicates() #remove valores duplicados
df = df.reset_index(drop=True) #reseta os indices
df['Code'] = df['Code'].str.lower() #converte para minusculo
df['Language'] = df['Language'].str.lower() #converte para minusculo
df = df.drop(columns=["Id"]) #remove coluna Id
#------------------fim da limpeza------------------

df['Language'] = LabelEncoder().fit_transform(df['Language']) #transforma as linguagens em numeros
vetorizarCode = TfidfVectorizer() #cria o vetorizador
X = vetorizarCode.fit_transform(df['Code']) #transforma o codigo em vetor
y = df['Language'] #variavel target

x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42) #divide os dados em treino e teste
print(df)