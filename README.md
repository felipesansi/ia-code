# Classificador de Linguagem de Programação com Machine Learning

Este projeto utiliza técnicas de **Machine Learning** para classificar trechos de código em diferentes linguagens de programação. Desenvolvido com **Flask**, **scikit-learn** e **pandas**, oferece uma API simples para predição.

---

## 🚀 Funcionalidades

- **Classificação de código**: Identifica a linguagem de programação de um trecho de código.
- **API RESTful**: Interface para integração com outras aplicações.
- **Deploy na Vercel**: Hospedagem contínua com deploy automático via GitHub.

---

## 🛠️ Tecnologias Utilizadas

- `Flask` - Framework web para construção da API.
- `scikit-learn` - Biblioteca para construção e treinamento do modelo de ML.
- `pandas` - Manipulação e análise de dados.
- `joblib` - Serialização de modelos.
- `Vercel` - Plataforma de deploy contínuo.

---

## 📦 Instalação

Clone o repositório:

```bash
git clone https://github.com/felipesansi/ia-code.git
cd ia-code


- Python 3.8+
- pandas
- scikit-learn
- (Opcional) Flask

Instale as dependências com:

```bash
pip install pandas scikit-learn flask
```

## Como funciona

1. **Leitura e limpeza dos dados**: Remove linhas nulas, duplicadas, padroniza textos para minúsculo e elimina a coluna de ID.
2. **Transformação**: 
   - Usa LabelEncoder para transformar o nome da linguagem em número.
   - Vetoriza os códigos usando TF-IDF.
3. **Treinamento dos modelos**:
   - Random Forest
   - KNN
4. **Avaliação**: Imprime a acurácia dos modelos.
5. **Classificação de código do usuário**: Recebe um código, faz a predição e exibe a linguagem prevista pelos dois modelos.

## Como rodar

1. Certifique-se de ter o arquivo `code_classification_dirty.csv` no mesmo diretório do script.
2. Execute o script Python:

```bash
python seu_script.py
```

3. Digite um trecho de código quando solicitado.

## Exemplo de uso

```
Digite o código que deseja classificar: print('Hello, world!')
Acurácia Random Forest: 0.93
Acurácia KNN: 0.90
Random Forest: python
KNN: python
```

## Estrutura do Código

- **Limpeza de dados:**  
  Remove nulos, duplicados e padroniza textos.
- **LabelEncoder:**  
  Transforma nomes de linguagens em números para o modelo.
- **TF-IDF Vectorizer:**  
  Vetoriza o texto do código-fonte.
- **Modelos:**  
  Treina RandomForestClassifier e KNeighborsClassifier.
- **Previsão:**  
  Recebe código do usuário, transforma e prevê a linguagem.



**Autor:** Felipe Freitas
**Licença:** MIT
