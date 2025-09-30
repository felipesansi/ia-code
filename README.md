# Classificador de Linguagem de Programação com Machine Learning

Este projeto utiliza machine learning para classificar trechos de código em diferentes linguagens de programação. Utiliza `pandas`, `scikit-learn` e `Flask` (caso queira API) para ler um dataset, treinar modelos e prever a linguagem de códigos fornecidos pelo usuário.

## Sumário

- [Descrição](#descrição)
- [Pré-requisitos](#pré-requisitos)
- [Como funciona](#como-funciona)
- [Como rodar](#como-rodar)
- [Exemplo de uso](#exemplo-de-uso)
- [Estrutura do Código](#estrutura-do-código)
- [Possíveis melhorias](#possíveis-melhorias)

---

## Descrição

O script realiza:
- Limpeza de dados de um arquivo CSV (`code_classification_dirty.csv`)
- Vetorização dos códigos usando TF-IDF
- Treinamento de dois modelos: Random Forest e KNN
- Previsão da linguagem de programação de um código informado pelo usuário

## Pré-requisitos

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
