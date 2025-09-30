import pandas as pd
import os
import joblib
import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, jsonify, render_template

# --- Configuração de Caminhos ---
MODEL_PATH = "best_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
DATA_FILE = "ExemplosDeLinguagensProgramacao.csv"
CLEAN_DATA_FILE = "ExemplosDeLinguagensProgramacaoLimpo.csv"

def load_artifacts():
    """Carrega os artefatos do modelo pré-treinado."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]):
        print("ERRO: Artefatos do modelo não encontrados. Execute o treinamento localmente primeiro.")
        return None, None, None
    
    print("Carregando artefatos do modelo pré-treinado...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Artefatos carregados com sucesso.")
    return model, vectorizer, le



def train_and_save_model():
    """Função para treinar e salvar o modelo e os pré-processadores."""
    print("--- Iniciando treinamento do modelo ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"ERRO: Arquivo de dados não encontrado em {DATA_FILE}. Por favor, crie o arquivo CSV.")
        return None, None, None

    # --- Carregamento e Limpeza dos Dados ---
    if os.path.exists(CLEAN_DATA_FILE):
        print(f"Carregando dados limpos de {CLEAN_DATA_FILE}")
        df = pd.read_csv(CLEAN_DATA_FILE)
    else:
        print(f"Lendo e limpando dados de {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        
        # Processamento e limpeza
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        df['Code'] = df['Code'].astype(str).str.lower()
        df['Language'] = df['Language'].astype(str).str.lower()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
            
        df.to_csv(CLEAN_DATA_FILE, index=False)

    # --- Pré-processamento e Treinamento ---
    if len(df['Language'].unique()) < 2:
        print("AVISO: O arquivo CSV tem apenas uma ou zero linguagens únicas. O treinamento será limitado.")
        if df.shape[0] > 0:
            # Configuração mínima para o caso de uma única classe
            le = LabelEncoder()
            y = le.fit_transform(df['Language'])
            
            vectorizer = TfidfVectorizer()
            x_vec = vectorizer.fit_transform(df['Code'].astype(str))
            
            best_model = RandomForestClassifier(random_state=42).fit(x_vec, y)
        else:
            print("ERRO: O arquivo CSV está vazio após a limpeza.")
            return None, None, None
    else:
        le = LabelEncoder()
        y = le.fit_transform(df['Language'])
        X_text = df['Code'] 
        n_classes = len(le.classes_)
        test_split_size = 0.2 # Tamanho de teste desejado

        # Verificar se a estratificação é possível
        class_counts = df['Language'].value_counts()
        if (class_counts < 2).any():
            print("AVISO: Uma ou mais classes têm apenas 1 amostra. Usando divisão não estratificada.")
            stratify_option = None
        else:
            stratify_option = y
            # Garantir que o test_size seja grande o suficiente para o número de classes
            if df.shape[0] * test_split_size < n_classes:
                # Calcula o tamanho mínimo necessário e adiciona uma pequena margem
                new_test_size = n_classes / df.shape[0] * 1.1 
                print(f"AVISO: O test_size de {test_split_size} é muito pequeno para {n_classes} classes. Ajustando para {new_test_size:.2f}.")
                test_split_size = new_test_size
            
            print("Usando divisão estratificada para manter a proporção das classes.")

        # Divisão dos dados
        x_treino_text, x_teste_text, y_treino, y_teste = train_test_split(
            X_text, y, test_size=test_split_size, random_state=42, stratify=stratify_option
        )

        vectorizer = TfidfVectorizer()
        x_treino_vec = vectorizer.fit_transform(x_treino_text)
        x_teste_vec = vectorizer.transform(x_teste_text)

        # Treinamento dos modelos
        modelo_arvore = RandomForestClassifier(random_state=42)
        modelo_knn = KNeighborsClassifier(n_neighbors=3)

        modelo_arvore.fit(x_treino_vec, y_treino)
        modelo_knn.fit(x_treino_vec, y_treino)

        # Avaliação e seleção do melhor modelo
        acuracia_arvore = modelo_arvore.score(x_teste_vec, y_teste)
        acuracia_knn = modelo_knn.score(x_teste_vec, y_teste)

        print("--- Avaliação do Modelo ---")
        if acuracia_arvore >= acuracia_knn:
            best_model = modelo_arvore
            print(f"Random Forest escolhido como melhor modelo (Acurácia: {acuracia_arvore:.2f})")
        else:
            best_model = modelo_knn
            print(f"KNN escolhido como melhor modelo (Acurácia: {acuracia_knn:.2f})")

    # --- Salvando os artefatos ---
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Artefatos salvos: {MODEL_PATH}, {VECTORIZER_PATH}, {LABEL_ENCODER_PATH}")
    print("---------------------------\n")

    return best_model, vectorizer, le

# --- TREINAMENTO FORÇADO ---
# Este bloco força o treinamento do modelo toda vez que o script é executado.
print("FORÇANDO NOVO TREINAMENTO DE MODELO...")
model, vectorizer, le = train_and_save_model()


# --- Configuração do Flask ---
app = Flask(__name__)

if model is not None:
    @app.route('/')
    def index():
        return render_template('index.html')

    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        codigo = data.get('code', '')
        
        if not codigo:
            return jsonify({'error': 'Nenhum código fornecido'}), 400
        
        try:
            # Pré-processamento do código de entrada
            codigo_transformado = vectorizer.transform([codigo.lower()])
            
            # Predição
            predicao = model.predict(codigo_transformado)
            linguagem_predita = le.inverse_transform(predicao)[0].capitalize()
            
            # Retorna a predição
            return jsonify({
                'best_prediction': linguagem_predita
            })
        except Exception as e:
            print(f"Erro na predição: {e}")
            return jsonify({'error': f'Erro interno na predição.'}), 500

    if __name__ == '__main__':
        # Este bloco é para desenvolvimento local.
        # Ele força o treinamento antes de iniciar o servidor.
        print("MODO DE DESENVOLVIMENTO LOCAL")
        print("Forçando novo treinamento do modelo...")
        model, vectorizer, le = train_and_save_model()
        if model is None:
            print("Falha no treinamento. A aplicação Flask não será iniciada.")
        else:
            print("Iniciando servidor Flask...")
        app.run(debug=True, port=5000)
else:
    print("A aplicação Flask não será iniciada, pois não foi possível carregar ou treinar o modelo.")
    # Este bloco será executado na Vercel
    print("MODO DE PRODUÇÃO (Vercel)")
    model, vectorizer, le = load_artifacts()
    if model is None:
        print("A aplicação Flask não será iniciada, pois não foi possível carregar o modelo.")
