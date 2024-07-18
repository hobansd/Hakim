import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dense, Flatten
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import poisson, binom
from multiprocessing import Pool

app = Flask(__name__)

# Configuração de logging
logging.basicConfig(level=logging.DEBUG)

# Função para carregar dados reais da Mega-Sena
def load_data(filepath='historico_megasena.csv'):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Exemplo de DataFrame se os dados reais não estiverem disponíveis
        df = pd.DataFrame({
            'numero_1': np.random.randint(1, 61, 100),
            'numero_2': np.random.randint(1, 61, 100),
            'numero_3': np.random.randint(1, 61, 100),
            'numero_4': np.random.randint(1, 61, 100),
            'numero_5': np.random.randint(1, 61, 100),
            'numero_6': np.random.randint(1, 61, 100)
        })
    return df

# Carregar dados
df = load_data()

# Transformando o histórico em um array de contagens de frequência
numero_counts = pd.Series(np.concatenate(df.values)).value_counts().sort_index()
total_sorteios = len(df)

# Distribuição de Poisson
lambda_poisson = numero_counts.mean()
poisson_probs = poisson.pmf(np.arange(1, 61), lambda_poisson)

# Distribuição Binomial
n = total_sorteios
p = 6 / 60  # Probabilidade de um número específico aparecer em um sorteio
binomial_probs = binom.pmf(np.arange(1, 61), n, p)

# Função para calcular a probabilidade Bayesiana
def bayesian_update(prior, likelihood):
    posterior = prior * likelihood
    return posterior / posterior.sum()

# Inicializando prior uniformemente
prior = np.ones(60) / 60

# Likelihood baseado nas contagens de frequência
likelihood = numero_counts.reindex(np.arange(1, 61), fill_value=0) / total_sorteios

# Atualizando a probabilidade Bayesiana
posterior = bayesian_update(prior, likelihood)

# Função de simulação Monte Carlo
def monte_carlo_simulation(num_simulations):
    results = []
    for _ in range(num_simulations):
        result = np.random.randint(1, 61, size=6)  # Simulação de sorteios
        results.append(result)
    return results

def parallel_monte_carlo(num_simulations):
    with Pool(processes=os.cpu_count()) as pool:  # Usar todos os núcleos disponíveis
        results = pool.map(monte_carlo_simulation, [num_simulations // os.cpu_count()] * os.cpu_count())
    return np.concatenate(results)

# Executando a simulação
num_simulations = 1000000
parallel_results = parallel_monte_carlo(num_simulations)

# Preparação dos dados para regressão linear múltipla
def prepare_data_for_regression(df):
    X = df.drop(columns=['numero_6'])
    y = df['numero_6']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data_for_regression(df)

# Treinamento do modelo de regressão linear múltipla
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Previsão e avaliação do modelo
y_pred = linear_reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
logging.info(f"Mean Squared Error of Linear Regression: {mse}")

# Definindo a rede neural adaptativa com 100 milhões de neurônios em 3 camadas
def create_adaptive_neural_network():
    model = Sequential()
    model.add(Dense(100000000, input_dim=5, activation='relu'))
    model.add(Dense(100000000, activation='relu'))
    model.add(Dense(100000000, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Definindo a rede convolucional com 100 milhões de kernels em 3 camadas
def create_convolutional_network():
    model = Sequential()
    model.add(Conv3D(100000000, (3, 3, 3), activation='relu', input_shape=(5, 5, 5, 1)))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(100000000, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(100000000, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Previsão com modelos estatísticos
        poisson_prediction = np.random.choice(np.arange(1, 61), size=6, p=poisson_probs / poisson_probs.sum())
        binomial_prediction = np.random.choice(np.arange(1, 61), size=6, p=binomial_probs / binomial_probs.sum())
        bayesian_prediction = np.random.choice(np.arange(1, 61), size=6, p=posterior)

        # Converter resultados de Monte Carlo para uma lista de listas
        monte_carlo_result = [result.tolist() for result in parallel_results[:10]]  # Exibindo apenas as 10 primeiras simulações

        # Previsão com regressão linear múltipla
        last_draw = df.drop(columns=['numero_6']).iloc[-1].values.reshape(1, -1)
        linear_reg_prediction = linear_reg_model.predict(last_draw).astype(int).tolist()

        return jsonify({
            'poisson_prediction': poisson_prediction.tolist(),
            'binomial_prediction': binomial_prediction.tolist(),
            'bayesian_prediction': bayesian_prediction.tolist(),
            'monte_carlo': monte_carlo_result,
            'linear_regression_prediction': linear_reg_prediction
        })
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        fig, ax = plt.subplots()
        sns.histplot(np.concatenate(df.values), bins=np.arange(1, 62), kde=True, ax=ax)
        ax.set_title('Distribuição de Frequência dos Números Sorteados')
        ax.set_xlabel('Número')
        ax.set_ylabel('Frequência')
        plt.savefig('static/distribution.png')
        plt.close(fig)
        return jsonify({'message': 'Visualização salva em static/distribution.png'})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)