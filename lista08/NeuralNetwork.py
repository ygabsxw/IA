import numpy as np


class NeuralNetwork:
    """
    Implementação de uma Rede Neural com backpropagation.
    Gabriel Diniz Reis Vianna
    25/10/25
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa os pesos com valores aleatórios pequenos
        # Pesos da camada oculta (Entrada -> Oculta)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        # Pesos da camada de saída (Oculta -> Saída)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    # --- Funções de Ativação e suas Derivadas ---
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    # --- Funções de Perda (Loss) ---

    def _mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    # --- Forward e Backward Pass ---

    def forward(self, X):
        """
        Executa o forward pass (propagação direta)
        X: (n_amostras, n_features_entrada)
        """
        # Camada Oculta
        self.z1 = np.dot(X, self.W1) + self.b1   # Entrada linear da camada oculta
        self.a1 = self._sigmoid(self.z1)        # Ativação da camada oculta
        
        # Camada de Saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # Entrada linear da camada de saída
        self.a2 = self._sigmoid(self.z2)        # Ativação da camada de saída (previsão)
        
        return self.a2

    def backward(self, X, y, y_pred, learning_rate):
        """
        Executa o backward pass (retropropagação) e atualiza os pesos.
        """
        n_samples = X.shape[0]

        # 1. Calcular o gradiente na Camada de Saída
        error_output = y - y_pred
        
        delta_output = error_output * self._sigmoid_derivative(y_pred)
        
        # 2. Calcular o gradiente na Camada Oculta
        # Propagar o erro para a camada oculta
        error_hidden = np.dot(delta_output, self.W2.T)
        
        delta_hidden = error_hidden * self._sigmoid_derivative(self.a1)
        
        # 3. Atualizar Pesos e Biases
        
        # Atualização W2 (Oculta -> Saída)
        self.W2 += np.dot(self.a1.T, delta_output) * learning_rate / n_samples
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate / n_samples
        
        # Atualização W1 (Entrada -> Oculta)
        self.W1 += np.dot(X.T, delta_hidden) * learning_rate / n_samples
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate / n_samples

    # --- Treinamento e Predição ---

    def train(self, X, y, epochs, learning_rate):
        """Treina a rede neural"""
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            loss = self._mse(y, y_pred)
            losses.append(loss)
            
            self.backward(X, y, y_pred, learning_rate)
            
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}')
        return losses

    def predict(self, X):
        """Faz previsões com a rede treinada"""
        return self.forward(X)

# --- Funções Auxiliares ---

def get_digit_from_output(output_vector):
    """Converte um vetor de saída (one-hot) no dígito correspondente"""
    return np.argmax(output_vector)

def add_noise(input_vector, n_bits_to_flip=1):
    """Simula ruído invertendo 'n_bits_to_flip' bits aleatórios no vetor de entrada"""
    noisy_vector = np.copy(input_vector)
    indices = np.random.choice(len(noisy_vector), n_bits_to_flip, replace=False)
    for idx in indices:
        noisy_vector[idx] = 1 - noisy_vector[idx] # Inverte o bit (0->1, 1->0)
    return noisy_vector

# --- 2. Execução: Problema XOR ---

print("="*40)
print("  Problema 1: XOR")
print("="*40)

# Dados de entrada (X) e saída (y) para o XOR
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Estrutura da rede: 2 neurônios de entrada, 2 na oculta, 1 na saída
nn_xor = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Treinar a rede
print("Iniciando treinamento da rede XOR...")
nn_xor.train(X_xor, y_xor, epochs=10000, learning_rate=3.0)
print("Treinamento XOR concluído.")

# Testar a rede XOR
print("\n--- Resultados do Teste XOR ---")
predictions_xor = nn_xor.predict(X_xor)
for i in range(len(X_xor)):
    print(f"Entrada: {X_xor[i]} | Esperado: {y_xor[i][0]} | Previsto: {predictions_xor[i][0]:.4f} (Rede: {np.round(predictions_xor[i][0])})")


# --- 3. Execução: Display de 7 Segmentos ---

print("\n" + "="*40)
print("  Problema 2: Display de 7 Segmentos")
print("="*40)

# Dados de entrada (Segmentos a, b, c, d, e, f, g)
X_seg = np.array([
    [1, 1, 1, 1, 1, 1, 0], # 0
    [0, 1, 1, 0, 0, 0, 0], # 1
    [1, 1, 0, 1, 1, 0, 1], # 2
    [1, 1, 1, 1, 0, 0, 1], # 3
    [0, 1, 1, 0, 0, 1, 1], # 4
    [1, 0, 1, 1, 0, 1, 1], # 5
    [1, 0, 1, 1, 1, 1, 1], # 6
    [1, 1, 1, 0, 0, 0, 0], # 7
    [1, 1, 1, 1, 1, 1, 1], # 8
    [1, 1, 1, 1, 0, 1, 1]  # 9
])

y_seg = np.identity(10)

# Estrutura da rede: 7 (entrada), 5 (oculta), 10 (saída)
nn_seg = NeuralNetwork(input_size=7, hidden_size=5, output_size=10)

# Treinar a rede
print("Iniciando treinamento da rede 7 Segmentos...")
nn_seg.train(X_seg, y_seg, epochs=20000, learning_rate=0.3) 
print("Treinamento 7 Segmentos concluído.")

# Testar a rede com dados originais (sem ruído)
print("\n--- Resultados do Teste 7 Segmentos (Dados Originais) ---")
predictions_seg = nn_seg.predict(X_seg)
correct_predictions = 0
for i in range(len(X_seg)):
    predicted_digit = get_digit_from_output(predictions_seg[i])
    expected_digit = get_digit_from_output(y_seg[i])
    is_correct = "CORRETO" if predicted_digit == expected_digit else "ERRADO"
    if is_correct == "CORRETO":
        correct_predictions += 1
    
    print(f"Dígito: {expected_digit} | Previsto: {predicted_digit} | Status: {is_correct}")
    # print(f"   Saída da rede: {[f'{x:.2f}' for x in predictions_seg[i]]}")

print(f"\nAcurácia (Dados Originais): {correct_predictions / len(X_seg) * 100:.1f}%")


# Testar a rede com ruído (Teste de Robustez)
print("\n--- Teste de Robustez (Dados com Ruído) ---")
print("Simulando falha em 1 segmento aleatório para cada dígito...")

correct_noisy_predictions = 0
for i in range(len(X_seg)):
    # Pega o dígito original e adiciona ruído
    original_input = X_seg[i]
    noisy_input = add_noise(original_input, n_bits_to_flip=1)
    
    # Faz a predição com o dado ruidoso
    noisy_prediction_vector = nn_seg.predict(noisy_input.reshape(1, -1))
    predicted_digit = get_digit_from_output(noisy_prediction_vector)
    expected_digit = get_digit_from_output(y_seg[i])
    
    is_correct = "CORRETO" if predicted_digit == expected_digit else "ERRADO"
    if is_correct == "CORRETO":
        correct_noisy_predictions += 1
        
    print(f"Dígito: {expected_digit} | Entrada com Ruído: {noisy_input} | Previsto: {predicted_digit} | Status: {is_correct}")

print(f"\nAcurácia (Dados com Ruído): {correct_noisy_predictions / len(X_seg) * 100:.1f}%")