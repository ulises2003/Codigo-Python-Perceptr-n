import random

# Función de activación (función escalón)
def activation_function(x):
    return 1 if x >= 0 else 0

# Inicializar los pesos W y el umbral θ con valores pequeños aleatorios entre [-1, 1]
def initialize_weights(n):
    weights = [random.uniform(-1, 1) for _ in range(n)]
    theta = random.uniform(-1, 1)
    return weights, theta

# Entrenamiento del Perceptrón
def train_perceptron(X, D, learning_rate, epochs):
    n_features = len(X[0])
    weights, theta = initialize_weights(n_features)

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # Paso 3: Calcular Y en función de X usando los pesos y el umbral
            x_i = X[i]
            weighted_sum = sum(x_i[j] * weights[j] for j in range(n_features)) - theta
            y = activation_function(weighted_sum)

            # Paso 4: Calcular el error
            error = D[i] - y
            total_error += abs(error)

            # Paso 5: Ajustar los pesos y el umbral según el error y el coeficiente de aprendizaje
            for j in range(n_features):
                weights[j] += learning_rate * error * x_i[j]
            theta -= learning_rate * error  # Ajuste del umbral

        # Imprimir el error en cada época
        print(f"Epoch {epoch+1}, Total Error: {total_error}")

        # Si el error total es cero, finalizamos el entrenamiento
        if total_error == 0:
            break

    return weights, theta

# Datos de entrenamiento (X) y salidas deseadas (D)
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
D = [0, 0, 0, 1]  # Salidas deseadas para la compuerta AND

# Parámetros de entrenamiento
learning_rate = 0.1  # Coeficiente de aprendizaje
epochs = 100  # Número máximo de iteraciones

# Entrenar el Perceptrón
weights, theta = train_perceptron(X, D, learning_rate, epochs)

# Probar el Perceptrón
def predict(x, weights, theta):
    weighted_sum = sum(x[j] * weights[j] for j in range(len(weights))) - theta
    return activation_function(weighted_sum)

# Realizar predicciones con el modelo entrenado
for x in X:
    print(f"Input: {x}, Prediction: {predict(x, weights, theta)}")
