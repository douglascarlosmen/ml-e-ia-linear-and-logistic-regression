# Importando as libs (bibliotecas) necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Horas de estudo (variável independente)
horas_estudo = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Notas nas provas (variável dependente)
notas_prova = np.array([60, 65, 70, 80, 85, 90])

# Visualizar os dados em gráfico
plt.scatter(horas_estudo, notas_prova, color="blue")
plt.title("Relação entre Horas de Estudo e Notas em uma Prova")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota na Prova")
plt.show()

# Criar uma instância do modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(horas_estudo, notas_prova)

# Fazer previsões para novas horas de estudo
horas_novas = np.array([[10]])
nota_predita = model.predict(horas_novas)

# Visualizar os dados e a linha de regressão
plt.scatter(horas_estudo, notas_prova, color="green")
plt.plot(
    horas_estudo,
    model.predict(horas_estudo),
    color="red",
    linewidth=2,
    label="Regressão Linear",
)
plt.scatter(
    horas_novas,
    nota_predita,
    color="purple",
    marker="X",
    s=100,
    label="Previsão para 7 horas",
)
plt.title("Regressão Linear: Previsão de Notas com Base nas Horas de Estudo")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota na Prova")
plt.legend()
plt.show()

# Exibir a nota prevista para 7 horas de estudo
print(f"A nota prevista para 7 horas de estudo é: {nota_predita[0]:2f}")
