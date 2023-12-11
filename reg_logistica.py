# Importando as libs necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dados de exemplo: Horas de estudo e aprovação (1) ou reprovação (0)
horas_estudo = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
aprovacao = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Reshape para se adequar aos requisitos do modelo
horas_estudo = horas_estudo.reshape(-1, 1)

# Criar e treinar o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(horas_estudo, aprovacao)

# Gerar valores para previsões
horas_para_prever = np.array([2, 5, 8, 6]).reshape(-1, 1)
previsoes = modelo.predict(horas_para_prever)

# Plotar os resultados
plt.scatter(horas_estudo, aprovacao, color='blue', marker='o', label='Dados de Treino')
plt.plot(horas_estudo, modelo.predict_proba(horas_estudo)[:, 1], color='red', label='Probabilidade de Aprovação')
plt.scatter(horas_para_prever, previsoes, color='green', marker='x', s=100, label='Previsões')
plt.xlabel('Horas de estudo')
plt.ylabel('Aprovação (1) / Reprovação (0)')
plt.legend()
plt.show()