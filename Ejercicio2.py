#EJERCICIO 2:
#Utilizando el conjunto de datos de casas del ejercicio realizado en clase, (Ejercicio
#de Regresión Lineal) aplique ahora, el algoritmo Árbol de decisión, evalúe los
#datos y grafique la relación entre las características, dibujando el diagrama del
#árbol obtenido, el % de precisión, y demás datos utilizados en clase.

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
#Nota: Aunque en clase conocimos el DecisionTreeClassifier, como estamos trabajando con regresion usaremos DecisionTreeRegressor 
#Pues este no es un ejercicio de clasificacion sino de regresion predecimos valores, no clasificamos
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

df = pd.read_csv('./housing.csv') 
print(df.head())

#Reutilizare el preprocesamiento de datos del ejercicio 1 obviando las graficas porque las correlaciones y distribuciones ya las conocemos

datos = df.dropna()#
datos['ocean_proximity'].value_counts()
dummies = pd.get_dummies(datos['ocean_proximity'], dtype=int)
datos = pd.concat([datos, dummies], axis=1)
datos.drop('ocean_proximity', axis=1, inplace=True)

datos = datos[datos['housing_median_age'] < 50]
datos = datos[datos['median_house_value'] < 500000]
datos = datos[datos['median_income'] < 15]
datos['bedrooms_per_room'] = datos['total_bedrooms'] / datos['total_rooms']
datos['households_per_population'] = datos['households'] / datos['population']
datos['rooms_per_household'] = datos['total_rooms'] / datos['households']


datos['log_median_income'] = np.log1p(datos['median_income'])
datos['log_median_house_value'] = np.log1p(datos['median_house_value'])

datos.corr()['median_house_value'].sort_values(ascending=False)
datos['room_ratio'] = datos['total_rooms'] / datos['total_bedrooms']


#Preparación de los datos
X = datos.drop(columns='median_house_value', axis=1) 
y = datos['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

modelo = DecisionTreeRegressor(max_depth=5) #Modelo a usar
#entrenamiento
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

#Precisión
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones) #

print(f'Error: {mse}')
print(f'R^2 Score: {r2}') #Metodo de sccore segun la documentacion que nos entrego: El
#El R^2 score La puntuación utilizada al llamar scorea un regresor se utiliza multioutput='uniform_average'desde la versión 0.23 para mantener la coherencia con el valor predeterminado de r2_score. Esto influye en el scoremétodo de todos los regresores de salida múltiple (excepto MultiOutputRegressor).
plt.figure(figsize=(20,10))
plot_tree(modelo, filled=True, feature_names=X.columns, rounded=True)
plt.title('Árbol de Decisión para la prediccion de costos de vivienda')
plt.show()



