#Ejercicio 1:
#En clase analizando el conjunto de set de casas, utilizamos regresión lineal para
#predecir el precio de las casas, sin embargo los datos cuentan con límites fijos, por
#ejemplo:
#• Si una casa tiene 50 años o más, el dato indica 50.
#• Si el precio de una casa es 500,000 o más, el dato indica 500,000.
#• Ingresos, con un máximo de 15.
#Para la tarea, deberán cargar los datos y procesarlos de manera tal, que él % de
#precisión de entrenamiento (score) llegue al menos un 80%. Puede eliminar
#columnas o registros, agregar nuevas características, etc., para llegar a este valor.
#Realicen el proceso con el conjunto de datos resultante; la separación en datos de
#entrenamiento y pruebas, el entrenamiento y predicción.

#NOTA: Para el ejercicio tomare como punto de partida el analisis inicial realizado en clase y comenzaremos ajustar el modelo buscando
#el mejor conjunto de caracteristicas.

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('./housing.csv')
df.head()

df.info()
print(df.describe())
#df.hist(bins=50, figsize=(20,15), edgecolor='black')

datos = df.dropna()#Eliminamos los datos nulos

sb.scatterplot(data=datos[datos['median_income'] > 14 ], x='latitude', y='longitude', hue='median_income', palette='coolwarm')
datos['ocean_proximity'].value_counts()
dummies = pd.get_dummies(datos['ocean_proximity'], dtype=int)
datos = pd.concat([datos, dummies], axis=1)
datos.drop('ocean_proximity', axis=1, inplace=True)

#datos.drop(columns=['longitude', 'latitude',], inplace=True) #Estos tenian una bajisima correlacion asi que al probar desecharlos bajo ligeramente el porcentaje asi que los dejaremos

#datos['old'] = np.where(datos['housing_median_age'] >= 50, 1, 0) #Crea una nueva columna con 1 si la edad es mayor o igual a 50 y 0 si es menor simplificando
#esa columna a una variable binaria, pero despues de pruebas eso mas bien bajo el porcentaje de acierto

#datos['bedrooms_per_population'] = datos['total_bedrooms'] / datos['population'] #Agregar caracteristicas entre la relacion de habitaciones con poblacion pero mas bien redujo el porcentaje tambien a un 62%


#Deshacerme de lo datos con limites fijos y crear caracteristicas adicionales entre Total_bedrooms, total_rooms y population aumento el acierto a un 67%
datos = datos[datos['housing_median_age'] < 50]
datos = datos[datos['median_house_value'] < 500000]
datos = datos[datos['median_income'] < 15]
datos['bedrooms_per_room'] = datos['total_bedrooms'] / datos['total_rooms']
datos['households_per_population'] = datos['households'] / datos['population']
datos['rooms_per_household'] = datos['total_rooms'] / datos['households']

#AQUI ELIMINE AGREGUE Y ELIMINE BASTANTE PERO NINGUN CAMBIO PARECIO INCREMENTAR PORCENTAJE
#PERO IDENTIFIQUE QUE LA PRECISION DEBIA ESTAR CON EL FACTOR CON MAYOR CORRELACION ARRIBA DE 0.68 QUE ERA EL DE MEDIAN_INCOME

sb.scatterplot(x=datos["median_house_value"], y=datos["median_income"]) # Grafico esto a pie de la alta correlacion que hay enter ambas variables
# 0.688355 es la correlacion entre estas dos variables Y con mas de 0.3 es considerable tomar en cuenta
#plt.show()

#AUMENTO AL PORCENTAJE CLARO: AL PROBAR Y PROBAR NUEVAS RELACIONES O QUITANDO O AGREGANDO CARACTERISTICAS NO AUMENTA EL PORCENTAJE DE ACIERTO ME 
#PARE A IMPRIMIR LA MAYORIA DE GRAFICAS POSIBLE E INVESTIGAR COMO DEBIA VERSE UNA DISTRIBUCION MAS NORMAL O FACIL DE ENTENDER Y AUMENTAR EL PORCENTAJE DE ACIERTO
# E INVESTIGANDO SUPE QUE VARIAS GRAFICAS COMO EN ESTA:
print(datos[['median_house_value', 'median_income']].hist(bins=50, figsize=(20,15), edgecolor='black'))
#plt.show()

#AHI PERDCIBI UNA DISTRIBUCION CON LARGAS COLAS SEGUN LEI ESO SIGNIFICA QUE HAY VALORES EXTREMOS, MUCHOS QUE PUEDEN INFLUIR DE MANERA "DESPROPORCIONADA" EN ELENTRENAMIENTO
# de esta fuente: https://fastercapital.com/es/tema/la-importancia-de-tener-en-cuenta-la-asimetr%C3%ADa-en-el-an%C3%A1lisis-de-datos.html#:~:text=La%20asimetr%C3%ADa%20puede%20ayudar%20a%20identificar%20valores%20at%C3%ADpicos%20y%20puntos,o%20tratarse%20de%20otra%20manera.
#donde habla de la importancia de los valores atipicos en el analisis de datos
#segun documentacion de numpy: la tranformacion logaritmica se reduce la asimetria en modelos de regresion: "Al aplicar la transformación logarítmica, estas relaciones pueden volverse más lineales, lo que facilita el ajuste de modelos de regresión.": 

datos['log_median_income'] = np.log1p(datos['median_income'])
datos['log_median_house_value'] = np.log1p(datos['median_house_value'])
#Esta accion aumento el score hasta en un 93% muy arriba de lo esperado: 
 #Volvamos a imprimir la grafica para ver la diferencia:
print(datos[['log_median_house_value', 'log_median_income']].hist(bins=50, figsize=(20,15), edgecolor='black'))
plt.show()
#Aca se imprimen ambas graficas, la primera con median_house_value y median_income y la segunda con log_median_house_value y log_median_income puede notar que pasan como a 
#aparentar una campana de Gauss de distribucion normal ya sin las colas a la izquierda este unico cambio aumento el procentaje hasta un 93% esto fue a base de investigacion
#Estaba a punto de desistir y concluir que el modelo de regresion lineal no era capaz de aumentar la precision y se necesitaba de otro 

sb.set(rc={'figure.figsize':(15,8)})
sb.heatmap(data=datos.corr(), annot=True, cmap='YlGnBu' )

datos.corr()['median_house_value'].sort_values(ascending=False)
datos['room_ratio'] = datos['total_rooms'] / datos['total_bedrooms']
#print(datos)
#print(datos.corr())

#Preparación de los datos
X = datos.drop(columns='median_house_value', axis=1) 
y = datos['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

modelo = LinearRegression() #Modelo 
#entrenamiento
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
comparativa = {"predicciones":predicciones, "valor real": y_test}
result = pd.DataFrame(comparativa)
#print(result)
scoretest =modelo.score(X_test, y_test)
scoretrain = modelo.score(X_train, y_train)
print("Score de test: ",scoretest, "Score de train: ",scoretrain)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mse = mean_squared_error(y_test, predicciones)
mse = np.sqrt(mse) 
print(mse)

