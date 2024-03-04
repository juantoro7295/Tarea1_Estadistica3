import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Cargar el dataset
# Profesor recuerda cambiar la ruta  para probar
df = pd.read_csv('C:\\Users\\juanp\\Downloads\\Trabajos Estadistica\\Src\\CARS.csv')

# Excluir columnas no numéricas antes de calcular la matriz de correlación
df_numeric = df.select_dtypes(include=['number'])

# Imprimir outliers
outliers = df[(df['MPG_City'] > df['MPG_City'].mean() + 4 * df['MPG_City'].std()) | 
              (df['MPG_City'] < df['MPG_City'].mean() - 4 * df['MPG_City'].std())]
print("Outliers:")
print(outliers[['Model', 'MPG_City']])

# Realizar prueba de normalidad
stat, p_value = shapiro(df['MPG_City'])
print(f"Estadístico de prueba: {stat}, Valor p: {p_value}")
if p_value < 0.05:
    print("La distribución no es normal.")
else:
    print("La distribución es normal.")

# 2.1 Distribución de cada variable
# 2.1.1 Para variables categóricas (por ejemplo, 'Make')
plt.figure(figsize=(10, 6))
sns.countplot(x='Make', data=df)
plt.title('Distribución de Make')
plt.xlabel('Make')
plt.ylabel('Número de observaciones')
plt.xticks(rotation=45)
plt.show()

# 2.1.2 Para variables numéricas (por ejemplo, 'MPG_City')
plt.figure(figsize=(10, 6))
sns.histplot(df['MPG_City'], bins=20, kde=True)
plt.title('Histograma de MPG_City')
plt.xlabel('MPG_City')
plt.ylabel('Frecuencia')
plt.show()

# Listar los modelos de carros que están más lejos de 4 estándares de desviación
outliers = df[(df['MPG_City'] > df['MPG_City'].mean() + 4 * df['MPG_City'].std()) | 
              (df['MPG_City'] < df['MPG_City'].mean() - 4 * df['MPG_City'].std())]
print("Outliers:")
print(outliers[['Model', 'MPG_City']])

# Test de normalidad
stat, p_value = shapiro(df['MPG_City'])
print(f"Estadístico de prueba: {stat}, Valor p: {p_value}")
if p_value < 0.05:
    print("La distribución no es normal.")
else:
    print("La distribución es normal.")

# 2.2 Gráfico de la relación con respecto a MPG_City
# 2.2.1 Boxplot para variables categóricas (por ejemplo, 'Origin')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Origin', y='MPG_City', data=df)
plt.title('Relación entre Origin y MPG_City (Boxplot)')
plt.xlabel('Origin')
plt.ylabel('MPG_City')
plt.show()

# 2.2.2 Scatter plot para variables numéricas (por ejemplo, 'Horsepower')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Horsepower', y='MPG_City', data=df)
plt.title('Relación entre Horsepower y MPG_City (Scatter plot)')
plt.xlabel('Horsepower')
plt.ylabel('MPG_City')
plt.show()

# 2.3 Matriz de correlación
# 2.3.1 Matriz de correlación
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# Variables más importantes para explicar la variabilidad de MPG_City
important_variables = correlation_matrix['MPG_City'].abs().sort_values(ascending=False)[1:]
print("Variables más importantes:")
print(important_variables)

# 2.3.2 Matriz de correlación excluyendo outliers
outliers_models = ["Civic Hybrid 4dr manual (gas/electric)", "Insight 2dr (gas/electric)", "Prius 4dr (gas/electric)"]
df_no_outliers = df[~df['Model'].isin(outliers_models)]
df_no_outliers_numeric = df_no_outliers.select_dtypes(include=['number'])
correlation_matrix_no_outliers = df_no_outliers_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_no_outliers, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación sin Outliers')
plt.show()
