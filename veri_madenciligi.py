import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('deneme.csv')

data = data.fillna('')

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['Author'])
y = data['units sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')

def yeni_kitap_tahmin(yazar):
    X_new = vectorizer.transform([yazar])
    satis_tahmini = model.predict(X_new)
    return satis_tahmini[0]



yazar = "Harper Lee"


print(f'Tahmini Satış Oranı: {yeni_kitap_tahmin(yazar)}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red')
plt.xlabel('Gerçek Satış Değerleri')
plt.ylabel('Tahmin Edilen Satış Değerleri')
plt.title('Gerçek ve Tahmin Edilen Satış Değerlerinin Karşılaştırılması')
plt.show()
