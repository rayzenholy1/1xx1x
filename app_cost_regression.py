import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

screens = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
cost = np.array([3.0, 5.5, 8.2, 10.4, 13.1, 15.3])

model = LinearRegression()
model.fit(screens, cost)

pred = model.predict(screens)
print("Коефіцієнт (вартість одного екрана, тис. $):", round(model.coef_[0], 3))
print("Вільний член (базова вартість, тис. $):", round(model.intercept_, 3))
print("R2:", round(r2_score(cost, pred), 4))
print("MAE:", round(mean_absolute_error(cost, pred), 3))

new_screens = np.array([[5], [7], [15]])
new_pred = model.predict(new_screens)
for s, p in zip(new_screens.ravel(), new_pred):
    print(f"Екранів: {s} -> орієнтовна вартість: {p:.2f} тис. $")

line = np.linspace(1, 16, 100).reshape(-1, 1)

plt.figure(figsize=(9, 5))
plt.scatter(screens, cost, color="blue", label="Історичні замовлення")
plt.plot(line, model.predict(line), color="red", label="Лінія регресії")
plt.scatter(new_screens, new_pred, color="green", marker="x", s=100, label="Прогноз для нових клієнтів")
plt.xlabel("Кількість екранів")
plt.ylabel("Вартість, тис. $")
plt.title("Залежність вартості розробки від кількості екранів")
plt.legend()
plt.grid(True)
plt.show()
