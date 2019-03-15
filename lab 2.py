import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.integrate as integrate
from scipy.stats import skew, kurtosis, t, norm
import statistics
from sklearn.linear_model import LinearRegression

sns.set()

data = [
    [6.0, 6.5, 6.3, 5.7, 6.2, 6.4, 6.2, 6.2, 7.3, 7.2, 6.3, 5.9, 6.6, 6.7, 6.5, 5.5, 5.2, 4.8, 5.7, 3.9, 5.5, 5.6, 8.6,
     5.7, 7.0, 5.3, 7.0, 4.4, 5.5, 4.7, 6.6, 5.7, 5.2, 5.5, 5.5, 5.6, 5.6],
    [6.8, 8.0, 6.6, 7.4, 7.0, 7.1, 7.1, 7.1, 7.8, 7.6, 7.5, 6.8, 7.4, 7.1, 6.8, 5.9, 6.7, 6.9, 9.9, 6.1, 6.4, 6.1, 8.6,
     6.7, 6.8, 5.5, 7.6, 6.1, 6.2, 6.4, 6.3, 6.8, 5.8, 5.3, 6.5, 5.0, 6.3],
    [8.2, 9.0, 7.9, 8.0, 8.1, 7.6, 7.8, 9.0, 8.4, 8.2, 8.2, 8.0, 8.4, 7.4, 9.1, 7.5, 8.3, 7.9, 8.3, 6.9, 8.7, 7.7, 9.1,
     8.8, 7.4, 7.2, 8.0, 7.3, 7.5, 7.2, 9.0, 7.9, 7.6, 6.6, 7.7, 6.6, 7.8]]
print(len(data[0]))
year = [1957 + i for i in range(37)]

df = pd.DataFrame(data=list(map(list, zip(*data))), columns=[1, 2, 3], index=year)
'''_ =  plt.plot(df.index, data[0])
plt.show()
'''

x_max, x_min = max(data[0]), min(data[0])
R = x_max - x_min
n = len(data[0])

tmp = int(math.log(n, 2))
N = 1 + tmp
print(N)
a = [x_min + i * R / N for i in range(N + 1)]
print(a)
m = [0] * N

for j in range(0, N):
    m[j] = sum((a[j] <= data[0][i] and data[0][i] <= a[j + 1]) for i in range(n))

print(m)

w = [m[i] / n for i in range(N)]
print(w)

med_x = [(a[i] + a[i + 1]) / 2 for i in range(N)]

sample_mean = sum(m[i] * med_x[i] for i in range(N)) / n
sample_std_deviation = math.sqrt(sum(m[i] * (med_x[i] - sample_mean) ** 2 for i in range(N)) / n)

print(sample_mean)

z = [0] * (N + 1)
z[0], z[N] = -math.inf, math.inf
for i in range(1, N):
    z[i] = (a[i] - sample_mean) / sample_std_deviation


def Laplace(z):
    if (z == -math.inf):
        return -0.5
    elif (z == math.inf):
        return 0.5
    return integrate.quad(lambda u: math.exp(- u ** 2 / 2), 0, z)[0] / math.sqrt(2 * math.pi)


P = [Laplace(z[i + 1]) - Laplace(z[i]) for i in range(N)]
print(P)

m2 = [P[i] * n for i in range(N)]

chi_square = sum((m[i] - m2[i]) ** 2 / m2[i] for i in range(N))
print("sdsafasf", chi_square)

print(sum(data[0]) / n)
var = np.var(data[0])
print(var)
print(np.mean(data[0]))
print(var * n / (n - 1))
print(math.sqrt(var))
print(skew(data[0]))
print(kurtosis(data[0]))
print(np.median(data[0]))

mat = np.zeros((8, 3))
for i in range(3):
    mat[0][i] = np.mean(data[i])
    mat[1][i] = np.var(data[i])
    mat[2][i] = mat[1][i] * n / (n - 1)
    mat[3][i] = math.sqrt(mat[1][i])
    mat[4][i] = mat[3][i] / mat[0][i]  # коэффициенты вариации
    mat[5][i] = skew(data[i])
    mat[6][i] = kurtosis(data[i])
    #   mat[7][i] = statistics.mode(data[i])
    mat[7][i] = np.around(np.median(data[i]), 2)

table = pd.DataFrame(data=mat)
table.columns = ['Апрель', 'Май', 'Июнь']
table.index = ['Среднее арифметическое', 'Выборочная дисперсия', 'Выборочная исправленная дисперсия',
               'Стандартное отклонение', 'Коэффициент вариации', 'Коэффициен асимметрии', 'Коэффициент эксцесса',
               'Медиана']
print(table)
print(type(np.median(data[0])))

_ = plt.scatter(data[1], data[2])
_ = plt.xlim([4.5, 10])
_ = plt.ylim([5.5, 10])
print(np.cov(data[1], data[2])[0, 1])
# print(sum((data[1][i] - np.mean(data[1])) * (data[2][i] - np.mean(data[2])) for i in range(n)) / (n - 1))

r_xy = np.corrcoef(data[1], data[2])[0, 1]

print(t.ppf(0.95, df=n - 2))

lin_reg = LinearRegression()
lin_reg.fit(np.array(data[1]).reshape(n, 1), np.array(data[2]).reshape(n, 1))
print("Linear regression: ", lin_reg.intercept_, lin_reg.coef_)
b0, b1 = lin_reg.intercept_, lin_reg.coef_
y_pred = b0 + b1 * np.array(data[1])
plt.plot(data[1], y_pred[0], color="g", label='Linear Regression')
plt.xlabel("Температура в мае")
plt.ylabel("Температура в июне")
plt.legend()
plt.show()
##alsjndjkasndkjasndjkasndjknasjkn
sigma2_eps = np.var(data[2]) * (1 - r_xy ** 2)
print(sigma2_eps)
sigma_eps = math.sqrt(sigma2_eps)
var_x = math.sqrt(np.var(data[1]))
mean_x = np.mean(data[1])
sigma_a = sigma_eps / (var_x * math.sqrt(n - 2))
sigma_b = sigma_eps * math.sqrt(1 + mean_x ** 2 / var_x ** 2) / math.sqrt(n - 2)
T_a = b1 / sigma_a
T_b = b0 / sigma_b
print(T_a, float(T_b))

y_star = b0 + b1 * np.array(data[1])

_ = plt.plot(year, data[2])
_ = plt.xlabel("годы")
_ = plt.ylabel("Температура воды")
_ = plt.title("Фактические и вычисленные значения температуры воды в мае в точке 8")
_ = plt.plot(year, y_star[0])
plt.show()
print(type(y_star))

print()