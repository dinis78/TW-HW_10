import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Задача 1. Провести дисперсионный анализ для определения того, есть ли различия среднего
# роста среди взрослых футболистов, хоккеистов и штангистов.
# Даны значения роста в трех группах случайно выбранных спортсменов:
# Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
# Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
# Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.

fb = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hoc = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
wei = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])
k = 3
n = len(fb) + len(hoc) + len(wei)
n1, n2, n3 = len(fb), len(hoc), len(wei)
mean_fb = np.mean(fb)
mean_hoc = np.mean(hoc)
mean_wei = np.mean(wei)
total = np.concatenate((fb, hoc, wei), axis=0)
total_mean = np.mean(total)
print(total_mean)
print(mean_fb, mean_hoc, mean_wei)

q1 = np.sum((fb-mean_fb)**2) + np.sum((hoc-mean_hoc)**2) + np.sum((wei-mean_wei)**2)
q2 = n1 * (mean_fb - total_mean)**2 + n2 * (mean_hoc - total_mean)**2 + n3 * (mean_wei - total_mean)**2

d_f = q2 / (k-1)
d_o = q1 / (n-k)

f_n = d_f/d_o
print(f_n)

x1 = stats.shapiro(fb)
print(x1)
x2 = stats.shapiro(hoc)
print(x2)
x3 = stats.shapiro(wei)
print(x3)
# нормальное распределение

d = stats.bartlett(fb, hoc, wei) # равенство дисперсий норм
print(d)

f = stats.f_oneway(fb, hoc, wei) # дисперсионный анализ
print(f)
# различия среднего роста среди взрослых футболистов, хоккеистов и штангистов нет.

alpha = 0.05
a = stats.f.ppf(1 - alpha, k-1, n-k)
print(a)