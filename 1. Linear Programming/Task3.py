from cvxopt.modeling import variable, op
# Zadanie 3: Leki
x_LekI = variable()
x_LekII = variable()
x_SurI = variable()
x_SurII = variable()

c1 = (0.01 * x_SurI + 0.02 * x_SurII - 0.50 * x_LekI - 0.60 * x_LekII == 0)
c2 = (x_SurI + x_SurII <= 1000)
c3 = (90.00 * x_LekI + 100.00 * x_LekII <= 2000)
c4 = (40.00 * x_LekI + 50.00 * x_LekII <= 800)
c5 = (100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII <= 100000)
c6 = (x_LekI >= 0)
c7 = (x_LekII >= 0)
c8 = (x_SurI >= 0)
c9 = (x_SurII >= 0)

p1 = op(100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII - (6500.00 * x_LekI + 7100.00 * x_LekII), [c1, c2, c3, c4, c5, c6, c7, c8, c9])
p1.solve()

print("Zadanie 3 - optymalne wartości:", round(x_LekI.value[0], 3), round(x_LekII.value[0], 3), round(x_SurI.value[0], 3), round(x_SurII.value[0], 3))
