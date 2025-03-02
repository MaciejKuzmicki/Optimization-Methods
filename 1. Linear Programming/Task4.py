from cvxopt.modeling import variable, op
import numpy as np
import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv('data01.csv')
x_values = data.iloc[:, 0].values.astype(float)
y_values = data.iloc[:, 1].values.astype(float)

X = np.vstack([x_values, np.ones(len(x_values))]).T
theta_ls = np.linalg.pinv(X) @ y_values

a = variable()
b = variable()
τ = [variable() for _ in range(len(x_values))]

c_list = [(a * float(x_values[i]) + b - float(y_values[i]) <= τ[i]) for i in range(len(x_values))] + \
         [(a * float(x_values[i]) + b - float(y_values[i]) >= -τ[i]) for i in range(len(x_values))]

p1 = op(sum(τ), c_list)
p1.solve()

ls_a, ls_b = round(theta_ls[0], 3), round(theta_ls[1], 3)

lp_a, lp_b = round(a.value[0], 3), round(b.value[0], 3)

print("Zadanie 4 - LS:", ls_a, ls_b)
print("Zadanie 4 - LP:", lp_a, lp_b)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Dane', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=x_values, y=ls_a * x_values + ls_b, mode='lines', name=f'Metoda LS: y = {ls_a}x + {ls_b}', line=dict(color='black')))
fig.add_trace(go.Scatter(x=x_values, y=lp_a * x_values + lp_b, mode='lines', name=f'Metoda LP: y = {lp_a}x + {lp_b}', line=dict(color='blue')))
fig.update_layout(
    xaxis=dict(range=[0, 10]),
    yaxis=dict(range=[0, 15])
)
fig.show()
