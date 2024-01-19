import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../predictions/metrics.csv')

# Agrupar os dados por modelo
groups = df.groupby('model')

# Criar subplots para cada grupo
fig, axes = plt.subplots(len(groups), 1, figsize=(10, 6), sharex=True)

# Iterar sobre cada grupo e plotar as métricas
for i, (model, group) in enumerate(groups):
    ax = axes[i] if len(groups) > 1 else axes
    group.plot(x='run_date', y=['mae', 'mse'], ax=ax)
    ax.set_title(model)

plt.tight_layout()
plt.savefig('../plots/metrics.png')

