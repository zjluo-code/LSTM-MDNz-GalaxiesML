import matplotlib.pyplot as plt
import numpy as np

# 1. Read data
data = np.loadtxt('all_samples.dat')
data1 = np.loadtxt('zconf_080_samples.dat')
data2 = np.loadtxt('zconf_090_samples.dat')
data3 = np.loadtxt('zconf_095_samples.dat')

x = data[:, 0]  

# 2. Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

column_indices = [1, 2, 3, 4]
# titles = ['$\sigma_{NMAD}$', 'Outlier Fraction', 'Bias', 'RMSE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 3. Loop to plot
for i, ax in enumerate(axes.flat):
    col_idx = column_indices[i]
    ax.plot(x, data[:, col_idx], color=colors[0], linestyle="-", label=f'Col {col_idx+1}')
    ax.plot(x, data1[:, col_idx], color=colors[1], linestyle=":", label=f'Col {col_idx+1}')
    ax.plot(x, data2[:, col_idx], color=colors[2], linestyle="-.", label=f'Col {col_idx+1}')
    ax.plot(x, data3[:, col_idx], color=colors[3], linestyle="--", label=f'Col {col_idx+1}')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    if i == 0:
        ax.set_ylabel('$\sigma_{NMAD}$')
        ax.set_ylim(0, 0.1)
    elif i == 1:
        ax.set_ylabel('Outlier Fraction')
        ax.set_ylim(0, 0.35)
    elif i == 2:
        ax.set_ylabel('Bias')
        ax.set_ylim(-0.08, 0.08)
    elif i == 3:
        ax.set_ylabel('RMSE')
        ax.set_ylim(0, 0.3)
    
    if i >= 0:
        ax.set_xlabel('Redshift bin')
        ax.set_xlim(0, 2.6)

plt.show()