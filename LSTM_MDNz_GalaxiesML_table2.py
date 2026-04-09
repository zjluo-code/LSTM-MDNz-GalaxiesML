import matplotlib.pyplot as plt
import numpy as np

# --- Global style beautification configuration ---
plt.rcParams.update({
    "font.family": "serif",       
    "font.size": 12,              
    "axes.linewidth": 1.5,        
    "xtick.direction": "in",      
    "ytick.direction": "in",      
    "xtick.major.width": 1.2,     
    "ytick.major.width": 1.2,
    "xtick.minor.visible": True,  
    "ytick.minor.visible": True,
    "xtick.top": True,            
    "ytick.right": True,          
    "legend.frameon": False,      
    "legend.fontsize": 10,
    "savefig.dpi": 300            
})

# 1. Read data
try:
    data = np.loadtxt('all_sample.dat')
    data1 = np.loadtxt('zconf_005_sample.dat')
    data2 = np.loadtxt('zconf_025_sample.dat')
    data3 = np.loadtxt('zconf_050_sample.dat')
except OSError:
    x = np.linspace(0.1, 2.5, 10)
    data = np.random.rand(10, 5) * 0.1
    data1, data2, data3 = data*0.8, data*0.6, data*0.4

x = data[:, 0]

# 2. Create figure and axes
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
fig.subplots_adjust(hspace=0.15, wspace=0.22)

column_indices = [1, 2, 3, 5]
colors = ['#000000', '#E69F00', '#56B4E9', '#D55E00'] 
labels = ['Full Sample', '$z_{\\rm conf} > 0.05$', '$z_{\\rm conf} > 0.25$', '$z_{\\rm conf} > 0.50$']
styles = ['-', ':', '-.', '--']
# List of subplot labels
panel_labels = ['(a)', '(b)', '(c)', '(d)']

# 3. Loop to plot
for i, ax in enumerate(axes.flat):
    col_idx = column_indices[i]
    
    ax.plot(x, data[:, col_idx],  color=colors[0], linestyle=styles[0], lw=2, label=labels[0], zorder=4)
    ax.plot(x, data1[:, col_idx], color=colors[1], linestyle=styles[1], lw=2, label=labels[1], zorder=3)
    ax.plot(x, data2[:, col_idx], color=colors[2], linestyle=styles[2], lw=2, label=labels[2], zorder=2)
    ax.plot(x, data3[:, col_idx], color=colors[3], linestyle=styles[3], lw=2, label=labels[3], zorder=1)
    
    # --- Core modification: add subplot labels ---
    # transform=ax.transAxes means using relative coordinates (0,0) bottom-left, (1,1) top-right
    ax.text(0.95, 0.9, panel_labels[i], transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='bottom', ha='right')
    
    # Fine-tuning of axes details
    if i == 0:
        ax.set_ylabel(r'$\sigma_{\rm NMAD}$', fontsize=14)
        ax.set_ylim(0, 0.1)
    elif i == 1:
        ax.set_ylabel(r'Outlier Fraction $O (\%)$', fontsize=14)
        ax.set_ylim(0, 0.3)
    elif i == 2:
        ax.set_ylabel(r'$bias$', fontsize=14)
        ax.set_ylim(-0.04, 0.08)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
    elif i == 3:
        ax.set_ylabel(r'CRPS (mean)', fontsize=14)
        ax.set_ylim(0, 0.25)
    
    ax.grid(ls=':', lw=0.5, color='gray', alpha=0.5)
    ax.set_xlabel(r'Redshift $z_{\rm spec}$', fontsize=14)
    ax.set_xlim(0, 2.6)
    ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
