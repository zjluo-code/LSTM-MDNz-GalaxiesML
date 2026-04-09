import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize

# ==========================================
# 1. Data loading and metric calculation
# ==========================================
data = np.genfromtxt('./GalaxiesML_pred_z.dat', skip_header=1)
photo_z_all = data[:, 0]
spect_z_all = data[:, 1]
zconf_all = data[:, 2]
crps_all = data[:,3]

zconf_threshold = 0.25
mask = zconf_all > zconf_threshold
photo_z_mask = photo_z_all[mask]
spect_z_mask = spect_z_all[mask]
crps_mask = crps_all[mask]

def get_metrics(p_z, s_z):
    delta_z = p_z - s_z
    sigma_nmad = 1.48 * np.median(np.abs((delta_z - np.median(delta_z)) / (1 + s_z)))
    f_out = np.sum(np.abs(delta_z) / (1 + s_z) > 0.15) / len(p_z)
    bias = np.median(-delta_z / (1 + s_z))
    return sigma_nmad, f_out, bias

sig_m, fout_m, bias_m = get_metrics(photo_z_mask, spect_z_mask)
sig_a, fout_a, bias_a = get_metrics(photo_z_all, spect_z_all)

# ==========================================
# 2. KDE density preparation
# ==========================================
def prepare_scatter_data(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    norm = Normalize(vmin=z.min(), vmax=z.max())
    return x[idx], y[idx], norm(z[idx])

s_m, p_m, d_m = prepare_scatter_data(spect_z_mask, photo_z_mask)
s_a, p_a, d_a = prepare_scatter_data(spect_z_all, photo_z_all)

# ==========================================
# 3. Plot layout (left: All | right: Quality)
# ==========================================
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

x_ref = np.linspace(0, 4, 100)
y_up = x_ref + 0.15 * (1 + x_ref)
y_down = x_ref - 0.15 * (1 + x_ref)

# --- Left plot (ax1): All Test Samples ---
sc1 = ax1.scatter(s_a, p_a, c=d_a, s=3, cmap='Spectral_r', rasterized=True)
ax1.set_title('All Test Samples', fontsize=14)
text_a = f"$\mathrm{{bias}} = {bias_a:.4f}$\n$f_{{\mathrm{{out}}}} = {fout_a:.3f}$\n$\sigma_{{\mathrm{{NMAD}}}} = {sig_a:.3f}$\n$N = {len(photo_z_all)}$"
ax1.text(0.1, 3.8, text_a, transform=ax1.transData, va='top', ha='left', linespacing=1.8)
# Bottom right label (a)
ax1.text(0.95, 0.05, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', ha='right', va='bottom')

# --- Right plot (ax2): Quality Samples ---
sc2 = ax2.scatter(s_m, p_m, c=d_m, s=3, cmap='Spectral_r', rasterized=True)
ax2.set_title(f'Quality Samples ($z_{{\mathrm{{conf}}}} > {zconf_threshold}$)', fontsize=14)
text_m = f"$\mathrm{{bias}} = {bias_m:.4f}$\n$f_{{\mathrm{{out}}}} = {fout_m:.3f}$\n$\sigma_{{\mathrm{{NMAD}}}} = {sig_m:.3f}$\n$N = {len(photo_z_mask)}$"
ax2.text(0.1, 3.8, text_m, transform=ax2.transData, va='top', ha='left', linespacing=1.8)
# Bottom right label (b)
ax2.text(0.95, 0.05, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='right', va='bottom')

for ax in [ax1, ax2]:
    ax.plot([0, 4], [0, 4], color='black', linewidth=1.2, zorder=10)
    ax.plot(x_ref, y_up, color='blue', linestyle='--', linewidth=1, alpha=0.4)
    ax.plot(x_ref, y_down, color='blue', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Spectroscopic Redshift ($z_{\mathrm{spec}}$)', fontsize=13)
    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

ax1.set_ylabel('Photometric Redshift ($z_{\mathrm{phot}}$)', fontsize=13)

# ==========================================
# 4. Color bar and margin adjustment
# ==========================================
fig.subplots_adjust(left=0.08, right=0.88, wspace=0.12)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
cbar = fig.colorbar(sc2, cax=cbar_ax)
cbar.set_label('Relative Point Density [0, 1]', fontsize=11)
cbar.ax.tick_params(width=1.2)

plt.show()

# ==========================================
# 5. Plot CRPS distribution comparison histogram (1x2 subplots)
# ==========================================
fig_crps, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

hist_kwargs = {'bins': 25, 'range': (0, 1), 'alpha': 0.7, 'edgecolor': 'black'}

ax_c1.hist(crps_all, color='orange', **hist_kwargs)
mean_all = np.mean(crps_all)
med_all = np.median(crps_all)
num_all = len(crps_all)

text_all = f"Mean: {mean_all:.4f}\nMedian: {med_all:.4f}\n$N$: {num_all}"
ax_c1.text(0.65, 0.92, text_all, transform=ax_c1.transAxes, va='top', ha='left')

ax_c1.axvline(mean_all, color='red', linestyle='--', linewidth=1.5)
ax_c1.axvline(med_all, color='blue', linestyle=':', linewidth=1.5)
ax_c1.set_title('All Test Samples', fontsize=14)
ax_c1.set_xlabel('CRPS', fontsize=12)
ax_c1.set_ylabel('Frequency', fontsize=12)
ax_c1.text(0.95, 0.05, '(a)', transform=ax_c1.transAxes, fontsize=16, fontweight='bold', ha='right')

ax_c2.hist(crps_mask, color='green', **hist_kwargs)
mean_m = np.mean(crps_mask)
med_m = np.median(crps_mask)
num_m = len(crps_mask)

text_m = f"Mean: {mean_m:.4f}\nMedian: {med_m:.4f}\n$N$: {num_m}"
ax_c2.text(0.65, 0.92, text_m, transform=ax_c2.transAxes, va='top', ha='left')

ax_c2.axvline(mean_m, color='red', linestyle='--', linewidth=1.5)
ax_c2.axvline(med_m, color='blue', linestyle=':', linewidth=1.5)
ax_c2.set_title(f'Quality Samples ($z_{{\mathrm{{conf}}}} > {zconf_threshold}$)', fontsize=14)
ax_c2.set_xlabel('CRPS', fontsize=12)
ax_c2.text(0.95, 0.05, '(b)', transform=ax_c2.transAxes, fontsize=16, fontweight='bold', ha='right')

for ax in [ax_c1, ax_c2]:
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

plt.tight_layout()
plt.show()
