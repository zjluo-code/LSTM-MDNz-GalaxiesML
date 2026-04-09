#!/usr/bin/env python
# coding=utf-8

from astropy.table import Table
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.layers import Activation, Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional, Concatenate, Conv1D, Flatten, Layer, Softmax
import tensorflow.keras.backend as K
from sklearn import preprocessing
from tensorflow.keras.models import Model, load_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from scipy.stats import uniform
import random
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error

gpu_list = tf.config.experimental.list_physical_devices('GPU')
if gpu_list:
    try:   
        tf.config.experimental.set_visible_devices(gpu_list[0], 'GPU')
    except RuntimeError as e:
        print(e)


# fits to dataframe
filename1 = './GalaxiesML/hsc_train.csv'
filename2 = './GalaxiesML/hsc_validation.csv'
filename3 = './GalaxiesML/hsc_test.csv'
df1 = pd.read_csv(filename1, sep=',', header=0)
df2 = pd.read_csv(filename2, sep=',', header=0)
df3 = pd.read_csv(filename3, sep=',', header=0)

print(len(df1), len(df2), len(df3))

# Plot histogram of df['Z'] distribution
plt.figure(figsize=(7, 6))
filename = './GalaxyiesML/HSC_v6.csv'
df = pd.read_csv(filename, sep=',', header=0)

bins = 80  # can be adjusted as needed

# Plot histogram
n, bins, patches = plt.hist(df['specz_redshift'], bins=bins, range=(0.01, 4), 
                            edgecolor='black', linewidth=1, alpha=0.7)

plt.xlabel('Redshift ($z_{spec}$)', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.xlim(-0.1, 4)

ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis='both', direction='in', width=2, labelsize=12)

plt.tight_layout()
plt.show()

n_band = 5

hsc_train = np.array(df1.iloc[:, np.r_[12, 17, 13, 18, 14, 19, 15, 20, 16, 21, 2]])
hsc_valid = np.array(df2.iloc[:, np.r_[12, 17, 13, 18, 14, 19, 15, 20, 16, 21, 2]])
hsc_test = np.array(df3.iloc[:, np.r_[12, 17, 13, 18, 14, 19, 15, 20, 16, 21, 2]])

hsc_x_train = hsc_train[:, :-1]
hsc_y_train = hsc_train[:, -1]
hsc_x_valid = hsc_valid[:, :-1]
hsc_y_valid = hsc_valid[:, -1]
hsc_x_test = hsc_test[:, :-1]
hsc_y_test = hsc_test[:, -1]

x_train_temp, y_train_temp, x_valid_temp, y_valid_temp, x_test_temp, y_test_temp = hsc_x_train, hsc_y_train, hsc_x_valid, hsc_y_valid, hsc_x_test, hsc_y_test

#######################################
# data normalize
#@@@@@@@@@@@@@@@@@@@@@@
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train_temp)
y_train = y_train_temp
x_valid = scaler.transform(x_valid_temp)
y_valid = y_valid_temp
x_test = scaler.transform(x_test_temp)
y_test = y_test_temp

n_train = x_train.shape[0]
n_test = x_test.shape[0] 
n_valid = x_valid.shape[0] 

print("train number = ", x_train.shape)
print("test number = ", x_test.shape)
print("valid number = ", x_valid.shape)

hsc_x_train = np.empty((n_train, n_band, 2))
hsc_y_train = y_train
for galaxy in range(n_train):
    for band in range(n_band):
        hsc_x_train[galaxy, band, 0] = x_train[galaxy, 2 * band]
        hsc_x_train[galaxy, band, 1] = x_train[galaxy, 2 * band + 1]

hsc_x_test = np.empty((n_test, n_band, 2))
hsc_y_test = y_test

for galaxy in range(n_test):
    for band in range(n_band):
        hsc_x_test[galaxy, band, 0] = x_test[galaxy, 2 * band]
        hsc_x_test[galaxy, band, 1] = x_test[galaxy, 2 * band + 1]

hsc_x_valid = np.empty((n_valid, n_band, 2))
hsc_y_valid = y_valid
for galaxy in range(n_valid):
    for band in range(n_band):
        hsc_x_valid[galaxy, band, 0] = x_valid[galaxy, 2 * band]
        hsc_x_valid[galaxy, band, 1] = x_valid[galaxy, 2 * band + 1]

X_input = Input(shape=(n_band, 2), name='input_layer')
dr = 0.25

X = Bidirectional(LSTM(128, return_sequences=True))(X_input)
X = BatchNormalization()(X)
X = Dropout(dr)(X, training=False)

X = Bidirectional(LSTM(64, return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(dr)(X, training=False)
X = Flatten()(X)

x = Dense(256, activation='relu', name='rep')(X)
X = BatchNormalization()(X)
X = Dropout(dr)(X)
X = Dense(128, activation='relu', name='rep1')(X)
X = BatchNormalization()(X)
X = Dropout(dr)(X)

################## below is for MDN ##############
KMIX = 10  # KMIX is the number of mixtures

NOUT = KMIX * 3  # number of pi, mu, std

op = Dense(KMIX, activation='linear', name='op')(X)
op = Softmax()(op)
ou = Dense(KMIX, activation='linear', name='ou')(X)
os = Dense(KMIX, activation='softplus', name='os')(X)

output_layer = Concatenate()([op, ou, os]) 

# output_layer = Dense(1, use_bias=True)(stack_layer)

model = Model(X_input, output_layer)

model.summary()

############ DIY the Loss Function ##############
def get_mixture_coef(output, Training=True):  # out_mu.shape=[,KMIX]
    out_pct = output[:, :KMIX]
    out_mu = output[:, KMIX:2 * KMIX]
    out_std = output[:, 2 * KMIX:]
    return out_pct, out_mu, out_std

def get_loss(pct, mu, std, y, epsilon=1e-8):
    # Add epsilon to prevent division by zero
    std = std + epsilon
    
    y = K.reshape(y, (-1, 1))  # ensure y is of shape (batch_size, 1)
    y = K.repeat_elements(y, KMIX, axis=1)  # repeat to (batch_size, KMIX)
    
    # Calculate Gaussian probability density
    factors = 1 / (math.sqrt(2 * math.pi) * std)
    exponent = K.exp(-0.5 * K.square((y - mu) / std))
    GMM_likelihood = K.sum(pct * factors * exponent, axis=1)
    # Add small constant to avoid log(0)
    log_likelihood = -K.log(GMM_likelihood + epsilon)
    return K.mean(log_likelihood)

def loss_func(y_true, y_pred):
    out_pct, out_mu, out_std = get_mixture_coef(y_pred)
    result = get_loss(out_pct, out_mu, out_std, y_true)
    return result

custom_objects = {'loss_func': loss_func}
model = load_model('./LSTM_MDNz_weights.h5', custom_objects=custom_objects)

lr = 0.001

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./DROP03_hsc_kmix10_{epoch}.h5',
                                                save_best_only=True,
                                                monitor='val_loss',
                                                mode='min'),

             tf.keras.callbacks.EarlyStopping(
                 monitor='val_loss',
                 patience=30,
                 restore_best_weights=True
                 ),
             tf.keras.callbacks.ReduceLROnPlateau(
                 monitor='val_loss',
                 factors=0.2,
                 patience=10,
                 min_lr=1e-9,
                 verbose=1
                 )]
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_func, metrics=[loss_func])

print("########", hsc_y_train.shape)
# model.fit(hsc_x_train, hsc_y_train, epochs=1000, batch_size=128, validation_data=(hsc_x_valid, hsc_y_valid), callbacks=callbacks)
########### Use tf.keras.Model.predict method to compute and predict on test data, output prediction results

n_pred = n_test 

m = n_pred  # X_train.shape[0]          # number of training samples
m_test = n_pred  # X_test.shape[0]    # number of test samples

y_pred_test_1 = model.predict(hsc_x_test[0:n_pred])
prob_test, mu_test, sigma_test = get_mixture_coef(y_pred_test_1) 
y_pred_train_1 = model.predict(hsc_x_train[0:n_pred])
prob, mu, sigma = get_mixture_coef(y_pred_train_1) 

# Function to calculate distribution statistics
def calculate_distribution_stats(prob, mu, sigma, x_range=(0, 10), num_points=5000):
    """Calculate distribution probability and the x value corresponding to the peak for each sample"""
    # Check if input matrices have consistent shapes
    n, m = prob.shape
    assert mu.shape == (n, m)
    assert sigma.shape == (n, m)
    
    # Create x values for distribution calculation
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate Gaussian distribution values at each x point using matrix operations for speed
    x_diff = x[:, np.newaxis, np.newaxis] - mu[np.newaxis, :, :]
    
    # Calculate values of each Gaussian at each x point
    exponent = -0.5 * (x_diff / sigma[np.newaxis, :, :]) ** 2
    gaussian_values = np.exp(exponent) / (sigma[np.newaxis, :, :] * np.sqrt(2 * np.pi))
    
    # Apply probability weights
    weighted_gaussian = gaussian_values * prob[np.newaxis, :, :]
    
    # Calculate distribution probability for each sample (sum of all Gaussians)
    distribution = np.sum(weighted_gaussian, axis=2).T
    
    # Calculate cumulative distribution function (CDF) for each sample
    dx = x[1] - x[0]
    cdf = np.cumsum(distribution * dx, axis=1)
    
    # Calculate x value corresponding to peak for each sample
    peak_indices = np.argmax(distribution, axis=1)
    peak_x = x[peak_indices]
    
    # Calculate mean x value for each sample
    epsilon = 1e-10
    normalized_distribution = distribution / (np.sum(distribution, axis=1, keepdims=True) + epsilon)
    mean_x = np.sum(x[np.newaxis, :] * normalized_distribution, axis=1)
    
    return distribution, cdf, peak_x, mean_x, x

prob = np.asarray(prob)
mu = np.asarray(mu)
sigma = np.asarray(sigma)

# Calculate distribution statistics
results_train = calculate_distribution_stats(prob, mu, sigma)
results_test = calculate_distribution_stats(prob_test, mu_test, sigma_test)
pdf = results_train[0]
cdf_train = results_train[1]  # Training set CDF
pdf_test = results_test[0]
cdf_test = results_test[1]    # Test set CDF
print(f"PDF shape: {pdf.shape}, CDF shape: {cdf_train.shape}")

# Get predicted values
y_pred_train = results_train[3]
y_pred_test = results_test[3]

# Add CRPS calculation function
def calculate_crps(pdf, cdf, x, y_true):
    """
    Calculate Continuous Ranked Probability Score (CRPS)
    
    Parameters:
    pdf: predicted probability density function, shape (n_samples, n_x)
    cdf: predicted cumulative distribution function, shape (n_samples, n_x)
    x: evaluation points, shape (n_x,)
    y_true: true values, shape (n_samples,)
    
    Returns:
    crps_values: CRPS for each sample
    mean_crps: mean CRPS
    """
    n_samples, n_x = pdf.shape
    dx = x[1] - x[0]  # grid spacing
    crps_values = []
    
    for i in range(n_samples):
        y = y_true[i]
        
        idx = np.searchsorted(x, y)
        
        part1 = np.sum(np.square(cdf[i, :idx]) * dx)
        
        part2 = np.sum(np.square(1 - cdf[i, idx:]) * dx)
        
        crps = part1 + part2
        crps_values.append(crps)
    
    crps_values = np.array(crps_values)
    return crps_values, np.mean(crps_values), np.median(crps_values)
    
# zConf calculation function
    
def calculate_zconf(pdf, x, mean_x):
    """
    Calculate zConf metric (Carrasco Kind & Brunner 2013)
    
    Parameters:
    pdf: predicted probability density function (n_samples, n_x)
    x: evaluation grid (n_x,)
    mean_x: mean of PDF for each sample (n_samples,)
    
    Returns:
    zconf_values: zConf for each sample
    """
    n_samples = pdf.shape[0]
    dx = x[1] - x[0]
    zconf_values = []
    
    for i in range(n_samples):
        z_m = mean_x[i]
        
        if z_m < 1:
            alpha = 0.05
        else:
            alpha = 0.05 
        
        delta = alpha * (1 + z_m)
        z_min = z_m - delta
        z_max = z_m + delta
        
        mask = (x >= z_min) & (x <= z_max)
        
        zconf = np.sum(pdf[i, mask] * dx)
        zconf_values.append(zconf)
        
    return np.array(zconf_values)
    

# Calculate evaluation metrics
coef = 0.15
photo_z = y_pred_train
spect_z = np.squeeze(hsc_y_train[0:n_pred])

print(f"Training prediction shape: {photo_z.shape}, Training true shape: {spect_z.shape}")

photo_z_test = y_pred_test
spect_z_test = np.squeeze(hsc_y_test[0:n_pred])

# Calculate CRPS for training and test sets
x_grid = results_train[4]  # Get x grid

# Calculate CRPS for training set
crps_train_values, mean_crps_train, median_crps_train = calculate_crps(
    pdf, cdf_train, x_grid, spect_z
)

# Calculate CRPS for test set
crps_test_values, mean_crps_test, median_crps_test = calculate_crps(
    pdf_test, cdf_test, x_grid, spect_z_test
)

# --- Calculate zConf ---
# Training set zConf
zconf_train = calculate_zconf(pdf, x_grid, y_pred_train)

# Test set zConf
zconf_test = calculate_zconf(pdf_test, x_grid, y_pred_test)

print(f"Training set mean zConf: {np.mean(zconf_train):.4f}")
print(f"Test set mean zConf: {np.mean(zconf_test):.4f}")

########## Save test set results to file #########
p_z = y_pred_test.flatten()
s_z = spect_z_test.flatten()
z_c = zconf_test.flatten()
crps_c = crps_test_values.flatten()

save_data = np.column_stack((p_z, s_z, z_c, crps_c))

np.savetxt('GalaxiesML_pred_z.dat', save_data, 
           fmt='%.6f', 
           header='photo_z_test spect_z_test zconf_test crps_test',
           comments='')

print("File GalaxiesML_pred_z.dat has been successfully saved.")

# Create mask to select records with zconf > 0.95
mask = zconf_test > 0

photo_z_test = photo_z_test[mask]
spect_z_test = spect_z_test[mask]
zconf_test = zconf_test[mask]
crps_test_values = crps_test_values[mask]
pdf_test = pdf_test[mask, :]
cdf_test = cdf_test[mask]
mean_crps_test = np.mean(crps_test_values)
median_crps_test = np.median(crps_test_values)

# Calculate outlier fraction
x = np.linspace(0, 4, 1000)
y_erro = x + coef * (1 + x)
y_erro1 = x - coef * (1 + x)
num_outlier = 0
num_outlier_test = 0
delta_z = []
delta_z_test = []

for p in range(len(photo_z)):
    delta_z.append(photo_z[p] - spect_z[p])
    if (photo_z[p] > (1 + spect_z[p]) * coef + spect_z[p] or 
        photo_z[p] < spect_z[p] - coef * (1 + spect_z[p])):
        num_outlier += 1

outlier_frac = round(num_outlier / len(photo_z), 3)  

for p in range(len(photo_z_test)):
    delta_z_test.append(photo_z_test[p] - spect_z_test[p])
    if (photo_z_test[p] > (1 + spect_z_test[p]) * coef + spect_z_test[p] or 
        photo_z_test[p] < spect_z_test[p] - coef * (1 + spect_z_test[p])):
        num_outlier_test += 1

outlier_frac_test = round(num_outlier_test / len(photo_z_test), 3)  

# --- Calculate Catastrophic Outlier (Oc) ---
# Formula: |z_phot - z_spec| > 1.0

# Training set
num_cata_outlier_train = np.sum(np.abs(photo_z - spect_z) > 1.0)
outlier_cata_frac_train = round(num_cata_outlier_train / len(photo_z), 4)

# Test set
num_cata_outlier_test = np.sum(np.abs(photo_z_test - spect_z_test) > 1.0)
outlier_cata_frac_test = round(num_cata_outlier_test / len(photo_z_test), 4)

# --- Print section ---
print(f"Training set Catastrophic Outlier (Oc): {outlier_cata_frac_train}")
print(f"Test set Catastrophic Outlier (Oc): {outlier_cata_frac_test}")

# Calculate sigma_NMAD
z_array = np.asarray(delta_z)
med_delta_z = np.median(z_array)

med_z = []
for q in range(len(photo_z)):
    med_z.append(abs((photo_z[q] - spect_z[q] - med_delta_z) / (1 + spect_z[q])))

# sigma_NMAD = round(1.48 * np.median(np.asarray(med_z)), 3)
sigma_NMAD = round(1.0 * np.median(np.asarray(med_z)), 3)

z_array_test = np.asarray(delta_z_test)
med_delta_z_test = np.median(z_array_test)

med_z_test = []
for q in range(len(photo_z_test)):
    med_z_test.append(abs((photo_z_test[q] - spect_z_test[q] - med_delta_z_test) / (1 + spect_z_test[q])))

# sigma_NMAD_test = round(1.48 * np.median(np.asarray(med_z_test)), 3)
sigma_NMAD_test = round(1.0 * np.median(np.asarray(med_z_test)), 3)

# Calculate bias
bias_temp_test = -np.asarray(delta_z_test) / (1 + spect_z_test)
bias_test = round(np.median(bias_temp_test), 4)
bias_temp = -np.asarray(delta_z) / (1 + spect_z)
bias = round(np.median(bias_temp), 4)

# Calculate MSE and MAE
mse_train = mean_squared_error(spect_z, photo_z)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(spect_z, photo_z)
mse_test = mean_squared_error(spect_z_test, photo_z_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(spect_z_test, photo_z_test)

# Calculate normalized squared error for training set
res_sq_train = ((photo_z - spect_z) / (1 + spect_z)) ** 2
rms_error_train = np.sqrt(np.mean(res_sq_train))

# Calculate normalized squared error for test set
res_sq_test = ((photo_z_test - spect_z_test) / (1 + spect_z_test)) ** 2
rms_error_test = np.sqrt(np.mean(res_sq_test))

print(f"Training set RMSE (standard): {rmse_train:.6f}")
print(f"Training set rms_error (normalized): {rms_error_train:.6f}") 
print(f"Test set RMSE (standard): {rmse_test:.6f}")
print(f"Test set rms_error (normalized): {rms_error_test:.6f}") 

print(f"Training set outlier fraction: {outlier_frac}, sigma_NMAD: {sigma_NMAD}, bias: {bias}")
print(f"Test set outlier fraction: {outlier_frac_test}, sigma_NMAD: {sigma_NMAD_test}, bias: {bias_test} ")

# Plot scatter plot of predictions
plt.figure(figsize=(7, 6))

plt.rcParams.update({'font.size': 12})

spph_test = np.vstack([spect_z_test, photo_z_test])
z_con_test = gaussian_kde(spph_test)(spph_test)
idx_test = z_con_test.argsort()
spect_z_test_con, photo_z_test_con, z_con_test = spect_z_test[idx_test], photo_z_test[idx_test], z_con_test[idx_test]
scatter_test = plt.scatter(spect_z_test_con, photo_z_test_con, c=z_con_test, s=2, cmap='Spectral')
cbar_test = plt.colorbar(scatter_test)
cbar_test.set_label('Density')

plt.plot(x, y_erro, color='b', linestyle=':')
plt.plot(x, y_erro1, color='b', linestyle=':')

plt.plot([0, 4], [0, 4], color='b', linestyle='-', linewidth=1.5)

plt.text(0.3, 3.7, r'$\mathrm{bias} = %.4f$' % bias_test)
plt.text(0.3, 3.5, r'$f_{\mathrm{out}} = %.3f$' % outlier_frac_test)
plt.text(0.3, 3.3, r'$\sigma_{\mathrm{NMAD}} = %.3f$' % sigma_NMAD_test)  # Greek letter σ
plt.text(0.3, 3.1, r'$N_{\mathrm{test}} = %d$' % len(photo_z_test))

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.xlabel('Spectroscopic Redshift', fontsize=14)
plt.ylabel('Photometric Redshift', fontsize=14)
# plt.title('Test Set Predictions', fontsize=16)

ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)  
    
ax.xaxis.set_ticks_position('both')  
ax.yaxis.set_ticks_position('both')  
ax.set_xticks(np.arange(0, 4, 1))    
ax.set_yticks(np.arange(0, 4, 1))    
    
ax.tick_params(axis='both', direction='in', width=2, labelsize=12)

plt.tight_layout()
plt.show()

# Plot PDF and CDF plots
plt.figure(figsize=(16, 8))

plt.rcParams.update({'font.size': 10})

selected_indices = np.array([5, 6, 9, 8])

legend_positions = ['upper right', 'upper right', 'upper right', 'upper left']

for idx, i in enumerate(selected_indices):
    ax1 = plt.subplot(2, 4, idx + 1)
    plt.plot(results_test[4], pdf_test[i, :], linewidth=1.5)
    plt.axvline(x=spect_z_test[i], color='r', linestyle='--', label=f'True: {spect_z_test[i]:.2f}')
    plt.axvline(x=photo_z_test[i], color='g', linestyle='--', label=f'Predicted: {photo_z_test[i]:.2f}')
    if idx == 0:
        plt.xlim(0.1, 1.1)
    elif idx == 1:
        plt.xlim(0, 3)
    elif idx == 2:
        plt.xlim(0, 0.6)
    elif idx == 3:
        plt.xlim(0.2, 0.6)
    plt.xlabel('Redshift', fontsize=12)
    plt.ylabel('PDF', fontsize=12)
    plt.title(f'Sample {idx + 1}', fontsize=14)  
    
    plt.legend(frameon=False, loc=legend_positions[idx])
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.tick_params(axis='both', direction='in', width=2, labelsize=10)
    
    ax2 = plt.subplot(2, 4, idx + 5)
    plt.plot(results_test[4], cdf_test[i, :])
    plt.axvline(x=spect_z_test[i], color='r', linestyle='--')
    plt.axhline(y=0.5, color='gray', linestyle=':')  
    
    plt.text(0.95, 0.05, f'CRPS: {crps_test_values[i]:.4f}', 
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=12)
    
    if idx == 0:
        plt.xlim(0.1, 1.1)
    elif idx == 1:
        plt.xlim(0, 3)
    elif idx == 2:
        plt.xlim(0, 0.6)
    elif idx == 3:
        plt.xlim(0.2, 0.6)
    plt.ylim(0, 1)
    plt.xlabel('Redshift', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title(f'Sample {idx + 1}', fontsize=14)  

    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
    
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params(axis='both', direction='in', width=2, labelsize=10)

plt.tight_layout()
plt.show()

# Plot histogram of CRPS distribution
plt.figure(figsize=(6, 6))
plt.hist(crps_test_values, bins=25, range=(0, 1), alpha=0.7, color='orange', edgecolor='black')
plt.axvline(x=mean_crps_test, color='r', linestyle='--', label=f'Mean CRPS: {mean_crps_test:.4f}')
plt.axvline(x=median_crps_test, color='blue', linestyle=':', label=f'Median CRPS: {median_crps_test:.4f}')
plt.title('Distribution of CRPS Values (Test Set)')
plt.xlabel('CRPS')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

def plot_pit(P, Z, S, bins=20, title="PIT Histogram", show_qq=True):
    """Plot Probability Integral Transform (PIT) histogram, optimized normalization step"""
    P = np.asarray(P)
    Z = np.asarray(Z)
    S = np.asarray(S)
    
    n, m = P.shape
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape != (n, 1):
        raise ValueError("Z must have shape n×1")
    if S.ndim == 1:
        S = S.reshape(1, -1)
    if S.shape != (1, m):
        raise ValueError("S must have shape 1×m")
    
    P_normalized = P / np.sum(P, axis=1, keepdims=True)
    
    pit_values = []
    for i in range(n):
        z_true = Z[i, 0]
        mask = S[0, :] <= z_true
        if np.sum(mask) == 0:
            cdf = 0.0
        elif np.sum(mask) == m:
            cdf = 1.0
        else:
            cdf = np.sum(P_normalized[i, mask])
        pit_values.append(cdf)
    
    pit_values = np.array(pit_values)
    
    if show_qq:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot PIT histogram
    ax1.hist(pit_values, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axhline(y=1, color='r', linestyle='--', label='Uniform(0,1)')
    ax1.set_title(title, fontsize=14)
    ax1.set_xlabel('PIT Values', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.legend()
    
    # Plot QQ plot
    if show_qq:
        uniform_quantiles = uniform.ppf(np.linspace(0, 1, n))
        pit_quantiles = np.sort(pit_values)
        
        ax2.scatter(uniform_quantiles, pit_quantiles, alpha=0.2, marker='o', facecolors='none', color='green')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2)  # reference line
        ax2.set_title('QQ Plot vs Uniform(0,1)', fontsize=14)
        ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax2.set_ylabel('Sample Quantiles', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig, pit_values

# Plot PIT plot
z_true = spect_z_test.reshape((-1, 1))
s = results_test[4].reshape((1, -1))
print(f"True shape: {z_true.shape}, Grid shape: {s.shape}, PDF shape: {pdf_test.shape}")

plot_samples = len(z_true)
random_indices = np.random.choice(len(z_true), plot_samples, replace=False)

fig, pit_values = plot_pit(
    pdf_test[random_indices], 
    z_true[random_indices], 
    s, 
    bins=100, 
    title="Probability Integral Transform (PIT) Histogram"
)

pit_mean = np.mean(pit_values)
pit_std = np.std(pit_values)
pit_ks, _ = scipy.stats.kstest(pit_values, 'uniform')

plt.figtext(0.5, 0.01, 
            f"PIT Mean: {pit_mean:.3f}, Std: {pit_std:.3f}, KS-statistic: {pit_ks:.3f}",
            ha="center", fontsize=12)

plt.show()
