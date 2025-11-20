import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# DATELE DIN PROBLEMĂ
# ------------------------------------------------------------
data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

# ------------------------------------------------------------
# (a) MODELUL BAYESIAN CU PRIORI SLABI
#     μ ~ N(x, 10^2), unde x = media eșantionului
# ------------------------------------------------------------
x = np.mean(data)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=x, sigma=10)      # μ ~ Normal(media_sample, 10)
    sigma = pm.HalfNormal("sigma", sigma=10)  # σ ~ HalfNormal(10)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

    trace = pm.sample(2000, tune=1000, chains=2)

print("\n--- (b) Posteriors & HDI 95% pentru μ și σ ---\n")
print(az.summary(trace, hdi_prob=0.95))

# ------------------------------------------------------------
# (c) Comparație cu estimările frecventiste
# ------------------------------------------------------------
freq_mean = np.mean(data)
freq_std = np.std(data, ddof=1)

print("\n--- Estimări Frecventiste ---")
print(f"Media eșantionului      = {freq_mean:.3f}")
print(f"Deviația standard (s)   = {freq_std:.3f}")

# ------------------------------------------------------------
# (d) Model cu PRIORI PUTERNICE
#     μ ~ N(50, 1^2), σ ~ HalfNormal(10)
# ------------------------------------------------------------
with pm.Model() as strong_prior_model:
    mu_sp = pm.Normal("mu", mu=50, sigma=1)
    sigma_sp = pm.HalfNormal("sigma", sigma=10)
    obs_sp = pm.Normal("obs", mu=mu_sp, sigma=sigma_sp, observed=data)

    trace_sp = pm.sample(2000, tune=1000, chains=2)

print("\n--- Posteriors cu priori puternice (μ ~ N(50,1²)) ---\n")
print(az.summary(trace_sp, hdi_prob=0.95))

# ------------------------------------------------------------
# Plot comparativ (opțional)
# ------------------------------------------------------------
az.plot_posterior(trace, var_names=["mu", "sigma"])
plt.suptitle("Posterior cu priori slabi")
plt.show()

az.plot_posterior(trace_sp, var_names=["mu", "sigma"])
plt.suptitle("Posterior cu priori puternici")
plt.show()
