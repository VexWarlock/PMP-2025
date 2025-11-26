import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

posteriors = {}
predictives = {}

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            n = pm.Poisson("n", mu=10)
            y_obs = pm.Binomial("y_obs", n=n, p=theta, observed=Y)
            
            # Sample posterior
            trace = pm.sample(2000, tune=1000, cores=1, progressbar=False)
            
            # Posterior predictive
            y_pred = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=False, random_seed=42)
        
        key = f"Y={Y}, θ={theta}"
        posteriors[key] = trace
        predictives[key] = y_pred

# Plot posterior distributions of n
fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 8))
for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        key = f"Y={Y}, θ={theta}"
        az.plot_posterior(posteriors[key].posterior["n"], ax=axes[i, j])
        axes[i, j].set_title(key)
plt.tight_layout()
plt.show()

# Plot predictive distributions of Y*
fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 8))
for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        key = f"Y={Y}, θ={theta}"
        y_samples = predictives[key].posterior_predictive["y_obs"].values.flatten()
        az.plot_dist(y_samples, ax=axes[i, j])
        axes[i, j].set_title(f"Predictive {key}")
plt.tight_layout()
plt.show()
