import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# All combinations
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

# Store posterior traces for visualization
posterior_results = {}

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            # Prior: n ~ Poisson(10)
            n = pm.Poisson("n", mu=10)

            # Likelihood: Y ~ Binomial(n, θ)
            Y_obs = pm.Binomial("Y_obs", n=n, p=theta, observed=Y)

            # Sample posterior
            trace = pm.sample(2000, tune=1000, cores=1, chains=2, progressbar=False)

            posterior_results[(Y, theta)] = trace

# One big posterior plot
_, axs = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        ax = axs[i, j]
        az.plot_posterior(
            posterior_results[(Y, theta)],
            var_names=["n"],
            ax=ax,
        )
        ax.set_title(f"Posterior for n | Y={Y}, θ={theta}")

plt.tight_layout()
plt.show()

# PART (c): Predictive posterior for Y*
for Y in Y_values:
    for theta in theta_values:
        trace = posterior_results[(Y, theta)]

        with pm.Model() as predictive_model:
            # n is drawn from posterior of n
            n_post = pm.Poisson("n", mu=10)
            # But we replace it with posterior samples
            pm.set_data({"n": trace.posterior["n"].values.flatten()})

            # Predictive Y*
            Y_future = pm.Binomial("Y_future", n=n_post, p=theta)

            pred_samples = pm.sample_prior_predictive(samples=3000)

        az.plot_dist(pred_samples["Y_future"], 
                     label=f"Y* | Y={Y}, θ={theta}")
        plt.legend()
        plt.show()
