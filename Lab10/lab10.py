import numpy as np
import pymc as pm
import arviz as az


#datele
publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

#model bayesian
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu = alpha + beta * publicity
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)

    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=1,
        cores=1,
        return_inferencedata=True
    )
#rezultate
print("\n-SUMMARY-")
print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))
print("\n-95% HDI-")
print(az.hdi(trace, hdi_prob=0.95, var_names=["alpha", "beta"]))

#predictii
new_publicity = np.array([12, 14, 16])

alpha_s = trace.posterior["alpha"].values.flatten()
beta_s = trace.posterior["beta"].values.flatten()
sigma_s = trace.posterior["sigma"].values.flatten()

print("\n-PREZICERI-")
for p in new_publicity:
    mu_pred = alpha_s + beta_s * p
    y_pred = np.random.normal(mu_pred, sigma_s)
    print(f"Publicity {p}: mean={y_pred.mean():.2f}, HDI={az.hdi(y_pred, hdi_prob=0.95)}")
