"""
In cod construim un model Bayesian pentru a intelege cum se distribuie numarul total 
de clienti n intr-un magazin, pornind de la informatia despre cati clienti au cumparat un produs Y 
si de la probabilitatea θ ca un client sa cumpere produsul. 

Pasi:
1. Definirea priorului: presupun ca n urmeaza o distributie Poisson cu media 10, ceea ce reflecta 
   credinta initiala despre cate persoane ar putea veni in magazin intr-o zi.
2. Definirea likelihood-ului: pentru fiecare valoare posibila a lui n, numarul de clienti care cumpara 
   produsul Y urmeaza o distributie Binomiala cu parametrii n si θ, unde θ este probabilitatea cunoscuta 
   ca un client sa cumpere produsul.
3. Efectuez inferenta Bayesiana folosind PyMC: esantionam posteriorul lui n,adica distributia lui n actualizata pe baza observatiilor Y.
4. Generam predictii pentru viitor: folosim valorile lui n din posterior pentru a calcula distributia 
   predictiva a viitorului numar de cumparatori Y*, astfel incat sa pot estima ce s-ar putea intampla 
   daca ziua s-ar repeta.
5. Vizualizare: afisez grafice pentru posteriorul lui n si pentru distributia predictiva a lui Y* 
   pentru fiecare combinatie de valori Y si θ, astfel incat sa pot compara cum se schimba distributiile 
   in functie de datele observate si de probabilitatea de cumparare.

Astfel vedem atat cat de probabil este sa vina un anumit numar de clienti, 
cat si ce asteptari am pentru viitorii cumparatori, intr-un mod complet Bayesian.
"""






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
