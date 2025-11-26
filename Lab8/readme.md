# Bayesian Inference for Customer Count (Posterior and Predictive)

A store receives an unknown number of customers `n` per day.  
Each customer independently buys a product with probability θ.  
Observed: Y customers bought the product.

- Likelihood: **Y ~ Binomial(n, θ)**
- Prior: **n ~ Poisson(10)**

Goal:  
Compute posterior distributions of `n` for combinations:  
- Y ∈ {0, 5, 10}  
- θ ∈ {0.2, 0.5}

Then generate predictive posterior distributions for **future buyers Y\***.

---

## (b) Effect of Y and θ on the Posterior of n

### Effect of Y  
- Higher Y pushes the posterior of n upward.  
- If many customers bought the product, the total visitors must have been higher.

### Effect of θ  
- With high θ, fewer customers are needed to explain Y → posterior for n shifts lower.  
- With low θ, more visitors are needed → posterior shifts higher.

**Summary:**  
- Increasing Y → increases inferred n.  
- Decreasing θ → increases inferred n.

---

## (d) Predictive Posterior vs Posterior for n

### Posterior for n
- Describes uncertainty about **how many customers visited today**.
- Conditioned only on observed Y.

### Predictive Posterior for Y\*
- Describes uncertainty about **future number of buyers**.
- Integrates over uncertainty in n:

  p(Y* | Y) = ∫ p(Y* | n, θ) p(n | Y) dn

- More uncertain (wider) because it accounts for both:
  - randomness in customer behavior, and
  - uncertainty in n.
