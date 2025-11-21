#Exercitiul 2

import numpy as np
from hmmlearn import hmm

#a
#hidden states: W=0, R=1,S=2
states = ["W","R","S"]

#observations: L=0, M=1, H=2
obs_map = {"L": 0, "M": 1, "H": 2}
obs_seq = ["M", "H", "L"]
obs_indices = np.array([obs_map[o] for o in obs_seq])

#convert observations to one-hot count vectors (required for n_trials=1)
observations = np.zeros((len(obs_indices), 3), dtype=int)
observations[np.arange(len(obs_indices)), obs_indices] = 1

#initial probabilities
pi = np.array([0.4, 0.3, 0.3])

#transition probabilities
A = np.array([
    [0.6, 0.3, 0.1],  # from W
    [0.2, 0.7, 0.1],  # from R
    [0.3, 0.2, 0.5]   # from S
])

#emission probabilities
B = np.array([
    [0.1, 0.7, 0.2],   # W emits L,M,H
    [0.05, 0.25, 0.7], # R emits L,M,H
    [0.8, 0.15, 0.05]  # S emits L,M,H
])

#create multinomial HMM
model = hmm.MultinomialHMM(n_components=3, n_trials=1, init_params="", params="")
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

#b
logprob = model.score(observations)
forward_probability = np.exp(logprob)
print("Forward probability P(M,H,L) =", forward_probability)

#c
logprob_v, hidden_states = model.decode(observations, algorithm="viterbi")
decoded_sequence = [states[s] for s in hidden_states]

print("Viterbi most likely state sequence:", decoded_sequence)
print("Viterbi log probability:", logprob_v)

#why is Viterbi preferred over brute force?
#because brute force requires checking all possible sequences of hidden states (3^T)
#which grows exponentially with sequence length. Viterbi runs in O(N^2*t)

#d
N = 10000
count = 0

target = tuple(obs_indices)

for _ in range(N):
    X, _ = model.sample(3)
    #flatten and convert to indices for comparison
    sampled_indices = np.argmax(X, axis=1)
    if tuple(sampled_indices) == target:
        count += 1

empirical_prob = count / N
print("\nempirical probability (10000 samples):", empirical_prob)
print("exact Forward probability:", forward_probability)
