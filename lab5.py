#Ex 1

import numpy as np
from hmmlearn import hmm

stari=["Difficult","Medium","Easy"]
nr_stari=len(stari)

observatii=["FB","B","S","NS"]
nr_observatii=len(observatii)

start_prob=np.array([1/3,1/3,1/3])

transition_prob=np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])


emission_prob=np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])


#cream modelul HMM
model=hmm.CategoricalHMM(n_components=nr_stari, init_params="")
model.startprob_=start_prob
model.transmat_=transition_prob
model.emissionprob_=emission_prob

obs_sequence=["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
obs_index=np.array([[observatii.index(o)] for o in obs_sequence])

logprob = model.score(obs_index)
prob=np.exp(logprob)
print(f"Probabilitate secventei de obs: {prob:.10f}")


logprob_viterbi, state_sequence = model.decode(obs_index, algorithm="viterbi")
decoded_states = [stari[i] for i in state_sequence]

print("\nCea mai probabila secventa de dificultati la test:")
print(decoded_states)
print(f"\nProbabilitatea celei mai probabile secvente: {np.exp(logprob_viterbi):.10f}")

#Bonus
