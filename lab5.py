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
import numpy as np

#definirea starilor si observarilor
stari = ["Difficult", "Medium", "Easy"]
observatii = ["FB", "B", "S", "NS"]

#probabilitatile initiale de start pentru fiecare stare
start_prob = np.array([1/3, 1/3, 1/3])

#matricea de tranzitie intre stari
transition_prob = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

#probabilitatile de emisie a fiecarei observatii din fiecare stare
emission_prob = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

#secventa de observatii de analizat
obs_sequence = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
#convertim observarile in indici numerici pentru calcul
obs_index = [observatii.index(o) for o in obs_sequence]

#numarul de stari si lungimea secventei
n_states = len(stari)
T = len(obs_index)

# matricele pentru algoritmul viterbi
#V[s, t] stocheaza probabilitatea maxima pentru traseul care ajunge in starea s la momentul t
V = np.zeros((n_states, T))
#backpointer[s, t] retine starea anterioara optima pentru starea s la momentul t
backpointer = np.zeros((n_states, T), dtype=int)

#initializare pentru t=0
for s in range(n_states):
    #probabilitatea traseului incepe cu start_prob * probabilitate de emisie pentru prima observatie
    V[s, 0] = start_prob[s] * emission_prob[s, obs_index[0]]
    backpointer[s, 0] = 0  # nu conteaza pentru prima coloana

#recursivitate pentru t>0
for t in range(1, T):
    for s in range(n_states):
        #probabilitatea de a ajunge in starea s la momentul t din fiecare stare precedenta
        prob_tranz = V[:, t-1] * transition_prob[:, s] * emission_prob[s, obs_index[t]]
        #alegem maximul pentru probabilitatea optima
        V[s, t] = np.max(prob_tranz)
        #retinem starea precedenta corespunzatoare
        backpointer[s, t] = np.argmax(prob_tranz)

#terminare si reconstruire traseu
#probabilitatea maxima a traseului final
best_path_prob = np.max(V[:, T-1])
#starea finala corespunzatoare probabilitatii maxime
best_last_state = np.argmax(V[:, T-1])

#reconstruim traseul optim mergand inapoi folosind backpointer
best_path = [best_last_state]
for t in range(T-1, 0, -1):
    best_path.insert(0, backpointer[best_path[0], t])

#convertim indicii in nume de stari
decoded_states_manual = [stari[i] for i in best_path]

#afisare rezultate
print("\nCea mai probabila secventa de dificultati la test:")
print(decoded_states_manual)
print(f"Probabilitatea traseului: {best_path_prob:.10f}")
