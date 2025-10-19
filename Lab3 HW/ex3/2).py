from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from math import comb
import numpy as np

model = BayesianNetwork([('Starter', 'SecondHeads'),
                         ('FirstRoll', 'SecondHeads')])

#CPD pt starter
cpd_starter = TabularCPD(
    variable='Starter',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'Starter': ['P0', 'P1']}
)

#CPD pt firstroll
cpd_first = TabularCPD(
    variable='FirstRoll',
    variable_card=6,
    values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]],
    state_names={'FirstRoll': [1, 2, 3, 4, 5, 6]}
)

#CPD ptr SecondHeads
max_heads = 12
values = []

for k in range(max_heads + 1):
    row = []
    for starter_state in ['P0', 'P1']:
        for n in range(1, 7):
            flips = 2 * n
            other_p = 4/7 if starter_state == 'P0' else 1/2
            prob = comb(flips, k) * (other_p ** k) * ((1 - other_p) ** (flips - k)) if k <= flips else 0
            row.append(prob)
    values.append(row)

cpd_second = TabularCPD(
    variable='SecondHeads',
    variable_card=max_heads + 1,
    values=values,
    evidence=['Starter', 'FirstRoll'],
    evidence_card=[2, 6],
    state_names={
        'SecondHeads': list(range(max_heads + 1)),
        'Starter': ['P0', 'P1'],
        'FirstRoll': [1, 2, 3, 4, 5, 6]
    }
)

#adaugam CPD-urile
model.add_cpds(cpd_starter, cpd_first, cpd_second)

#verificare si afisare
print("Verificat?:", model.check_model())
print("Structura retea:", model.edges())
print("Variabile:", model.nodes())
