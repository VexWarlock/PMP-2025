import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definim structura
model = BayesianModel([
    ('S', 'O'), 
    ('S', 'L'), 
    ('S', 'M'),
    ('L', 'M')
])

# Definim CPD-urile
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])
cpd_O = TabularCPD(variable='O', variable_card=2, 
                   values=[[0.3, 0.9],  
                           [0.7, 0.1]],
                   evidence=['S'], evidence_card=[2])
cpd_L = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.2, 0.7],   
                           [0.8, 0.3]],
                   evidence=['S'], evidence_card=[2])
cpd_M = TabularCPD(variable='M', variable_card=2,
                   values=[
                       [0.1, 0.5, 0.4, 0.8],  
                       [0.9, 0.5, 0.6, 0.2]
                   ],
                   evidence=['S', 'L'], evidence_card=[2, 2])

# adaugam CPD-urile si verificam modelul
model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
assert model.check_model()

# afisam independentele
indep = model.get_independencies()
print("Independente:")
print(indep)

# Graficul retelei
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue', arrowsize=20)
plt.title("Bayesian Network Structure")
plt.show()
