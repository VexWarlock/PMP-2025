from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import JointProbabilityDistribution
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#definim modelul Markov
model = MarkovModel()

#adaugam nodurile 
nodes = ['A1', 'A2', 'A3', 'A4', 'A5']
model.add_nodes_from(nodes)

#adaugam muchiile
model.add_edges_from([
    ('A1', 'A2'),
    ('A1', 'A3'),
    ('A2', 'A4'),
    ('A2', 'A5'),
    ('A3', 'A4'),
    ('A4', 'A5') 
])

plt.figure(figsize=(6, 5))
pos = nx.spring_layout(model)
nx.draw_networkx_nodes(model, pos, node_size=700)
nx.draw_networkx_edges(model, pos, width=2)
nx.draw_networkx_labels(model, pos, font_size=10, font_weight='bold')
plt.title("Markov Network Graph")
plt.axis('off')
plt.show()


maximal_cliques = list(nx.find_cliques(model.to_undirected()))
print("\nMaximal Cliques al modellui Markov:")
for clique in maximal_cliques:
    print(clique)
