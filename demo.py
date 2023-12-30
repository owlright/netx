from .netx.util.plotters import plot,highlight
import networkx as nx
import matplotlib.pyplot as plt
import random


g = nx.random_graphs.random_regular_graph(4, 15, seed=1234)
weights = { e : random.randint(10, 50) for e in g.edges()}
pos = nx.kamada_kawai_layout(g)
nx.set_node_attributes(g, pos, 'pos')
nx.set_edge_attributes(g, weights, "cost")

# plot(g, edge_label_name="cost")
p = nx.shortest_path(g, 5, 8)
fig, ax = plt.subplots(1, 2, figsize=(14,5))
plt.subplots_adjust(wspace=0.1)
plot(g, ax=ax[0])
highlight(g, p, ax=ax[1])
fig.savefig("demo.png", bbox_inches='tight')