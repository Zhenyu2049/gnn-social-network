import pandas as pd
from pyvis.network import Network

net = Network()
nodes=[]
label=[]
edges=[]
nodes_number=100

df = pd.read_csv("data/musae_git_edges.csv")
for i in range(nodes_number):
    nodes.append(df.iloc[i]["id_1"])
    nodes.append(df.iloc[i]["id_2"])
    edges.append((int(df.iloc[i]["id_1"]),int(df.iloc[i]["id_2"])))
nodes=list(set(nodes))
nodes.sort()
print(nodes)

for element in nodes:
    label.append(str(element))
print(label)
print(edges)

net.add_nodes(nodes=nodes,label=label)
net.add_edges(edges=edges)
net.repulsion(node_distance=500, spring_length=200)
net.show_buttons(filter_=True)

net.show('edges.html')



