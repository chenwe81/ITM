import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load station data
stations = pd.read_csv('/Users/weichen/Desktop/BERLIN-stations.csv')

# Load connection data
connections = pd.read_csv('/Users/weichen/Desktop/BERLIN-connections.csv')

# common = set(connections['station']).union(set(connections['adjacent']))
# print(len(common))
# # Find stations in the connections dataset that are not in the stations dataset
# missing_stations = set(connections['station']).union(set(connections['adjacent'])) - set(stations['name'])
# print(len(missing_stations))
#
# print("Stations present in connections dataset but not in stations dataset:")
# for station in missing_stations:
#     print(station)
# Create an empty graph
G = nx.Graph()

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph using station data
for _, station in stations.iterrows():
    G.add_node(station['name'], pos=(station['longitude'], station['latitude']))

# Filter connections to exclude missing stations
connections = connections[connections['station'].isin(stations['name']) & connections['adjacent'].isin(stations['name'])]

# Add edges to the graph using filtered connection data
for _, connection in connections.iterrows():
    G.add_edge(connection['station'], connection['adjacent'], line=connection['line'])

# Plot the graph based on node position
plt.figure(figsize=(10, 8))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.title("Berlin Stations Graph")
plt.show()

# Plot the graph as a circular layout
plt.figure(figsize=(10, 8))
nx.draw_circular(G, with_labels=True, node_size=50, font_size=8)
plt.title("Circular Layout")
plt.show()

# Plot the graph as a Kamada-Kawai layout
plt.figure(figsize=(10, 8))
nx.draw_kamada_kawai(G, with_labels=True, node_size=50, font_size=8)
plt.title("Kamada-Kawai Layout")
plt.show()

# Plot the adjacency matrix of the graph
plt.figure(figsize=(10, 8))
nx.draw_networkx(G, pos=pos, with_labels=False, node_size=50, font_size=8)
plt.matshow(nx.to_numpy_array(G), cmap=plt.cm.Blues, aspect='auto')
plt.title("Adjacency Matrix")
plt.colorbar(label="Edge Weight")
plt.show()

# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)
highest_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)

print("Node with the highest betweenness centrality:", highest_betweenness_node)

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)
highest_degree_node = max(degree_centrality, key=degree_centrality.get)

print("Node with the highest degree centrality:", highest_degree_node)

# Calculate average clustering coefficient
average_clustering_coefficient = nx.average_clustering(G)
print("Average clustering coefficient:", average_clustering_coefficient)
# Calculate transitivity score
transitivity_score = nx.transitivity(G)

print("Transitivity score:", transitivity_score)

# Calculate graph density
graph_density = nx.density(G)

print("Graph density:", graph_density)

# Calculate number of modules and number of nodes in each module
modules = list(nx.connected_components(G))
num_modules = len(modules)
nodes_per_module = [len(module) for module in modules]

print("Number of modules:", num_modules)
print("Nodes per module:", nodes_per_module)

# Calculate average clustering coefficient and small worldness for modules with more than 50 nodes
large_modules = [module for module in modules if len(module) > 50]
# print(large_modules)
# print(len(large_modules))
average_clustering_coefficient_large = []
small_worldness = []

for module in large_modules:
    subgraph = G.subgraph(module)
    average_clustering_coefficient_large.append(nx.average_clustering(subgraph))
    small_worldness.append(nx.smallworld.sigma(subgraph))

print("Average clustering coefficient of large modules:", average_clustering_coefficient_large)
print("Small worldness of large modules:", small_worldness)
