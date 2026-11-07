import osmnx as ox

# Configure OSMnx to fetch driveable public streets
ox.settings.log_console = True
ox.settings.use_cache = True

# # Download the entire street network within the administrative boundaries of Chicago
# print("Downloading all Chicago streets... This will take a moment.")
# G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")

# # Save the real network directly to a GraphML file
output_filename = "chicago_real_complete_network.graphml"
# ox.save_graphml(G, filepath=output_filename)

G = ox.io.load_graphml(output_filename)

# print(f"Success! Saved complete graph to {output_filename}")
ox.plot_graph(G, filepath="chicago_graph.pdf")