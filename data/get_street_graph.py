import osmnx as ox

# Configure OSMnx to fetch driveable public streets
ox.settings.log_console = True

# # Download the entire street network within the administrative boundaries of Chicago
print("Downloading all Chicago streets... This will take a moment.")
G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")
output_filename = "chicago_real_complete_network.graphml"
G = ox.io.load_graphml(output_filename)
print(f"Success! Saved complete graph to {output_filename}")
