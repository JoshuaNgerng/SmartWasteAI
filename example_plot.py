import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys

# Data
locations = {
    1: (451, 205),
    2: (242, 327),
    3: (577, 345),
}

requests = {
    1: 40,
    2: 25,
    3: 6
}

# The routes for trucks
routes = {
    1: [1, 3, 2],
    2: [3, 1, 2]
}

# Function to generate colors dynamically using HSL space
def generate_colors(n):
    # We will generate n distinct colors by modifying the hue
    colors = []
    for i in range(n):
        # Vary the hue between 0 and 1 (360 degrees on the color wheel)
        hue = i / n  # This will give us a distinct hue for each color
        # Convert HSL to RGB, keeping saturation and lightness fixed
        rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)  # Saturation = 0.5, Lightness = 0.7
        colors.append(rgb)
    return colors

# Assuming all locations have positive coordinates (x, y)
def plot_vehicle_routing(locations, routes, max_trucks):
    # Generate a list of colors for all trucks
    colors = generate_colors(max_trucks)
    
    plt.figure(figsize=(10, 8))
    
    # Plot all the locations first
    for loc_id, (x, y) in locations.items():
        plt.scatter(x, y, color='red', s=100, label=f'Location {loc_id}' if loc_id not in [l[2] for l in routes.values()] else "")
        plt.text(x + 5, y, f'{loc_id}', fontsize=12, color='black')

    # Plot the routes for each truck
    for idx, route in routes.items():
        truck_color = colors[idx - 1]  # Get the color for the current truck
        
        # Plot the route line
        for i in range(len(route) - 1):
            start_loc = route[i]
            end_loc = route[i+1]
            
            # Get coordinates of the starting and ending location
            start_x, start_y = locations[start_loc]
            end_x, end_y = locations[end_loc]
            
            # Plot a line between the start and end locations
            plt.plot([start_x, end_x], [start_y, end_y], color=truck_color, linestyle='-', linewidth=2)

    # Add labels and title
    plt.title('Vehicle Routing Problem - Routes and Requests')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    
    # Display plot
    plt.savefig("test.png")

# Calling the function to plot the routing
matplotlib.use('Agg')
plot_vehicle_routing(locations, routes, 2)
