import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_alphabet_mapping():
    """Load the alphabet-to-vector mapping from JSON file"""
    with open("data/alphabet_vectors.json", "r") as f:
        mapping = json.load(f)
    return mapping

def create_3d_vector_graph():
    """Create a 3D graph showing vectors as lines with 4th value as labels"""
    # Load the data
    mapping = load_alphabet_mapping()
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each letter's vector
    for letter, vector in mapping.items():
        x, y, z, fourth_val = vector
        
        # Draw line from origin (0,0,0) to (x,y,z)
        ax.plot([0, x], [0, y], [0, z], 'b-', alpha=0.7, linewidth=2)
        
        # Add arrow head at the tip
        ax.scatter(x, y, z, c='red', s=50, alpha=0.8)
        
        # Add label showing letter and 4th value
        ax.text(x, y, z, f'{letter.upper()}({fourth_val})', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('X (1st value)')
    ax.set_ylabel('Y (2nd value)')
    ax.set_zlabel('Z (3rd value)')
    ax.set_title('3D Visualization of Alphabet Vectors\n(Lines show first 3 values, labels show 4th value)')
    
    # Set equal aspect ratio
    max_range = max([max(vector[:3]) for vector in mapping.values()]) + 1
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def create_4d_color_graph():
    """Create a 3D graph with 4th dimension mapped to color intensity"""
    mapping = load_alphabet_mapping()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all 4th values for color scaling
    fourth_values = [vector[3] for vector in mapping.values()]
    min_val, max_val = min(fourth_values), max(fourth_values)
    
    for letter, vector in mapping.items():
        x, y, z, fourth_val = vector
        
        # Map 4th value to color intensity (0-1)
        color_intensity = (fourth_val - min_val) / (max_val - min_val)
        color = plt.cm.viridis(color_intensity)  # Blue to yellow gradient
        
        # Draw line and point with mapped color
        ax.plot([0, x], [0, y], [0, z], color=color, linewidth=3, alpha=0.8)
        ax.scatter(x, y, z, c=[color], s=80, alpha=0.9)
        ax.text(x, y, z, f'{letter.upper()}', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('X (1st value)')
    ax.set_ylabel('Y (2nd value)')
    ax.set_zlabel('Z (3rd value)')
    ax.set_title('3D Alphabet Vectors\n(Color represents 4th dimension: Blue=Low, Yellow=High)')
    
    # Set equal aspect ratio
    max_range = max([max(vector[:3]) for vector in mapping.values()]) + 1
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])
    
    # Add colorbar to show 4th dimension scale
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(fourth_values)
    cbar = plt.colorbar(mappable, ax=ax, label='4th Dimension Value', shrink=0.5, pad=0.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create the basic 3D graph
    # create_3d_vector_graph()
    
    # Create the 4D color-mapped graph
    create_4d_color_graph()
    
