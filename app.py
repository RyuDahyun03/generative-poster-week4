# generative-poster-week4

# Week 4 - 3D-like Generative Poster
# Concepts: shadow, transparency, layering, depth cues
# Method: Coding with Prompt approach

import random, math
import numpy as np
import matplotlib.pyplot as plt

def blob(center=(0.5,0.5), r=0.3, points=200, wobble=0.15, shape="circle", concavity=0):
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    radii  = r * (1 + wobble*(np.random.rand(points)-0.5) - concavity * np.sin(angles * 2)**2) # Added concavity term

    if shape == "square":
        # Generate points on a square and then add wobble
        half_side = r * math.sqrt(2)
        points_per_side = points // 4
        x_square = []
        y_square = []

        # Top side
        x_square.extend(np.linspace(center[0] - half_side/2, center[0] + half_side/2, points_per_side))
        y_square.extend([center[1] + half_side/2] * points_per_side)

        # Right side
        x_square.extend([center[0] + half_side/2] * points_per_side)
        y_square.extend(np.linspace(center[1] + half_side/2, center[1] - half_side/2, points_per_side))

        # Bottom side
        x_square.extend(np.linspace(center[0] + half_side/2, center[0] - half_side/2, points_per_side))
        y_square.extend([center[1] - half_side/2] * points_per_side)

        # Left side
        x_square.extend([center[0] - half_side/2] * points_per_side)
        y_square.extend(np.linspace(center[1] - half_side/2, center[1] + half_side/2, points_per_side))

        # Add wobble and concavity to square points
        x = np.array(x_square) + wobble * r * (np.random.rand(points)-0.5)
        y = np.array(y_square) + wobble * r * (np.random.rand(points)-0.5)
        # Applying concavity to square might need a different approach, this is a simple radial one
        # For now, let's apply it to the radial distance from the center for simplicity
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        angles_square = np.arctan2(y - center[1], x - center[0])
        radii_square = dist_from_center - concavity * r * np.sin(angles_square * 2)**2
        x = center[0] + radii_square * np.cos(angles_square)
        y = center[1] + radii_square * np.sin(angles_square)


    elif shape == "diamond":
        # Generate points on a diamond (rotated square) and then add wobble
        half_diag = r * math.sqrt(2) / 2
        points_per_side = points // 4
        x_diamond = []
        y_diamond = []

        # Top point to right point
        x_diamond.extend(np.linspace(center[0], center[0] + half_diag, points_per_side))
        y_diamond.extend(np.linspace(center[1] + half_diag, center[1], points_per_side))

        # Right point to bottom point
        x_diamond.extend(np.linspace(center[0] + half_diag, center[0], points_per_side))
        y_diamond.extend(np.linspace(center[1], center[1] - half_diag, points_per_side))

        # Bottom point to left point
        x_diamond.extend(np.linspace(center[0], center[0] - half_diag, points_per_side))
        y_diamond.extend(np.linspace(center[1] - half_diag, center[1], points_per_side))

        # Left point to top point
        x_diamond.extend(np.linspace(center[0] - half_diag, center[0], points_per_side))
        y_diamond.extend(np.linspace(center[1], center[1] + half_diag, points_per_side))

        # Add wobble and concavity to diamond points
        x = np.array(x_diamond) + wobble * r * (np.random.rand(points)-0.5)
        y = np.array(y_diamond) + wobble * r * (np.random.rand(points)-0.5)
        # Applying concavity to diamond might need a different approach, this is a simple radial one
        # For now, let's apply it to the radial distance from the center for simplicity
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        angles_diamond = np.arctan2(y - center[1], x - center[0])
        radii_diamond = dist_from_center - concavity * r * np.sin(angles_diamond * 2)**2
        x = center[0] + radii_diamond * np.cos(angles_diamond)
        y = center[1] + radii_diamond * np.sin(angles_diamond)


    else: # circle shape
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
    return x, y

def generate_3d_poster(n_layers=6, seed=0, shape="circle", concavity=0):
    random.seed(seed); np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.axis('off')
    ax.set_facecolor((0.95,0.95,0.95))

    for depth in range(n_layers):
        # Position & radius - now using a normal distribution centered at 0.5
        cx = np.random.normal(0.5, 0.1) # Mean 0.5, standard deviation 0.1
        cy = np.random.normal(0.5, 0.1) # Mean 0.5, standard deviation 0.1
        # Clamp values to be within [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))

        rr = random.uniform(0.15, 0.45)

        # Shadow (slightly shifted, dark, semi-transparent)
        x_shadow, y_shadow = blob((cx,cy), r=rr, wobble=0.12, shape=shape, concavity=concavity)
        ax.fill(x_shadow+0.02, y_shadow-0.02, color=(0,0,0), alpha=0.2)

        # Main shape with depth-dependent color/opacity
        x_main, y_main = blob((cx,cy), r=rr, wobble=0.12, shape=shape, concavity=concavity) # Use same shape for main
        color = (random.random(), random.random(), random.random())
        alpha = 0.4 + depth*0.08  # closer = stronger opacity
        ax.fill(x_main, y_main, color=color, alpha=min(alpha,1.0))

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"3D-like Generative Poster ({shape.capitalize()}) with Concavity: {concavity:.2f}", fontsize=14, weight="bold")
    plt.show()

# Example run with diamond shape and some concavity
generate_3d_poster(n_layers=10, seed=40, shape="diamond", concavity=0.15)
