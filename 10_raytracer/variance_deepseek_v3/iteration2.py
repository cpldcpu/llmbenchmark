import numpy as np
from PIL import Image

# Define the size of the image
width = 800
height = 600

# Create an empty image array
image = np.zeros((height, width, 3), dtype=np.uint8)

# Define the camera position
camera_pos = np.array([0, 0, -5])

# Define the screen dimensions
screen_width = 2.0
screen_height = 1.5
screen_z = 0

# Define the light sources with their positions and colors
lights = [
    {"position": np.array([2, 3, -10]), "color": np.array([255, 0, 0])},    # Red light
    {"position": np.array([-2, -1, -15]), "color": np.array([0, 255, 0])},  # Green light
    {"position": np.array([0, 4, -5]), "color": np.array([0, 0, 255])},     # Blue light
]

# Define a simple sphere
sphere = {
    "center": np.array([0, 0, 0]),
    "radius": 1.0,
    "color": np.array([255, 255, 255])  # White color
}

# Function to calculate the intersection of a ray with the sphere
def intersect_sphere(ray_origin, ray_direction, sphere):
    oc = ray_origin - sphere["center"]
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere["radius"] ** 2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    else:
        t = (-b - np.sqrt(discriminant)) / (2*a)
        return t

# Function to compute the color at a given point
def compute_color(point, normal, light_sources, sphere_color):
    color = np.zeros(3)
    for light in light_sources:
        light_dir = light["position"] - point
        light_distance = np.linalg.norm(light_dir)
        light_dir = light_dir / light_distance
        intensity = max(0, np.dot(normal, light_dir)) / (light_distance ** 2)
        color += sphere_color * light["color"] * intensity
    return np.clip(color, 0, 255)

# Raytracing loop
for y in range(height):
    for x in range(width):
        # Convert pixel coordinate to world space
        px = (x / width - 0.5) * screen_width
        py = (0.5 - y / height) * screen_height
        ray_direction = np.array([px, py, screen_z]) - camera_pos
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Check for intersection with the sphere
        t = intersect_sphere(camera_pos, ray_direction, sphere)
        if t is not None:
            # Compute the intersection point and normal
            intersection_point = camera_pos + t * ray_direction
            normal = intersection_point - sphere["center"]
            normal = normal / np.linalg.norm(normal)
            
            # Compute the color at the intersection point
            color = compute_color(intersection_point, normal, lights, sphere["color"])
            
            # Set the pixel color
            image[y, x] = color

# Save the image as a PNG file
Image.fromarray(image, 'RGB').save('raytraced_scene2.png')
