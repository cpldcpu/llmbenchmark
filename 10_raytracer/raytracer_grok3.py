import numpy as np
from PIL import Image
import time

# Vector operations
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def dot(v1, v2):
    return np.clip(np.dot(v1, v2), 0, 1)

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, reflectivity):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflectivity = reflectivity

class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Ray tracing functions
def ray_sphere_intersect(origin, direction, sphere):
    oc = origin - sphere.center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return float('inf'), None
    
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        t = (-b + np.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        return float('inf'), None
        
    hit_point = origin + direction * t
    normal = normalize(hit_point - sphere.center)
    return t, (hit_point, normal)

def trace_ray(origin, direction, objects, lights, depth=0, max_depth=3):
    if depth >= max_depth:
        return np.array([0.1, 0.1, 0.1])  # Ambient background color
    
    closest_t = float('inf')
    hit_obj = None
    hit_data = None
    
    # Find closest intersection
    for obj in objects:
        t, data = ray_sphere_intersect(origin, direction, obj)
        if t < closest_t:
            closest_t = t
            hit_obj = obj
            hit_data = data
    
    if hit_obj is None:
        return np.array([0.1, 0.1, 0.1])
    
    hit_point, normal = hit_data
    color = np.zeros(3)
    
    # Calculate lighting
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        light_dist = np.linalg.norm(light.position - hit_point)
        
        # Shadow test
        shadow_origin = hit_point + normal * 0.001
        shadow_t, _ = min(
            (ray_sphere_intersect(shadow_origin, light_dir, obj) 
             for obj in objects if obj != hit_obj),
            key=lambda x: x[0],
            default=(float('inf'), None)
        )
        
        if shadow_t < light_dist:
            continue
            
        # Diffuse lighting
        diffuse = dot(normal, light_dir)
        color += hit_obj.color * light.color * diffuse * light.intensity / (light_dist * light_dist)
    
    # Reflection
    if hit_obj.reflectivity > 0:
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflect_origin = hit_point + normal * 0.001
        reflect_color = trace_ray(reflect_origin, reflect_dir, objects, lights, depth + 1, max_depth)
        color += reflect_color * hit_obj.reflectivity
    
    return np.clip(color, 0, 1)

# Scene setup
width, height = 800, 600
camera = np.array([0, 0, -5])
objects = [
    Sphere([0, 0, 0], 1, [0.8, 0.2, 0.2], 0.7),    # Red sphere
    Sphere([-2, 1, 2], 0.8, [0.2, 0.8, 0.2], 0.5),  # Green sphere
    Sphere([2, -1, 3], 0.6, [0.2, 0.2, 0.8], 0.6),  # Blue sphere
]

lights = [
    Light([-3, 3, -3], [1, 0, 0], 50),    # Red light
    Light([3, 3, -3], [0, 1, 0], 50),     # Green light
    Light([0, -3, -3], [0, 0, 1], 50),    # Blue light
    Light([0, 5, -2], [1, 1, 1], 30),     # White light
]

# Render
image = np.zeros((height, width, 3))
aspect_ratio = width / height

print("Rendering...")
start_time = time.time()

for y in range(height):
    for x in range(width):
        # Convert to normalized device coordinates
        ndc_x = (2 * x / width - 1) * aspect_ratio
        ndc_y = 1 - 2 * y / height
        
        # Ray direction from camera
        direction = normalize(np.array([ndc_x, ndc_y, 1]))
        color = trace_ray(camera, direction, objects, lights)
        image[y, x] = color

# Save image
image = (image * 255).astype(np.uint8)
img = Image.fromarray(image, 'RGB')
img.save('raytraced_scene.png')
print(f"Render completed in {time.time() - start_time:.2f} seconds")
print("Image saved as 'raytraced_scene.png'")