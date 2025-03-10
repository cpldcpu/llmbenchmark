import numpy as np
from PIL import Image
from dataclasses import dataclass
import math

@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray
    specular: float
    reflective: float = 0.5

@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float

def normalize(vector):
    return vector / np.linalg.norm(vector)

def intersect_ray_sphere(origin, direction, sphere):
    oc = origin - sphere.center
    a = np.dot(direction, direction)
    b = 2 * np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    disc = b * b - 4 * a * c
    
    if disc < 0:
        return np.inf
    
    t = (-b - np.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b + np.sqrt(disc)) / (2 * a)
    if t < 0:
        return np.inf
    
    return t

def compute_lighting(point, normal, view, specular, objects, lights):
    intensity = 0.1  # ambient light
    
    for light in lights:
        light_dir = normalize(light.position - point)
        
        # Shadow check
        shadow_origin = point + normal * 0.001
        shadow_ray = light_dir
        shadow = False
        
        for obj in objects:
            if intersect_ray_sphere(shadow_origin, shadow_ray, obj) < np.inf:
                shadow = True
                break
        
        if not shadow:
            # Diffuse lighting
            n_dot_l = np.dot(normal, light_dir)
            if n_dot_l > 0:
                intensity += light.intensity * n_dot_l
            
            # Specular lighting
            if specular != -1:
                reflection = 2 * normal * np.dot(normal, light_dir) - light_dir
                r_dot_v = np.dot(reflection, view)
                if r_dot_v > 0:
                    intensity += light.intensity * pow(r_dot_v, specular)
    
    return intensity

def trace_ray(origin, direction, objects, lights, depth=3):
    closest_t = np.inf
    closest_obj = None
    
    for obj in objects:
        t = intersect_ray_sphere(origin, direction, obj)
        if t < closest_t:
            closest_t = t
            closest_obj = obj
    
    if closest_obj is None:
        return np.array([0.2, 0.3, 0.5])  # Sky color
    
    point = origin + closest_t * direction
    normal = normalize(point - closest_obj.center)
    
    color = closest_obj.color * compute_lighting(point, normal, -direction, 
                                               closest_obj.specular, objects, lights)
    
    if depth <= 0 or closest_obj.reflective <= 0:
        return np.clip(color, 0, 1)
    
    reflection_dir = direction - 2 * np.dot(direction, normal) * normal
    reflection_origin = point + normal * 0.001
    reflection_color = trace_ray(reflection_origin, reflection_dir, objects, lights, depth - 1)
    
    return np.clip((1 - closest_obj.reflective) * color + 
                   closest_obj.reflective * reflection_color, 0, 1)

def render_scene():
    width, height = 800, 600
    fov = 60
    aspect_ratio = width / height
    
    # Scene objects
    objects = [
        Sphere(np.array([0, -1, 3]), 1, np.array([1, 0.2, 0.2]), 50),  # Red sphere
        Sphere(np.array([2, 0, 4]), 1, np.array([0.2, 1, 0.2]), 50),   # Green sphere
        Sphere(np.array([-2, 0, 4]), 1, np.array([0.2, 0.2, 1]), 50),  # Blue sphere
        Sphere(np.array([0, -5001, 0]), 5000, np.array([0.5, 0.5, 0.5]), 10)  # Ground
    ]
    
    # Colored lights
    lights = [
        Light(np.array([2, 1, 0]), np.array([1, 0.2, 0.2]), 0.6),    # Red light
        Light(np.array([-2, 1, 0]), np.array([0.2, 1, 0.2]), 0.6),   # Green light
        Light(np.array([0, 1, -1]), np.array([0.2, 0.2, 1]), 0.6),   # Blue light
    ]
    
    image = np.zeros((height, width, 3))
    
    for i in range(height):
        for j in range(width):
            x = (2 * (j + 0.5) / width - 1) * aspect_ratio * math.tan(fov / 2 * math.pi / 180)
            y = -(2 * (i + 0.5) / height - 1) * math.tan(fov / 2 * math.pi / 180)
            direction = normalize(np.array([x, y, 1]))
            
            color = trace_ray(np.array([0, 0, 0]), direction, objects, lights)
            image[i, j] = np.clip(color * 255, 0, 255)
    
    Image.fromarray(image.astype('uint8')).save('raytraced_scene.png')

if __name__ == "__main__":
    render_scene()