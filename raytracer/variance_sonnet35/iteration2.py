import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray
    specular: float
    reflection: float

@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def intersect_ray_sphere(origin: np.ndarray, direction: np.ndarray, sphere: Sphere) -> Tuple[Optional[float], Optional[float]]:
    b = 2 * np.dot(direction, origin - sphere.center)
    c = np.linalg.norm(origin - sphere.center) ** 2 - sphere.radius ** 2
    delta = b ** 2 - 4 * c
    if delta < 0:
        return None, None
    t1 = (-b + np.sqrt(delta)) / 2
    t2 = (-b - np.sqrt(delta)) / 2
    return t1, t2

def compute_lighting(point: np.ndarray, normal: np.ndarray, view: np.ndarray, specular: float, lights: List[Light], spheres: List[Sphere]) -> np.ndarray:
    intensity = np.zeros(3)
    
    for light in lights:
        light_dir = normalize(light.position - point)
        # Shadow check
        shadow = False
        for sphere in spheres:
            t1, t2 = intersect_ray_sphere(point + normal * 0.001, light_dir, sphere)
            if t1 is not None and t1 > 0:
                shadow = True
                break
        
        if not shadow:
            # Diffuse lighting
            n_dot_l = np.dot(normal, light_dir)
            if n_dot_l > 0:
                intensity += light.color * light.intensity * n_dot_l
            
            # Specular lighting
            if specular != -1:
                reflection = 2 * normal * n_dot_l - light_dir
                r_dot_v = np.dot(reflection, view)
                if r_dot_v > 0:
                    intensity += light.color * light.intensity * (r_dot_v ** specular)
    
    return np.clip(intensity, 0, 1)

def trace_ray(origin: np.ndarray, direction: np.ndarray, spheres: List[Sphere], lights: List[Light], depth: int) -> np.ndarray:
    if depth <= 0:
        return np.zeros(3)
    
    closest_t = float('inf')
    closest_sphere = None
    
    for sphere in spheres:
        t1, t2 = intersect_ray_sphere(origin, direction, sphere)
        if t1 is not None:
            if t1 > 0.001 and t1 < closest_t:
                closest_t = t1
                closest_sphere = sphere
    
    if closest_sphere is None:
        return np.zeros(3)
    
    point = origin + closest_t * direction
    normal = normalize(point - closest_sphere.center)
    
    color = closest_sphere.color * compute_lighting(point, normal, -direction, closest_sphere.specular, lights, spheres)
    
    if closest_sphere.reflection > 0:
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflection = trace_ray(point + normal * 0.001, reflect_dir, spheres, lights, depth - 1)
        color = color * (1 - closest_sphere.reflection) + reflection * closest_sphere.reflection
    
    return np.clip(color, 0, 1)

def render_scene(width: int, height: int) -> np.ndarray:
    aspect_ratio = width / height
    fov = np.pi / 3
    
    # Scene setup
    spheres = [
        Sphere(np.array([0, -1, 3]), 1, np.array([1, 0.2, 0.2]), 50, 0.2),  # Red sphere
        Sphere(np.array([2, 0, 4]), 1, np.array([0.2, 1, 0.2]), 50, 0.3),   # Green sphere
        Sphere(np.array([-2, 0, 4]), 1, np.array([0.2, 0.2, 1]), 50, 0.3),  # Blue sphere
        Sphere(np.array([0, -5001, 0]), 5000, np.array([0.5, 0.5, 0.5]), 10, 0.1),  # Ground
    ]
    
    lights = [
        Light(np.array([5, 5, -5]), np.array([1, 0.3, 0.3]), 0.6),   # Red light
        Light(np.array([-5, 5, -5]), np.array([0.3, 1, 0.3]), 0.6),  # Green light
        Light(np.array([0, 5, -5]), np.array([0.3, 0.3, 1]), 0.6),   # Blue light
    ]
    
    image = np.zeros((height, width, 3))
    
    for i in range(height):
        for j in range(width):
            x = (2 * (j + 0.5) / width - 1) * np.tan(fov / 2) * aspect_ratio
            y = -(2 * (i + 0.5) / height - 1) * np.tan(fov / 2)
            direction = normalize(np.array([x, y, 1]))
            
            color = trace_ray(np.array([0, 0, -5]), direction, spheres, lights, depth=3)
            image[i, j] = color
    
    return (image * 255).astype(np.uint8)

# Render and save the image
image_array = render_scene(800, 600)
image = Image.fromarray(image_array)
image.save('raytraced_scene2.png')