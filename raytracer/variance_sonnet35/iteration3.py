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
    reflective: float

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
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return None
    
    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)
    
    if t1 < 0 and t2 < 0:
        return None
    
    return min(t for t in (t1, t2) if t > 0)

def compute_lighting(point, normal, view, specular, lights, spheres):
    intensity = np.zeros(3)
    
    for light in lights:
        light_dir = normalize(light.position - point)
        
        # Shadow check
        shadow = False
        for sphere in spheres:
            if intersect_ray_sphere(point + light_dir * 0.001, light_dir, sphere):
                shadow = True
                break
        
        if not shadow:
            # Diffuse
            n_dot_l = np.dot(normal, light_dir)
            if n_dot_l > 0:
                intensity += light.color * light.intensity * n_dot_l
            
            # Specular
            if specular != -1:
                reflection = 2 * normal * n_dot_l - light_dir
                r_dot_v = np.dot(reflection, view)
                if r_dot_v > 0:
                    intensity += light.color * light.intensity * pow(r_dot_v, specular)
    
    return np.clip(intensity, 0, 1)

def trace_ray(origin, direction, spheres, lights, depth=3):
    closest_t = float('inf')
    closest_sphere = None
    
    for sphere in spheres:
        t = intersect_ray_sphere(origin, direction, sphere)
        if t and t < closest_t:
            closest_t = t
            closest_sphere = sphere
    
    if not closest_sphere:
        return np.array([0.1, 0.1, 0.2])  # Sky color
    
    point = origin + closest_t * direction
    normal = normalize(point - closest_sphere.center)
    
    color = closest_sphere.color * compute_lighting(
        point, normal, -direction, closest_sphere.specular, lights, spheres
    )
    
    if depth <= 0 or closest_sphere.reflective <= 0:
        return color
    
    reflection_dir = direction - 2 * np.dot(direction, normal) * normal
    reflection_color = trace_ray(
        point + reflection_dir * 0.001, 
        reflection_dir, 
        spheres, 
        lights, 
        depth - 1
    )
    
    return color * (1 - closest_sphere.reflective) + reflection_color * closest_sphere.reflective

def render_scene():
    width, height = 800, 600
    fov = 90
    aspect_ratio = width / height
    
    # Scene setup
    spheres = [
        Sphere(np.array([0, -1, 3]), 1, np.array([1, 0.2, 0.2]), 50, 0.2),      # Red sphere
        Sphere(np.array([2, 0, 4]), 1, np.array([0.2, 1, 0.2]), 50, 0.3),       # Green sphere
        Sphere(np.array([-2, 0, 4]), 1, np.array([0.2, 0.2, 1]), 50, 0.3),      # Blue sphere
        Sphere(np.array([0, -5001, 0]), 5000, np.array([0.5, 0.5, 0.5]), 10, 0.1),  # Ground
    ]
    
    lights = [
        Light(np.array([2, 1, 0]), np.array([1, 0.2, 0.2]), 0.6),    # Red light
        Light(np.array([-2, 1, 0]), np.array([0.2, 0.2, 1]), 0.6),   # Blue light
        Light(np.array([0, 4, -1]), np.array([1, 1, 1]), 0.4),       # White light
    ]
    
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            screen_x = (2 * (x + 0.5) / width - 1) * math.tan(fov/2 * math.pi/180) * aspect_ratio
            screen_y = (1 - 2 * (y + 0.5) / height) * math.tan(fov/2 * math.pi/180)
            direction = normalize(np.array([screen_x, screen_y, 1]))
            
            # Trace ray
            color = trace_ray(np.array([0, 0, 0]), direction, spheres, lights)
            image[y, x] = np.clip(color * 255, 0, 255)
    
    # Save the image
    Image.fromarray(image.astype('uint8')).save('raytraced_scene3.png')

if __name__ == "__main__":
    render_scene()