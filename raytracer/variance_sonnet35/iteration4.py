import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray
    reflectivity: float

@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def intersect_sphere(ray: Ray, sphere: Sphere) -> Tuple[float, float]:
    oc = ray.origin - sphere.center
    a = np.dot(ray.direction, ray.direction)
    b = 2 * np.dot(oc, ray.direction)
    c = np.dot(oc, oc) - sphere.radius**2
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return float('inf'), float('inf')
    
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    return t1, t2

def get_color(ray: Ray, spheres: List[Sphere], lights: List[Light], depth: int = 3) -> np.ndarray:
    if depth <= 0:
        return np.zeros(3)

    closest_t = float('inf')
    closest_sphere = None

    # Find closest intersection
    for sphere in spheres:
        t1, t2 = intersect_sphere(ray, sphere)
        if t1 > 0.001 and t1 < closest_t:
            closest_t = t1
            closest_sphere = sphere

    if closest_sphere is None:
        return np.array([0.1, 0.1, 0.2])  # Sky color

    # Calculate intersection point and normal
    hit_point = ray.origin + ray.direction * closest_t
    normal = normalize(hit_point - closest_sphere.center)

    color = np.zeros(3)
    
    # Calculate lighting
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        shadow_ray = Ray(hit_point + normal * 0.001, light_dir)
        
        # Check if point is in shadow
        in_shadow = False
        for sphere in spheres:
            t1, t2 = intersect_sphere(shadow_ray, sphere)
            if t1 > 0.001 and t1 < np.linalg.norm(light.position - hit_point):
                in_shadow = True
                break

        if not in_shadow:
            diffuse = max(0, np.dot(normal, light_dir))
            color += closest_sphere.color * light.color * diffuse * light.intensity

    # Calculate reflection
    if closest_sphere.reflectivity > 0:
        reflect_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        reflect_ray = Ray(hit_point + normal * 0.001, normalize(reflect_dir))
        reflect_color = get_color(reflect_ray, spheres, lights, depth - 1)
        color = color * (1 - closest_sphere.reflectivity) + reflect_color * closest_sphere.reflectivity

    return np.clip(color, 0, 1)

def render(width: int, height: int, spheres: List[Sphere], lights: List[Light]) -> Image.Image:
    aspect_ratio = width / height
    image = np.zeros((height, width, 3))
    
    camera_pos = np.array([0, 0, -1])
    
    for y in range(height):
        for x in range(width):
            u = (x + 0.5) / width * 2 - 1
            v = -((y + 0.5) / height * 2 - 1)
            
            direction = normalize(np.array([u * aspect_ratio, v, 1.0]))
            ray = Ray(camera_pos, direction)
            
            color = get_color(ray, spheres, lights)
            image[y, x] = color

    return Image.fromarray((image * 255).astype(np.uint8))

# Create scene
spheres = [
    Sphere(np.array([0, 0, 5]), 1, np.array([0.7, 0.3, 0.3]), 0.5),
    Sphere(np.array([-2, 0, 7]), 1, np.array([0.3, 0.7, 0.3]), 0.3),
    Sphere(np.array([2, 0, 7]), 1, np.array([0.3, 0.3, 0.7]), 0.3),
    Sphere(np.array([0, -1001, 0]), 1000, np.array([0.5, 0.5, 0.5]), 0.1),
]

lights = [
    Light(np.array([-5, 5, -5]), np.array([1.0, 0.0, 0.0]), 0.7),
    Light(np.array([5, 5, -5]), np.array([0.0, 1.0, 0.0]), 0.7),
    Light(np.array([0, 5, -5]), np.array([0.0, 0.0, 1.0]), 0.7),
]

# Render and save image
image = render(800, 600, spheres, lights)
image.save('raytraced_scene4.png')