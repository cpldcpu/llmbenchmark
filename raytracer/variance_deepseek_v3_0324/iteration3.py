import numpy as np
from PIL import Image
import math
import random

# Vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def plane_intersect(normal, point, ray_origin, ray_direction):
    denom = np.dot(normal, ray_direction)
    if abs(denom) > 1e-6:
        t = np.dot(normal, point - ray_origin) / denom
        if t > 0:
            return t
    return None

# Materials
class Material:
    def __init__(self, color, diffuse=1.0, specular=1.0, reflection=0.5):
        self.color = np.array(color)
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection

# Objects
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

class Plane:
    def __init__(self, normal, point, material):
        self.normal = np.array(normal)
        self.point = np.array(point)
        self.material = material

# Light sources
class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Scene setup
def create_scene():
    materials = [
        Material(color=(0.4, 0.4, 0.4), reflection=0.6),  # Floor
        Material(color=(1, 0, 0), diffuse=0.9, specular=0.1),  # Red sphere
        Material(color=(0, 1, 0), diffuse=0.9, specular=0.1),  # Green sphere
        Material(color=(0, 0, 1), diffuse=0.9, specular=0.1),  # Blue sphere
        Material(color=(1, 1, 0), diffuse=0.9, specular=0.1),  # Yellow sphere
        Material(color=(1, 1, 1), diffuse=0.2, specular=0.8, reflection=0.8)  # Mirror sphere
    ]
    
    objects = [
        Plane(normal=(0, 1, 0), point=(0, -0.5, 0), material=materials[0]),
        Sphere(center=(-1, 0, -1), radius=0.5, material=materials[1]),
        Sphere(center=(1, 0, -1), radius=0.5, material=materials[2]),
        Sphere(center=(0, 0, -3), radius=0.5, material=materials[3]),
        Sphere(center=(-2, 0.5, 1), radius=0.5, material=materials[4]),
        Sphere(center=(2, 0.5, 1), radius=0.5, material=materials[5])
    ]
    
    lights = [
        Light(position=(-3, 2, 2), color=(1, 0, 0), intensity=5),  # Red light
        Light(position=(3, 2, 2), color=(0, 1, 0), intensity=5),    # Green light
        Light(position=(0, 5, -5), color=(0, 0, 1), intensity=3),   # Blue light
        Light(position=(0, 2, 5), color=(1, 1, 1), intensity=4),    # White light
        Light(position=(-2, 1, -2), color=(1, 0, 1), intensity=3),  # Purple light
        Light(position=(2, 1, -2), color=(0, 1, 1), intensity=3)    # Cyan light
    ]
    
    return objects, lights

# Raytracing functions
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = []
    for obj in objects:
        if isinstance(obj, Sphere):
            distances.append(sphere_intersect(obj.center, obj.radius, ray_origin, ray_direction))
        elif isinstance(obj, Plane):
            distances.append(plane_intersect(obj.normal, obj.point, ray_origin, ray_direction))
    
    nearest_object = None
    min_distance = np.inf
    for i, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[i]
    
    return nearest_object, min_distance

def trace_ray(ray_origin, ray_direction, objects, lights, max_depth=3):
    if max_depth == 0:
        return np.array([0, 0, 0])
    
    nearest_object, distance = nearest_intersected_object(objects, ray_origin, ray_direction)
    if nearest_object is None:
        return np.array([0, 0, 0])  # Background color
    
    # Compute intersection point and normal
    intersection = ray_origin + distance * ray_direction
    
    if isinstance(nearest_object, Sphere):
        normal = normalize(intersection - nearest_object.center)
    elif isinstance(nearest_object, Plane):
        normal = nearest_object.normal
    
    # Offset to prevent self-intersection
    offset_point = intersection + 1e-5 * normal
    
    # Compute lighting
    material = nearest_object.material
    color = np.zeros(3)
    
    for light in lights:
        # Check if light is visible
        light_direction = normalize(light.position - offset_point)
        light_distance = np.linalg.norm(light.position - offset_point)
        
        # Shadow check
        shadow_intersection, shadow_distance = nearest_intersected_object(
            objects, offset_point, light_direction)
        
        if shadow_intersection and shadow_distance < light_distance:
            continue
        
        # Diffuse component
        diffuse_intensity = max(0, np.dot(normal, light_direction))
        diffuse = material.diffuse * diffuse_intensity * material.color * light.color * light.intensity
        
        # Specular component (Blinn-Phong)
        view_direction = normalize(-ray_direction)
        halfway = normalize(light_direction + view_direction)
        specular_intensity = max(0, np.dot(normal, halfway)) ** 50
        specular = material.specular * specular_intensity * light.color * light.intensity
        
        # Attenuation
        attenuation = 1 / (1 + 0.1 * light_distance + 0.01 * light_distance ** 2)
        
        color += (diffuse + specular) * attenuation
    
    # Reflection
    if material.reflection > 0:
        reflection_direction = reflected(ray_direction, normal)
        reflection_color = trace_ray(offset_point, reflection_direction, objects, lights, max_depth - 1)
        color += material.reflection * reflection_color
    
    return np.clip(color, 0, 1)

# Main rendering function
def render(width=800, height=600):
    # Camera setup
    camera = np.array([0, 0, 1])
    ratio = width / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # Left, top, right, bottom
    
    # Create scene
    objects, lights = create_scene()
    
    # Create image buffer
    image = np.zeros((height, width, 3))
    
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # Screen coordinates to world coordinates
            pixel = np.array([x, y, 0])
            ray_direction = normalize(pixel - camera)
            
            # Trace the ray
            color = trace_ray(camera, ray_direction, objects, lights)
            
            # Store the color
            image[i, j] = color
    
    # Convert to 8-bit and save
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image, 'RGB')
    img.save('raytraced_scene3.png')
    img.show()

if __name__ == "__main__":
    render()