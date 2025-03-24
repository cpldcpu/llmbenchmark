import numpy as np
from PIL import Image
import math
import random

# Vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, reflection=0.5, transparency=0, refraction=1.0):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = reflection
        self.transparency = transparency
        self.refraction = refraction
    
    def intersect(self, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)
        c = np.linalg.norm(ray_origin - self.center) ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4 * c
        
        if discriminant > 0:
            t1 = (-b + math.sqrt(discriminant)) / 2
            t2 = (-b - math.sqrt(discriminant)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Raytracing functions
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    
    return nearest_object, min_distance

def trace_ray(ray_origin, ray_direction, objects, lights, max_depth=3):
    # Find nearest object
    nearest_object, min_distance = nearest_intersected_object(objects, ray_origin, ray_direction)
    
    if nearest_object is None:
        return np.array([0.1, 0.1, 0.1])  # Background color
    
    # Compute intersection point and normal
    intersection = ray_origin + min_distance * ray_direction
    normal = normalize(intersection - nearest_object.center)
    
    # Compute color
    color = np.zeros(3)
    
    # Ambient light
    color += 0.1 * nearest_object.color
    
    # Diffuse and specular lighting
    for light in lights:
        light_direction = normalize(light.position - intersection)
        
        # Shadow check
        shadow_intersection, shadow_distance = nearest_intersected_object(
            objects, intersection + normal * 1e-5, light_direction)
        
        if shadow_intersection is None or shadow_distance > np.linalg.norm(light.position - intersection):
            # Diffuse
            color += (light.color * light.intensity * nearest_object.color * 
                     max(0, np.dot(normal, light_direction)))
            
            # Specular
            reflection = reflected(-light_direction, normal)
            color += (light.color * light.intensity * 
                     max(0, np.dot(reflection, -ray_direction)) ** 50)
    
    # Reflection
    if max_depth > 0 and nearest_object.reflection > 0:
        reflection_direction = reflected(ray_direction, normal)
        reflection_color = trace_ray(
            intersection + normal * 1e-5, 
            reflection_direction, 
            objects, lights, max_depth - 1)
        color = color * (1 - nearest_object.reflection) + reflection_color * nearest_object.reflection
    
    # Refraction (simplified)
    if max_depth > 0 and nearest_object.transparency > 0:
        refraction_direction = ray_direction  # Simplified - should use Snell's law
        refraction_color = trace_ray(
            intersection - normal * 1e-5,
            refraction_direction,
            objects, lights, max_depth - 1)
        color = color * (1 - nearest_object.transparency) + refraction_color * nearest_object.transparency
    
    return np.clip(color, 0, 1)

# Create scene
def create_scene():
    objects = [
        # Ground (large sphere)
        Sphere([0, -10004, -20], 10000, [0.2, 0.2, 0.2], reflection=0.3),
        
        # Main spheres
        Sphere([0, 0, -20], 4, [1, 0.32, 0.36], reflection=0.75),
        Sphere([5, -1, -15], 2, [0.9, 0.76, 0.46], reflection=0.5),
        Sphere([-5, 0, -25], 3, [0.65, 0.77, 0.97], reflection=0.5),
        Sphere([-2, 5, -15], 3, [0.9, 0.9, 0.9], transparency=0.8, refraction=1.5),
        
        # Small colorful spheres
        Sphere([-4, -2, -10], 1, [0.8, 0.4, 0.9], reflection=0.3),
        Sphere([3, 3, -18], 1.5, [0.3, 0.8, 0.4], reflection=0.3),
        Sphere([6, 2, -12], 1, [0.2, 0.6, 0.9], reflection=0.3),
    ]
    
    lights = [
        Light([-20, 20, 20], [1, 0.6, 0.6], 1.5),  # Pink
        Light([30, 50, -25], [0.7, 0.7, 1], 1.8),    # Blue
        Light([30, 20, 30], [0.9, 0.9, 0.6], 1.2),   # Yellow
        Light([0, 30, 0], [0.6, 1, 0.6], 1.0),       # Green
    ]
    
    return objects, lights

# Render function
def render(width=800, height=600):
    # Camera setup
    camera = np.array([0, 0, 1])
    ratio = width / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    
    # Create scene
    objects, lights = create_scene()
    
    # Create image buffer
    image = np.zeros((height, width, 3))
    
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # Screen coordinates
            pixel = np.array([x, y, 0])
            ray_direction = normalize(pixel - camera)
            
            # Trace ray
            color = trace_ray(camera, ray_direction, objects, lights)
            
            # Store color
            image[i, j] = color
    
    # Convert to 8-bit and save
    image = np.clip(image * 255, 0, 255).astype('uint8')
    img = Image.fromarray(image, 'RGB')
    img.save('raytraced_scene.png')
    print("Image saved as 'raytraced_scene2.png'")

# Run the renderer
if __name__ == "__main__":
    render(800, 600)