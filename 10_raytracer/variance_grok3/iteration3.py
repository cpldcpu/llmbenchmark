import numpy as np
from PIL import Image
import math

# Vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def dot(v1, v2):
    return np.clip(np.dot(v1, v2), 0, 1)

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, reflection=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)  # RGB 0-1
        self.reflection = reflection

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)  # RGB 0-1
        self.intensity = intensity

# Ray tracing functions
def intersect_sphere(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return float('inf')
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else float('inf')

def trace_ray(ray_origin, ray_dir, objects, lights, depth=0, max_depth=3):
    if depth > max_depth:
        return np.array([0.1, 0.1, 0.1])  # Background color
    
    # Find closest intersection
    closest_t = float('inf')
    closest_sphere = None
    
    for obj in objects:
        t = intersect_sphere(ray_origin, ray_dir, obj)
        if t < closest_t:
            closest_t = t
            closest_sphere = obj
    
    if closest_sphere is None:
        return np.array([0.1, 0.1, 0.1])
    
    # Calculate intersection point and normal
    hit_point = ray_origin + ray_dir * closest_t
    normal = normalize(hit_point - closest_sphere.center)
    
    # Lighting calculation
    color = np.zeros(3)
    ambient = 0.1
    
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        light_dist = np.linalg.norm(light.position - hit_point)
        
        # Shadow test (simplified)
        in_shadow = False
        shadow_origin = hit_point + normal * 0.001
        for obj in objects:
            if obj != closest_sphere:
                t = intersect_sphere(shadow_origin, light_dir, obj)
                if t < light_dist:
                    in_shadow = True
                    break
        
        if not in_shadow:
            diffuse = dot(normal, light_dir)
            color += closest_sphere.color * light.color * diffuse * light.intensity
    
    # Add ambient light and clamp
    color = color * (1 - ambient) + closest_sphere.color * ambient
    
    # Reflection
    if closest_sphere.reflection > 0 and depth < max_depth:
        reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
        reflect_origin = hit_point + normal * 0.001
        reflect_color = trace_ray(reflect_origin, reflect_dir, objects, lights, depth + 1)
        color = color * (1 - closest_sphere.reflection) + reflect_color * closest_sphere.reflection
    
    return np.clip(color, 0, 1)

def render_scene():
    # Scene setup
    width, height = 800, 600
    aspect = width / height
    fov = math.tan(math.radians(60) / 2)
    
    # Camera
    camera = np.array([0, 0, -5])
    
    # Objects
    spheres = [
        Sphere([0, 0, 0], 1.0, [1.0, 0.2, 0.2], 0.7),    # Red sphere
        Sphere([-2, 1, 1], 0.7, [0.2, 1.0, 0.2], 0.5),   # Green sphere
        Sphere([2, -1, 2], 0.5, [0.2, 0.2, 1.0], 0.6),   # Blue sphere
        Sphere([0, -101, 0], 100.0, [0.5, 0.5, 0.5], 0.3) # Ground plane
    ]
    
    # Lights
    lights = [
        Light([-3, 3, -3], [1.0, 0.0, 0.0], 0.8),  # Red light
        Light([3, 2, -2], [0.0, 1.0, 0.0], 0.7),   # Green light
        Light([0, 4, -1], [0.0, 0.0, 1.0], 0.6),   # Blue light
    ]
    
    # Render
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # Convert to camera space
            px = (2 * (x + 0.5) / width - 1) * fov * aspect
            py = (1 - 2 * (y + 0.5) / height) * fov
            ray_dir = normalize(np.array([px, py, 1]))
            
            color = trace_ray(camera, ray_dir, spheres, lights)
            image[y, x] = color
    
    # Convert to 8-bit and save
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save("raytraced_scene3.png")
    print("Image saved as 'raytraced_scene3.png'")

if __name__ == "__main__":
    render_scene()