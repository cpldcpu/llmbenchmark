import numpy as np
from PIL import Image
import math

# Vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def dot(v1, v2):
    return np.dot(v1, v2)

# Scene objects
class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Ray tracing functions
def ray_sphere_intersect(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return float('inf'), None
        
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        return float('inf'), None
        
    hit_point = ray_origin + ray_dir * t
    normal = normalize(hit_point - sphere.center)
    return t, normal

def trace_ray(ray_origin, ray_dir, spheres, lights, depth=0):
    if depth > 3:  # Maximum recursion depth
        return np.array([0, 0, 0])
        
    closest_t = float('inf')
    hit_sphere = None
    hit_normal = None
    
    # Find closest intersection
    for sphere in spheres:
        t, normal = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t < closest_t:
            closest_t = t
            hit_sphere = sphere
            hit_normal = normal
            
    if hit_sphere is None:
        return np.array([0.1, 0.1, 0.1])  # Background color
        
    hit_point = ray_origin + ray_dir * closest_t
    
    # Calculate lighting
    color = np.array([0.0, 0.0, 0.0])
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        light_dist = np.linalg.norm(light.position - hit_point)
        
        # Shadow test
        shadow_origin = hit_point + hit_normal * 0.001
        shadow_t = float('inf')
        for s in spheres:
            t, _ = ray_sphere_intersect(shadow_origin, light_dir, s)
            if t < shadow_t and t < light_dist:
                shadow_t = t
                break
                
        if shadow_t == float('inf'):  # No shadow
            diffuse = max(0, dot(hit_normal, light_dir))
            intensity = light.intensity / (light_dist * light_dist)
            color += hit_sphere.color * light.color * diffuse * intensity
            
    return np.clip(color, 0, 1)

def render():
    # Scene setup
    width, height = 800, 600
    aspect_ratio = width / height
    
    # Camera setup
    camera_pos = np.array([0, 0, -5])
    fov = 60
    tan_fov = math.tan(math.radians(fov/2))
    
    # Scene objects
    spheres = [
        Sphere([0, -1, 5], 1, [0.8, 0.2, 0.2]),    # Red sphere
        Sphere([-2, 0, 4], 0.7, [0.2, 0.8, 0.2]),  # Green sphere
        Sphere([2, 0, 6], 0.9, [0.2, 0.2, 0.8]),   # Blue sphere
        Sphere([0, -1001, 0], 1000, [0.5, 0.5, 0.5])  # Floor
    ]
    
    lights = [
        Light([-3, 3, 0], [1, 0, 0], 50),    # Red light
        Light([3, 2, -2], [0, 1, 0], 50),    # Green light
        Light([0, 4, 5], [0, 0, 1], 50),     # Blue light
    ]
    
    # Render
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # Convert to normalized device coordinates (-1 to 1)
            ndc_x = (2 * x / width - 1) * aspect_ratio * tan_fov
            ndc_y = (1 - 2 * y / height) * tan_fov
            ray_dir = normalize(np.array([ndc_x, ndc_y, 1]))
            
            color = trace_ray(camera_pos, ray_dir, spheres, lights)
            image[y, x] = color
    
    # Convert to 8-bit RGB and save
    img = (image * 255).astype(np.uint8)
    Image.fromarray(img).save('raytraced_scene1.png')

if __name__ == "__main__":
    render()
    print("Image saved as 'raytraced_scene1.png'")