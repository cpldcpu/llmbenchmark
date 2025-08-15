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
    def __init__(self, center, radius, color, specular=50):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Ray tracing functions
def intersect_sphere(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return float('inf')
    t = (-b - math.sqrt(discriminant)) / (2.0*a)
    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2.0*a)
    return t if t > 0 else float('inf')

def trace_ray(ray_origin, ray_dir, spheres, lights, depth=0):
    if depth > 3:  # Max recursion depth
        return np.array([0, 0, 0])
    
    min_t = float('inf')
    hit_sphere = None
    
    # Find closest intersection
    for sphere in spheres:
        t = intersect_sphere(ray_origin, ray_dir, sphere)
        if t < min_t:
            min_t = t
            hit_sphere = sphere
    
    if hit_sphere is None:
        return np.array([0.1, 0.1, 0.2])  # Background color
    
    # Calculate intersection point and normal
    hit_point = ray_origin + ray_dir * min_t
    normal = normalize(hit_point - hit_sphere.center)
    
    # Lighting calculation
    color = np.zeros(3)
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        light_dist = np.linalg.norm(light.position - hit_point)
        
        # Shadow test
        shadow = False
        for s in spheres:
            if s != hit_sphere:
                t = intersect_sphere(hit_point + normal * 0.001, light_dir, s)
                if t < light_dist:
                    shadow = True
                    break
        
        if not shadow:
            # Diffuse
            diffuse = max(0, dot(normal, light_dir))
            # Specular
            view_dir = normalize(ray_origin - hit_point)
            reflect_dir = reflect(-light_dir, normal)
            specular = pow(max(0, dot(reflect_dir, view_dir)), sphere.specular)
            
            color += (diffuse * sphere.color * light.color * light.intensity + 
                     specular * light.color * light.intensity * 0.5)
    
    return np.clip(color, 0, 1)

def reflect(v, n):
    return v - 2 * dot(v, n) * n

# Main rendering function
def render():
    width, height = 800, 600
    aspect_ratio = width / height
    image = np.zeros((height, width, 3))
    
    # Camera setup
    camera_pos = np.array([0, 0, -5])
    fov = 60
    tan_fov = math.tan(math.radians(fov/2))
    
    # Scene setup
    spheres = [
        Sphere([0, -1, 3], 1, [0.9, 0.1, 0.1], 100),    # Red sphere
        Sphere([2, 0, 4], 0.8, [0.1, 0.9, 0.1], 50),    # Green sphere
        Sphere([-2, 0, 4], 0.8, [0.1, 0.1, 0.9], 50),   # Blue sphere
        Sphere([0, -1001, 0], 1000, [0.5, 0.5, 0.5], 10) # Floor
    ]
    
    lights = [
        Light([2, 2, 0], [1, 0, 0], 0.8),    # Red light
        Light([-2, 2, 0], [0, 1, 0], 0.8),   # Green light
        Light([0, 5, 5], [0, 0, 1], 0.8),    # Blue light
        Light([0, 5, -2], [1, 1, 1], 0.5)    # White light
    ]
    
    # Render loop
    for y in range(height):
        for x in range(width):
            # Convert to camera space
            px = (2 * (x + 0.5) / width - 1) * tan_fov * aspect_ratio
            py = (1 - 2 * (y + 0.5) / height) * tan_fov
            ray_dir = normalize(np.array([px, py, 1]))
            
            color = trace_ray(camera_pos, ray_dir, spheres, lights)
            image[y, x] = color
            
        if y % 50 == 0:
            print(f"Progress: {y/height*100:.1f}%")
    
    # Convert to 8-bit RGB and save
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image, 'RGB')
    img.save('raytraced_scene2.png')
    print("Image saved as 'raytraced_scene2.png'")

if __name__ == "__main__":
    render()