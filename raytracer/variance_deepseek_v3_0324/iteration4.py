import numpy as np
from PIL import Image
import math
import random

# Image dimensions
WIDTH = 800
HEIGHT = 600

# Camera settings
CAMERA_POS = np.array([0, 1, 5])
FOV = math.pi / 3  # 60 degrees

# Scene settings
MAX_DEPTH = 3
SAMPLES = 4

class Sphere:
    def __init__(self, center, radius, color, reflection=0.5, transparency=0, refraction=1.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = reflection
        self.transparency = transparency
        self.refraction = refraction
    
    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return float('inf')
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        if t1 > 0.001:
            return t1
        elif t2 > 0.001:
            return t2
        else:
            return float('inf')

class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

def refracted(vector, normal, eta):
    cosi = -max(-1, min(1, np.dot(vector, normal)))
    if cosi < 0:
        return refracted(vector, -normal, 1/eta)
    
    eta = 1/eta
    k = 1 - eta**2 * (1 - cosi**2)
    if k < 0:
        return None
    return eta * vector + (eta * cosi - math.sqrt(k)) * normal

def trace_ray(ray_origin, ray_dir, objects, lights, depth=0):
    if depth > MAX_DEPTH:
        return np.array([0, 0, 0])  # Black
    
    # Find closest intersection
    closest_obj = None
    closest_t = float('inf')
    
    for obj in objects:
        t = obj.intersect(ray_origin, ray_dir)
        if t < closest_t:
            closest_t = t
            closest_obj = obj
    
    if closest_obj is None:
        return np.array([0.2, 0.2, 0.2])  # Background color
    
    # Calculate intersection point and normal
    point = ray_origin + closest_t * ray_dir
    normal = normalize(point - closest_obj.center)
    
    # Offset the point to avoid self-intersection
    offset_point = point + normal * 0.001
    
    # Calculate lighting
    color = np.zeros(3)
    
    # Ambient light
    ambient = 0.1
    color += closest_obj.color * ambient
    
    # Diffuse and specular lighting from all lights
    for light in lights:
        light_dir = normalize(light.position - point)
        light_distance = np.linalg.norm(light.position - point)
        
        # Shadow check
        shadow_ray_origin = offset_point
        shadow_ray_dir = light_dir
        in_shadow = False
        
        for obj in objects:
            t = obj.intersect(shadow_ray_origin, shadow_ray_dir)
            if t < light_distance:
                in_shadow = True
                break
        
        if not in_shadow:
            # Diffuse
            diffuse = max(0, np.dot(normal, light_dir))
            color += closest_obj.color * light.color * diffuse * light.intensity
            
            # Specular
            reflection_dir = reflected(-light_dir, normal)
            specular = max(0, np.dot(reflection_dir, -ray_dir))**50
            color += np.array([1, 1, 1]) * specular * light.intensity
    
    # Reflection
    if closest_obj.reflection > 0:
        reflection_dir = reflected(ray_dir, normal)
        reflection_color = trace_ray(offset_point, reflection_dir, objects, lights, depth + 1)
        color = color * (1 - closest_obj.reflection) + reflection_color * closest_obj.reflection
    
    # Refraction
    if closest_obj.transparency > 0:
        refraction_dir = refracted(ray_dir, normal, closest_obj.refraction)
        if refraction_dir is not None:
            refraction_origin = point - normal * 0.001 if np.dot(ray_dir, normal) < 0 else point + normal * 0.001
            refraction_color = trace_ray(refraction_origin, refraction_dir, objects, lights, depth + 1)
            color = color * (1 - closest_obj.transparency) + refraction_color * closest_obj.transparency
    
    return np.clip(color, 0, 1)

def render_scene():
    # Create scene objects
    objects = [
        # Ground plane (represented as a large sphere)
        Sphere([0, -1000, 0], 1000, [0.3, 0.3, 0.3], reflection=0.2),
        
        # Central sphere
        Sphere([0, 1, 0], 1, [0.8, 0.2, 0.2], reflection=0.7),
        
        # Glass sphere
        Sphere([-2, 0.7, -1], 0.7, [0.9, 0.9, 0.9], reflection=0.1, transparency=0.9),
        
        # Metal sphere
        Sphere([2, 0.5, -1], 0.5, [0.3, 0.3, 0.8], reflection=0.8),
        
        # Small colorful spheres
        Sphere([-1, 0.3, -3], 0.3, [0.2, 0.8, 0.2]),
        Sphere([1.5, 0.2, -2], 0.2, [0.8, 0.8, 0.2]),
        Sphere([0, 0.4, -4], 0.4, [0.8, 0.2, 0.8])
    ]
    
    # Create lights
    lights = [
        Light([-3, 5, 5], [1, 0, 0], 1.5),    # Red light
        Light([3, 5, 5], [0, 1, 0], 1.5),      # Green light
        Light([0, 5, -5], [0, 0, 1], 1.5),     # Blue light
        Light([0, 10, 0], [1, 1, 1], 2.0),     # White light
        Light([-5, 3, 0], [1, 1, 0], 1.0),     # Yellow light
        Light([5, 3, 0], [0, 1, 1], 1.0)       # Cyan light
    ]
    
    # Create image buffer
    image = np.zeros((HEIGHT, WIDTH, 3))
    
    # Calculate pixel size
    aspect_ratio = WIDTH / HEIGHT
    pixel_width = 2 * math.tan(FOV / 2) / WIDTH
    pixel_height = pixel_width / aspect_ratio
    
    # Render each pixel
    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = np.zeros(3)
            
            # Anti-aliasing with multiple samples
            for _ in range(SAMPLES):
                # Calculate ray direction with jitter for anti-aliasing
                jitter_x = random.uniform(-0.5, 0.5)
                jitter_y = random.uniform(-0.5, 0.5)
                
                # Calculate ray direction in camera space
                ray_x = (x + jitter_x - WIDTH / 2) * pixel_width
                ray_y = -(y + jitter_y - HEIGHT / 2) * pixel_height
                ray_z = -1  # Camera looks along negative Z
                
                ray_dir = normalize(np.array([ray_x, ray_y, ray_z]))
                
                # Trace the ray
                color += trace_ray(CAMERA_POS, ray_dir, objects, lights)
            
            # Average the samples
            color /= SAMPLES
            image[y, x] = color
    
    # Convert to 8-bit color and save as PNG
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image, 'RGB')
    img.save('raytraced_scene.png')
    print("Rendering complete! Saved as 'raytraced_scene.png'")

if __name__ == "__main__":
    render_scene()