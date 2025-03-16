import numpy as np
from PIL import Image
import math

# Vector class for 3D operations
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def normalize(self):
        mag = math.sqrt(self.dot(self))
        return Vector3(self.x/mag, self.y/mag, self.z/mag)

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, reflectivity):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflectivity = reflectivity

class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

# Ray tracing functions
def ray_sphere_intersect(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return float('inf')
    t = (-b - math.sqrt(discriminant)) / (2.0*a)
    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2.0*a)
    return t if t > 0 else float('inf')

def checkered_floor(pos):
    x = int(math.floor(pos.x))
    z = int(math.floor(pos.z))
    return Vector3(0.8, 0.8, 0.8) if (x + z) % 2 == 0 else Vector3(0.2, 0.2, 0.2)

def trace_ray(ray_origin, ray_dir, objects, lights, depth=0):
    if depth > 3:  # Max reflection depth
        return Vector3(0, 0, 0)
    
    # Find nearest intersection
    nearest_t = float('inf')
    nearest_obj = None
    
    for obj in objects:
        t = ray_sphere_intersect(ray_origin, ray_dir, obj)
        if t < nearest_t:
            nearest_t = t
            nearest_obj = obj
    
    # Check floor intersection
    if ray_dir.y < -0.01:
        t = -ray_origin.y / ray_dir.y
        if t > 0 and t < nearest_t:
            hit_point = ray_origin + ray_dir * t
            if abs(hit_point.x) < 20 and abs(hit_point.z) < 20:
                nearest_t = t
                nearest_obj = "floor"
    
    if nearest_t == float('inf'):
        return Vector3(0.1, 0.1, 0.2)  # Background color
    
    hit_point = ray_origin + ray_dir * nearest_t
    normal = None
    base_color = None
    
    if nearest_obj == "floor":
        normal = Vector3(0, 1, 0)
        base_color = checkered_floor(hit_point)
    else:
        normal = (hit_point - nearest_obj.center).normalize()
        base_color = nearest_obj.color
    
    # Lighting calculation
    color = Vector3(0, 0, 0)
    for light in lights:
        light_dir = (light.position - hit_point).normalize()
        light_dist = (light.position - hit_point).dot(light.position - hit_point)
        
        # Shadow check
        shadow = False
        for obj in objects:
            if ray_sphere_intersect(hit_point + normal * 0.001, light_dir, obj) < float('inf'):
                shadow = True
                break
        
        if not shadow:
            diffuse = max(0, normal.dot(light_dir))
            color = color + light.color * (diffuse * light.intensity / light_dist)
    
    # Ambient
    color = color + base_color * 0.1
    
    # Reflection
    if nearest_obj != "floor" and nearest_obj.reflectivity > 0:
        reflect_dir = ray_dir - normal * 2 * ray_dir.dot(normal)
        reflect_color = trace_ray(hit_point + normal * 0.001, reflect_dir, objects, lights, depth + 1)
        color = color + reflect_color * nearest_obj.reflectivity
    
    return Vector3(min(color.x, 1), min(color.y, 1), min(color.z, 1))

# Main rendering function
def render():
    width, height = 800, 600
    aspect = width / height
    image = np.zeros((height, width, 3))
    
    # Camera setup
    camera = Vector3(0, 2, -5)
    
    # Scene setup
    objects = [
        Sphere(Vector3(-1.5, 1, 0), 1, Vector3(0.8, 0.2, 0.2), 0.3),
        Sphere(Vector3(1.5, 0.8, 1), 0.8, Vector3(0.2, 0.8, 0.2), 0.3),
        Sphere(Vector3(0, 0.5, 2), 0.5, Vector3(0.2, 0.2, 0.8), 0.3)
    ]
    
    lights = [
        Light(Vector3(-3, 3, -2), Vector3(1, 0.2, 0.2), 50),  # Red light
        Light(Vector3(3, 2, -1), Vector3(0.2, 1, 0.2), 40),   # Green light
        Light(Vector3(0, 4, 1), Vector3(0.2, 0.2, 1), 30),    # Blue light
    ]
    
    # Render loop
    for y in range(height):
        for x in range(width):
            # Convert to -1..1 range with aspect ratio
            u = (2 * x / width - 1) * aspect
            v = 1 - 2 * y / height
            
            # Ray direction
            ray_dir = Vector3(u, v, 1).normalize()
            color = trace_ray(camera, ray_dir, objects, lights)
            
            image[y, x] = [color.x * 255, color.y * 255, color.z * 255]
        
        if y % 50 == 0:
            print(f"Progress: {y/height*100:.1f}%")
    
    # Save image
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img.save('raytraced_scene4.png')
    print("Image saved as 'raytraced_scene4.png'")

if __name__ == "__main__":
    render()