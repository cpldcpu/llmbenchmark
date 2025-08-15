import numpy as np
from PIL import Image

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def normalize(self):
        length = np.sqrt(self.dot(self))
        return self * (1.0 / length) if length > 0 else self

class Sphere:
    def __init__(self, center, radius, color, specular=-1, reflection=0.0):
        self.center = center
        self.radius = radius
        self.color = color
        self.specular = specular
        self.reflection = reflection
    
    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = ray_dir.dot(ray_dir)
        b = 2 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius * self.radius
        disc = b * b - 4 * a * c
        
        if disc < 0:
            return None
        
        t = (-b - np.sqrt(disc)) / (2 * a)
        if t < 0:
            t = (-b + np.sqrt(disc)) / (2 * a)
            if t < 0:
                return None
        
        return t

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

def reflect_ray(ray_dir, normal):
    return ray_dir - normal * (2 * ray_dir.dot(normal))

def trace_ray(ray_origin, ray_dir, spheres, lights, depth=0):
    closest_t = float('inf')
    closest_sphere = None
    
    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_dir)
        if t and t < closest_t:
            closest_t = t
            closest_sphere = sphere
    
    if not closest_sphere:
        return Vec3(0, 0, 0)  # Background color (black)
    
    hit_point = ray_origin + ray_dir * closest_t
    normal = (hit_point - closest_sphere.center).normalize()
    
    color = Vec3(0, 0, 0)
    for light in lights:
        light_dir = (light.position - hit_point).normalize()
        
        # Shadow check
        in_shadow = False
        shadow_origin = hit_point + normal * 0.001
        for sphere in spheres:
            if sphere != closest_sphere:
                t = sphere.intersect(shadow_origin, light_dir)
                if t:
                    in_shadow = True
                    break
        
        if not in_shadow:
            # Diffuse lighting
            diff = max(0, normal.dot(light_dir))
            color = color + closest_sphere.color * light.color * (diff * light.intensity)
            
            # Specular lighting
            if closest_sphere.specular > 0:
                reflected = reflect_ray(light_dir * -1, normal)
                spec = max(0, reflected.dot(ray_dir * -1)) ** closest_sphere.specular
                color = color + light.color * (spec * light.intensity)
    
    # Reflection
    if depth < 3 and closest_sphere.reflection > 0:
        reflected_dir = reflect_ray(ray_dir, normal)
        reflected_origin = hit_point + normal * 0.001
        reflected_color = trace_ray(reflected_origin, reflected_dir, spheres, lights, depth + 1)
        color = color * (1 - closest_sphere.reflection) + reflected_color * closest_sphere.reflection
    
    return color

def render_scene(width, height):
    aspect_ratio = width / height
    
    # Define scene objects
    spheres = [
        Sphere(Vec3(0, -1, 3), 1, Vec3(0.7, 0.3, 0.3), specular=100, reflection=0.2),  # Red sphere
        Sphere(Vec3(2, 0, 4), 1, Vec3(0.3, 0.7, 0.3), specular=100, reflection=0.3),   # Green sphere
        Sphere(Vec3(-2, 0, 4), 1, Vec3(0.3, 0.3, 0.7), specular=100, reflection=0.3),  # Blue sphere
        Sphere(Vec3(0, -5001, 0), 5000, Vec3(0.5, 0.5, 0.5), specular=10, reflection=0.05)  # Ground
    ]
    
    # Define colored lights
    lights = [
        Light(Vec3(2, 1, 0), Vec3(1.0, 0.3, 0.3), intensity=0.6),    # Red light
        Light(Vec3(-2, 1, 0), Vec3(0.3, 1.0, 0.3), intensity=0.6),   # Green light
        Light(Vec3(0, 1, -2), Vec3(0.3, 0.3, 1.0), intensity=0.6),   # Blue light
        Light(Vec3(0, 5, 5), Vec3(1.0, 1.0, 1.0), intensity=0.4)     # White light
    ]
    
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            screen_x = (-1 + 2 * (x + 0.5) / width) * aspect_ratio
            screen_y = (1 - 2 * (y + 0.5) / height)
            ray_dir = Vec3(screen_x, screen_y, 1).normalize()
            
            # Trace ray and get color
            color = trace_ray(Vec3(0, 0, 0), ray_dir, spheres, lights)
            
            # Apply simple tone mapping
            image[y, x] = [
                min(1, max(0, color.x)),
                min(1, max(0, color.y)),
                min(1, max(0, color.z))
            ]
    
    return (image * 255).astype(np.uint8)

# Render the scene
width, height = 800, 600
image_array = render_scene(width, height)
image = Image.fromarray(image_array)
image.save('raytraced_scene.png')
print("Image rendered and saved as 'raytraced_scene.png'")