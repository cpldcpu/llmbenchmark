import numpy as np
from PIL import Image
import math
import random

# Vector Operations
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def random_in_unit_sphere():
    while True:
        p = 2.0 * np.array([random.random(), random.random(), random.random()]) - 1.0
        if np.dot(p, p) < 1.0:
            return p

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)
    
    def point_at_parameter(self, t):
        return self.origin + t * self.direction

# Material classes
class Material:
    def scatter(self, ray_in, hit_record, attenuation, scattered):
        pass

class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo
    
    def scatter(self, ray_in, hit_record, attenuation, scattered):
        target = hit_record.p + hit_record.normal + random_in_unit_sphere()
        scattered[0] = Ray(hit_record.p, target - hit_record.p)
        attenuation[0] = self.albedo
        return True

class Metal(Material):
    def __init__(self, albedo, fuzz=0.0):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0)
    
    def scatter(self, ray_in, hit_record, attenuation, scattered):
        reflected = reflect(ray_in.direction, hit_record.normal)
        scattered[0] = Ray(hit_record.p, reflected + self.fuzz * random_in_unit_sphere())
        attenuation[0] = self.albedo
        return np.dot(scattered[0].direction, hit_record.normal) > 0

# Hit record class
class HitRecord:
    def __init__(self):
        self.t = 0.0
        self.p = np.zeros(3)
        self.normal = np.zeros(3)
        self.material = None

# Sphere class
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def hit(self, ray, t_min, t_max, hit_record):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - a * c
        
        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / a
            if temp < t_max and temp > t_min:
                hit_record.t = temp
                hit_record.p = ray.point_at_parameter(temp)
                hit_record.normal = (hit_record.p - self.center) / self.radius
                hit_record.material = self.material
                return True
            
            temp = (-b + math.sqrt(discriminant)) / a
            if temp < t_max and temp > t_min:
                hit_record.t = temp
                hit_record.p = ray.point_at_parameter(temp)
                hit_record.normal = (hit_record.p - self.center) / self.radius
                hit_record.material = self.material
                return True
        
        return False

# Scene class (list of hittable objects)
class Scene:
    def __init__(self):
        self.objects = []
    
    def add(self, obj):
        self.objects.append(obj)
    
    def hit(self, ray, t_min, t_max, hit_record):
        hit_anything = False
        closest_so_far = t_max
        
        for obj in self.objects:
            if obj.hit(ray, t_min, closest_so_far, hit_record):
                hit_anything = True
                closest_so_far = hit_record.t
        
        return hit_anything

# Light source class
class PointLight:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# Camera class
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect):
        theta = vfov * math.pi / 180
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height
        
        self.origin = lookfrom
        w = normalize(lookfrom - lookat)
        u = normalize(np.cross(vup, w))
        v = np.cross(w, u)
        
        self.lower_left_corner = self.origin - half_width * u - half_height * v - w
        self.horizontal = 2 * half_width * u
        self.vertical = 2 * half_height * v
    
    def get_ray(self, s, t):
        return Ray(self.origin, self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin)

# Color functions
def color(ray, world, lights, depth=0):
    hit_record = HitRecord()
    
    if world.hit(ray, 0.001, float('inf'), hit_record):
        scattered = [None]
        attenuation = [None]
        
        if depth < 50 and hit_record.material.scatter(ray, hit_record, attenuation, scattered):
            # Calculate direct lighting from all light sources
            direct_light = np.zeros(3)
            for light in lights:
                light_dir = normalize(light.position - hit_record.p)
                shadow_ray = Ray(hit_record.p, light_dir)
                shadow_rec = HitRecord()
                
                # Check if point is in shadow
                if not world.hit(shadow_ray, 0.001, float('inf'), shadow_rec):
                    light_distance = np.linalg.norm(light.position - hit_record.p)
                    intensity = light.intensity / (light_distance * light_distance)
                    direct_light += light.color * max(0, np.dot(hit_record.normal, light_dir)) * intensity
            
            # Combine direct lighting with reflected color
            return attenuation[0] * (0.6 * direct_light + 0.4 * color(scattered[0], world, lights, depth + 1))
        else:
            return np.zeros(3)
    else:
        # Sky gradient
        unit_direction = ray.direction
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])

# Set up the scene
def create_scene():
    world = Scene()
    lights = []
    
    # Materials
    ground_material = Lambertian(np.array([0.5, 0.5, 0.5]))
    red_material = Lambertian(np.array([0.9, 0.2, 0.2]))
    green_material = Lambertian(np.array([0.2, 0.8, 0.2]))
    blue_material = Lambertian(np.array([0.2, 0.2, 0.8]))
    yellow_material = Lambertian(np.array([0.8, 0.8, 0.0]))
    purple_material = Lambertian(np.array([0.8, 0.2, 0.8]))
    cyan_material = Lambertian(np.array([0.2, 0.8, 0.8]))
    metal_material = Metal(np.array([0.7, 0.6, 0.5]), 0.0)
    shiny_material = Metal(np.array([0.8, 0.8, 0.8]), 0.1)
    
    # Add spheres
    world.add(Sphere(np.array([0, -100.5, -1]), 100, ground_material))  # Ground
    world.add(Sphere(np.array([0, 0, -1]), 0.5, shiny_material))  # Center
    world.add(Sphere(np.array([-1, 0, -1]), 0.5, red_material))  # Left
    world.add(Sphere(np.array([1, 0, -1]), 0.5, green_material))  # Right
    world.add(Sphere(np.array([0, 0, -2]), 0.5, blue_material))  # Back
    world.add(Sphere(np.array([-0.5, -0.4, -0.5]), 0.1, purple_material))  # Small front left
    world.add(Sphere(np.array([0.5, -0.4, -0.5]), 0.1, yellow_material))  # Small front right
    world.add(Sphere(np.array([0, -0.4, -0.25]), 0.1, cyan_material))  # Small front center
    world.add(Sphere(np.array([0, 0.8, -1]), 0.3, metal_material))  # Top
    
    # Add lights
    lights.append(PointLight(np.array([3, 3, 2]), np.array([1.0, 0.2, 0.2]), 5.0))  # Red light
    lights.append(PointLight(np.array([-3, 2, 2]), np.array([0.2, 0.2, 1.0]), 5.0))  # Blue light
    lights.append(PointLight(np.array([0, 2, 3]), np.array([0.2, 1.0, 0.2]), 5.0))  # Green light
    lights.append(PointLight(np.array([2, 1, 0]), np.array([1.0, 1.0, 0.0]), 4.0))  # Yellow light
    lights.append(PointLight(np.array([-2, 1, 0]), np.array([1.0, 0.0, 1.0]), 4.0))  # Purple light
    
    return world, lights

# Main rendering function
def render(width=800, height=600, samples=100):
    world, lights = create_scene()
    
    # Set up camera
    lookfrom = np.array([0, 1, 2])
    lookat = np.array([0, 0, -1])
    vup = np.array([0, 1, 0])
    camera = Camera(lookfrom, lookat, vup, 90, width / height)
    
    # Create image array
    img = np.zeros((height, width, 3))
    
    # Render the scene
    for j in range(height-1, -1, -1):
        print("Scanlines remaining: " + str(j))
        for i in range(width):
            col = np.zeros(3)
            
            # Anti-aliasing with multiple samples
            for s in range(samples):
                u = (i + random.random()) / width
                v = (j + random.random()) / height
                ray = camera.get_ray(u, v)
                col += color(ray, world, lights)
            
            col /= samples
            
            # Gamma correction
            col = np.sqrt(col)
            
            img[height-1-j, i] = np.clip(col * 255, 0, 255)
    
    # Save the image
    final_img = Image.fromarray(img.astype(np.uint8))
    final_img.save("colorful_raytraced_scene.png")
    print("Image saved as colorful_raytraced_scene.png")
    return final_img

# Render with fewer samples for faster results
# Increase samples for better quality (but slower rendering)
if __name__ == "__main__":
    render(samples=10)