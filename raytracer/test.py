import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import random
import math

# Vector operations
class Vec3:
    def __init__(self, x=0, y=0, z=0):
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
        else:
            return Vec3(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return Vec3(self.x / other, self.y / other, self.z / other)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.dot(self))
    
    def squared_length(self):
        return self.dot(self)
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return self / l
        return Vec3()
    
    def clamp(self, min_val=0, max_val=1):
        return Vec3(
            min(max(self.x, min_val), max_val),
            min(max(self.y, min_val), max_val),
            min(max(self.z, min_val), max_val)
        )

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def point_at(self, t):
        return self.origin + self.direction * t

# Material classes
class Material:
    def scatter(self, ray_in, hit_record):
        return None
    
    def emit(self):
        return Vec3(0, 0, 0)

class Diffuse(Material):
    def __init__(self, albedo):
        self.albedo = albedo
    
    def scatter(self, ray_in, hit_record):
        scatter_direction = hit_record.normal + random_in_unit_sphere()
        
        # Catch degenerate scatter direction
        if scatter_direction.squared_length() < 0.001:
            scatter_direction = hit_record.normal
        
        scattered = Ray(hit_record.p, scatter_direction)
        attenuation = self.albedo
        return attenuation, scattered

class Metal(Material):
    def __init__(self, albedo, fuzz=0):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1)
    
    def scatter(self, ray_in, hit_record):
        reflected = reflect(ray_in.direction, hit_record.normal)
        scattered = Ray(hit_record.p, reflected + self.fuzz * random_in_unit_sphere())
        attenuation = self.albedo
        if scattered.direction.dot(hit_record.normal) > 0:
            return attenuation, scattered
        return None

class Emissive(Material):
    def __init__(self, emit_color, strength=1.0):
        self.emit_color = emit_color
        self.strength = strength
    
    def scatter(self, ray_in, hit_record):
        return None
    
    def emit(self):
        return self.emit_color * self.strength

# Helper functions
def random_in_unit_sphere():
    while True:
        p = 2.0 * Vec3(random.random(), random.random(), random.random()) - Vec3(1, 1, 1)
        if p.squared_length() < 1:
            return p

def random_in_unit_disk():
    while True:
        p = 2.0 * Vec3(random.random(), random.random(), 0) - Vec3(1, 1, 0)
        if p.squared_length() < 1:
            return p

def reflect(v, n):
    return v - 2 * v.dot(n) * n

# Hit record class
class HitRecord:
    def __init__(self):
        self.p = Vec3()
        self.normal = Vec3()
        self.material = None
        self.t = 0.0

# Hittable objects
class Hittable:
    def hit(self, ray, t_min, t_max, hit_record):
        pass

class Sphere(Hittable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def hit(self, ray, t_min, t_max, hit_record):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - a * c
        
        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                hit_record.t = temp
                hit_record.p = ray.point_at(hit_record.t)
                hit_record.normal = (hit_record.p - self.center) / self.radius
                hit_record.material = self.material
                return True
            
            temp = (-b + math.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                hit_record.t = temp
                hit_record.p = ray.point_at(hit_record.t)
                hit_record.normal = (hit_record.p - self.center) / self.radius
                hit_record.material = self.material
                return True
        
        return False

class HittableList(Hittable):
    def __init__(self):
        self.objects = []
    
    def add(self, object):
        self.objects.append(object)
    
    def hit(self, ray, t_min, t_max, hit_record):
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = t_max
        
        for obj in self.objects:
            if obj.hit(ray, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                hit_record.t = temp_rec.t
                hit_record.p = temp_rec.p
                hit_record.normal = temp_rec.normal
                hit_record.material = temp_rec.material
        
        return hit_anything

# Camera class
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
        theta = math.radians(vfov)
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height
        
        self.w = (lookfrom - lookat).normalize()
        self.u = vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)
        
        self.origin = lookfrom
        self.lower_left_corner = self.origin - half_width * focus_dist * self.u - half_height * focus_dist * self.v - focus_dist * self.w
        self.horizontal = 2 * half_width * focus_dist * self.u
        self.vertical = 2 * half_height * focus_dist * self.v
        self.lens_radius = aperture / 2
    
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u * rd.x + self.v * rd.y
        return Ray(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset
        )

# Path tracing
def color(ray, world, depth=0):
    hit_record = HitRecord()
    
    if world.hit(ray, 0.001, float('inf'), hit_record):
        emitted = hit_record.material.emit()
        
        if depth < 50:
            scatter_result = hit_record.material.scatter(ray, hit_record)
            if scatter_result:
                attenuation, scattered = scatter_result
                return emitted + attenuation * color(scattered, world, depth + 1)
        
        return emitted
    
    # Sky color - gradient from light blue to dark blue
    unit_direction = ray.direction
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * Vec3(0.1, 0.1, 0.1) + t * Vec3(0.3, 0.3, 0.5)

# Create an interesting scene with multiple colored light sources
def create_colorful_scene():
    world = HittableList()
    
    # Floor
    world.add(Sphere(Vec3(0, -1000, 0), 1000, Diffuse(Vec3(0.7, 0.7, 0.7))))
    
    # Colored light sources - creating a rainbow effect
    world.add(Sphere(Vec3(-5, 3, 0), 0.7, Emissive(Vec3(1, 0, 0), 5.0)))       # Red
    world.add(Sphere(Vec3(-3, 3, 0), 0.7, Emissive(Vec3(1, 0.5, 0), 5.0)))     # Orange
    world.add(Sphere(Vec3(-1, 3, 0), 0.7, Emissive(Vec3(1, 1, 0), 5.0)))       # Yellow
    world.add(Sphere(Vec3(1, 3, 0), 0.7, Emissive(Vec3(0, 1, 0), 5.0)))        # Green
    world.add(Sphere(Vec3(3, 3, 0), 0.7, Emissive(Vec3(0, 0.5, 1), 5.0)))      # Blue
    world.add(Sphere(Vec3(5, 3, 0), 0.7, Emissive(Vec3(0.5, 0, 1), 5.0)))      # Purple
    
    # Additional lights for more interesting effects
    world.add(Sphere(Vec3(0, 5, -3), 1.0, Emissive(Vec3(1, 1, 1), 4.0)))       # White overhead
    world.add(Sphere(Vec3(-4, 1, -4), 0.5, Emissive(Vec3(0, 1, 1), 3.0)))      # Cyan
    world.add(Sphere(Vec3(4, 1, -4), 0.5, Emissive(Vec3(1, 0, 1), 3.0)))       # Magenta
    
    # Main objects
    world.add(Sphere(Vec3(0, 1, 0), 1.0, Metal(Vec3(0.8, 0.8, 0.8), 0.0)))     # Mirror sphere center
    world.add(Sphere(Vec3(-2.5, 1, 2), 1.0, Metal(Vec3(0.8, 0.6, 0.2), 0.2)))  # Gold sphere
    world.add(Sphere(Vec3(2.5, 1, 2), 1.0, Metal(Vec3(0.6, 0.8, 0.8), 0.1)))   # Silver-blue sphere
    
    # Small spheres arranged in a grid pattern
    for a in range(-5, 6, 2):
        for b in range(-5, 6, 2):
            # Skip positions near the main spheres
            if (abs(a) < 2 and abs(b) < 2) or (abs(a-2.5) < 2 and abs(b-2) < 2) or (abs(a+2.5) < 2 and abs(b-2) < 2):
                continue
                
            center = Vec3(a, 0.3, b)
            material_choice = random.random()
            
            if material_choice < 0.6:  # Diffuse
                albedo = Vec3(random.random() * 0.8 + 0.2, 
                             random.random() * 0.8 + 0.2, 
                             random.random() * 0.8 + 0.2)
                world.add(Sphere(center, 0.3, Diffuse(albedo)))
            elif material_choice < 0.9:  # Metal
                albedo = Vec3(random.random() * 0.5 + 0.5, 
                             random.random() * 0.5 + 0.5, 
                             random.random() * 0.5 + 0.5)
                fuzz = random.random() * 0.4
                world.add(Sphere(center, 0.3, Metal(albedo, fuzz)))
            else:  # Small lights
                hue = random.random()
                if hue < 0.167:
                    emit = Vec3(1, 0, 0)  # Red
                elif hue < 0.333:
                    emit = Vec3(1, 0.5, 0)  # Orange
                elif hue < 0.5:
                    emit = Vec3(1, 1, 0)  # Yellow
                elif hue < 0.667:
                    emit = Vec3(0, 1, 0)  # Green
                elif hue < 0.833:
                    emit = Vec3(0, 0, 1)  # Blue
                else:
                    emit = Vec3(0.5, 0, 1)  # Purple
                
                world.add(Sphere(center, 0.3, Emissive(emit, 2.0)))
    
    return world

# Main rendering function
def render_scene(width, height, samples_per_pixel=100):
    # Create the image
    img = np.zeros((height, width, 3))
    
    # Create scene
    world = create_colorful_scene()
    
    # Camera setup
    lookfrom = Vec3(0, 3, 10)
    lookat = Vec3(0, 1, 0)
    vup = Vec3(0, 1, 0)
    dist_to_focus = 10.0
    aperture = 0.1
    camera = Camera(lookfrom, lookat, vup, 30, width / height, aperture, dist_to_focus)
    
    start_time = time.time()
    
    # Render each pixel
    for j in tqdm(range(height)):
        for i in range(width):
            col = Vec3(0, 0, 0)
            for s in range(samples_per_pixel):
                u = (i + random.random()) / width
                v = (j + random.random()) / height
                ray = camera.get_ray(u, v)
                col = col + color(ray, world, 0)
            
            # Average the samples and apply gamma correction
            col = col / samples_per_pixel
            col = Vec3(math.sqrt(col.x), math.sqrt(col.y), math.sqrt(col.z))
            
            # Clamp and convert to 0-255 range
            col = col.clamp(0, 1)
            ir = int(255.99 * col.x)
            ig = int(255.99 * col.y)
            ib = int(255.99 * col.z)
            
            img[height - j - 1, i] = [ir, ig, ib]  # Flip vertically
    
    end_time = time.time()
    print(f"Rendering completed in {end_time - start_time:.2f} seconds")
    
    # Save the image
    pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
    pil_img.save("colorful_path_traced_scene.png")
    print("Image saved as colorful_path_traced_scene.png")

if __name__ == "__main__":
    width = 800
    height = 600
    samples = 100  # Increase for better quality, decrease for faster rendering
    
    print(f"Rendering a {width}x{height} image with {samples} samples per pixel")
    render_scene(width, height, samples)