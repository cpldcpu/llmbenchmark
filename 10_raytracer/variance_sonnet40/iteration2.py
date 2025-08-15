import numpy as np
from PIL import Image
import math

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)
    
    def reflect(self, normal):
        return self - normal * (2 * self.dot(normal))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def at(self, t):
        return self.origin + self.direction * t

class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.7, specular=0.8, shininess=50, reflectivity=0.2):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflectivity = reflectivity

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        t = t1 if t1 > 0.001 else (t2 if t2 > 0.001 else None)
        return t
    
    def normal_at(self, point):
        return (point - self.center).normalize()

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material
    
    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 0.0001:
            return None
        
        t = (self.point - ray.origin).dot(self.normal) / denom
        return t if t > 0.001 else None
    
    def normal_at(self, point):
        return self.normal

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Camera:
    def __init__(self, position, target, up, fov, aspect_ratio):
        self.position = position
        self.forward = (target - position).normalize()
        self.right = self.forward.cross_product(up).normalize()
        self.up = self.right.cross_product(self.forward)
        
        self.half_width = math.tan(math.radians(fov) / 2)
        self.half_height = self.half_width / aspect_ratio
    
    def get_ray(self, x, y, width, height):
        # Convert pixel coordinates to camera coordinates
        u = (2 * x / width - 1) * self.half_width
        v = (2 * y / height - 1) * self.half_height
        
        # Calculate ray direction
        direction = self.forward + self.right * u + self.up * v
        return Ray(self.position, direction)

# Helper method for cross product
def cross_product(a, b):
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )

Vec3.cross_product = cross_product

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vec3(0.1, 0.1, 0.2)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def intersect(self, ray):
        closest_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_t = t
                closest_obj = obj
        
        return (closest_t, closest_obj) if closest_obj else (None, None)

def calculate_lighting(point, normal, view_dir, material, lights, scene):
    color = Vec3(0, 0, 0)
    
    # Ambient lighting
    ambient = material.color * material.ambient
    color = color + ambient
    
    for light in lights:
        # Shadow ray
        light_dir = (light.position - point).normalize()
        shadow_ray = Ray(point, light_dir)
        shadow_t, shadow_obj = scene.intersect(shadow_ray)
        
        # Check if in shadow
        light_distance = (light.position - point).length()
        if shadow_t is not None and shadow_t < light_distance:
            continue  # In shadow, skip this light
        
        # Diffuse lighting
        diff_intensity = max(0, normal.dot(light_dir))
        diffuse = material.color * material.diffuse * diff_intensity * light.intensity
        
        # Apply light color
        diffuse = Vec3(
            diffuse.x * light.color.x,
            diffuse.y * light.color.y,
            diffuse.z * light.color.z
        )
        
        # Specular lighting
        reflect_dir = (-light_dir).reflect(normal)
        spec_intensity = max(0, view_dir.dot(reflect_dir)) ** material.shininess
        specular = Vec3(1, 1, 1) * material.specular * spec_intensity * light.intensity
        
        # Apply light color to specular
        specular = Vec3(
            specular.x * light.color.x,
            specular.y * light.color.y,
            specular.z * light.color.z
        )
        
        color = color + diffuse + specular
    
    return color

def trace_ray(ray, scene, depth=0, max_depth=3):
    if depth >= max_depth:
        return scene.background_color
    
    t, obj = scene.intersect(ray)
    if obj is None:
        return scene.background_color
    
    hit_point = ray.at(t)
    normal = obj.normal_at(hit_point)
    view_dir = (ray.origin - hit_point).normalize()
    
    # Calculate direct lighting
    color = calculate_lighting(hit_point, normal, view_dir, obj.material, scene.lights, scene)
    
    # Add reflection
    if obj.material.reflectivity > 0:
        reflect_dir = (-ray.direction).reflect(normal)
        reflect_ray = Ray(hit_point, reflect_dir)
        reflect_color = trace_ray(reflect_ray, scene, depth + 1, max_depth)
        color = color + reflect_color * obj.material.reflectivity
    
    return color

def clamp_color(color):
    return Vec3(
        max(0, min(1, color.x)),
        max(0, min(1, color.y)),
        max(0, min(1, color.z))
    )

def render_scene():
    width, height = 800, 600
    aspect_ratio = width / height
    
    # Create scene
    scene = Scene()
    
    # Create materials
    red_material = Material(Vec3(0.8, 0.2, 0.2), reflectivity=0.3)
    blue_material = Material(Vec3(0.2, 0.2, 0.8), reflectivity=0.4)
    green_material = Material(Vec3(0.2, 0.8, 0.2), reflectivity=0.2)
    yellow_material = Material(Vec3(0.8, 0.8, 0.2), reflectivity=0.3)
    purple_material = Material(Vec3(0.6, 0.2, 0.8), reflectivity=0.35)
    white_material = Material(Vec3(0.9, 0.9, 0.9), reflectivity=0.1)
    mirror_material = Material(Vec3(0.9, 0.9, 0.9), reflectivity=0.8)
    
    # Add spheres
    scene.add_object(Sphere(Vec3(0, 0, -5), 1.5, red_material))
    scene.add_object(Sphere(Vec3(-3, -1, -6), 1, blue_material))
    scene.add_object(Sphere(Vec3(3, -0.5, -4), 0.8, green_material))
    scene.add_object(Sphere(Vec3(-1, 2, -7), 1.2, yellow_material))
    scene.add_object(Sphere(Vec3(2, 1.5, -8), 1, purple_material))
    scene.add_object(Sphere(Vec3(0, -1.5, -3), 0.7, mirror_material))
    
    # Add floor plane
    scene.add_object(Plane(Vec3(0, -2.5, 0), Vec3(0, 1, 0), white_material))
    
    # Add colorful lights
    scene.add_light(Light(Vec3(-5, 5, -2), Vec3(1.0, 0.3, 0.3), 0.8))  # Red light
    scene.add_light(Light(Vec3(5, 4, -3), Vec3(0.3, 0.3, 1.0), 0.7))   # Blue light
    scene.add_light(Light(Vec3(0, 6, -4), Vec3(0.3, 1.0, 0.3), 0.6))   # Green light
    scene.add_light(Light(Vec3(3, 2, 0), Vec3(1.0, 1.0, 0.3), 0.5))    # Yellow light
    scene.add_light(Light(Vec3(-2, 3, -1), Vec3(1.0, 0.3, 1.0), 0.6))  # Magenta light
    scene.add_light(Light(Vec3(1, 8, -6), Vec3(1.0, 1.0, 1.0), 0.4))   # White light
    
    # Create camera
    camera = Camera(
        position=Vec3(0, 1, 2),
        target=Vec3(0, 0, -5),
        up=Vec3(0, 1, 0),
        fov=60,
        aspect_ratio=aspect_ratio
    )
    
    # Render image
    image_data = np.zeros((height, width, 3), dtype=np.uint8)
    
    print("Rendering... This may take a moment.")
    for y in range(height):
        if y % 50 == 0:
            print(f"Progress: {y}/{height} lines")
        for x in range(width):
            ray = camera.get_ray(x, y, width, height)
            color = trace_ray(ray, scene)
            color = clamp_color(color)
            
            image_data[y, x] = [
                int(color.x * 255),
                int(color.y * 255),
                int(color.z * 255)
            ]
    
    # Save image
    image = Image.fromarray(image_data)
    image.save('raytraced_scene.png')
    print("Rendered image saved as 'raytraced_scene2.png'")
    
    return image

if __name__ == "__main__":
    render_scene()