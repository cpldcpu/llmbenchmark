import numpy as np
from PIL import Image
import math

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return Vec3(self.x/l, self.y/l, self.z/l)
        return Vec3(0, 0, 0)
    
    def reflect(self, normal):
        return self - normal * (2 * self.dot(normal))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def point_at(self, t):
        return self.origin + self.direction * t

class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.7, specular=0.2, shininess=50, reflectivity=0.0):
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
        
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        
        t = t1 if t1 > 0.001 else t2
        if t > 0.001:
            point = ray.point_at(t)
            normal = (point - self.center).normalize()
            return {'t': t, 'point': point, 'normal': normal, 'material': self.material}
        return None

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
        if t > 0.001:
            point = ray.point_at(t)
            return {'t': t, 'point': point, 'normal': self.normal, 'material': self.material}
        return None

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

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
        closest_hit = None
        min_t = float('inf')
        
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit['t'] < min_t:
                min_t = hit['t']
                closest_hit = hit
        
        return closest_hit
    
    def is_shadowed(self, point, light_pos):
        light_dir = (light_pos - point).normalize()
        shadow_ray = Ray(point, light_dir)
        hit = self.intersect(shadow_ray)
        
        if hit:
            light_dist = (light_pos - point).length()
            return hit['t'] < light_dist
        return False

def clamp(value, min_val=0, max_val=1):
    return max(min_val, min(max_val, value))

def color_multiply(color1, color2):
    return Vec3(color1.x * color2.x, color1.y * color2.y, color1.z * color2.z)

def raytrace(scene, ray, depth=0, max_depth=3):
    if depth >= max_depth:
        return scene.background_color
    
    hit = scene.intersect(ray)
    if not hit:
        return scene.background_color
    
    material = hit['material']
    point = hit['point']
    normal = hit['normal']
    
    # Ambient lighting
    color = color_multiply(material.color, Vec3(material.ambient, material.ambient, material.ambient))
    
    # Process each light source
    for light in scene.lights:
        if scene.is_shadowed(point, light.position):
            continue
        
        light_dir = (light.position - point).normalize()
        light_dist = (light.position - point).length()
        attenuation = 1.0 / (1.0 + 0.1 * light_dist + 0.01 * light_dist**2)
        
        # Diffuse lighting
        diff_intensity = max(0, normal.dot(light_dir))
        diffuse_color = color_multiply(material.color, light.color)
        diffuse_contribution = diffuse_color * (material.diffuse * diff_intensity * light.intensity * attenuation)
        
        # Specular lighting
        view_dir = (ray.origin - point).normalize()
        reflect_dir = (-light_dir).reflect(normal)
        spec_intensity = max(0, view_dir.dot(reflect_dir)) ** material.shininess
        specular_contribution = light.color * (material.specular * spec_intensity * light.intensity * attenuation)
        
        color = color + diffuse_contribution + specular_contribution
    
    # Reflection
    if material.reflectivity > 0:
        reflect_dir = ray.direction.reflect(normal)
        reflect_ray = Ray(point, reflect_dir)
        reflect_color = raytrace(scene, reflect_ray, depth + 1, max_depth)
        color = color * (1 - material.reflectivity) + reflect_color * material.reflectivity
    
    return Vec3(clamp(color.x), clamp(color.y), clamp(color.z))

def render_scene():
    width, height = 800, 600
    fov = 60
    aspect_ratio = width / height
    
    # Create scene
    scene = Scene()
    
    # Add colorful spheres
    spheres = [
        Sphere(Vec3(0, 0, -5), 1.0, Material(Vec3(0.8, 0.2, 0.2), reflectivity=0.3)),  # Red
        Sphere(Vec3(-2.5, 0, -4), 0.8, Material(Vec3(0.2, 0.8, 0.2), reflectivity=0.1)),  # Green
        Sphere(Vec3(2.5, 0, -4), 0.8, Material(Vec3(0.2, 0.2, 0.8), reflectivity=0.1)),   # Blue
        Sphere(Vec3(-1, -1.5, -3), 0.6, Material(Vec3(0.8, 0.8, 0.2), reflectivity=0.2)), # Yellow
        Sphere(Vec3(1, -1.5, -3), 0.6, Material(Vec3(0.8, 0.2, 0.8), reflectivity=0.2)),  # Magenta
        Sphere(Vec3(0, 1.5, -6), 0.7, Material(Vec3(0.2, 0.8, 0.8), reflectivity=0.5)),   # Cyan
        Sphere(Vec3(-3, 1, -6), 0.5, Material(Vec3(0.9, 0.6, 0.1), reflectivity=0.4)),    # Orange
        Sphere(Vec3(3, 1, -6), 0.5, Material(Vec3(0.6, 0.1, 0.9), reflectivity=0.4)),     # Purple
    ]
    
    for sphere in spheres:
        scene.add_object(sphere)
    
    # Add ground plane
    ground = Plane(Vec3(0, -2, 0), Vec3(0, 1, 0), 
                   Material(Vec3(0.6, 0.6, 0.6), ambient=0.2, diffuse=0.6, reflectivity=0.1))
    scene.add_object(ground)
    
    # Add multiple colored lights
    lights = [
        Light(Vec3(-4, 4, -2), Vec3(1.0, 0.2, 0.2), 0.8),    # Red light
        Light(Vec3(4, 4, -2), Vec3(0.2, 1.0, 0.2), 0.8),     # Green light
        Light(Vec3(0, 5, -3), Vec3(0.2, 0.2, 1.0), 0.8),     # Blue light
        Light(Vec3(-2, 3, -1), Vec3(1.0, 1.0, 0.2), 0.6),    # Yellow light
        Light(Vec3(2, 3, -1), Vec3(1.0, 0.2, 1.0), 0.6),     # Magenta light
        Light(Vec3(0, 2, -8), Vec3(1.0, 1.0, 1.0), 0.4),     # White light (back)
        Light(Vec3(-6, 2, -4), Vec3(0.2, 1.0, 1.0), 0.5),    # Cyan light
        Light(Vec3(6, 2, -4), Vec3(1.0, 0.6, 0.1), 0.5),     # Orange light
    ]
    
    for light in lights:
        scene.add_light(light)
    
    # Camera setup
    camera_pos = Vec3(0, 0, 0)
    
    # Render image
    image_data = []
    
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to world coordinates
            px = (2 * (x + 0.5) / width - 1) * aspect_ratio * math.tan(math.radians(fov/2))
            py = (1 - 2 * (y + 0.5) / height) * math.tan(math.radians(fov/2))
            
            ray_dir = Vec3(px, py, -1).normalize()
            ray = Ray(camera_pos, ray_dir)
            
            color = raytrace(scene, ray)
            
            # Convert to RGB values
            r = int(clamp(color.x) * 255)
            g = int(clamp(color.y) * 255)
            b = int(clamp(color.z) * 255)
            
            image_data.extend([r, g, b])
    
    # Create and save image
    img = Image.frombytes('RGB', (width, height), bytes(image_data))
    img.save('raytraced_scene.png')
    print("Rendered image saved as 'raytraced_scene.png'")
    
    return img

if __name__ == "__main__":
    render_scene()