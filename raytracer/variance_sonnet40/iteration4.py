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
    
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)
    
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
        return self - normal * 2 * self.dot(normal)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def point_at(self, t):
        return self.origin + self.direction * t

class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.7, specular=0.2, shininess=20, reflectivity=0.0):
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
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        t = t1 if t1 > 0.001 else t2
        if t > 0.001:
            return t
        return None
    
    def normal_at(self, point):
        return (point - self.center).normalize()

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material
    
    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 1e-6:
            return None
        
        t = (self.point - ray.origin).dot(self.normal) / denom
        if t > 0.001:
            return t
        return None
    
    def normal_at(self, point):
        return self.normal

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vec3(0.05, 0.05, 0.1)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def intersect(self, ray):
        closest_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t and t < closest_t:
                closest_t = t
                closest_obj = obj
        
        if closest_obj:
            return closest_t, closest_obj
        return None, None

def shade_point(scene, ray, hit_point, obj, depth=0):
    if depth > 3:  # Limit recursion
        return Vec3(0, 0, 0)
    
    normal = obj.normal_at(hit_point)
    material = obj.material
    
    # Ambient lighting
    color = material.color * material.ambient
    
    # Direct lighting from each light source
    for light in scene.lights:
        light_dir = (light.position - hit_point).normalize()
        
        # Check for shadows
        shadow_ray = Ray(hit_point, light_dir)
        shadow_t, shadow_obj = scene.intersect(shadow_ray)
        light_distance = (light.position - hit_point).length()
        
        if not shadow_obj or shadow_t > light_distance:
            # Diffuse lighting
            diffuse_intensity = max(0, normal.dot(light_dir))
            diffuse_color = Vec3(
                material.color.x * light.color.x,
                material.color.y * light.color.y,
                material.color.z * light.color.z
            )
            color = color + diffuse_color * material.diffuse * diffuse_intensity * light.intensity
            
            # Specular lighting
            if diffuse_intensity > 0:
                reflect_dir = (-light_dir).reflect(normal)
                view_dir = (-ray.direction).normalize()
                specular_intensity = max(0, reflect_dir.dot(view_dir)) ** material.shininess
                specular_color = light.color * material.specular * specular_intensity * light.intensity
                color = color + specular_color
    
    # Reflection
    if material.reflectivity > 0 and depth < 3:
        reflect_dir = ray.direction.reflect(normal)
        reflect_ray = Ray(hit_point, reflect_dir)
        reflect_color = trace_ray(scene, reflect_ray, depth + 1)
        color = color + reflect_color * material.reflectivity
    
    return color

def trace_ray(scene, ray, depth=0):
    t, obj = scene.intersect(ray)
    if obj:
        hit_point = ray.point_at(t)
        return shade_point(scene, ray, hit_point, obj, depth)
    return scene.background_color

def render(scene, width, height, camera_pos, camera_target, fov=60):
    # Set up camera
    forward = (camera_target - camera_pos).normalize()
    right = Vec3(0, 1, 0).cross_product(forward).normalize()  # We'll implement cross product
    up = forward.cross_product(right).normalize()
    
    # Calculate screen dimensions
    aspect_ratio = width / height
    screen_height = 2 * math.tan(math.radians(fov) / 2)
    screen_width = screen_height * aspect_ratio
    
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            # Convert pixel to world coordinates
            u = (x / width - 0.5) * screen_width
            v = -(y / height - 0.5) * screen_height
            
            ray_dir = (forward + right * u + up * v).normalize()
            ray = Ray(camera_pos, ray_dir)
            
            color = trace_ray(scene, ray)
            
            # Tone mapping and gamma correction
            r = min(1.0, max(0.0, color.x)) ** (1/2.2)
            g = min(1.0, max(0.0, color.y)) ** (1/2.2)
            b = min(1.0, max(0.0, color.z)) ** (1/2.2)
            
            image[y, x] = [r, g, b]
    
    return (image * 255).astype(np.uint8)

# Add cross product method to Vec3
def cross_product(self, other):
    return Vec3(
        self.y * other.z - self.z * other.y,
        self.z * other.x - self.x * other.z,
        self.x * other.y - self.y * other.x
    )
Vec3.cross_product = cross_product

def create_scene():
    scene = Scene()
    
    # Create materials with different colors
    red_shiny = Material(Vec3(0.8, 0.2, 0.2), diffuse=0.7, specular=0.3, shininess=50, reflectivity=0.3)
    blue_matte = Material(Vec3(0.2, 0.4, 0.8), diffuse=0.9, specular=0.1, shininess=10)
    green_glossy = Material(Vec3(0.2, 0.8, 0.3), diffuse=0.6, specular=0.4, shininess=30, reflectivity=0.2)
    yellow_bright = Material(Vec3(0.9, 0.8, 0.1), diffuse=0.8, specular=0.2, shininess=20)
    purple_metallic = Material(Vec3(0.6, 0.2, 0.8), diffuse=0.4, specular=0.6, shininess=80, reflectivity=0.4)
    white_reflective = Material(Vec3(0.9, 0.9, 0.9), diffuse=0.3, specular=0.4, shininess=100, reflectivity=0.6)
    
    # Floor with checkerboard pattern (simplified as gray)
    floor_material = Material(Vec3(0.5, 0.5, 0.5), diffuse=0.8, specular=0.1, reflectivity=0.1)
    floor = Plane(Vec3(0, -2, 0), Vec3(0, 1, 0), floor_material)
    scene.add_object(floor)
    
    # Add spheres at different positions
    scene.add_object(Sphere(Vec3(0, 0, -5), 1.2, red_shiny))
    scene.add_object(Sphere(Vec3(-3, -0.5, -6), 0.8, blue_matte))
    scene.add_object(Sphere(Vec3(3, 0.2, -4), 1.0, green_glossy))
    scene.add_object(Sphere(Vec3(-1.5, 1, -3), 0.6, yellow_bright))
    scene.add_object(Sphere(Vec3(1.8, -0.8, -7), 0.9, purple_metallic))
    scene.add_object(Sphere(Vec3(0, 2, -8), 0.7, white_reflective))
    scene.add_object(Sphere(Vec3(-2.5, 0.5, -4), 0.5, red_shiny))
    
    # Add multiple colored lights
    scene.add_light(Light(Vec3(-4, 4, -2), Vec3(1.0, 0.7, 0.7), 0.8))  # Warm red-orange
    scene.add_light(Light(Vec3(4, 3, -1), Vec3(0.7, 0.7, 1.0), 0.7))   # Cool blue
    scene.add_light(Light(Vec3(0, 5, -3), Vec3(0.9, 1.0, 0.8), 0.6))   # Bright white-green
    scene.add_light(Light(Vec3(2, 1, -10), Vec3(0.8, 0.4, 0.9), 0.5))  # Purple
    scene.add_light(Light(Vec3(-1, 2, 0), Vec3(1.0, 0.9, 0.4), 0.4))   # Yellow
    
    return scene

def main():
    print("Creating scene...")
    scene = create_scene()
    
    print("Rendering image...")
    camera_pos = Vec3(0, 1, 2)
    camera_target = Vec3(0, 0, -5)
    
    image_array = render(scene, 800, 600, camera_pos, camera_target, fov=60)
    
    print("Saving image...")
    image = Image.fromarray(image_array)
    image.save('raytraced_scene.png')
    print("Image saved as 'raytraced_scene4.png'")

if __name__ == "__main__":
    main()