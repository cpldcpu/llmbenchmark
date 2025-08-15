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
        return self - normal * 2 * self.dot(normal)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def point_at(self, t):
        return self.origin + self.direction * t

class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.6, specular=0.3, shininess=32, reflectance=0.0):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflectance = reflectance

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
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        t = t1 if t1 > 0.001 else (t2 if t2 > 0.001 else None)
        if t is None:
            return None
        
        point = ray.point_at(t)
        normal = (point - self.center).normalize()
        return {'t': t, 'point': point, 'normal': normal, 'material': self.material}

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
        if t < 0.001:
            return None
        
        point = ray.point_at(t)
        return {'t': t, 'point': point, 'normal': self.normal, 'material': self.material}

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = position
        self.forward = (look_at - position).normalize()
        self.right = self.forward.cross(up).normalize() if hasattr(Vec3, 'cross') else Vec3(1, 0, 0)
        self.up = up.normalize()
        
        # Manual cross product for right vector
        self.right = Vec3(
            self.forward.y * up.z - self.forward.z * up.y,
            self.forward.z * up.x - self.forward.x * up.z,
            self.forward.x * up.y - self.forward.y * up.x
        ).normalize()
        
        # Recalculate up to ensure orthogonality
        self.up = Vec3(
            self.right.y * self.forward.z - self.right.z * self.forward.y,
            self.right.z * self.forward.x - self.right.x * self.forward.z,
            self.right.x * self.forward.y - self.right.y * self.forward.x
        ).normalize()
        
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        
        # Calculate viewport dimensions
        viewport_height = 2.0 * math.tan(math.radians(fov) / 2)
        viewport_width = aspect_ratio * viewport_height
        
        self.viewport_u = self.right * viewport_width
        self.viewport_v = self.up * (-viewport_height)
        
        self.pixel_delta_u = self.viewport_u * (1.0 / 800)
        self.pixel_delta_v = self.viewport_v * (1.0 / 600)
        
        viewport_upper_left = self.position + self.forward - self.viewport_u * 0.5 - self.viewport_v * 0.5
        self.pixel00_loc = viewport_upper_left + (self.pixel_delta_u + self.pixel_delta_v) * 0.5

def clamp(value, min_val=0, max_val=1):
    return max(min_val, min(max_val, value))

def color_multiply(c1, c2):
    return Vec3(c1.x * c2.x, c1.y * c2.y, c1.z * c2.z)

class Raytracer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.max_depth = 5
        
        # Create scene objects
        self.objects = []
        self.lights = []
        
        # Materials
        red_shiny = Material(Vec3(0.8, 0.2, 0.2), ambient=0.1, diffuse=0.6, specular=0.8, shininess=64, reflectance=0.3)
        blue_matte = Material(Vec3(0.2, 0.4, 0.8), ambient=0.2, diffuse=0.8, specular=0.2, shininess=16)
        green_metal = Material(Vec3(0.2, 0.8, 0.3), ambient=0.1, diffuse=0.4, specular=0.9, shininess=128, reflectance=0.6)
        yellow_plastic = Material(Vec3(0.9, 0.8, 0.1), ambient=0.15, diffuse=0.7, specular=0.5, shininess=32)
        purple_glass = Material(Vec3(0.6, 0.2, 0.8), ambient=0.1, diffuse=0.3, specular=0.9, shininess=256, reflectance=0.8)
        
        # Floor
        floor_material = Material(Vec3(0.5, 0.5, 0.5), ambient=0.2, diffuse=0.6, specular=0.2, shininess=8, reflectance=0.1)
        self.objects.append(Plane(Vec3(0, -2, 0), Vec3(0, 1, 0), floor_material))
        
        # Spheres
        self.objects.append(Sphere(Vec3(0, 0, -5), 1.0, red_shiny))
        self.objects.append(Sphere(Vec3(-3, -1, -4), 0.8, blue_matte))
        self.objects.append(Sphere(Vec3(3, -0.5, -6), 1.2, green_metal))
        self.objects.append(Sphere(Vec3(-1, 1, -3), 0.6, yellow_plastic))
        self.objects.append(Sphere(Vec3(1.5, 0.5, -4), 0.7, purple_glass))
        self.objects.append(Sphere(Vec3(-2, -0.2, -7), 0.9, red_shiny))
        self.objects.append(Sphere(Vec3(2, -1.2, -3), 0.5, blue_matte))
        
        # Colorful lights
        self.lights.append(Light(Vec3(-5, 5, -2), Vec3(1.0, 0.4, 0.4), 0.8))  # Red light
        self.lights.append(Light(Vec3(5, 4, -1), Vec3(0.4, 1.0, 0.4), 0.9))   # Green light
        self.lights.append(Light(Vec3(0, 6, -8), Vec3(0.4, 0.4, 1.0), 0.7))   # Blue light
        self.lights.append(Light(Vec3(3, 3, -3), Vec3(1.0, 1.0, 0.4), 0.6))   # Yellow light
        self.lights.append(Light(Vec3(-3, 2, -6), Vec3(1.0, 0.4, 1.0), 0.5))  # Magenta light
        self.lights.append(Light(Vec3(0, 8, -5), Vec3(0.8, 0.8, 0.8), 0.4))   # White light
        
        # Camera
        self.camera = Camera(
            Vec3(0, 2, 2),      # position
            Vec3(0, 0, -5),     # look at
            Vec3(0, 1, 0),      # up
            45,                 # fov
            width / height      # aspect ratio
        )
    
    def intersect_scene(self, ray):
        closest_hit = None
        min_distance = float('inf')
        
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit['t'] < min_distance:
                min_distance = hit['t']
                closest_hit = hit
        
        return closest_hit
    
    def calculate_lighting(self, hit_point, normal, view_dir, material):
        color = Vec3(0, 0, 0)
        
        # Ambient lighting
        ambient = material.color * material.ambient
        color = color + ambient
        
        for light in self.lights:
            # Shadow ray
            light_dir = (light.position - hit_point).normalize()
            shadow_ray = Ray(hit_point, light_dir)
            
            # Check for shadows
            shadow_hit = self.intersect_scene(shadow_ray)
            light_distance = (light.position - hit_point).length()
            
            if shadow_hit and shadow_hit['t'] < light_distance:
                continue  # In shadow
            
            # Diffuse lighting
            diff = max(0, normal.dot(light_dir))
            diffuse = color_multiply(material.color, light.color) * (material.diffuse * diff * light.intensity)
            color = color + diffuse
            
            # Specular lighting
            if diff > 0:
                reflect_dir = (light_dir * -1).reflect(normal)
                spec = max(0, view_dir.dot(reflect_dir)) ** material.shininess
                specular = light.color * (material.specular * spec * light.intensity)
                color = color + specular
        
        return color
    
    def trace_ray(self, ray, depth=0):
        if depth >= self.max_depth:
            return Vec3(0.1, 0.1, 0.2)  # Dark blue background
        
        hit = self.intersect_scene(ray)
        if not hit:
            # Sky gradient
            t = 0.5 * (ray.direction.y + 1.0)
            return Vec3(0.1, 0.1, 0.3) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t
        
        material = hit['material']
        hit_point = hit['point']
        normal = hit['normal']
        view_dir = (ray.direction * -1).normalize()
        
        # Calculate direct lighting
        color = self.calculate_lighting(hit_point, normal, view_dir, material)
        
        # Add reflection
        if material.reflectance > 0:
            reflect_dir = ray.direction.reflect(normal)
            reflect_ray = Ray(hit_point, reflect_dir)
            reflect_color = self.trace_ray(reflect_ray, depth + 1)
            color = color * (1 - material.reflectance) + reflect_color * material.reflectance
        
        return color
    
    def get_ray(self, x, y):
        pixel_center = self.camera.pixel00_loc + (self.camera.pixel_delta_u * x) + (self.camera.pixel_delta_v * y)
        ray_direction = pixel_center - self.camera.position
        return Ray(self.camera.position, ray_direction)
    
    def render(self):
        image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            if y % 50 == 0:
                print(f"Rendering line {y}/{self.height}")
            
            for x in range(self.width):
                ray = self.get_ray(x, y)
                color = self.trace_ray(ray)
                
                # Gamma correction and tone mapping
                color.x = clamp(math.sqrt(color.x))
                color.y = clamp(math.sqrt(color.y))
                color.z = clamp(math.sqrt(color.z))
                
                image_data[y, x] = [
                    int(color.x * 255),
                    int(color.y * 255),
                    int(color.z * 255)
                ]
        
        return image_data

def main():
    print("Starting raytracer...")
    raytracer = Raytracer(800, 600)
    
    print("Rendering scene...")
    image_data = raytracer.render()
    
    print("Saving image...")
    image = Image.fromarray(image_data, 'RGB')
    image.save('raytraced_scene.png')
    print("Image saved as 'raytraced_scene3.png'")

if __name__ == "__main__":
    main()