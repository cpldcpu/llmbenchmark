import numpy as np
from PIL import Image
import math

class Vector3:
    def __init__(self, x=0, y=0, z=0):
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
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.dot(self))
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return self * (1.0 / length)
        return self
    
    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Material:
    def __init__(self, color, diffuse=1.0, specular=1.0, shininess=25.0, reflection=0.5):
        self.color = color
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_discr = math.sqrt(discriminant)
        t1 = (-b - sqrt_discr) / (2 * a)
        t2 = (-b + sqrt_discr) / (2 * a)
        
        t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2)
        if t < 0:
            return None
        
        point = ray.origin + ray.direction * t
        normal = (point - self.center).normalize()
        
        return {'t': t, 'point': point, 'normal': normal, 'object': self}

class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vector3(0.1, 0.1, 0.1)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def trace_ray(self, ray, depth=0, max_depth=3):
        if depth > max_depth:
            return self.background_color
        
        closest_intersection = None
        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection and (not closest_intersection or intersection['t'] < closest_intersection['t']):
                closest_intersection = intersection
        
        if not closest_intersection:
            return self.background_color
        
        material = closest_intersection['object'].material
        point = closest_intersection['point']
        normal = closest_intersection['normal']
        
        # Calculate direct illumination
        color = Vector3(0, 0, 0)
        
        for light in self.lights:
            light_dir = (light.position - point).normalize()
            light_distance = (light.position - point).length()
            
            # Shadow check
            shadow_ray = Ray(point + normal * 0.001, light_dir)
            in_shadow = False
            for obj in self.objects:
                shadow_intersect = obj.intersect(shadow_ray)
                if shadow_intersect and shadow_intersect['t'] < light_distance:
                    in_shadow = True
                    break
            
            if not in_shadow:
                # Diffuse component
                diffuse = max(0, normal.dot(light_dir))
                diffuse_color = Vector3(
                    material.color.x * light.color.x * diffuse * material.diffuse,
                    material.color.y * light.color.y * diffuse * material.diffuse,
                    material.color.z * light.color.z * diffuse * material.diffuse
                ) * light.intensity
                
                # Specular component (Phong)
                view_dir = (ray.origin - point).normalize()
                reflect_dir = (normal * (2 * normal.dot(light_dir)) - light_dir).normalize()
                specular = math.pow(max(0, view_dir.dot(reflect_dir)), material.shininess)
                specular_color = Vector3(
                    light.color.x * specular * material.specular,
                    light.color.y * specular * material.specular,
                    light.color.z * specular * material.specular
                ) * light.intensity
                
                # Attenuation
                attenuation = 1.0 / (1.0 + 0.1 * light_distance + 0.01 * light_distance * light_distance)
                
                color += (diffuse_color + specular_color) * attenuation
        
        # Reflection
        reflected_color = Vector3(0, 0, 0)
        if material.reflection > 0:
            reflect_dir = ray.direction - normal * 2 * ray.direction.dot(normal)
            reflect_ray = Ray(point + normal * 0.001, reflect_dir)
            reflected_color = self.trace_ray(reflect_ray, depth + 1, max_depth)
            color += reflected_color * material.reflection
        
        return color

def render(scene, width, height):
    aspect_ratio = width / height
    camera_pos = Vector3(0, 1, 5)
    fov = math.pi / 3
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Normalized device coordinates
            ndc_x = (x + 0.5) / width
            ndc_y = (y + 0.5) / height
            
            # Screen space coordinates
            screen_x = (2 * ndc_x - 1) * math.tan(fov / 2) * aspect_ratio
            screen_y = (1 - 2 * ndc_y) * math.tan(fov / 2)
            
            ray_dir = Vector3(screen_x, screen_y, -1).normalize()
            ray = Ray(camera_pos, ray_dir)
            
            color = scene.trace_ray(ray)
            
            # Clamp and convert to 0-255
            r = min(255, max(0, int(color.x * 255)))
            g = min(255, max(0, int(color.y * 255)))
            b = min(255, max(0, int(color.z * 255)))
            
            image[y, x] = [r, g, b]
    
    return image

def create_scene():
    scene = Scene()
    
    # Materials
    red_mat = Material(Vector3(1, 0.2, 0.2), reflection=0.3)
    blue_mat = Material(Vector3(0.2, 0.2, 1), diffuse=0.9, specular=0.5, shininess=50)
    green_mat = Material(Vector3(0.2, 1, 0.2), diffuse=0.7, specular=0.8, shininess=100)
    yellow_mat = Material(Vector3(1, 1, 0.2), diffuse=0.8, specular=0.3, shininess=25)
    purple_mat = Material(Vector3(0.8, 0.2, 0.8), diffuse=0.6, specular=0.9, shininess=75)
    mirror_mat = Material(Vector3(1, 1, 1), diffuse=0.1, specular=0.9, shininess=150, reflection=0.9)
    
    # Spheres
    scene.add_object(Sphere(Vector3(-2, 0.5, -1), 1, red_mat))
    scene.add_object(Sphere(Vector3(0, 0.5, -2), 1, blue_mat))
    scene.add_object(Sphere(Vector3(2, 0.5, -1), 1, green_mat))
    scene.add_object(Sphere(Vector3(-1, 0.2, 0), 0.7, yellow_mat))
    scene.add_object(Sphere(Vector3(1, 0.3, 0), 0.8, purple_mat))
    scene.add_object(Sphere(Vector3(0, -1000, 0), 1000, Material(Vector3(0.5, 0.5, 0.5))))
    
    # Lights - colorful lights at different positions
    scene.add_light(Light(Vector3(-3, 2, 3), Vector3(1, 0.5, 0.5), 1.5))  # Pinkish
    scene.add_light(Light(Vector3(3, 2, 3), Vector3(0.5, 1, 0.5), 1.5))    # Greenish
    scene.add_light(Light(Vector3(0, 5, 5), Vector3(0.8, 0.8, 1), 2.0))    # Bluish
    scene.add_light(Light(Vector3(0, 2, -3), Vector3(1, 1, 0.7), 1.2))     # Yellowish
    scene.add_light(Light(Vector3(0, 10, 0), Vector3(1, 1, 1), 0.8))       # White fill light
    
    return scene

def main():
    width = 800
    height = 600
    
    scene = create_scene()
    image = render(scene, width, height)
    
    # Save the image
    img = Image.fromarray(image, 'RGB')
    img.save('raytraced_scene.png')
    print("Rendering complete! Saved as raytraced_scene.png")

if __name__ == "__main__":
    main()