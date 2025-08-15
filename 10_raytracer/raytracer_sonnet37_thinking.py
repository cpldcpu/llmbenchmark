import numpy as np
from PIL import Image
import math

# Vector operations
def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

# Material class
class Material:
    def __init__(self, color, ambient, diffuse, specular, reflection, shininess):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.shininess = shininess

# Light class
class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

# Sphere class
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        
        return None
    
    def normal(self, point):
        return normalize(point - self.center)

# Plane class
class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal_vector = normalize(normal)
        self.material = material
    
    def intersect(self, ray):
        denominator = np.dot(ray.direction, self.normal_vector)
        
        if abs(denominator) < 1e-6:
            return None
        
        t = np.dot(self.point - ray.origin, self.normal_vector) / denominator
        
        if t > 0:
            return t
        
        return None
    
    def normal(self, point):
        return self.normal_vector

# Scene class
class Scene:
    def __init__(self, width, height, fov):
        self.width = width
        self.height = height
        self.fov = fov
        self.objects = []
        self.lights = []
        self.ambient_color = np.array([0.05, 0.05, 0.05])
        self.background_color = np.array([0.0, 0.0, 0.0])
        self.max_depth = 3
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def closest_intersection(self, ray, min_distance, max_distance):
        closest_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and min_distance < t < max_distance and t < closest_t:
                closest_t = t
                closest_obj = obj
        
        if closest_obj is None:
            return None, None
        
        return closest_obj, closest_t
    
    def trace_ray(self, ray, min_distance, max_distance, depth):
        closest_obj, closest_t = self.closest_intersection(ray, min_distance, max_distance)
        
        if closest_obj is None:
            return self.background_color
        
        hit_point = ray.origin + closest_t * ray.direction
        normal = closest_obj.normal(hit_point)
        
        # Ensure normal points toward the ray
        if np.dot(normal, ray.direction) > 0:
            normal = -normal
        
        # Move the hit point slightly away from the surface to avoid self-intersection
        hit_point = hit_point + 1e-4 * normal
        
        color = self.ambient_color * closest_obj.material.ambient * closest_obj.material.color
        
        # Calculate illumination from all lights
        for light in self.lights:
            light_dir = normalize(light.position - hit_point)
            shadow_ray = Ray(hit_point, light_dir)
            shadow_obj, shadow_t = self.closest_intersection(shadow_ray, 1e-4, float('inf'))
            
            # Check if the point is in shadow
            light_distance = np.linalg.norm(light.position - hit_point)
            is_shadowed = shadow_obj is not None and shadow_t < light_distance
            
            if not is_shadowed:
                # Diffuse component
                diffuse = max(0, np.dot(normal, light_dir))
                diffuse_color = closest_obj.material.diffuse * diffuse * closest_obj.material.color * light.color * light.intensity
                
                # Specular component
                view_dir = normalize(-ray.direction)
                reflect_dir = reflect(-light_dir, normal)
                specular = max(0, np.dot(view_dir, reflect_dir)) ** closest_obj.material.shininess
                specular_color = closest_obj.material.specular * specular * light.color * light.intensity
                
                color += diffuse_color + specular_color
        
        # Calculate reflection
        if depth < self.max_depth and closest_obj.material.reflection > 0:
            reflect_dir = reflect(ray.direction, normal)
            reflect_ray = Ray(hit_point, reflect_dir)
            reflect_color = self.trace_ray(reflect_ray, 1e-4, float('inf'), depth + 1)
            color += closest_obj.material.reflection * reflect_color
        
        # Clamp color values
        color = np.clip(color, 0, 1)
        
        return color
    
    def render(self, filename):
        aspect_ratio = self.width / self.height
        image = np.zeros((self.height, self.width, 3))
        
        for y in range(self.height):
            for x in range(self.width):
                # Calculate the ray direction for the current pixel
                pixel_x = (2 * ((x + 0.5) / self.width) - 1) * aspect_ratio * math.tan(self.fov / 2)
                pixel_y = (1 - 2 * ((y + 0.5) / self.height)) * math.tan(self.fov / 2)
                
                ray_origin = np.array([0, 0, 0])
                ray_direction = normalize(np.array([pixel_x, pixel_y, -1]))
                
                ray = Ray(ray_origin, ray_direction)
                color = self.trace_ray(ray, 0.1, float('inf'), 0)
                
                image[y, x] = color
        
        # Convert the image to 8-bit RGB format
        image = (image * 255).astype(np.uint8)
        
        # Save the image
        Image.fromarray(image).save(filename)
        print(f"Image saved as {filename}")

# Setup and render a scene
def main():
    # Create scene (800x600 with 60 degree FOV)
    scene = Scene(800, 600, math.radians(60))
    
    # Define materials
    red_material = Material(np.array([1.0, 0.2, 0.2]), 0.2, 0.7, 0.9, 0.3, 100)
    green_material = Material(np.array([0.2, 1.0, 0.2]), 0.2, 0.6, 0.8, 0.3, 100)
    blue_material = Material(np.array([0.2, 0.2, 1.0]), 0.2, 0.6, 0.8, 0.3, 100)
    yellow_material = Material(np.array([1.0, 1.0, 0.2]), 0.2, 0.8, 0.5, 0.1, 50)
    purple_material = Material(np.array([0.8, 0.2, 0.8]), 0.2, 0.8, 0.5, 0.2, 80)
    mirror_material = Material(np.array([0.9, 0.9, 0.9]), 0.1, 0.3, 0.9, 0.8, 200)
    floor_material = Material(np.array([0.9, 0.9, 0.9]), 0.2, 0.8, 0.1, 0.2, 10)
    
    # Add objects to the scene
    scene.add_object(Sphere(np.array([-1.5, 0, -6]), 1.0, red_material))
    scene.add_object(Sphere(np.array([0, 0, -6]), 1.0, blue_material))
    scene.add_object(Sphere(np.array([1.5, 0, -6]), 1.0, green_material))
    scene.add_object(Sphere(np.array([0, -1.5, -4]), 0.5, yellow_material))
    scene.add_object(Sphere(np.array([0, 1.5, -7]), 0.7, purple_material))
    scene.add_object(Sphere(np.array([3, 0.5, -8]), 1.2, mirror_material))
    
    # Add a floor plane
    scene.add_object(Plane(np.array([0, -1.0, 0]), np.array([0, 1, 0]), floor_material))
    
    # Add colorful lights
    scene.add_light(Light(np.array([-5, 5, -3]), np.array([1.0, 0.3, 0.3]), 1.0))  # Red light
    scene.add_light(Light(np.array([5, 5, -3]), np.array([0.3, 1.0, 0.3]), 1.0))   # Green light
    scene.add_light(Light(np.array([0, 5, -10]), np.array([0.3, 0.3, 1.0]), 1.0))  # Blue light
    scene.add_light(Light(np.array([0, -3, -2]), np.array([1.0, 1.0, 0.5]), 0.7))  # Yellow light
    scene.add_light(Light(np.array([3, 1, -2]), np.array([0.8, 0.3, 0.8]), 0.8))   # Purple light
    
    # Render the scene
    scene.render("colorful_raytraced_scene.png")

if __name__ == "__main__":
    main()