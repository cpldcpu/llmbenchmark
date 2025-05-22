import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def length(self):
        return math.sqrt(self.dot(self))
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return Vector3(self.x / l, self.y / l, self.z / l)
        return Vector3(0, 0, 0)
    
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
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        
        if t1 > 0.001:
            return t1
        elif t2 > 0.001:
            return t2
        return None
    
    def normal_at(self, point):
        return (point - self.center).normalize()

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vector3(0.05, 0.05, 0.1)
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def intersect(self, ray):
        closest_t = float('inf')
        closest_object = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_t = t
                closest_object = obj
        
        if closest_object:
            return closest_t, closest_object
        return None, None

class Raytracer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.max_depth = 3
    
    def trace_ray(self, scene, ray, depth=0):
        if depth >= self.max_depth:
            return scene.background_color
        
        t, obj = scene.intersect(ray)
        if obj is None:
            return scene.background_color
        
        hit_point = ray.point_at(t)
        normal = obj.normal_at(hit_point)
        material = obj.material
        
        # Ambient lighting
        color = Vector3(
            material.color.x * material.ambient,
            material.color.y * material.ambient,
            material.color.z * material.ambient
        )
        
        # Process each light source
        for light in scene.lights:
            light_dir = (light.position - hit_point).normalize()
            
            # Check for shadows
            shadow_ray = Ray(hit_point + normal * 0.001, light_dir)
            shadow_t, shadow_obj = scene.intersect(shadow_ray)
            light_distance = (light.position - hit_point).length()
            
            if shadow_obj is None or shadow_t > light_distance:
                # Diffuse lighting
                diffuse_intensity = max(0, normal.dot(light_dir))
                diffuse_color = Vector3(
                    material.color.x * material.diffuse * diffuse_intensity * light.color.x * light.intensity,
                    material.color.y * material.diffuse * diffuse_intensity * light.color.y * light.intensity,
                    material.color.z * material.diffuse * diffuse_intensity * light.color.z * light.intensity
                )
                color = color + diffuse_color
                
                # Specular lighting
                view_dir = (ray.origin - hit_point).normalize()
                reflect_dir = (-light_dir).reflect(normal)
                specular_intensity = max(0, view_dir.dot(reflect_dir)) ** material.shininess
                specular_color = Vector3(
                    material.specular * specular_intensity * light.color.x * light.intensity,
                    material.specular * specular_intensity * light.color.y * light.intensity,
                    material.specular * specular_intensity * light.color.z * light.intensity
                )
                color = color + specular_color
        
        # Reflection
        if material.reflectivity > 0:
            reflect_dir = ray.direction.reflect(normal)
            reflect_ray = Ray(hit_point + normal * 0.001, reflect_dir)
            reflect_color = self.trace_ray(scene, reflect_ray, depth + 1)
            color = color + reflect_color * material.reflectivity
        
        return color
    
    def render(self, scene, camera_pos, camera_target):
        image = np.zeros((self.height, self.width, 3))
        
        # Camera setup
        camera_dir = (camera_target - camera_pos).normalize()
        camera_right = Vector3(0, 1, 0).dot(camera_dir)
        if abs(camera_right) > 0.9:
            camera_up = Vector3(1, 0, 0)
        else:
            camera_up = Vector3(0, 1, 0)
        
        camera_right = Vector3(
            camera_dir.y * camera_up.z - camera_dir.z * camera_up.y,
            camera_dir.z * camera_up.x - camera_dir.x * camera_up.z,
            camera_dir.x * camera_up.y - camera_dir.y * camera_up.x
        ).normalize()
        
        camera_up = Vector3(
            camera_right.y * camera_dir.z - camera_right.z * camera_dir.y,
            camera_right.z * camera_dir.x - camera_right.x * camera_dir.z,
            camera_right.x * camera_dir.y - camera_right.y * camera_dir.x
        ).normalize()
        
        fov = 45
        aspect_ratio = self.width / self.height
        scale = math.tan(math.radians(fov) * 0.5)
        
        for y in range(self.height):
            for x in range(self.width):
                # Convert pixel coordinates to normalized device coordinates
                px = (2 * (x + 0.5) / self.width - 1) * aspect_ratio * scale
                py = (1 - 2 * (y + 0.5) / self.height) * scale
                
                # Calculate ray direction
                ray_dir = camera_dir + camera_right * px + camera_up * py
                ray = Ray(camera_pos, ray_dir)
                
                # Trace ray and get color
                color = self.trace_ray(scene, ray)
                
                # Clamp color values
                r = min(1.0, max(0.0, color.x))
                g = min(1.0, max(0.0, color.y))
                b = min(1.0, max(0.0, color.z))
                
                image[y, x] = [r, g, b]
            
            # Progress indicator
            if y % 50 == 0:
                print(f"Rendering... {y}/{self.height} lines completed")
        
        return image

def create_scene():
    scene = Scene()
    
    # Create materials with different properties
    red_shiny = Material(Vector3(0.8, 0.2, 0.2), reflectivity=0.3)
    green_matte = Material(Vector3(0.2, 0.8, 0.2), diffuse=0.9, specular=0.1)
    blue_mirror = Material(Vector3(0.3, 0.3, 0.9), reflectivity=0.6, specular=0.4)
    yellow_plastic = Material(Vector3(0.9, 0.9, 0.3), shininess=100)
    purple_metal = Material(Vector3(0.7, 0.3, 0.8), reflectivity=0.4, specular=0.6)
    orange_rubber = Material(Vector3(0.9, 0.5, 0.1), diffuse=0.8, specular=0.0)
    cyan_glass = Material(Vector3(0.2, 0.8, 0.8), reflectivity=0.2, specular=0.8)
    
    # Add spheres to create an interesting scene
    scene.add_object(Sphere(Vector3(0, 0, -5), 1.0, red_shiny))
    scene.add_object(Sphere(Vector3(-2.5, -1, -6), 0.8, green_matte))
    scene.add_object(Sphere(Vector3(2.5, -1, -6), 0.8, blue_mirror))
    scene.add_object(Sphere(Vector3(-1.5, 1.5, -4), 0.6, yellow_plastic))
    scene.add_object(Sphere(Vector3(1.5, 1.5, -4), 0.6, purple_metal))
    scene.add_object(Sphere(Vector3(0, -1.5, -3), 0.5, orange_rubber))
    scene.add_object(Sphere(Vector3(-3, 0.5, -7), 0.7, cyan_glass))
    scene.add_object(Sphere(Vector3(3, 0.5, -7), 0.7, Material(Vector3(0.9, 0.7, 0.9), reflectivity=0.1)))
    
    # Add a large sphere as ground
    scene.add_object(Sphere(Vector3(0, -1001, -5), 1000, Material(Vector3(0.8, 0.8, 0.8), diffuse=0.6)))
    
    # Add multiple colorful light sources
    scene.add_light(Light(Vector3(-5, 5, -2), Vector3(1.0, 0.3, 0.3), 0.8))  # Red light
    scene.add_light(Light(Vector3(5, 5, -2), Vector3(0.3, 1.0, 0.3), 0.8))   # Green light
    scene.add_light(Light(Vector3(0, 8, -3), Vector3(0.3, 0.3, 1.0), 1.0))   # Blue light
    scene.add_light(Light(Vector3(-3, 3, 0), Vector3(1.0, 1.0, 0.3), 0.6))   # Yellow light
    scene.add_light(Light(Vector3(3, 3, 0), Vector3(1.0, 0.3, 1.0), 0.6))    # Magenta light
    scene.add_light(Light(Vector3(0, 2, -8), Vector3(0.3, 1.0, 1.0), 0.7))   # Cyan light
    
    return scene

def main():
    print("Starting raytracer...")
    
    # Create scene
    scene = create_scene()
    
    # Create raytracer
    raytracer = Raytracer(800, 600)
    
    # Set camera position and target
    camera_pos = Vector3(0, 2, 2)
    camera_target = Vector3(0, 0, -5)
    
    # Render the scene
    print("Rendering scene...")
    image = raytracer.render(scene, camera_pos, camera_target)
    
    # Convert to 8-bit and save
    image_8bit = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_8bit)
    img.save('raytraced_scene1.png')
    
    print("Rendering complete! Image saved as 'raytraced_scene.png'")
    
    # Display the image
    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.title('Raytraced Scene with Multiple Colored Light Sources')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()