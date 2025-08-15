import numpy as np
from PIL import Image
import time

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
        if isinstance(scalar, Vector3):
            # Component-wise multiplication for Vector3 * Vector3
            return Vector3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        # Scalar multiplication for Vector3 * number
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return np.sqrt(self.dot(self))
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return self
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, array):
        return cls(array[0], array[1], array[2])

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Material:
    def __init__(self, color=Vector3(1, 1, 1), ambient=0.1, diffuse=0.7, specular=0.2, shininess=50, reflectivity=0.0):
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
        
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        
        if t1 < 0 and t2 < 0:
            return None
        
        if t1 < 0:
            return t2
        if t2 < 0:
            return t1
        
        return min(t1, t2)
    
    def normal_at(self, point):
        return (point - self.center).normalize()

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material
    
    def intersect(self, ray):
        denom = ray.direction.dot(self.normal)
        
        if abs(denom) < 1e-6:  # Ray is parallel to the plane
            return None
        
        t = (self.point - ray.origin).dot(self.normal) / denom
        
        if t < 0:  # Intersection is behind the ray's origin
            return None
        
        return t
    
    def normal_at(self, point):
        return self.normal

class PointLight:
    def __init__(self, position, intensity=1.0, color=Vector3(1, 1, 1)):
        self.position = position
        self.intensity = intensity
        self.color = color

class Scene:
    def __init__(self, width, height, fov=60):
        self.width = width
        self.height = height
        self.fov = fov
        self.objects = []
        self.lights = []
        self.ambient_color = Vector3(0.1, 0.1, 0.1)
        self.background_color = Vector3(0.05, 0.05, 0.1)
        self.max_ray_depth = 3
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def cast_ray(self, ray, depth=0):
        if depth > self.max_ray_depth:
            return self.background_color
        
        closest_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t and t < closest_t:
                closest_t = t
                closest_obj = obj
        
        if closest_obj is None:
            return self.background_color
        
        hit_point = ray.origin + ray.direction * closest_t
        normal = closest_obj.normal_at(hit_point)
        
        # Ensure normal points towards viewer
        if normal.dot(ray.direction) > 0:
            normal = normal * -1
        
        material = closest_obj.material
        
        # Initialize with ambient light
        color = material.color * material.ambient * self.ambient_color
        
        # Slightly offset hit point to avoid shadow acne
        hit_point_offset = hit_point + normal * 0.001
        
        for light in self.lights:
            light_dir = (light.position - hit_point).normalize()
            light_distance = (light.position - hit_point).length()
            
            # Shadow check
            shadow_ray = Ray(hit_point_offset, light_dir)
            in_shadow = False
            
            for obj in self.objects:
                shadow_t = obj.intersect(shadow_ray)
                if shadow_t and shadow_t < light_distance:
                    in_shadow = True
                    break
            
            if not in_shadow:
                # Diffuse (Lambert) shading
                nl_dot = max(normal.dot(light_dir), 0)
                diffuse = material.color * material.diffuse * nl_dot * light.intensity * light.color
                
                # Specular (Phong) shading
                view_dir = (ray.origin - hit_point).normalize()
                reflect_dir = light_dir - normal * 2 * light_dir.dot(normal)
                spec_angle = max(view_dir.dot(reflect_dir), 0)
                specular = Vector3(1, 1, 1) * material.specular * (spec_angle ** material.shininess) * light.intensity * light.color
                
                color = color + diffuse + specular
        
        # Handle reflection
        if material.reflectivity > 0 and depth < self.max_ray_depth:
            reflect_dir = ray.direction - normal * 2 * ray.direction.dot(normal)
            reflection_ray = Ray(hit_point_offset, reflect_dir)
            reflection_color = self.cast_ray(reflection_ray, depth + 1)
            color = color * (1 - material.reflectivity) + reflection_color * material.reflectivity
        
        # Clamp color values to [0, 1]
        return Vector3(
            min(max(color.x, 0), 1),
            min(max(color.y, 0), 1),
            min(max(color.z, 0), 1)
        )
    
    def render(self):
        # Create image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        aspect_ratio = self.width / self.height
        fov_radians = self.fov * np.pi / 180.0
        camera_pos = Vector3(0, 1, -5)
        
        start_time = time.time()
        
        # Render each pixel
        for y in range(self.height):
            for x in range(self.width):
                # Map pixel coordinates to [-1, 1] range
                screen_x = (2 * x / self.width - 1) * aspect_ratio * np.tan(fov_radians / 2)
                screen_y = (1 - 2 * y / self.height) * np.tan(fov_radians / 2)
                
                # Create ray from camera to pixel
                direction = Vector3(screen_x, screen_y, 1).normalize()
                ray = Ray(camera_pos, direction)
                
                # Cast ray and get color
                color = self.cast_ray(ray)
                
                # Convert color to RGB and store in image array
                image[y, x] = np.clip(np.array([color.x, color.y, color.z]) * 255, 0, 255).astype(np.uint8)
            
            # Print progress
            if (y + 1) % 20 == 0:
                progress = (y + 1) / self.height * 100
                elapsed = time.time() - start_time
                print(f"Rendering: {progress:.1f}% complete, elapsed: {elapsed:.1f}s")
        
        return image

def main():
    # Create scene
    scene = Scene(800, 600, 60)
    
    # Add objects
    # Floor
    scene.add_object(Plane(
        Vector3(0, -1, 0), 
        Vector3(0, 1, 0), 
        Material(Vector3(0.8, 0.8, 0.8), diffuse=0.6, specular=0.4, reflectivity=0.2)
    ))
    
    # Ceiling
    scene.add_object(Plane(
        Vector3(0, 5, 0), 
        Vector3(0, -1, 0), 
        Material(Vector3(0.8, 0.8, 0.8), diffuse=0.6, specular=0.1)
    ))
    
    # Back wall
    scene.add_object(Plane(
        Vector3(0, 0, 10), 
        Vector3(0, 0, -1), 
        Material(Vector3(0.8, 0.8, 0.8), diffuse=0.6, specular=0.1)
    ))
    
    # Center sphere
    scene.add_object(Sphere(
        Vector3(0, 0.5, 2), 
        1.5, 
        Material(Vector3(0.8, 0.2, 0.2), diffuse=0.7, specular=0.3, shininess=100, reflectivity=0.3)
    ))
    
    # Left reflective sphere
    scene.add_object(Sphere(
        Vector3(-2.5, 0, 3), 
        1, 
        Material(Vector3(0.2, 0.8, 0.2), diffuse=0.6, specular=0.4, shininess=60, reflectivity=0.3)
    ))
    
    # Right glass-like sphere
    scene.add_object(Sphere(
        Vector3(2.5, 0, 3), 
        1, 
        Material(Vector3(0.2, 0.2, 0.8), diffuse=0.6, specular=0.6, shininess=150, reflectivity=0.4)
    ))
    
    # Small spheres in the back
    scene.add_object(Sphere(
        Vector3(-1.5, -0.5, 4.5), 
        0.5, 
        Material(Vector3(0.9, 0.9, 0.1), diffuse=0.7, specular=0.3, shininess=50, reflectivity=0.1)
    ))
    
    scene.add_object(Sphere(
        Vector3(1.5, -0.5, 4.5), 
        0.5, 
        Material(Vector3(0.1, 0.9, 0.9), diffuse=0.7, specular=0.3, shininess=50, reflectivity=0.1)
    ))
    
    # Add colorful lights
    scene.add_light(PointLight(Vector3(0, 4, 0), 0.8, Vector3(1.0, 0.9, 0.9)))  # Main light (slightly warm)
    scene.add_light(PointLight(Vector3(-3, 3, -2), 0.6, Vector3(0.2, 0.2, 1.0)))  # Blue light
    scene.add_light(PointLight(Vector3(3, 3, -2), 0.6, Vector3(1.0, 0.2, 0.2)))  # Red light
    scene.add_light(PointLight(Vector3(0, 1, -3), 0.5, Vector3(0.2, 1.0, 0.2)))  # Green light
    scene.add_light(PointLight(Vector3(0, 2, 6), 0.7, Vector3(1.0, 1.0, 0.2)))  # Yellow light
    scene.add_light(PointLight(Vector3(-3, 0.5, 3), 0.4, Vector3(1.0, 0.5, 0.0)))  # Orange light
    scene.add_light(PointLight(Vector3(3, 0.5, 3), 0.4, Vector3(0.8, 0.2, 1.0)))  # Purple light
    
    # Render scene
    print("Starting render...")
    start_time = time.time()
    image_array = scene.render()
    end_time = time.time()
    print(f"Render completed in {end_time - start_time:.1f} seconds")
    
    # Convert to PIL Image and save
    img = Image.fromarray(image_array)
    output_filename = "colorful_scene2.png"
    img.save(output_filename)
    print(f"Image saved as {output_filename}")

if __name__ == "__main__":
    main()