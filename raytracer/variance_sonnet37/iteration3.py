import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

@dataclass
class Vec3:
    x: float
    y: float
    z: float
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
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
    
    def reflect(self, normal):
        return self - 2 * self.dot(normal) * normal
    
    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2])


@dataclass
class Ray:
    origin: Vec3
    direction: Vec3
    
    def point_at(self, t):
        return self.origin + self.direction * t


@dataclass
class Material:
    color: Vec3
    ambient: float = 0.05
    diffuse: float = 0.8
    specular: float = 0.3
    shininess: float = 30
    reflective: float = 0.0
    
    
@dataclass
class Sphere:
    center: Vec3
    radius: float
    material: Material
    
    def intersect(self, ray: Ray) -> Tuple[bool, float]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False, float('inf')
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t > 0.001:  # Avoid self-intersection
            return True, t
        
        t = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t > 0.001:
            return True, t
        
        return False, float('inf')
    
    def normal_at(self, point):
        return (point - self.center).normalize()


@dataclass
class PointLight:
    position: Vec3
    color: Vec3
    intensity: float = 1.0


class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vec3(0.05, 0.05, 0.08)  # Dark blue background
        self.max_reflection_depth = 4
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)
    
    def closest_intersection(self, ray):
        min_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            hit, t = obj.intersect(ray)
            if hit and t < min_t:
                min_t = t
                closest_obj = obj
        
        return closest_obj, min_t

    def is_shadowed(self, point, light):
        light_dir = (light.position - point).normalize()
        shadow_ray = Ray(point + light_dir * 0.001, light_dir)
        
        for obj in self.objects:
            hit, t = obj.intersect(shadow_ray)
            distance_to_light = (light.position - point).length()
            if hit and t < distance_to_light:
                return True
        
        return False
    
    def calculate_lighting(self, point, normal, view_dir, material, obj):
        result = material.color * material.ambient  # Ambient component
        
        for light in self.lights:
            if self.is_shadowed(point, light):
                continue
                
            light_dir = (light.position - point).normalize()
            distance_to_light = (light.position - point).length()
            
            # Diffuse reflection
            n_dot_l = max(normal.dot(light_dir), 0)
            diffuse = material.diffuse * n_dot_l
            
            # Specular reflection (Blinn-Phong)
            half_vector = (light_dir + view_dir).normalize()
            n_dot_h = max(normal.dot(half_vector), 0)
            specular = material.specular * (n_dot_h ** material.shininess)
            
            # Light attenuation
            attenuation = light.intensity / (1 + 0.1 * distance_to_light + 0.01 * distance_to_light * distance_to_light)
            
            # Apply light color and add the lighting contribution
            light_contribution = light.color * attenuation * (diffuse + specular)
            result = result + Vec3.from_array(material.color.to_array() * light_contribution.to_array())
            
        return result
    
    def ray_color(self, ray, depth=0):
        if depth >= self.max_reflection_depth:
            return self.background_color
        
        obj, t = self.closest_intersection(ray)
        if obj is None:
            return self.background_color
        
        hit_point = ray.point_at(t)
        normal = obj.normal_at(hit_point)
        view_dir = ray.direction * -1
        
        material = obj.material
        color = self.calculate_lighting(hit_point, normal, view_dir, material, obj)
        
        # Calculate reflections if material is reflective
        if material.reflective > 0 and depth < self.max_reflection_depth:
            reflect_dir = ray.direction.reflect(normal)
            reflect_ray = Ray(hit_point + reflect_dir * 0.001, reflect_dir)
            reflect_color = self.ray_color(reflect_ray, depth + 1)
            color = color * (1 - material.reflective) + reflect_color * material.reflective
            
        # Ensure color values are in valid range
        return Vec3(
            min(max(color.x, 0), 1),
            min(max(color.y, 0), 1),
            min(max(color.z, 0), 1)
        )


class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = position
        self.forward = (look_at - position).normalize()
        self.right = self.forward.cross(up).normalize()
        self.up = self.right.cross(self.forward)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
    
    def get_ray(self, u, v):
        # Convert to NDC space
        ndc_x = (2 * u - 1) * self.aspect_ratio * np.tan(np.radians(self.fov) / 2)
        ndc_y = (1 - 2 * v) * np.tan(np.radians(self.fov) / 2)
        
        ray_dir = (self.forward + ndc_x * self.right + ndc_y * self.up).normalize()
        return Ray(self.position, ray_dir)


def render(scene, camera, width, height, samples=1):
    aspect_ratio = width / height
    image = np.zeros((height, width, 3))
    
    start_time = time.time()
    
    for y in range(height):
        print(f"Rendering... {y}/{height} lines complete ({100*y/height:.1f}%)", end='\r')
        
        for x in range(width):
            color = Vec3(0, 0, 0)
            
            # Anti-aliasing with random sampling
            for _ in range(samples):
                if samples > 1:
                    u = (x + np.random.random()) / width
                    v = (y + np.random.random()) / height
                else:
                    u = (x + 0.5) / width
                    v = (y + 0.5) / height
                    
                ray = camera.get_ray(u, v)
                color = color + scene.ray_color(ray)
            
            # Average samples
            color = color / samples
            
            # Store pixel color
            image[y, x] = [color.x, color.y, color.z]
    
    print(f"\nRendering complete in {time.time() - start_time:.2f} seconds")
    return image


def create_scene():
    scene = Scene()
    
    # Materials
    red_material = Material(Vec3(0.9, 0.2, 0.1), ambient=0.05, diffuse=0.7, specular=0.4, shininess=40, reflective=0.2)
    green_material = Material(Vec3(0.1, 0.9, 0.2), ambient=0.05, diffuse=0.8, specular=0.3, shininess=30, reflective=0.1)
    blue_material = Material(Vec3(0.2, 0.3, 0.9), ambient=0.05, diffuse=0.8, specular=0.3, shininess=30, reflective=0.1)
    gold_material = Material(Vec3(0.9, 0.8, 0.2), ambient=0.1, diffuse=0.6, specular=0.8, shininess=60, reflective=0.4)
    silver_material = Material(Vec3(0.8, 0.8, 0.8), ambient=0.1, diffuse=0.5, specular=0.9, shininess=60, reflective=0.6)
    purple_material = Material(Vec3(0.6, 0.1, 0.7), ambient=0.05, diffuse=0.7, specular=0.3, shininess=30, reflective=0.1)
    cyan_material = Material(Vec3(0.1, 0.7, 0.7), ambient=0.05, diffuse=0.8, specular=0.3, shininess=30, reflective=0.1)
    
    # Floor - Large sphere
    scene.add_object(Sphere(Vec3(0, -100.5, -1), 100, green_material))
    
    # Main spheres
    scene.add_object(Sphere(Vec3(0, 0, -1), 0.5, silver_material))
    scene.add_object(Sphere(Vec3(-1, 0.3, -1.5), 0.3, red_material))
    scene.add_object(Sphere(Vec3(1, -0.2, -1.2), 0.3, blue_material))
    scene.add_object(Sphere(Vec3(0.3, -0.3, -0.5), 0.2, gold_material))
    scene.add_object(Sphere(Vec3(-0.6, -0.2, -0.8), 0.3, purple_material))
    scene.add_object(Sphere(Vec3(0.5, 0.2, -2), 0.3, cyan_material))
    
    # Add colorful lights
    scene.add_light(PointLight(Vec3(2, 2, 0), Vec3(1.0, 0.9, 0.9), 1.0))  # Warm white
    scene.add_light(PointLight(Vec3(-2, 1, 0), Vec3(0.8, 0.2, 0.2), 0.8))  # Red
    scene.add_light(PointLight(Vec3(0, 1, 1), Vec3(0.2, 0.8, 0.2), 0.8))   # Green
    scene.add_light(PointLight(Vec3(-1, 2, 2), Vec3(0.2, 0.2, 0.8), 0.8))  # Blue
    scene.add_light(PointLight(Vec3(1, 1, -3), Vec3(0.8, 0.6, 0.2), 0.7))  # Amber
    scene.add_light(PointLight(Vec3(-1, 0.5, -1), Vec3(0.6, 0.3, 0.8), 0.6)) # Purple
    
    return scene


def main():
    width, height = 800, 600
    samples = 4  # Anti-aliasing samples per pixel
    
    # Create scene
    scene = create_scene()
    
    # Set up camera
    camera = Camera(
        position=Vec3(0, 0.5, 2),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        fov=60,
        aspect_ratio=width / height
    )
    
    # Render scene
    print("Raytracer starting...")
    image = render(scene, camera, width, height, samples)
    
    # Save output
    plt.figure(figsize=(10, 7.5))
    plt.imshow(np.clip(image, 0, 1))
    plt.axis('off')
    plt.savefig('raytraced_scene3.png', bbox_inches='tight', dpi=100)
    print("Image saved as 'raytraced_scene.png'")


if __name__ == "__main__":
    main()