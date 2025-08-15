import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random

@dataclass
class Vector3:
    x: float
    y: float
    z: float
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
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
        return self / self.length()
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    def reflect(self, normal):
        return self - normal * 2 * self.dot(normal)

@dataclass
class Ray:
    origin: Vector3
    direction: Vector3
    
    def point_at(self, t):
        return self.origin + self.direction * t

@dataclass
class Material:
    color: Vector3
    ambient: float = 0.1
    diffuse: float = 0.7
    specular: float = 0.2
    shininess: float = 30.0
    reflectivity: float = 0.0
    
@dataclass
class Sphere:
    center: Vector3
    radius: float
    material: Material
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                return None
        
        return t
    
    def normal_at(self, point):
        return (point - self.center).normalize()

@dataclass
class Light:
    position: Vector3
    intensity: float
    color: Vector3

@dataclass
class Scene:
    objects: List[Sphere]
    lights: List[Light]
    # Fix: Use default_factory to create a new Vector3 instance each time
    ambient_light: Vector3 = field(default_factory=lambda: Vector3(0.1, 0.1, 0.1))
    
    def intersect(self, ray):
        closest_t = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_t = t
                closest_obj = obj
        
        if closest_obj is None:
            return None
        
        return closest_obj, closest_t

@dataclass
class Camera:
    position: Vector3
    look_at: Vector3
    up: Vector3
    fov: float
    aspect_ratio: float
    
    def get_ray(self, u, v):
        # Calculate camera coordinate system
        w = (self.position - self.look_at).normalize()
        u_axis = self.up.cross(w).normalize()
        v_axis = w.cross(u_axis)
        
        # Calculate viewport dimensions
        theta = self.fov * np.pi / 180
        half_height = np.tan(theta / 2)
        half_width = self.aspect_ratio * half_height
        
        # Calculate ray direction
        direction = (u_axis * (2 * u - 1) * half_width + 
                    v_axis * (1 - 2 * v) * half_height - w).normalize()
        
        return Ray(self.position, direction)

def clamp(x, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, x))

def trace_ray(ray, scene, depth=0, max_depth=3):
    if depth > max_depth:
        return Vector3(0, 0, 0)
    
    intersection = scene.intersect(ray)
    if intersection is None:
        return Vector3(0.1, 0.1, 0.2)  # Sky color
    
    obj, t = intersection
    hit_point = ray.point_at(t)
    normal = obj.normal_at(hit_point)
    color = obj.material.color * obj.material.ambient * scene.ambient_light.x
    
    # Calculate lighting from each light source
    for light in scene.lights:
        light_dir = (light.position - hit_point).normalize()
        
        # Check if point is in shadow
        shadow_ray = Ray(hit_point + normal * 0.001, light_dir)
        shadow_intersection = scene.intersect(shadow_ray)
        
        if shadow_intersection is not None:
            shadow_obj, shadow_t = shadow_intersection
            light_distance = (light.position - hit_point).length()
            if shadow_t < light_distance:
                continue  # In shadow
        
        # Diffuse lighting
        light_dot_normal = light_dir.dot(normal)
        if light_dot_normal > 0:
            diffuse = obj.material.color * obj.material.diffuse * light_dot_normal * light.intensity
            diffuse = Vector3(diffuse.x * light.color.x, 
                             diffuse.y * light.color.y, 
                             diffuse.z * light.color.z)
            
            # Specular lighting
            view_dir = (ray.origin - hit_point).normalize()
            reflect_dir = light_dir.reflect(normal)
            spec_factor = max(0, view_dir.dot(reflect_dir)) ** obj.material.shininess
            specular = obj.material.specular * spec_factor * light.intensity
            specular = Vector3(specular * light.color.x, 
                              specular * light.color.y, 
                              specular * light.color.z)
            
            color = color + diffuse + specular
    
    # Handle reflections
    if obj.material.reflectivity > 0 and depth < max_depth:
        reflect_dir = ray.direction.reflect(normal)
        reflect_ray = Ray(hit_point + normal * 0.001, reflect_dir)
        reflect_color = trace_ray(reflect_ray, scene, depth + 1, max_depth)
        color = color * (1 - obj.material.reflectivity) + reflect_color * obj.material.reflectivity
    
    return color

def render(scene, camera, width, height):
    image = np.zeros((height, width, 3))
    
    for j in range(height):
        for i in range(width):
            u = i / width
            v = j / height
            ray = camera.get_ray(u, v)
            color = trace_ray(ray, scene)
            
            # Convert color to RGB and clamp values
            image[j, i, 0] = clamp(color.x)
            image[j, i, 1] = clamp(color.y)
            image[j, i, 2] = clamp(color.z)
            
        if j % 50 == 0:
            print(f"Rendering progress: {j/height*100:.1f}%")
    
    return image

def main():
    # Create materials
    red_metal = Material(Vector3(1.0, 0.3, 0.3), reflectivity=0.6)
    blue_matte = Material(Vector3(0.3, 0.3, 1.0), diffuse=0.9, specular=0.1)
    green_glossy = Material(Vector3(0.3, 0.8, 0.3), specular=0.5, shininess=100.0, reflectivity=0.2)
    gold = Material(Vector3(1.0, 0.8, 0.2), reflectivity=0.7, specular=0.7)
    white = Material(Vector3(0.9, 0.9, 0.9), diffuse=0.8, specular=0.2, reflectivity=0.1)
    
    # Create scene
    spheres = [
        Sphere(Vector3(0, -0.2, -1), 0.7, red_metal),
        Sphere(Vector3(-1.5, 0.3, -1.5), 0.5, blue_matte),
        Sphere(Vector3(1.5, 0, -2), 0.7, green_glossy),
        Sphere(Vector3(0.3, -0.3, -0.5), 0.3, gold),
        Sphere(Vector3(0, -101, -1), 100, white),  # Floor
    ]
    
    # Create colorful lights
    lights = [
        Light(Vector3(-2, 3, 0), 0.8, Vector3(1.0, 0.2, 0.2)),    # Red light
        Light(Vector3(2, 2, 0), 0.8, Vector3(0.2, 0.2, 1.0)),     # Blue light
        Light(Vector3(0, 2, 2), 0.7, Vector3(0.2, 1.0, 0.2)),     # Green light
        Light(Vector3(0, 5, -5), 1.0, Vector3(1.0, 1.0, 1.0)),    # White light
        Light(Vector3(3, 1, -2), 0.5, Vector3(1.0, 0.8, 0.0)),    # Yellow light
        Light(Vector3(-2, 1, -3), 0.5, Vector3(1.0, 0.0, 1.0)),   # Purple light
    ]
    
    scene = Scene(spheres, lights)
    
    # Create camera
    camera = Camera(
        position=Vector3(0, 1, 4),
        look_at=Vector3(0, 0, -1),
        up=Vector3(0, 1, 0),
        fov=60.0,
        aspect_ratio=800/600
    )
    
    # Render scene
    print("Starting rendering...")
    image = render(scene, camera, 800, 600)
    print("Rendering complete!")
    
    # Save image
    plt.imsave('raytraced_scene4.png', image)
    
    # Display image
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()