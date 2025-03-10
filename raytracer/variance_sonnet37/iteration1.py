import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

@dataclass
class Vector:
    x: float
    y: float
    z: float
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def normalize(self):
        norm = np.sqrt(self.dot(self))
        return self * (1.0 / norm) if norm > 0 else self
    
    def length(self):
        return np.sqrt(self.dot(self))

@dataclass
class Ray:
    origin: Vector
    direction: Vector

@dataclass
class Material:
    diffuse: Vector  # RGB color
    specular: Vector  # RGB specular color
    reflection: float  # 0-1, reflection coefficient
    shininess: float  # specular exponent

@dataclass
class Sphere:
    center: Vector
    radius: float
    material: Material

@dataclass
class PointLight:
    position: Vector
    color: Vector
    intensity: float

@dataclass
class Scene:
    objects: List[Sphere]
    lights: List[PointLight]
    ambient: Vector
    max_recursion: int

@dataclass
class Camera:
    position: Vector
    look_at: Vector
    up: Vector
    fov: float
    width: int
    height: int

def ray_sphere_intersect(ray: Ray, sphere: Sphere) -> Tuple[bool, float, float]:
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False, float('inf'), float('inf')
    
    t1 = (-b + np.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    
    return True, t1, t2

def cast_ray(ray: Ray, scene: Scene, depth: int = 0) -> Vector:
    if depth > scene.max_recursion:
        return Vector(0, 0, 0)
    
    closest_t = float('inf')
    closest_obj = None
    
    # Find the closest intersection
    for obj in scene.objects:
        hit, t1, t2 = ray_sphere_intersect(ray, obj)
        if hit:
            if 0.001 < t2 < closest_t:
                closest_t = t2
                closest_obj = obj
            elif 0.001 < t1 < closest_t:
                closest_t = t1
                closest_obj = obj
    
    if closest_obj is None:
        return Vector(0, 0, 0)  # Background color
    
    # Compute intersection point and normal
    hit_point = ray.origin + ray.direction * closest_t
    normal = (hit_point - closest_obj.center).normalize()
    
    # Ensure normal points toward the ray origin
    if normal.dot(ray.direction) > 0:
        normal = normal * (-1)
    
    material = closest_obj.material
    color = Vector(0, 0, 0)
    
    # Ambient component
    color = Vector(
        scene.ambient.x * material.diffuse.x,
        scene.ambient.y * material.diffuse.y,
        scene.ambient.z * material.diffuse.z
    )
    
    # For each light
    for light in scene.lights:
        light_dir = (light.position - hit_point).normalize()
        
        # Shadow ray to check if the point is in shadow
        shadow_ray = Ray(hit_point, light_dir)
        shadow = False
        
        for obj in scene.objects:
            hit, t1, t2 = ray_sphere_intersect(shadow_ray, obj)
            if hit and t2 > 0.001 and t2 < (light.position - hit_point).length():
                shadow = True
                break
        
        if not shadow:
            # Diffuse component
            diffuse_intensity = max(0, normal.dot(light_dir))
            diffuse = Vector(
                material.diffuse.x * light.color.x * diffuse_intensity * light.intensity,
                material.diffuse.y * light.color.y * diffuse_intensity * light.intensity,
                material.diffuse.z * light.color.z * diffuse_intensity * light.intensity
            )
            
            # Specular component (Blinn-Phong)
            view_dir = (ray.origin - hit_point).normalize()
            half_dir = (light_dir + view_dir).normalize()
            specular_intensity = max(0, normal.dot(half_dir)) ** material.shininess
            specular = Vector(
                material.specular.x * light.color.x * specular_intensity * light.intensity,
                material.specular.y * light.color.y * specular_intensity * light.intensity,
                material.specular.z * light.color.z * specular_intensity * light.intensity
            )
            
            color = color + diffuse + specular
    
    # Reflection component
    if material.reflection > 0 and depth < scene.max_recursion:
        reflect_dir = ray.direction - normal * (2 * ray.direction.dot(normal))
        reflect_ray = Ray(hit_point, reflect_dir.normalize())
        reflect_color = cast_ray(reflect_ray, scene, depth + 1)
        color = color + Vector(
            reflect_color.x * material.reflection,
            reflect_color.y * material.reflection,
            reflect_color.z * material.reflection
        )
    
    # Clamp color values to [0, 1]
    return Vector(
        min(1, max(0, color.x)),
        min(1, max(0, color.y)),
        min(1, max(0, color.z))
    )

def render(scene: Scene, camera: Camera) -> np.ndarray:
    # Create image buffer
    image = np.zeros((camera.height, camera.width, 3))
    
    # Calculate camera basis vectors
    forward = (camera.look_at - camera.position).normalize()
    right = forward.cross(camera.up).normalize()
    up = right.cross(forward).normalize()
    
    # Calculate the aspect ratio and field of view
    aspect_ratio = camera.width / camera.height
    tan_fov = np.tan(camera.fov * 0.5 * np.pi / 180)
    
    # Render each pixel
    for y in range(camera.height):
        for x in range(camera.width):
            # Map pixel coordinates to [-1, 1]
            screen_x = (2 * (x + 0.5) / camera.width - 1) * aspect_ratio * tan_fov
            screen_y = (1 - 2 * (y + 0.5) / camera.height) * tan_fov
            
            # Calculate ray direction
            ray_dir = (forward + right * screen_x + up * screen_y).normalize()
            ray = Ray(camera.position, ray_dir)
            
            # Cast ray and get color
            color = cast_ray(ray, scene)
            
            # Set pixel color
            image[y, x] = [color.x, color.y, color.z]
    
    return image

def create_scene():
    # Create materials
    red_material = Material(
        diffuse=Vector(0.9, 0.1, 0.1),
        specular=Vector(0.8, 0.8, 0.8),
        reflection=0.2,
        shininess=50
    )
    
    blue_material = Material(
        diffuse=Vector(0.1, 0.3, 0.9),
        specular=Vector(0.8, 0.8, 0.8),
        reflection=0.3,
        shininess=100
    )
    
    green_material = Material(
        diffuse=Vector(0.1, 0.8, 0.2),
        specular=Vector(0.8, 0.8, 0.8),
        reflection=0.2,
        shininess=75
    )
    
    gold_material = Material(
        diffuse=Vector(0.9, 0.8, 0.1),
        specular=Vector(0.9, 0.9, 0.8),
        reflection=0.4,
        shininess=200
    )
    
    mirror_material = Material(
        diffuse=Vector(0.1, 0.1, 0.1),
        specular=Vector(0.9, 0.9, 0.9),
        reflection=0.8,
        shininess=300
    )
    
    glass_material = Material(
        diffuse=Vector(0.8, 0.8, 0.9),
        specular=Vector(0.9, 0.9, 0.9),
        reflection=0.6,
        shininess=150
    )

    # Create objects
    objects = [
        Sphere(center=Vector(0, 0, 0), radius=1.0, material=mirror_material),
        Sphere(center=Vector(-2.5, 0.5, 1), radius=1.5, material=blue_material),
        Sphere(center=Vector(2.5, 0, 0.5), radius=1.2, material=red_material),
        Sphere(center=Vector(0, -401, 0), radius=400, material=green_material),  # Ground
        Sphere(center=Vector(0.5, 1.5, -1), radius=0.5, material=gold_material),
        Sphere(center=Vector(-1.5, 0.5, -2), radius=0.7, material=glass_material),
    ]

    # Create colorful lights
    lights = [
        PointLight(position=Vector(-5, 5, -5), color=Vector(1.0, 0.3, 0.3), intensity=1.0),
        PointLight(position=Vector(5, 5, -5), color=Vector(0.3, 0.3, 1.0), intensity=1.0),
        PointLight(position=Vector(0, 5, 5), color=Vector(0.3, 1.0, 0.3), intensity=1.0),
        PointLight(position=Vector(0, 7, 0), color=Vector(1.0, 1.0, 1.0), intensity=1.5),
    ]

    # Create scene
    scene = Scene(
        objects=objects,
        lights=lights,
        ambient=Vector(0.1, 0.1, 0.1),
        max_recursion=3
    )

    return scene

def main():
    # Set up camera
    camera = Camera(
        position=Vector(0, 3, -10),
        look_at=Vector(0, 0, 0),
        up=Vector(0, 1, 0),
        fov=60,
        width=800,
        height=600
    )

    # Create scene
    scene = create_scene()
    
    print("Rendering image...")
    image = render(scene, camera)
    
    # Apply simple tone mapping (gamma correction)
    image = np.power(image, 1/2.2)
    
    # Save image
    plt.imsave('raytraced_scene1.png', image)
    print("Image saved as 'raytraced_scene1.png'")

if __name__ == "__main__":
    main()