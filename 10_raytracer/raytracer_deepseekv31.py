import numpy as np
import math
from PIL import Image

# Vector math utilities
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))

# Material class
class Material:
    def __init__(self, color, diffuse=1.0, specular=1.0, shininess=25, reflection=0.5):
        self.color = np.array(color, dtype=float)
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

# Sphere class
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.material = material
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return float('inf')
        
        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            t = (-b + math.sqrt(discriminant)) / (2.0 * a)
        
        return t if t >= 0 else float('inf')

# Plane class
class Plane:
    def __init__(self, point, normal, material, checkerboard=False):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.material = material
        self.checkerboard = checkerboard
    
    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            return t if t >= 0 else float('inf')
        return float('inf')

# Light class
class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)
        self.intensity = intensity

# Scene setup
def create_scene():
    materials = {
        'red': Material([1, 0.2, 0.2]),
        'blue': Material([0.2, 0.2, 1]),
        'green': Material([0.2, 1, 0.2]),
        'yellow': Material([1, 1, 0.2]),
        'purple': Material([0.8, 0.2, 1]),
        'cyan': Material([0.2, 1, 1]),
        'white': Material([1, 1, 1], reflection=0.2),
        'checker': Material([1, 1, 1], reflection=0.3)
    }
    
    spheres = [
        Sphere([-2, 0, -8], 1.5, materials['red']),
        Sphere([2, 0, -8], 1.5, materials['blue']),
        Sphere([0, 3, -10], 1.5, materials['green']),
        Sphere([-4, 1, -12], 1.5, materials['yellow']),
        Sphere([4, 1, -12], 1.5, materials['purple']),
        Sphere([0, -1.5, -6], 1.0, materials['cyan'])
    ]
    
    plane = Plane([0, -3, 0], [0, 1, 0], materials['checker'], checkerboard=True)
    
    lights = [
        Light([-5, 5, -3], [1, 0.5, 0.5], 1.0),  # Pink light
        Light([5, 5, -3], [0.5, 1, 0.5], 1.0),   # Greenish light
        Light([0, 10, 0], [0.5, 0.5, 1], 1.2),    # Blueish light
        Light([-3, 3, -15], [1, 1, 0.7], 0.8),    # Yellow light
        Light([3, 2, -15], [0.8, 0.3, 1], 0.8)    # Purple light
    ]
    
    return spheres + [plane], lights

# Checkerboard pattern for the plane
def checkerboard_pattern(point):
    scale = 2.0
    x = int((point[0] + 1000) * scale) % 2
    z = int((point[2] + 1000) * scale) % 2
    return (x + z) % 2

# Ray tracing function
def trace_ray(ray, scene, lights, depth=0, max_depth=3):
    if depth > max_depth:
        return np.array([0, 0, 0])
    
    # Find closest intersection
    closest_t = float('inf')
    closest_obj = None
    
    for obj in scene:
        t = obj.intersect(ray)
        if t < closest_t:
            closest_t = t
            closest_obj = obj
    
    if closest_obj is None:
        return np.array([0.1, 0.1, 0.1])  # Background color
    
    # Calculate intersection point and normal
    point = ray.origin + closest_t * ray.direction
    
    if isinstance(closest_obj, Sphere):
        normal = normalize(point - closest_obj.center)
    else:  # Plane
        normal = closest_obj.normal
        # Apply checkerboard pattern if needed
        if closest_obj.checkerboard:
            if checkerboard_pattern(point) == 0:
                base_color = np.array([0.8, 0.8, 0.8])
            else:
                base_color = np.array([0.2, 0.2, 0.2])
            closest_obj.material.color = base_color
    
    # Calculate color with Phong illumination
    color = np.array([0.1, 0.1, 0.1])  # Ambient light
    
    for light in lights:
        # Calculate light direction
        light_dir = normalize(light.position - point)
        
        # Check for shadows
        shadow_ray = Ray(point + 0.001 * normal, light_dir)
        shadow = False
        
        for obj in scene:
            t = obj.intersect(shadow_ray)
            if t < float('inf') and t > 0.001:
                shadow = True
                break
        
        if not shadow:
            # Diffuse component
            diffuse = max(0, np.dot(normal, light_dir))
            diffuse_color = closest_obj.material.color * diffuse * light.color * light.intensity
            
            # Specular component
            view_dir = normalize(-ray.direction)
            reflect_dir = reflect(-light_dir, normal)
            specular = max(0, np.dot(view_dir, reflect_dir)) ** closest_obj.material.shininess
            specular_color = np.array([1, 1, 1]) * specular * closest_obj.material.specular * light.intensity
            
            color += diffuse_color + specular_color
    
    # Reflection
    if closest_obj.material.reflection > 0:
        reflect_dir = reflect(ray.direction, normal)
        reflect_ray = Ray(point + 0.001 * normal, reflect_dir)
        reflect_color = trace_ray(reflect_ray, scene, lights, depth + 1, max_depth)
        color = color * (1 - closest_obj.material.reflection) + reflect_color * closest_obj.material.reflection
    
    return np.clip(color, 0, 1)

# Render function
def render(scene, lights, width, height):
    image = np.zeros((height, width, 3))
    
    # Camera setup
    camera_pos = np.array([0, 0, 0])
    screen_z = -1
    screen_width = 2
    screen_height = 2 * height / width
    
    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            screen_x = (x / width - 0.5) * screen_width
            screen_y = -(y / height - 0.5) * screen_height  # Negative for correct orientation
            
            ray_dir = normalize(np.array([screen_x, screen_y, screen_z]))
            ray = Ray(camera_pos, ray_dir)
            
            # Trace ray
            color = trace_ray(ray, scene, lights)
            image[y, x] = color
    
    return image

# Main function
def main():
    width, height = 800, 600
    scene, lights = create_scene()
    
    print("Rendering image...")
    image = render(scene, lights, width, height)
    
    # Convert to 8-bit and save
    image_8bit = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_8bit, 'RGB')
    img.save('deepseek_v31.png')
    print("Image saved as 'raytraced_scene.png'")

if __name__ == "__main__":
    main()