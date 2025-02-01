import numpy as np
from PIL import Image

# Define constants
WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3

# Define colors
BLACK = np.array([0, 0, 0])
WHITE = np.array([255, 255, 255])
RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
BLUE = np.array([0, 0, 255])
YELLOW = np.array([255, 255, 0])
PURPLE = np.array([255, 0, 255])
CYAN = np.array([0, 255, 255])

# Sphere class for our objects
class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant > 0:
            t = (-b - np.sqrt(discriminant)) / (2.0 * a)
            if t > 0:
                return t
        return None

# Light source
class Light:
    def __init__(self, position, color):
        self.position = np.array(position)
        self.color = np.array(color)

# Scene setup
spheres = [
    Sphere([0, -1, 3], 1, RED),
    Sphere([2, 0, 4], 1, GREEN),
    Sphere([-2, 0, 4], 1, BLUE)
]

lights = [
    Light([-5, 5, -10], YELLOW),
    Light([5, 5, -10], PURPLE),
    Light([0, 5, -10], CYAN)
]

def cast_ray(ray_origin, ray_direction, depth=0):
    if depth > MAX_DEPTH:
        return BLACK
    
    nearest_t = float('inf')
    nearest_sphere = None
    
    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_direction)
        if t and t < nearest_t:
            nearest_t = t
            nearest_sphere = sphere
    
    if nearest_sphere:
        hit_point = ray_origin + nearest_t * ray_direction
        normal = (hit_point - nearest_sphere.center) / nearest_sphere.radius
        local_color = np.zeros(3)
        
        # Ambient light
        local_color = 0.1 * nearest_sphere.color
        
        for light in lights:
            # Light vector
            light_dir = light.position - hit_point
            light_distance = np.linalg.norm(light_dir)
            light_dir /= light_distance
            
            # Shadow check
            shadow = False
            for shadow_sphere in spheres:
                if shadow_sphere != nearest_sphere:
                    t = shadow_sphere.intersect(hit_point + 0.001 * normal, light_dir)
                    if t and t < light_distance:
                        shadow = True
                        break
            
            if not shadow:
                # Diffuse lighting
                diffuse_intensity = max(0, np.dot(normal, light_dir))
                local_color += nearest_sphere.color * light.color * diffuse_intensity / 255.0
            
            # Reflection
            reflection_vector = ray_direction - 2 * np.dot(ray_direction, normal) * normal
            reflected_color = cast_ray(hit_point + 0.001 * reflection_vector, reflection_vector, depth + 1)
            local_color += 0.5 * reflected_color  # Reflective value
        
        return np.clip(local_color, 0, 255)
    
    return BLACK

# Camera setup
origin = np.array([0, 0, 0])
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

for y in range(HEIGHT):
    for x in range(WIDTH):
        # Convert pixel coordinate to world space
        ray_direction = np.array([
            (x + 0.5 - WIDTH / 2) / (WIDTH / 2),
            -(y + 0.5 - HEIGHT / 2) / (HEIGHT / 2),
            1
        ])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize
        color = cast_ray(origin, ray_direction)
        image[y, x] = color

# Save the image
Image.fromarray(image.astype(np.uint8)).save('raytraced_scene.png')