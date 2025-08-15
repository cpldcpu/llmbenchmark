import numpy as np
from PIL import Image
from tqdm import tqdm
import math

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 3  # Max recursion depth for reflections
EPSILON = 1e-4 # To avoid self-intersection / shadow acne

# --- Helper Vector Functions ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(incident, normal):
    return incident - 2 * np.dot(incident, normal) * normal

# --- Ray ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))

# --- Material ---
class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.7, specular=0.3, shininess=100, reflection=0.0):
        self.color = np.array(color, dtype=float) # Base color [R, G, B] 0-1
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection # 0 for no reflection, 1 for perfect mirror

# --- Objects ---
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray):
        # Ray-sphere intersection
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction) # Should be 1 if direction is normalized
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None, None # No intersection
        else:
            t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t1 > EPSILON: # Prefer smallest positive t
                return t1, self
            elif t2 > EPSILON:
                return t2, self
            return None, None

    def normal_at(self, point):
        return normalize(point - self.center)

class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point, dtype=float) # A point on the plane
        self.normal = normalize(np.array(normal, dtype=float))
        self.material = material

    def intersect(self, ray):
        # Ray-plane intersection
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > EPSILON: # Avoid division by zero (ray parallel to plane)
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t > EPSILON:
                return t, self
        return None, None

    def normal_at(self, point):
        return self.normal # Normal is constant for a plane

# --- Light ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float) # [R, G, B] 0-1
        self.intensity = intensity

# --- Scene ---
def setup_scene():
    objects = [
        # Ground Plane (reflective)
        Plane([0, -1, 0], [0, 1, 0], Material(color=[0.4, 0.4, 0.4], ambient=0.1, diffuse=0.8, specular=0.2, shininess=50, reflection=0.3)),

        # Spheres
        Sphere([-1.5, 0, -5], 1, Material(color=[1, 0, 0], ambient=0.1, diffuse=0.9, specular=0.5, shininess=100, reflection=0.1)), # Red
        Sphere([0, 0, -4], 1, Material(color=[0, 1, 0], ambient=0.1, diffuse=0.9, specular=0.8, shininess=200, reflection=0.2)), # Green
        Sphere([1.5, 0, -3], 1, Material(color=[0, 0, 1], ambient=0.1, diffuse=0.9, specular=0.3, shininess=50, reflection=0.05)),  # Blue

        Sphere([-0.5, 1.2, -3.5], 0.5, Material(color=[1, 1, 0], ambient=0.1, diffuse=0.7, specular=0.9, shininess=500, reflection=0.4)), # Yellow, reflective
        Sphere([0.8, -0.5, -2], 0.5, Material(color=[1, 0, 1], ambient=0.1, diffuse=0.7, specular=0.6, shininess=150, reflection=0.1)), # Magenta
    ]

    lights = [
        Light([5, 5, 0], [1, 1, 1], intensity=0.8),      # White light from right-top-front
        Light([-5, 3, -2], [0.8, 0.2, 0.2], intensity=0.6), # Reddish light from left-top-back
        Light([0, 10, -5], [0.2, 0.8, 0.2], intensity=0.5), # Greenish light from top
        Light([2, 2, 5], [0.3, 0.3, 1.0], intensity=0.7),   # Bluish light from front-right (for highlights)
        Light([-3, 1, 2], [1.0, 0.5, 0.0], intensity=0.4) # Orange light
    ]
    return objects, lights

# --- Ray Tracing Core ---
def find_nearest_intersection(ray, objects):
    min_t = float('inf')
    intersected_object = None
    for obj in objects:
        t, obj_hit = obj.intersect(ray)
        if t is not None and t < min_t:
            min_t = t
            intersected_object = obj_hit
    if intersected_object:
        return min_t, intersected_object
    return None, None

def trace_ray(ray, objects, lights, depth):
    if depth > MAX_DEPTH:
        return np.array([0, 0, 0]) # Background color (black) if max depth reached

    t, obj = find_nearest_intersection(ray, objects)

    if obj is None:
        return np.array([0.1, 0.1, 0.2]) # Background color (dark blueish)

    # Hit point and normal
    hit_point = ray.origin + t * ray.direction
    normal = obj.normal_at(hit_point)
    view_dir = -ray.direction # Direction from hit point to camera

    # Lighting calculation
    color = np.array([0, 0, 0], dtype=float)
    
    # Ambient component (once per object)
    color += obj.material.color * obj.material.ambient

    for light in lights:
        light_dir = normalize(light.position - hit_point)
        
        # Shadow check
        shadow_ray_origin = hit_point + normal * EPSILON # Offset to avoid self-shadowing
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        shadow_t, shadow_obj = find_nearest_intersection(shadow_ray, objects)
        
        # Check if shadow ray hits something *before* the light
        light_distance = np.linalg.norm(light.position - hit_point)
        in_shadow = shadow_obj is not None and shadow_t < light_distance

        if not in_shadow:
            # Diffuse (Lambertian)
            diffuse_intensity = max(0, np.dot(normal, light_dir))
            diffuse_color = obj.material.color * obj.material.diffuse * diffuse_intensity * light.color * light.intensity
            color += diffuse_color

            # Specular (Blinn-Phong)
            # halfway_dir = normalize(light_dir + view_dir)
            # specular_intensity = max(0, np.dot(normal, halfway_dir)) ** obj.material.shininess
            
            # Specular (Phong - using reflection of light vector)
            reflected_light_dir = reflect(-light_dir, normal)
            specular_intensity = max(0, np.dot(view_dir, reflected_light_dir)) ** obj.material.shininess

            specular_color = light.color * obj.material.specular * specular_intensity * light.intensity
            color += specular_color
            
    # Reflection
    if obj.material.reflection > 0:
        reflection_ray_dir = reflect(ray.direction, normal)
        reflection_ray_origin = hit_point + normal * EPSILON # Offset
        reflection_ray = Ray(reflection_ray_origin, reflection_ray_dir)
        reflected_color = trace_ray(reflection_ray, objects, lights, depth + 1)
        color = color * (1 - obj.material.reflection) + reflected_color * obj.material.reflection

    return np.clip(color, 0, 1)


# --- Main Rendering Loop ---
def render(width, height, objects, lights):
    image = np.zeros((height, width, 3), dtype=float)
    
    # Camera setup
    aspect_ratio = float(width) / height
    fov_angle = math.pi / 3.0 # 60 degrees
    
    # Simple camera at origin looking along -Z
    camera_origin = np.array([0, 1, 5], dtype=float) # Move camera back and up a bit

    print("Rendering...")
    for j in tqdm(range(height)):
        for i in range(width):
            # Screen space to NDC space (Normalized Device Coordinates)
            # x from -1 to 1, y from -1 to 1 (approximately, depends on aspect ratio)
            # Pixel centers: (i + 0.5) / width, (j + 0.5) / height
            # Remap to [-1, 1] for x, and corrected for y
            # NDC x: (2 * (i + 0.5) / width - 1)
            # NDC y: (1 - 2 * (j + 0.5) / height) -> Y is often inverted in screen coords
            
            px = (2 * ((i + 0.5) / width) - 1) * aspect_ratio * math.tan(fov_angle / 2)
            py = (1 - 2 * ((j + 0.5) / height)) * math.tan(fov_angle / 2)
            
            # Ray direction assuming camera looks along -Z
            ray_direction = normalize(np.array([px, py, -1.0]))
            
            # If camera is not at origin or looking along -Z, transform ray_direction
            # For now, camera is at camera_origin, looking towards (0,0,-1) relative to its position
            # A proper camera would use a view matrix transformation.
            # Simple 'look_at' like transformation:
            look_at = np.array([0, 0, -1]) # Point camera is looking at
            cam_dir = normalize(look_at - camera_origin)
            
            # This is a simplified camera for now, let's just use origin for ray.
            # A full camera model involves view, and projection matrices.
            # For a camera at `camera_origin` looking at `target` with `up` vector:
            # forward = normalize(target - camera_origin)
            # right = normalize(np.cross(forward, up)) # up should be normalized
            # true_up = np.cross(right, forward)
            # ray_direction_world = normalize(px * right + py * true_up + forward)
            # For this setup, we'll assume a fixed camera orientation and just adjust origin.
            # The ray_direction calculated above is in camera space.
            # If camera is only translated, direction remains same in world space.
            # If camera is rotated, we need to rotate ray_direction.

            # For this example, let's assume the [px, py, -1] is already in a coordinate system
            # where the camera is at `camera_origin` and looking down the negative Z axis
            # of that system. If the camera itself is oriented, this direction needs to be
            # transformed by the camera's orientation matrix.
            # For simplicity, let's consider the px, py, -1 as world direction relative to where camera is pointing.
            # The current [px, py, -1] is a direction from origin for a camera at origin.
            # If camera is at camera_origin, the ray direction stays the same.
            
            ray = Ray(camera_origin, ray_direction)
            pixel_color = trace_ray(ray, objects, lights, 0)
            image[j, i] = pixel_color

    # Convert to 8-bit color range and save
    image_uint8 = (image * 255).astype(np.uint8)
    img_pil = Image.fromarray(image_uint8, 'RGB')
    img_pil.save("raytraced_scene4.png")
    print("Scene saved to raytraced_scene.png")

if __name__ == '__main__':
    scene_objects, scene_lights = setup_scene()
    render(WIDTH, HEIGHT, scene_objects, scene_lights)