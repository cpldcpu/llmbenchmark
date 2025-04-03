import numpy as np
from PIL import Image
import math
import time

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 3  # Max reflection bounces (kept low for speed, not fully utilized in this version)
AMBIENT_LIGHT = 0.1 # Global ambient light factor

# --- Vector Math Helpers ---
def normalize(vector):
    """Normalizes a numpy vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector # Avoid division by zero
    return vector / norm

def reflect(vector, normal):
    """Calculates the reflection of a vector off a surface with a given normal."""
    return vector - 2 * np.dot(vector, normal) * normal

# --- Scene Objects ---
class Material:
    """Represents the material properties of an object."""
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.3, shininess=50):
        self.color = np.array(color) # Base color (RGB)
        self.ambient = ambient       # Ambient reflection coefficient
        self.diffuse = diffuse       # Diffuse reflection coefficient
        self.specular = specular     # Specular reflection coefficient
        self.shininess = shininess   # Controls the size of specular highlights

class Sphere:
    """Represents a sphere in the scene."""
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_direction):
        """Checks if a ray intersects the sphere.
           Returns distance to intersection or None.
        """
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None # No intersection
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2.0*a)
            t2 = (-b + sqrt_discriminant) / (2.0*a)

            # Return the smallest positive intersection distance
            if t1 >= 1e-4 and (t2 < 1e-4 or t1 < t2):
                 return t1
            if t2 >= 1e-4:
                 return t2
            return None # Intersection is behind the ray origin

    def normal_at(self, point):
        """Calculates the surface normal at a point on the sphere."""
        return normalize(point - self.center)

class Plane:
    """Represents an infinite plane in the scene."""
    def __init__(self, point, normal, material):
        self.point = np.array(point) # A point on the plane
        self.normal = normalize(np.array(normal))
        self.material = material

    def intersect(self, ray_origin, ray_direction):
        """Checks if a ray intersects the plane.
           Returns distance to intersection or None.
        """
        denom = np.dot(ray_direction, self.normal)
        # Use a small epsilon to avoid floating point issues and self-intersection
        if abs(denom) > 1e-6:
            t = np.dot(self.point - ray_origin, self.normal) / denom
            if t >= 1e-4: # Check if intersection is in front of the ray
                return t
        return None

    def normal_at(self, point):
        """Normal is constant for a plane."""
        return self.normal

class Light:
    """Represents a point light source."""
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color) * intensity

# --- Scene Definition ---
scene_objects = [
    # Spheres
    Sphere([-0.5, 0.1, -1.5], 0.6, Material(color=[0.8, 0.3, 0.3], specular=0.6, shininess=100)), # Reddish
    Sphere([0.7, -0.2, -1.0], 0.3, Material(color=[0.3, 0.8, 0.3], specular=0.8, shininess=200)), # Greenish
    Sphere([0.0, -0.1, -0.5], 0.2, Material(color=[0.3, 0.3, 0.8], specular=0.9, shininess=500)), # Bluish
    Sphere([-0.8, -0.4, -0.8], 0.1, Material(color=[0.8, 0.8, 0.3], specular=0.5, shininess=80)), # Yellowish Small
    Sphere([1.5, 0.5, -2.5], 1.0, Material(color=[0.5, 0.5, 0.5], specular=0.2, shininess=30)),  # Greyish Large

    # Ground Plane
    Plane([0, -0.5, 0], [0, 1, 0], Material(color=[0.7, 0.7, 0.7], diffuse=0.8, specular=0.1, shininess=20)) # Grey floor
]

lights = [
    Light([-2, 2.5, 1.0], color=[1.0, 0.5, 0.5], intensity=1.5), # Reddish light
    Light([1.5, 3.0, -1.5], color=[0.5, 1.0, 0.5], intensity=1.5), # Greenish light
    Light([0.5, 1.5, 2.0], color=[0.5, 0.5, 1.0], intensity=1.5), # Bluish light
    Light([0, 5, -1], color=[1.0, 1.0, 1.0], intensity=0.8),   # White overhead light (softer)
]

# --- Camera ---
camera_pos = np.array([0, 0.5, 2.5])  # Camera position slightly elevated and back
look_at = np.array([0, 0, -1])    # Point the camera is looking at
up_vector = np.array([0, 1, 0])     # World's up direction

# Camera basis vectors
forward = normalize(look_at - camera_pos)
right = normalize(np.cross(forward, up_vector))
# Recompute camera's actual up vector to ensure orthogonality
cam_up = normalize(np.cross(right, forward))

# Screen dimensions and field of view
aspect_ratio = WIDTH / HEIGHT
fov_rad = math.pi / 3.0 # 60 degrees field of view
screen_height_world = 2.0 * math.tan(fov_rad / 2.0)
screen_width_world = screen_height_world * aspect_ratio

# --- Raytracing Core ---

def find_nearest_intersection(ray_origin, ray_direction, objects):
    """Finds the closest object intersected by the ray."""
    min_distance = float('inf')
    nearest_object = None
    for obj in objects:
        distance = obj.intersect(ray_origin, ray_direction)
        if distance is not None and distance < min_distance:
            min_distance = distance
            nearest_object = obj
    return nearest_object, min_distance

def trace_ray(ray_origin, ray_direction, objects, lights, depth):
    """Traces a single ray and returns the calculated color."""
    nearest_object, distance = find_nearest_intersection(ray_origin, ray_direction, objects)

    if nearest_object is None:
        # No intersection, return background color (e.g., a simple gradient or black)
        # Simple sky gradient based on y-direction
        # sky_color_top = np.array([0.5, 0.7, 1.0]) # Light blue
        # sky_color_horizon = np.array([0.8, 0.8, 0.8]) # Greyish
        # t = 0.5 * (ray_direction[1] + 1.0) # Map y-direction (-1 to 1) to t (0 to 1)
        # return sky_color_horizon * (1.0 - t) + sky_color_top * t
        return np.array([0.0, 0.0, 0.0]) # Black background

    intersection_point = ray_origin + ray_direction * distance
    surface_normal = nearest_object.normal_at(intersection_point)
    material = nearest_object.material

    # Ensure normal points outwards from the surface relative to the ray
    if np.dot(ray_direction, surface_normal) > 0:
        surface_normal = -surface_normal

    final_color = np.zeros(3)

    # Ambient component (global illumination approximation)
    final_color += material.color * material.ambient * AMBIENT_LIGHT

    # Lighting calculations (Diffuse and Specular)
    view_dir = normalize(ray_origin - intersection_point) # Direction from intersection to camera

    for light in lights:
        light_dir = normalize(light.position - intersection_point)
        light_distance = np.linalg.norm(light.position - intersection_point)

        # Shadow check: cast a ray from intersection point towards the light
        shadow_ray_origin = intersection_point + surface_normal * 1e-4 # Offset to avoid self-intersection
        shadow_object, shadow_dist = find_nearest_intersection(shadow_ray_origin, light_dir, objects)

        # If no object blocks the light or the blocking object is further than the light itself
        if shadow_object is None or shadow_dist > light_distance:
            # Calculate Diffuse (Lambertian)
            diffuse_intensity = max(0, np.dot(surface_normal, light_dir))
            diffuse_color = material.color * light.color * material.diffuse * diffuse_intensity
            final_color += diffuse_color

            # Calculate Specular (Blinn-Phong)
            half_vector = normalize(light_dir + view_dir)
            specular_intensity = max(0, np.dot(surface_normal, half_vector)) ** material.shininess
            specular_color = light.color * material.specular * specular_intensity
            final_color += specular_color

    # Simple reflection (can be disabled by setting MAX_DEPTH low)
    # Note: This basic reflection isn't fully integrated with material properties yet
    # if depth < MAX_DEPTH:
    #     reflection_dir = reflect(ray_direction, surface_normal)
    #     reflection_origin = intersection_point + surface_normal * 1e-4 # Offset
    #     reflection_color = trace_ray(reflection_origin, reflection_dir, objects, lights, depth + 1)
    #     # Add reflection weighted by specular component (simplistic approach)
    #     final_color += reflection_color * material.specular

    return final_color

# --- Main Rendering Loop ---
print(f"Rendering {WIDTH}x{HEIGHT} image...")
start_time = time.time()

image = Image.new("RGB", (WIDTH, HEIGHT))
pixels = image.load()

for y in range(HEIGHT):
    for x in range(WIDTH):
        # Map pixel coordinates (0 to WIDTH-1, 0 to HEIGHT-1) to screen coordinates (-1 to 1)
        # Adjust y because image coordinates usually start from top-left, world from bottom-left
        screen_x = (2 * (x + 0.5) / WIDTH - 1.0) * screen_width_world / 2.0
        screen_y = (1.0 - 2 * (y + 0.5) / HEIGHT) * screen_height_world / 2.0

        # Calculate ray direction in world space
        ray_direction = normalize(forward + right * screen_x + cam_up * screen_y)

        # Trace the ray
        color = trace_ray(camera_pos, ray_direction, scene_objects, lights, 0)

        # Clamp color values to [0, 1] and convert to 8-bit RGB
        color = np.clip(color, 0, 1)
        rgb = tuple(int(c * 255) for c in color)
        pixels[x, y] = rgb

    # Progress indicator
    if (y + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  ...Row {y+1}/{HEIGHT} complete ({elapsed:.1f}s elapsed)")


end_time = time.time()
print(f"Rendering finished in {end_time - start_time:.2f} seconds.")

# --- Save Image ---
output_filename = "raytraced_scene_colorful_lights.png"
image.save(output_filename)
print(f"Image saved as {output_filename}")