import numpy as np
from PIL import Image
import math
import time

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
OUTPUT_FILENAME = "raytraced_scene.png"
MAX_DEPTH = 3         # Maximum reflection recursion depth
EPSILON = 1e-6       # Small value to avoid self-intersection

# --- Vector Operations (using numpy) ---
def normalize(v):
    """Normalize a numpy vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(incident, normal):
    """Calculate the reflection vector."""
    return incident - 2 * np.dot(incident, normal) * normal

# --- Scene Elements ---
class Material:
    """Defines the surface properties of an object."""
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.9, shininess=200, reflection=0.3):
        self.color = np.array(color) # Base color (RGB, 0-1)
        self.ambient = ambient       # Ambient light coefficient
        self.diffuse = diffuse       # Diffuse reflection coefficient
        self.specular = specular     # Specular reflection coefficient
        self.shininess = shininess   # Controls the size of the specular highlight
        self.reflection = reflection # Reflection coefficient (0=none, 1=perfect mirror)

class Sphere:
    """Represents a sphere in the scene."""
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_direction):
        """Check if a ray intersects the sphere.
        Returns distance 't' if intersects, None otherwise.
        Uses the quadratic formula approach.
        """
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction) # Should be 1 if direction is normalized
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None  # No intersection
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2.0*a)
            t2 = (-b + sqrt_discriminant) / (2.0*a)

            # Return the smallest positive t value (closest intersection in front of ray origin)
            if t1 > EPSILON and t2 > EPSILON:
                return min(t1, t2)
            elif t1 > EPSILON:
                return t1
            elif t2 > EPSILON:
                return t2
            else:
                return None # Intersection is behind the ray origin

    def normal_at(self, point):
        """Calculate the surface normal vector at a given point on the sphere."""
        return normalize(point - self.center)

class Light:
    """Represents a point light source."""
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color) * intensity # Incorporate intensity into color

# --- Scene Setup ---
scene_objects = [
    # Central reflective sphere
    Sphere([-0.5, 0, -3], 0.6, Material(color=[0.8, 0.3, 0.3], reflection=0.6, specular=1.0, shininess=300)),
    # Green sphere
    Sphere([1.5, -0.2, -4], 0.8, Material(color=[0.2, 0.7, 0.2], reflection=0.1, diffuse=0.7, specular=0.5, shininess=100)),
    # Blue sphere
    Sphere([-1.5, -0.5, -5], 0.5, Material(color=[0.3, 0.4, 0.9], reflection=0.4, diffuse=0.6, specular=0.8, shininess=150)),
    # Small yellow sphere
    Sphere([0.5, -0.8, -2.5], 0.3, Material(color=[0.9, 0.9, 0.2], reflection=0.0, diffuse=0.8, specular=0.3, shininess=50)),
    # Large "ground" sphere (approximates a plane)
    Sphere([0, -100.5, -1], 100, Material(color=[0.7, 0.7, 0.7], reflection=0.1, diffuse=0.8, specular=0.1, shininess=10)),
     # Metallic-looking sphere
    Sphere([2.5, 0.5, -3.5], 0.7, Material(color=[0.6, 0.6, 0.65], ambient=0.05, diffuse=0.3, specular=1.0, shininess=500, reflection=0.7)),
]

lights = [
    Light([-5, 5, 0], [1.0, 1.0, 1.0], intensity=0.8),   # White light from left/top/front
    Light([5, 3, -2], [0.2, 0.5, 1.0], intensity=0.7),   # Bluish light from right/top/back
    Light([0, -3, -1], [1.0, 0.3, 0.3], intensity=0.5),  # Reddish light from bottom/front
    Light([-2, 1, -6], [0.5, 1.0, 0.5], intensity=0.6),  # Greenish light from left/back
]

ambient_light_color = np.array([0.1, 0.1, 0.1]) # General ambient light in the scene

# --- Camera Setup ---
camera_position = np.array([0, 0.5, 1])
look_at = np.array([0, 0, -3])
up_vector = np.array([0, 1, 0])

# Calculate camera basis vectors
forward = normalize(look_at - camera_position)
right = normalize(np.cross(forward, up_vector))
camera_up = np.cross(right, forward) # Recalculate up to ensure orthogonality

# Screen dimensions based on FOV (approx 60 degrees vertical)
fov_rad = math.pi / 3.0 # 60 degrees
aspect_ratio = WIDTH / HEIGHT
screen_height = 2 * math.tan(fov_rad / 2.0)
screen_width = aspect_ratio * screen_height

# --- Raytracing Core ---

def find_nearest_intersection(ray_origin, ray_direction, objects):
    """Find the nearest object intersected by the ray."""
    nearest_object = None
    min_distance = float('inf')

    for obj in objects:
        distance = obj.intersect(ray_origin, ray_direction)
        if distance is not None and distance < min_distance:
            min_distance = distance
            nearest_object = obj

    return nearest_object, min_distance

def trace_ray(ray_origin, ray_direction, objects, lights, depth):
    """Trace a single ray and return the calculated color."""
    if depth > MAX_DEPTH:
        return np.array([0, 0, 0]) # Black background or recursion limit hit

    nearest_object, distance = find_nearest_intersection(ray_origin, ray_direction, objects)

    if nearest_object is None:
        return np.array([0, 0, 0]) # Background color (black)

    # Calculate intersection point and surface normal
    intersection_point = ray_origin + ray_direction * distance
    normal = nearest_object.normal_at(intersection_point)
    view_direction = normalize(-ray_direction) # Direction from intersection point to camera

    material = nearest_object.material
    final_color = np.zeros(3)

    # Ambient component
    final_color += material.color * material.ambient * ambient_light_color

    # Iterate through each light source for diffuse and specular
    for light in lights:
        light_direction = normalize(light.position - intersection_point)
        light_distance = np.linalg.norm(light.position - intersection_point)

        # Shadow check: Cast a ray from intersection point towards the light
        shadow_origin = intersection_point + normal * EPSILON # Offset slightly to avoid self-shadowing
        shadow_object, shadow_dist = find_nearest_intersection(shadow_origin, light_direction, objects)

        # If no shadow object or the shadow object is further than the light source
        if shadow_object is None or shadow_dist > light_distance:
            # Diffuse component (Lambertian)
            diffuse_intensity = max(0.0, np.dot(normal, light_direction))
            final_color += material.color * material.diffuse * light.color * diffuse_intensity

            # Specular component (Phong/Blinn-Phong)
            # Using Blinn-Phong (Halfway vector) for generally better highlights
            halfway_vector = normalize(light_direction + view_direction)
            specular_intensity = max(0.0, np.dot(normal, halfway_vector)) ** material.shininess
            final_color += material.specular * light.color * specular_intensity

    # Reflection component
    if material.reflection > 0:
        reflection_direction = reflect(ray_direction, normal)
        # Offset origin slightly along the normal to avoid self-intersection
        reflection_origin = intersection_point + normal * EPSILON
        reflection_color = trace_ray(reflection_origin, reflection_direction, objects, lights, depth + 1)
        final_color += reflection_color * material.reflection

    return np.clip(final_color, 0, 1) # Ensure color values stay within [0, 1]


# --- Rendering Loop ---
print(f"Starting raytracing ({WIDTH}x{HEIGHT})...")
start_time = time.time()

image = Image.new("RGB", (WIDTH, HEIGHT))
pixels = image.load()

for j in range(HEIGHT):
    for i in range(WIDTH):
        # Map pixel coordinates (i, j) to screen space coordinates (x, y)
        # Screen space is centered at (0,0), extends from -sw/2 to +sw/2 and -sh/2 to +sh/2
        x = (i + 0.5) / WIDTH   # Normalize i to [0, 1] then add 0.5 for pixel center
        y = (j + 0.5) / HEIGHT  # Normalize j to [0, 1] then add 0.5 for pixel center

        # Convert to range [-1, 1] for x and [1, -1] for y (image coordinates top-left is 0,0)
        screen_x = (2 * x - 1) * screen_width / 2
        screen_y = (1 - 2 * y) * screen_height / 2 # Y is inverted

        # Calculate ray direction through this pixel
        ray_direction = normalize(forward + right * screen_x + camera_up * screen_y)

        # Trace the ray
        color = trace_ray(camera_position, ray_direction, scene_objects, lights, 0)

        # Convert color from [0, 1] float to [0, 255] int and set pixel
        r, g, b = [int(c * 255) for c in color]
        pixels[i, j] = (r, g, b)

    # Print progress
    if (j + 1) % 50 == 0 or j == HEIGHT - 1:
        elapsed = time.time() - start_time
        print(f"  Row {j+1}/{HEIGHT} completed. Time elapsed: {elapsed:.2f}s")

# --- Save Image ---
image.save(OUTPUT_FILENAME)
end_time = time.time()
print(f"Rendering complete. Image saved as '{OUTPUT_FILENAME}'.")
print(f"Total time taken: {end_time - start_time:.2f} seconds.")