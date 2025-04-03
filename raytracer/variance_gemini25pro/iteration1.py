import numpy as np
from PIL import Image
import math
import time

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 4  # Maximum recursion depth for reflections
SHADOW_BIAS = 1e-4 # Offset for shadow rays to avoid self-intersection

# --- Vector Operations ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(incident, normal):
    return incident - 2 * np.dot(incident, normal) * normal

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float64)
        self.direction = normalize(np.array(direction, dtype=np.float64))

# --- Material Class ---
class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.9, reflection=0.3, shininess=200):
        self.color = np.array(color, dtype=np.float64) # Base color (RGB, 0-1)
        self.ambient = ambient         # Ambient reflection coefficient
        self.diffuse = diffuse         # Diffuse reflection coefficient
        self.specular = specular       # Specular reflection coefficient
        self.reflection = reflection   # Mirror reflection coefficient (0-1)
        self.shininess = shininess     # Specular highlight shininess

# --- Light Class ---
class Light:
    def __init__(self, position, color):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64) # Light color/intensity (RGB, 0-1 or higher)

# --- Geometric Objects (Base and Implementations) ---
class SceneObject:
    def __init__(self, material):
        self.material = material

    def intersect(self, ray):
        raise NotImplementedError("Intersect method must be implemented by subclasses")

    def normal_at(self, point):
        raise NotImplementedError("NormalAt method must be implemented by subclasses")

class Sphere(SceneObject):
    def __init__(self, center, radius, material):
        super().__init__(material)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)

    def intersect(self, ray):
        # Ray-sphere intersection formula (solve quadratic equation for t)
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction) # Should be 1 if direction is normalized
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None  # No intersection
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2.0 * a)
            t2 = (-b + sqrt_discriminant) / (2.0 * a)

            if t1 > SHADOW_BIAS:  # Return smallest positive t
                 return t1
            if t2 > SHADOW_BIAS:
                 return t2
            return None # Both intersections are behind the ray origin

    def normal_at(self, point):
        return normalize(point - self.center)

class Plane(SceneObject):
    def __init__(self, point, normal, material):
        super().__init__(material)
        self.point = np.array(point, dtype=np.float64) # A point on the plane
        self.normal = normalize(np.array(normal, dtype=np.float64))

    def intersect(self, ray):
        # Ray-plane intersection formula
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > 1e-6: # Avoid division by zero (ray parallel to plane)
            p0l0 = self.point - ray.origin
            t = np.dot(p0l0, self.normal) / denom
            if t > SHADOW_BIAS: # Intersection must be in front of the ray
                return t
        return None

    def normal_at(self, point):
        # Normal is constant for a plane
        return self.normal

# --- Scene Class ---
class Scene:
     def __init__(self, objects, lights, ambient_color=np.array([0.1, 0.1, 0.1])):
         self.objects = objects
         self.lights = lights
         self.ambient_color = np.array(ambient_color, dtype=np.float64)

# --- Ray Tracing Core ---
def find_nearest_intersection(ray, scene):
    min_distance = float('inf')
    hit_object = None
    for obj in scene.objects:
        distance = obj.intersect(ray)
        if distance is not None and distance < min_distance:
            min_distance = distance
            hit_object = obj
    if hit_object:
        return hit_object, min_distance
    else:
        return None, None

def trace_ray(ray, scene, depth):
    if depth > MAX_DEPTH:
        return np.array([0, 0, 0])  # Return black if max depth reached

    hit_object, distance = find_nearest_intersection(ray, scene)

    if hit_object is None:
        # No intersection, return background color (e.g., a simple gradient or solid color)
        # Simple sky gradient:
        unit_dir = ray.direction
        t = 0.5 * (unit_dir[1] + 1.0) # Based on Y component
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0]) # White to light blue

    # Intersection found
    intersection_point = ray.origin + ray.direction * distance
    normal = hit_object.normal_at(intersection_point)
    view_dir = normalize(-ray.direction)
    material = hit_object.material

    # --- Lighting Calculation (Phong Model) ---
    final_color = np.zeros(3)

    # Ambient component
    final_color += material.color * material.ambient * scene.ambient_color

    # Diffuse and Specular components (for each light)
    for light in scene.lights:
        light_dir = normalize(light.position - intersection_point)

        # Shadow check
        shadow_ray_origin = intersection_point + normal * SHADOW_BIAS # Offset origin
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        shadow_hit_obj, shadow_dist = find_nearest_intersection(shadow_ray, scene)

        # Check if the path to the light is blocked
        light_distance = np.linalg.norm(light.position - intersection_point)
        in_shadow = shadow_hit_obj is not None and shadow_dist < light_distance

        if not in_shadow:
            # Diffuse (Lambertian)
            diffuse_intensity = max(0.0, np.dot(normal, light_dir))
            final_color += material.color * material.diffuse * diffuse_intensity * light.color

            # Specular (Phong)
            reflect_dir = reflect(-light_dir, normal)
            specular_intensity = max(0.0, np.dot(reflect_dir, view_dir)) ** material.shininess
            final_color += material.specular * specular_intensity * light.color

    # --- Reflection ---
    if material.reflection > 0:
        reflect_dir = reflect(ray.direction, normal)
        reflect_ray_origin = intersection_point + normal * SHADOW_BIAS # Offset origin
        reflect_ray = Ray(reflect_ray_origin, reflect_dir)
        reflection_color = trace_ray(reflect_ray, scene, depth + 1)
        final_color += material.reflection * reflection_color

    return np.clip(final_color, 0, 1) # Clamp color values to [0, 1]

# --- Rendering Function ---
def render(scene, camera_pos, camera_lookat, camera_up, fov_degrees, width, height):
    aspect_ratio = width / height
    fov_rad = math.radians(fov_degrees)
    viewport_height = 2 * math.tan(fov_rad / 2)
    viewport_width = aspect_ratio * viewport_height

    # Camera coordinate system
    w = normalize(camera_pos - camera_lookat) # Forward (negative Z)
    u = normalize(np.cross(camera_up, w))     # Right (X)
    v = np.cross(w, u)                        # Up (Y)

    origin = np.array(camera_pos, dtype=np.float64)
    horizontal = viewport_width * u
    vertical = viewport_height * v
    lower_left_corner = origin - horizontal/2 - vertical/2 - w

    # Create image buffer
    image = np.zeros((height, width, 3), dtype=np.float64)

    start_time = time.time()
    print(f"Rendering {width}x{height} image...")

    for y in range(height):
        # Print progress
        if y % 50 == 0 and y > 0:
             elapsed = time.time() - start_time
             print(f"  Scanline {y}/{height} ({elapsed:.2f}s elapsed)")

        for x in range(width):
            # Calculate ray direction through pixel (x, y)
            u_param = x / (width - 1)
            v_param = (height - 1 - y) / (height - 1) # Invert Y for image coordinates
            ray_direction = lower_left_corner + u_param * horizontal + v_param * vertical - origin
            primary_ray = Ray(origin, ray_direction)

            # Trace the ray and get the color
            pixel_color = trace_ray(primary_ray, scene, 0)
            image[y, x] = pixel_color

    end_time = time.time()
    print(f"Rendering finished in {end_time - start_time:.2f} seconds.")
    return image

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define Materials ---
    mat_red_shiny = Material(color=[1, 0.1, 0.1], ambient=0.1, diffuse=0.8, specular=1.0, reflection=0.3, shininess=250)
    mat_blue_matte = Material(color=[0.1, 0.1, 1], ambient=0.1, diffuse=0.9, specular=0.2, reflection=0.1, shininess=50)
    mat_green_reflective = Material(color=[0.1, 1, 0.1], ambient=0.05, diffuse=0.6, specular=0.8, reflection=0.6, shininess=150)
    mat_purple = Material(color=[0.6, 0.1, 0.8], ambient=0.1, diffuse=0.9, specular=0.5, reflection=0.2, shininess=100)
    mat_orange = Material(color=[1.0, 0.5, 0.1], ambient=0.1, diffuse=0.9, specular=0.5, reflection=0.1, shininess=80)
    mat_floor = Material(color=[0.8, 0.8, 0.8], ambient=0.05, diffuse=0.5, specular=0.9, reflection=0.7, shininess=300) # Reflective floor

    # --- Define Objects ---
    objects = [
        Sphere(center=[0, -0.5, -3], radius=1.0, material=mat_red_shiny),
        Sphere(center=[-2.5, 0, -4], radius=0.8, material=mat_blue_matte),
        Sphere(center=[2.5, 0.5, -4.5], radius=1.2, material=mat_green_reflective),
        Sphere(center=[0.5, 1.5, -2.5], radius=0.5, material=mat_purple),
        Sphere(center=[-1.0, 0.8, -2.0], radius=0.3, material=mat_orange),
        Plane(point=[0, -1.5, 0], normal=[0, 1, 0], material=mat_floor) # Ground plane at y=-1.5
    ]

    # --- Define Lights ---
    lights = [
        Light(position=[-5, 5, -1], color=[1.0, 1.0, 1.0]), # White light upper left
        Light(position=[5, 3, -1], color=[0.8, 0.3, 0.3]), # Reddish light upper right
        Light(position=[0, 8, -3], color=[0.3, 0.8, 0.3]), # Greenish light top middle
        Light(position=[2, 1, 1], color=[0.3, 0.3, 0.9]),  # Bluish light front right
        Light(position=[-2, 0.5, 1], color=[0.9, 0.9, 0.2]) # Yellowish light front left
    ]

    # --- Define Scene ---
    scene = Scene(objects, lights, ambient_color=[0.05, 0.05, 0.07]) # Slightly blue ambient

    # --- Camera Setup ---
    camera_pos = np.array([0, 1.5, 2.5])
    camera_lookat = np.array([0, -0.2, -3]) # Look slightly down towards the spheres
    camera_up = np.array([0, 1, 0])
    fov = 60 # Field of View in degrees

    # --- Render the Scene ---
    image_data = render(scene, camera_pos, camera_lookat, camera_up, fov, WIDTH, HEIGHT)

    # --- Save the Image ---
    # Convert float64[0,1] to uint8[0,255]
    image_data_uint8 = (image_data * 255).astype(np.uint8)
    img = Image.fromarray(image_data_uint8, 'RGB')
    output_filename = "raytraced_scene1.png"
    img.save(output_filename)
    print(f"Image saved as {output_filename}")