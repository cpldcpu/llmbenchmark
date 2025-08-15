import numpy as np
from PIL import Image
import math
import time
from multiprocessing import Pool, cpu_count

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
FOV = 60  # degrees
MAX_DEPTH = 3 # Maximum recursion depth for reflections
AMBIENT_LIGHT = 0.1
EPSILON = 1e-6 # Small value to avoid self-intersection

# --- Vector Utilities (using numpy) ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float64)
        self.direction = normalize(np.array(direction, dtype=np.float64))

# --- Material Class ---
class Material:
    def __init__(self, diffuse_color, reflectivity=0.0):
        self.diffuse_color = np.array(diffuse_color, dtype=np.float64) / 255.0 # Store as 0-1
        self.reflectivity = reflectivity

# --- Object Classes ---
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction) # Should be 1 if direction is normalized
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None # No intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # Return the closest positive intersection distance
        if t1 > EPSILON:
            return t1
        if t2 > EPSILON:
            return t2
        return None

    def normal_at(self, point):
        return normalize(point - self.center)

class Plane:
    def __init__(self, point, normal, material_even, material_odd=None):
        self.point = np.array(point, dtype=np.float64) # A point on the plane
        self.normal = normalize(np.array(normal, dtype=np.float64))
        self.material_even = material_even
        self.material_odd = material_odd if material_odd else material_even # Use even if odd not specified
        self.d = -np.dot(self.point, self.normal) # Plane equation constant: ax + by + cz + d = 0

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > EPSILON: # Avoid parallel rays or rays inside the plane
            t = -(np.dot(self.normal, ray.origin) + self.d) / denom
            if t > EPSILON:
                return t
        return None

    def normal_at(self, point):
        # Normal is constant for a plane
        return self.normal

    def get_material_at(self, point):
        # Create a checkerboard pattern
        scale = 2.0 # Size of the checker squares
        # Project point onto plane axes (find two orthogonal vectors on the plane)
        u_axis = normalize(np.cross(self.normal, np.array([1,0,0]) if abs(self.normal[0]) < 0.9 else np.array([0,1,0])))
        v_axis = normalize(np.cross(self.normal, u_axis))

        u = np.dot(point - self.point, u_axis) * scale
        v = np.dot(point - self.point, v_axis) * scale

        if (math.floor(u) + math.floor(v)) % 2 == 0:
            return self.material_even
        else:
            return self.material_odd

# --- Light Source Class ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64) / 255.0 # Store as 0-1
        self.intensity = intensity

# --- Scene Definition ---
# Materials
mat_red = Material(diffuse_color=[255, 50, 50], reflectivity=0.2)
mat_green = Material(diffuse_color=[50, 255, 50], reflectivity=0.3)
mat_blue = Material(diffuse_color=[50, 50, 255], reflectivity=0.4)
mat_yellow = Material(diffuse_color=[255, 255, 50], reflectivity=0.1)
mat_magenta = Material(diffuse_color=[255, 50, 255], reflectivity=0.5)
mat_cyan = Material(diffuse_color=[50, 255, 255], reflectivity=0.0)

mat_plane_white = Material(diffuse_color=[200, 200, 200], reflectivity=0.05)
mat_plane_black = Material(diffuse_color=[30, 30, 30], reflectivity=0.05)


# Objects
scene_objects = [
    Sphere(center=[-2, 0, -7], radius=1.0, material=mat_red),
    Sphere(center=[0, -0.25, -5], radius=0.75, material=mat_green),
    Sphere(center=[2.5, 0.5, -8], radius=1.5, material=mat_blue),
    Sphere(center=[0.5, 1.5, -4], radius=0.5, material=mat_yellow),
    Sphere(center=[-1.5, -0.5, -4], radius=0.5, material=mat_magenta),
    Sphere(center=[1, -0.7, -3.5], radius=0.3, material=mat_cyan),
    Plane(point=[0, -1, 0], normal=[0, 1, 0], material_even=mat_plane_white, material_odd=mat_plane_black) # Ground plane
]

# Lights
lights = [
    Light(position=[-10, 10, 0], color=[255, 255, 255], intensity=1.0),
    Light(position=[10, 5, -10], color=[255, 100, 100], intensity=0.8), # Reddish light
    Light(position=[0, 15, -5], color=[100, 100, 255], intensity=0.7),  # Bluish light
    Light(position=[5, 0, 0], color=[100, 255, 100], intensity=0.5),   # Greenish light
]

# --- Camera Setup ---
camera_pos = np.array([0, 1.5, 2], dtype=np.float64)
look_at = np.array([0, 0, -5], dtype=np.float64)
up_vector = np.array([0, 1, 0], dtype=np.float64)

# Calculate camera basis vectors
forward = normalize(look_at - camera_pos)
right = normalize(np.cross(forward, up_vector))
# Recompute up vector to ensure orthogonality
camera_up = normalize(np.cross(right, forward))

# Calculate viewport dimensions based on FOV
aspect_ratio = WIDTH / HEIGHT
fov_rad = math.radians(FOV)
viewport_height = 2.0 * math.tan(fov_rad / 2.0)
viewport_width = aspect_ratio * viewport_height

# Calculate vectors across viewport horizontal and vertical edges
viewport_u = right * viewport_width
viewport_v = -camera_up * viewport_height # Negative because image coords usually start top-left

# Calculate horizontal and vertical delta vectors between pixels
pixel_delta_u = viewport_u / WIDTH
pixel_delta_v = viewport_v / HEIGHT

# Calculate the location of the upper-left corner of the viewport
viewport_upper_left = camera_pos + forward - viewport_u / 2.0 - viewport_v / 2.0
# Calculate the location of the center of the upper-left pixel
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

# --- Raytracing Core ---
def find_nearest_intersection(ray, objects):
    min_distance = float('inf')
    hit_object = None
    for obj in objects:
        distance = obj.intersect(ray)
        if distance is not None and distance < min_distance:
            min_distance = distance
            hit_object = obj
    return hit_object, min_distance

def compute_shading(point, normal, material, scene_objects, lights, camera_pos):
    final_color = np.zeros(3, dtype=np.float64) # Start with black

    # Ambient component
    final_color += material.diffuse_color * AMBIENT_LIGHT

    for light in lights:
        light_dir = normalize(light.position - point)
        light_dist = np.linalg.norm(light.position - point)

        # Shadow check
        shadow_ray = Ray(point + normal * EPSILON, light_dir) # Start slightly off surface
        shadow_hit_object, shadow_dist = find_nearest_intersection(shadow_ray, scene_objects)

        # If no object is hit OR the hit object is farther than the light source
        if shadow_hit_object is None or shadow_dist > light_dist:
            # Diffuse component (Lambertian)
            diffuse_intensity = max(0.0, np.dot(normal, light_dir))
            diffuse_color = material.diffuse_color * light.color * diffuse_intensity * light.intensity
            final_color += diffuse_color

            # Specular component could be added here (e.g., Blinn-Phong)

    return np.clip(final_color, 0.0, 1.0) # Ensure color is within [0, 1] range

def trace_ray(ray, objects, lights, camera_pos, depth):
    if depth > MAX_DEPTH:
        return np.zeros(3) # Return black if max depth reached

    hit_object, distance = find_nearest_intersection(ray, objects)

    if hit_object is None:
        # No object hit, return background color (e.g., simple gradient or solid color)
        # Simple sky gradient (blueish top to lighter bottom)
        # unit_direction = normalize(ray.direction)
        # t = 0.5 * (unit_direction[1] + 1.0) # Map y-component (-1 to 1) to (0 to 1)
        # return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0]) # White to light blue
        return np.zeros(3) # Black background


    intersection_point = ray.origin + ray.direction * distance
    normal = hit_object.normal_at(intersection_point)

    # Get the correct material (important for checkerboard plane)
    if isinstance(hit_object, Plane):
        material = hit_object.get_material_at(intersection_point)
    else:
        material = hit_object.material

    # Adjust normal if ray hits from inside (though less likely with solid spheres)
    if np.dot(ray.direction, normal) > 0:
        normal = -normal

    # Calculate color from direct illumination
    direct_color = compute_shading(intersection_point, normal, material, objects, lights, camera_pos)

    # Reflection
    reflection_color = np.zeros(3)
    if material.reflectivity > 0:
        reflection_dir = normalize(ray.direction - 2 * np.dot(ray.direction, normal) * normal)
        reflection_ray = Ray(intersection_point + normal * EPSILON, reflection_dir) # Start slightly off surface
        reflection_color = trace_ray(reflection_ray, objects, lights, camera_pos, depth + 1)

    # Combine direct color and reflection color
    final_color = direct_color * (1 - material.reflectivity) + reflection_color * material.reflectivity

    return np.clip(final_color, 0.0, 1.0)

# --- Rendering Function for a Single Pixel ---
def render_pixel(args):
    j, i = args # Pixel coordinates (row, column)
    pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
    ray_direction = normalize(pixel_center - camera_pos)
    ray = Ray(camera_pos, ray_direction)
    color = trace_ray(ray, scene_objects, lights, camera_pos, 0)
    # Convert color from [0, 1] float to [0, 255] int
    return (j, i, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Raytracer ({WIDTH}x{HEIGHT})...")
    start_time = time.time()

    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    # Create a list of pixel coordinates to process
    pixel_coords = [(j, i) for j in range(HEIGHT) for i in range(WIDTH)]

    num_cores = cpu_count()
    print(f"Using {num_cores} cores for rendering.")

    # Use multiprocessing Pool
    with Pool(processes=num_cores) as pool:
        results = pool.map(render_pixel, pixel_coords)

    # Populate the image from results
    print("Processing results...")
    for j, i, color in results:
         # Need to access pixels via [i, j] because Pillow uses (width, height) coordinates
         pixels[i, j] = color


    end_time = time.time()
    print(f"Rendering finished in {end_time - start_time:.2f} seconds.")

    # Save the image
    output_filename = "raytraced_scene3.png"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")

    # Optionally show the image
    # image.show()