import numpy as np
from PIL import Image

# --- Vector Math Helpers ---
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def subtract_vectors(v1, v2):
    return np.array(v1) - np.array(v2)

def add_vectors(v1, v2):
    return np.array(v1) + np.array(v2)

def dot_product(v1, v2):
    return np.dot(np.array(v1), np.array(v2))

def multiply_vector(v, scalar):
    return np.array(v) * scalar

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalize_vector(np.array(direction))

    def point_at_parameter(self, t):
        return self.origin + multiply_vector(self.direction, t)

# --- Sphere Class ---
class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

    def hit(self, ray, t_min, t_max):
        oc = subtract_vectors(ray.origin, self.center)
        a = dot_product(ray.direction, ray.direction)
        b = 2.0 * dot_product(oc, ray.direction)
        c = dot_product(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            sqrt_discriminant = np.sqrt(discriminant)
            t = (-b - sqrt_discriminant) / (2*a)
            if t_min < t < t_max:
                return t, self.color, normalize_vector(subtract_vectors(ray.point_at_parameter(t), self.center))
            t = (-b + sqrt_discriminant) / (2*a)
            if t_min < t < t_max:
                return t, self.color, normalize_vector(subtract_vectors(ray.point_at_parameter(t), self.center))
        return None

# --- Light Class ---
class Light:
    def __init__(self, position, color):
        self.position = np.array(position)
        self.color = np.array(color)

# --- Scene Class ---
class Scene:
    def __init__(self, objects, lights, ambient_light):
        self.objects = objects
        self.lights = lights
        self.ambient_light = np.array(ambient_light)

    def trace_ray(self, ray, t_min=0.001, t_max=float('inf')):
        closest_hit = None
        closest_t = t_max

        for obj in self.objects:
            hit_result = obj.hit(ray, t_min, closest_t)
            if hit_result:
                t, color, normal = hit_result
                closest_t = t
                closest_hit = (color, normal, ray.point_at_parameter(t))

        if closest_hit:
            hit_color, normal, hit_point = closest_hit
            final_color = self.ambient_light * hit_color  # Start with ambient
            for light in self.lights:
                light_direction = normalize_vector(subtract_vectors(light.position, hit_point))
                diffuse_intensity = max(0, dot_product(normal, light_direction))
                final_color += diffuse_intensity * light.color * hit_color
            return np.clip(final_color, 0, 1) # Ensure color is in [0, 1] range
        else:
            return np.array([0, 0, 0]) # Background color (black)

# --- Camera Setup ---
def setup_camera(look_from, look_at, vup, vfov_degrees, aspect_ratio):
    lens_radius = 0  # No lens radius for simple camera
    vfov = np.radians(vfov_degrees)
    h = np.tan(vfov/2)
    viewport_height = 2.0 * h
    viewport_width = aspect_ratio * viewport_height
    w = normalize_vector(subtract_vectors(look_from, look_at))
    u = normalize_vector(np.cross(vup, w))
    v = np.cross(w, u)

    origin = np.array(look_from)
    horizontal = multiply_vector(u, viewport_width)
    vertical = multiply_vector(v, viewport_height)
    lower_left_corner = origin - horizontal/2 - vertical/2 - w

    return origin, horizontal, vertical, lower_left_corner

# --- Rendering Function ---
def render_scene(scene, image_width, image_height, samples_per_pixel=1):
    image = np.zeros((image_height, image_width, 3))
    aspect_ratio = image_width / image_height

    # Camera Setup
    look_from = [3, 3, 3]
    look_at = [0, 0, -1]
    vup = [0, 1, 0]
    vfov = 45.0
    camera_origin, horizontal, vertical, lower_left_corner = setup_camera(look_from, look_at, vup, vfov, aspect_ratio)

    for y in range(image_height):
        for x in range(image_width):
            pixel_color = np.array([0.0, 0.0, 0.0])
            for _ in range(samples_per_pixel): # Simple anti-aliasing (averaging)
                u = (x + np.random.rand()) / (image_width - 1)
                v = (image_height - 1 - y + np.random.rand()) / (image_height - 1) # Flip y for image coordinates
                ray_origin = camera_origin
                ray_direction = lower_left_corner + multiply_vector(horizontal, u) + multiply_vector(vertical, v) - camera_origin
                ray = Ray(ray_origin, ray_direction)
                pixel_color += scene.trace_ray(ray)
            image[y, x] = pixel_color / samples_per_pixel # Average color

    return (image * 255).astype(np.uint8) # Convert to 8-bit RGB

# --- Scene Definition ---
def create_interesting_scene():
    spheres = [
        Sphere([0, 0, -1], 0.5, [0.8, 0.3, 0.3]),       # Red sphere
        Sphere([0, -100.5, -1], 100, [0.8, 0.8, 0.0]),  # Yellow ground plane (large sphere)
        Sphere([1, 0, -1], 0.5, [0.3, 0.8, 0.3]),       # Green sphere
        Sphere([-1, 0, -1], 0.5, [0.3, 0.3, 0.8]),      # Blue sphere
        Sphere([-0.5, 1, -1.5], 0.3, [0.8, 0.8, 0.8])  # White small sphere
    ]

    lights = [
        Light([2, 3, 1], [1, 0, 0]),    # Red light
        Light([-2, 3, 1], [0, 1, 0]),   # Green light
        Light([0, 5, 3], [0, 0, 1]),    # Blue light
        Light([0, 1, -3], [1, 1, 1])   # White light behind
    ]

    ambient_light = [0.1, 0.1, 0.1] # Dim ambient light

    return Scene(spheres, lights, ambient_light)

# --- Main Rendering ---
if __name__ == "__main__":
    image_width = 800
    image_height = 600
    scene = create_interesting_scene()
    rendered_image = render_scene(scene, image_width, image_height, samples_per_pixel=4) # Increased samples for smoother image

    img = Image.fromarray(rendered_image)
    img.save("colorful_scene.png")
    print("Image saved as colorful_scene.png")