import math
from PIL import Image
import time # To time the render

# --- Configuration ---
WIDTH = 800  # Increased from 400 to 800
HEIGHT = 600 # Increased from 300 to 600
MAX_DEPTH = 3
EPSILON = 1e-4 # Small value to avoid self-intersection (shadow acne)
BACKGROUND_COLOR = (0.1, 0.1, 0.2) # Dark blueish background

# --- Vector Math ---
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, Vec3): # Element-wise product (for colors)
            return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length_squared(self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    def length(self):
        return math.sqrt(self.length_squared())

    def normalize(self):
        l = self.length()
        if l == 0: return Vec3(0,0,0) # Avoid division by zero
        return Vec3(self.x / l, self.y / l, self.z / l)

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def clamp(self, min_val=0.0, max_val=1.0):
        return Vec3(
            max(min_val, min(self.x, max_val)),
            max(min_val, min(self.y, max_val)),
            max(min_val, min(self.z, max_val))
        )

# --- Ray ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

# --- Material ---
class Material:
    def __init__(self, diffuse_color, ambient_coeff=0.1, diffuse_coeff=0.7,
                 specular_coeff=0.2, shininess=32, reflection_coeff=0.0):
        self.diffuse_color = diffuse_color
        self.ambient_coeff = ambient_coeff
        self.diffuse_coeff = diffuse_coeff
        self.specular_coeff = specular_coeff
        self.shininess = shininess
        self.reflection_coeff = reflection_coeff

# --- Scene Objects ---
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction) # Should be 1 if direction is normalized
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)

        if t1 > EPSILON:
            t = t1
        elif t2 > EPSILON:
            t = t2
        else:
            return None # Intersection is behind the ray origin or too close

        hit_point = ray.origin + ray.direction * t
        normal = (hit_point - self.center).normalize()
        return t, hit_point, normal, self

class Plane:
    def __init__(self, point, normal, material):
        self.point = point # A point on the plane
        self.normal = normal.normalize()
        self.material = material

    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > EPSILON: # Ray is not parallel to the plane
            t = (self.point - ray.origin).dot(self.normal) / denom
            if t > EPSILON:
                hit_point = ray.origin + ray.direction * t
                # Ensure normal points towards the ray origin side for correct lighting
                normal_to_use = self.normal if self.normal.dot(ray.direction) < 0 else self.normal * -1
                return t, hit_point, normal_to_use, self
        return None

# --- Light Source ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# --- Scene Definition ---
def setup_scene():
    objects = []
    lights = []

    # Materials
    mat_red_shiny = Material(Vec3(1, 0.1, 0.1), reflection_coeff=0.2, shininess=100, specular_coeff=0.8)
    mat_green_matte = Material(Vec3(0.1, 0.8, 0.1), diffuse_coeff=0.9, specular_coeff=0.1, shininess=10)
    mat_blue_reflective = Material(Vec3(0.2, 0.2, 0.9), reflection_coeff=0.7, shininess=200, specular_coeff=0.9)
    mat_yellow_bright = Material(Vec3(0.9, 0.9, 0.2), diffuse_coeff=0.8, specular_coeff=0.5, shininess=50)
    mat_purple_velvet = Material(Vec3(0.6, 0.1, 0.8), reflection_coeff=0.05, diffuse_coeff=0.8, specular_coeff=0.2, shininess=20)
    mat_cyan_glassy = Material(Vec3(0.1, 0.7, 0.7), reflection_coeff=0.5, specular_coeff=0.7, shininess=150)
    mat_ground = Material(Vec3(0.5, 0.5, 0.5), diffuse_coeff=0.8, specular_coeff=0.1, shininess=5, reflection_coeff=0.1)

    # Objects
    objects.append(Sphere(Vec3(0, -1, -5), 1, mat_blue_reflective))
    objects.append(Sphere(Vec3(-2.5, 0, -6), 1.2, mat_red_shiny))
    objects.append(Sphere(Vec3(2.5, 0.5, -7), 1.5, mat_green_matte))
    objects.append(Sphere(Vec3(0.5, 1.5, -4), 0.8, mat_yellow_bright))
    objects.append(Sphere(Vec3(-1.5, -0.5, -3.5), 0.7, mat_purple_velvet))
    objects.append(Sphere(Vec3(1.8, -0.8, -4.5), 0.9, mat_cyan_glassy))

    # Ground Plane
    objects.append(Plane(Vec3(0, -2.2, 0), Vec3(0, 1, 0), mat_ground)) # y=-2.2, normal pointing up

    # Lights
    lights.append(Light(Vec3(-10, 10, 0), Vec3(1, 1, 1), intensity=0.8))  # White bright light from left-top
    lights.append(Light(Vec3(5, 5, -2), Vec3(1, 0.5, 0.5), intensity=0.7)) # Reddish light from right-front
    lights.append(Light(Vec3(0, 8, -10), Vec3(0.5, 0.5, 1), intensity=0.6)) # Bluish light from top-back
    lights.append(Light(Vec3(-3, 2, 2), Vec3(0.5, 1, 0.5), intensity=0.5)) # Greenish light from front-left
    lights.append(Light(Vec3(8, -5, -5), Vec3(1,0,1), intensity=0.4)) # Magenta light from right-bottom-ish (for ground highlights)

    # Ambient light (global)
    ambient_light_color = Vec3(0.1, 0.1, 0.1) # Soft global ambient

    return objects, lights, ambient_light_color

# --- Ray Tracing Core ---
def find_closest_intersection(ray, objects):
    closest_hit = None
    min_t = float('inf')

    for obj in objects:
        hit_info = obj.intersect(ray)
        if hit_info:
            t, hit_point, normal, intersected_obj = hit_info
            if t < min_t:
                min_t = t
                closest_hit = (t, hit_point, normal, intersected_obj)
    return closest_hit

def trace_ray(ray, objects, lights, ambient_light_color, depth):
    if depth > MAX_DEPTH:
        return Vec3(*BACKGROUND_COLOR)

    hit_info = find_closest_intersection(ray, objects)

    if not hit_info:
        return Vec3(*BACKGROUND_COLOR)

    _t, hit_point, normal, obj_hit = hit_info
    material = obj_hit.material

    # Effective color starts with ambient contribution
    effective_color = material.diffuse_color * material.ambient_coeff * ambient_light_color

    view_dir = (ray.origin - hit_point).normalize() # Direction from hit point to camera

    for light in lights:
        light_dir = (light.position - hit_point).normalize()
        light_dist = (light.position - hit_point).length()

        # Shadow check
        shadow_ray_origin = hit_point + normal * EPSILON # Offset to avoid self-shadowing
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        shadow_hit = find_closest_intersection(shadow_ray, objects)

        in_shadow = False
        if shadow_hit:
            shadow_t, _, _, _ = shadow_hit
            if shadow_t < light_dist: # Object is between hit_point and light
                in_shadow = True

        if not in_shadow:
            # Diffuse (Lambertian)
            diffuse_intensity = max(0, normal.dot(light_dir))
            diffuse = material.diffuse_color * light.color * diffuse_intensity * material.diffuse_coeff * light.intensity
            effective_color += diffuse

            # Specular (Phong)
            # R = 2 * (N.L) * N - L
            reflect_dir = (normal * (2 * normal.dot(light_dir))) - light_dir
            reflect_dir = reflect_dir.normalize()
            specular_intensity = max(0, view_dir.dot(reflect_dir)) ** material.shininess
            specular = light.color * specular_intensity * material.specular_coeff * light.intensity
            effective_color += specular

    # Reflection
    if material.reflection_coeff > 0:
        # R = I - 2 * (I.N) * N  (I is incoming ray direction)
        reflection_ray_dir = ray.direction - normal * (2 * ray.direction.dot(normal))
        reflection_ray_origin = hit_point + normal * EPSILON # Offset
        reflection_ray = Ray(reflection_ray_origin, reflection_ray_dir)
        reflected_color = trace_ray(reflection_ray, objects, lights, ambient_light_color, depth + 1)
        effective_color += reflected_color * material.reflection_coeff

    return effective_color.clamp()


# --- Camera and Rendering ---
def render(width, height, objects, lights, ambient_light_color):
    img = Image.new("RGB", (width, height))
    pixels = img.load()

    # Camera setup
    eye = Vec3(0, 0.5, 2) # Camera position
    look_at = Vec3(0, -0.5, -5) # Point camera is looking at
    up = Vec3(0, 1, 0)    # Up direction for camera

    # Camera coordinate system
    forward = (look_at - eye).normalize()
    right = up.cross(forward).normalize() # Careful with order for handedness
    camera_up = forward.cross(right).normalize()


    aspect_ratio = width / height
    fov_rad = math.pi / 3.0  # 60 degrees field of view
    scale = math.tan(fov_rad * 0.5)

    start_time = time.time()
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to Normalized Device Coordinates (NDC) [-1, 1]
            # then to screen space coordinates
            # (0,0) is top-left, (width-1, height-1) is bottom-right
            # NDC x from -1 (left) to 1 (right)
            # NDC y from 1 (top) to -1 (bottom)
            px = (2 * (x + 0.5) / width - 1) * aspect_ratio * scale
            py = (1 - 2 * (y + 0.5) / height) * scale

            # Ray direction in world space
            ray_dir_camera_space = Vec3(px, py, -1).normalize() # Assuming camera looks along -Z in its own space
            
            # Transform ray direction from camera space to world space
            # This is a common way: ray_dir = right*px + camera_up*py - forward (if forward is -w)
            # Or simpler: ray_dir = (right * px + camera_up * py + forward).normalize() -- needs check
            # Let's use the matrix method conceptually:
            # M = [right, camera_up, forward_vector_pointing_from_eye_to_scene]
            # ray_dir_world = M * ray_dir_camera_space
            # Assuming forward is (look_at - eye).normalize(), then the point on the view plane is:
            # eye + forward * (some_focal_length) + right * px + camera_up * py
            # So ray_dir = ( (forward * 1.0) + (right * px) + (camera_up * py) ).normalize()
            # The -1 for Z in camera space means the view plane is at Z=-1.
            # The 'forward' vector points *into* the scene from the eye.
            
            ray_direction = (right * px + camera_up * py + forward).normalize() # Corrected this often tricky part

            ray = Ray(eye, ray_direction)
            color_vec = trace_ray(ray, objects, lights, ambient_light_color, 0)

            # Convert to 0-255 RGB
            r = int(color_vec.x * 255)
            g = int(color_vec.y * 255)
            b = int(color_vec.z * 255)
            pixels[x, y] = (r, g, b)

        # Progress update
        if (y + 1) % 50 == 0 or y == height - 1:
            elapsed = time.time() - start_time
            print(f"Rendered line {y+1}/{height}. Time: {elapsed:.2f}s. "
                  f"Est. total: {elapsed / (y+1) * height:.2f}s")

    end_time = time.time()
    print(f"Rendering finished in {end_time - start_time:.2f} seconds.")
    return img

# --- Main ---
if __name__ == "__main__":
    print("Setting up scene...")
    scene_objects, scene_lights, global_ambient = setup_scene()

    print(f"Starting raytracer ({WIDTH}x{HEIGHT}, max_depth={MAX_DEPTH})...")
    image = render(WIDTH, HEIGHT, scene_objects, scene_lights, global_ambient)

    output_filename = "raytraced_scene_colorful_lights.png"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
    # Try to open the image (platform dependent)
    try:
        import os, sys
        if sys.platform == "win32":
            os.startfile(output_filename)
        elif sys.platform == "darwin": # macOS
            os.system(f"open {output_filename}")
        else: # Linux
            os.system(f"xdg-open {output_filename}")
    except Exception as e:
        print(f"Could not auto-open image: {e}")