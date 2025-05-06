import numpy as np
from PIL import Image
from tqdm import tqdm
import math

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
FOV = math.pi / 3  # 60 degrees
MAX_DEPTH = 3     # Max recursion depth for reflections
AMBIENT_LIGHT = 0.1
EPSILON = 1e-4     # To prevent self-intersection

# --- Helper Vector Functions (using numpy) ---
def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(incident, normal):
    return incident - 2 * np.dot(incident, normal) * normal

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float64)
        self.direction = normalize(np.array(direction, dtype=np.float64))

# --- Material Class ---
class Material:
    def __init__(self, diffuse_color, specular_albedo=0.8, diffuse_albedo=0.8, shininess=32, reflection_coeff=0.3):
        self.diffuse_color = np.array(diffuse_color, dtype=np.float64) # Base color
        self.specular_albedo = specular_albedo # How much specular light is reflected
        self.diffuse_albedo = diffuse_albedo   # How much diffuse light is reflected
        self.shininess = shininess             # Specular highlight exponent
        self.reflection_coeff = reflection_coeff # How much light is reflected (0-1)

# --- Light Class ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.intensity = intensity

# --- Object Classes ---
class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.material = material

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = np.dot(L, ray.direction)
        if tca < 0: return None  # Sphere is behind the ray

        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius: return None # Ray misses sphere

        thc = math.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc

        if t0 < EPSILON and t1 < EPSILON: return None # Both intersections are too close or behind
        if t0 < EPSILON: t = t1 # t0 is behind or too close, use t1
        else: t = t0 # t0 is the first valid intersection

        if t < EPSILON: return None # Intersection is too close

        hit_point = ray.origin + ray.direction * t
        normal = normalize(hit_point - self.center)
        return t, hit_point, normal

class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point, dtype=np.float64) # A point on the plane
        self.normal = normalize(np.array(normal, dtype=np.float64))
        self.material = material

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > EPSILON: # Ray is not parallel to the plane
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t >= EPSILON:
                hit_point = ray.origin + ray.direction * t
                return t, hit_point, self.normal
        return None

# --- Scene Definition ---
camera_position = np.array([0, 1.5, -4])
look_at = np.array([0, 0, 0])
up_vector = np.array([0, 1, 0])

# Materials
mat_red_shiny = Material(diffuse_color=[0.8, 0.1, 0.1], reflection_coeff=0.4, shininess=100)
mat_green_matte = Material(diffuse_color=[0.1, 0.7, 0.1], reflection_coeff=0.1, shininess=10)
mat_blue_reflective = Material(diffuse_color=[0.2, 0.2, 0.9], reflection_coeff=0.7, shininess=200)
mat_yellow_bright = Material(diffuse_color=[0.9, 0.9, 0.2], reflection_coeff=0.2, shininess=50)
mat_floor = Material(diffuse_color=[0.7, 0.7, 0.7], reflection_coeff=0.1, shininess=5, diffuse_albedo=0.9)
mat_mirror = Material(diffuse_color=[0.9, 0.9, 0.9], reflection_coeff=0.9, shininess=1000)


objects = [
    Sphere(center=[-1.5, 0.2, 1], radius=0.7, material=mat_red_shiny),
    Sphere(center=[0, -0.1, 0.5], radius=0.4, material=mat_blue_reflective),
    Sphere(center=[1.5, 0.5, 1.5], radius=1.0, material=mat_green_matte),
    Sphere(center=[0.5, -0.5, -1], radius=0.3, material=mat_yellow_bright),
    Sphere(center=[-0.8, 1.0, -0.5], radius=0.5, material=mat_mirror), # A mirror sphere
    Plane(point=[0, -0.8, 0], normal=[0, 1, 0], material=mat_floor) # Ground plane
]

lights = [
    Light(position=[5, 5, -5], color=[1, 1, 1], intensity=0.8),
    Light(position=[-5, 3, -2], color=[1, 0.5, 0.5], intensity=0.6), # Reddish light
    Light(position=[0, 10, 0], color=[0.5, 0.5, 1], intensity=0.7),  # Bluish top light
    Light(position=[2, 1, 5], color=[0.5, 1, 0.5], intensity=0.5)    # Greenish back light
]

# --- Ray Tracing Core ---
def find_closest_intersection(ray, objects_list):
    closest_t = float('inf')
    closest_object = None
    hit_point_info = None

    for obj in objects_list:
        intersection_result = obj.intersect(ray)
        if intersection_result:
            t, hit_point, normal = intersection_result
            if t < closest_t:
                closest_t = t
                closest_object = obj
                hit_point_info = (hit_point, normal)
    
    if closest_object:
        return closest_object, closest_t, hit_point_info[0], hit_point_info[1]
    return None, None, None, None


def trace_ray(ray, scene_objects, scene_lights, depth):
    if depth > MAX_DEPTH:
        return np.array([0,0,0]) # Background color (black) or environment map

    hit_object, t, hit_point, normal = find_closest_intersection(ray, scene_objects)

    if not hit_object:
        # Simple sky gradient (can be more complex)
        # t_sky = 0.5 * (ray.direction[1] + 1.0) # y-component for gradient
        # return (1.0 - t_sky) * np.array([1.0, 1.0, 1.0]) + t_sky * np.array([0.5, 0.7, 1.0])
        return np.array([0.1, 0.1, 0.2]) # Dark blue background

    material = hit_object.material
    final_color = np.zeros(3, dtype=np.float64)
    
    # Ambient contribution
    final_color += material.diffuse_color * AMBIENT_LIGHT

    view_dir = normalize(ray.origin - hit_point)

    for light in scene_lights:
        light_dir = normalize(light.position - hit_point)
        distance_to_light = np.linalg.norm(light.position - hit_point)

        # Shadow check
        shadow_ray_origin = hit_point + normal * EPSILON # Offset to avoid self-intersection
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        
        # Check if any object is between hit_point and light source
        in_shadow = False
        shadow_hit_obj, shadow_t, _, _ = find_closest_intersection(shadow_ray, scene_objects)
        if shadow_hit_obj and shadow_t < distance_to_light:
            in_shadow = True
        
        if not in_shadow:
            # Diffuse
            diffuse_intensity = max(0, np.dot(normal, light_dir))
            diffuse_contrib = material.diffuse_albedo * material.diffuse_color * light.color * diffuse_intensity * light.intensity
            final_color += diffuse_contrib

            # Specular
            reflected_light_dir = reflect(-light_dir, normal)
            specular_intensity = max(0, np.dot(view_dir, reflected_light_dir)) ** material.shininess
            specular_contrib = material.specular_albedo * light.color * specular_intensity * light.intensity
            final_color += specular_contrib

    # Reflection
    if material.reflection_coeff > 0 and depth < MAX_DEPTH:
        reflection_dir = reflect(ray.direction, normal)
        reflection_ray_origin = hit_point + normal * EPSILON
        reflection_ray = Ray(reflection_ray_origin, reflection_dir)
        reflected_color = trace_ray(reflection_ray, scene_objects, scene_lights, depth + 1)
        final_color += material.reflection_coeff * reflected_color
        
    return np.clip(final_color, 0, 1)


# --- Main Rendering Loop ---
def render(width, height, cam_pos, look_at_pt, up_vec, fov_rad, scene_obj, scene_lgt):
    aspect_ratio = float(width) / height
    
    # Camera setup
    forward = normalize(look_at_pt - cam_pos)
    right = normalize(np.cross(forward, up_vec))
    if np.linalg.norm(right) < EPSILON: # if forward and up are collinear
        # Choose a different temporary up vector if camera looks straight up/down
        alt_up = np.array([0,0,1]) if abs(up_vec[1]) > 0.99 else np.array([0,1,0])
        right = normalize(np.cross(forward, alt_up))
    
    cam_up = normalize(np.cross(right, forward)) # Recalculate actual up vector

    image = Image.new("RGB", (width, height))
    pixels = image.load()

    tan_half_fov = math.tan(fov_rad / 2.0)

    for y in tqdm(range(height)):
        for x in range(width):
            # Screen space coordinates to NDC (Normalized Device Coordinates) [-1, 1]
            # (0,0) is top-left, (width-1, height-1) is bottom-right
            # Map x from [0, W-1] to [-1, 1] (approx)
            # Map y from [0, H-1] to [1, -1] (approx, y is inverted in screen space)
            px_ndc_x = (2 * (x + 0.5) / width - 1.0) 
            px_ndc_y = (1.0 - 2 * (y + 0.5) / height)

            # Camera space coordinates
            px_cam_x = px_ndc_x * aspect_ratio * tan_half_fov
            px_cam_y = px_ndc_y * tan_half_fov
            
            # Ray direction in world space
            ray_direction_cam_space = np.array([px_cam_x, px_cam_y, 1.0]) # Z is 1.0 (forward)
            
            # Transform ray direction from camera space to world space
            # This can be done by multiplying with the camera-to-world matrix
            # C2W = [right, up, forward] (as column vectors for point transformation)
            # For direction, it's simpler:
            ray_dir_world = normalize(right * ray_direction_cam_space[0] +
                                      cam_up * ray_direction_cam_space[1] +
                                      forward * ray_direction_cam_space[2])

            primary_ray = Ray(cam_pos, ray_dir_world)
            color = trace_ray(primary_ray, scene_obj, scene_lgt, 0)
            
            # Convert color from [0,1] float to [0,255] int
            r = int(color[0] * 255)
            g = int(color[1] * 255)
            b = int(color[2] * 255)
            pixels[x, y] = (r, g, b)
    
    return image

if __name__ == "__main__":
    print(f"Rendering {WIDTH}x{HEIGHT} image with MAX_DEPTH={MAX_DEPTH}...")
    rendered_image = render(WIDTH, HEIGHT, camera_position, look_at, up_vector, FOV, objects, lights)
    rendered_image.save("raytraced_scene1.png")
    print("Image saved as raytraced_scene_colorful_lights.png")
    rendered_image.show()