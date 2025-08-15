import math
from PIL import Image
import time # To estimate render time
from tqdm import tqdm # For a progress bar

# --- Vector Class ---
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, Vec3): # Element-wise multiplication for colors
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
        return self / l

    def __str__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

# --- Material Class ---
class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.9, shininess=200):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

# --- Hittable Objects ---
class Hittable:
    def __init__(self, material):
        self.material = material

    def intersect(self, ray):
        # Returns distance t if hit, or None
        raise NotImplementedError

class Sphere(Hittable):
    def __init__(self, center, radius, material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction) # Should be 1 if direction is normalized
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        else:
            t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t1 > EPSILON: return t1 # Smallest positive t
            if t2 > EPSILON: return t2
            return None

    def normal_at(self, point):
        return (point - self.center).normalize()

class Plane(Hittable):
    def __init__(self, point, normal, material):
        super().__init__(material)
        self.point = point # A point on the plane
        self.normal = normal.normalize()

    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > EPSILON: # Not parallel
            t = (self.point - ray.origin).dot(self.normal) / denom
            if t > EPSILON:
                return t
        return None

    def normal_at(self, point):
        return self.normal # Normal is constant for a plane

# --- Light Source ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# --- Constants ---
WIDTH = 800
HEIGHT = 600
FOV = math.pi / 3  # 60 degrees
ASPECT_RATIO = WIDTH / HEIGHT
EPSILON = 1e-4 # To avoid self-intersection / shadow acne
MAX_DEPTH = 3 # For future reflections/refractions, not used much here

# --- Scene Setup ---
camera_origin = Vec3(0, 1.5, 4)
look_at = Vec3(0, 0, 0)
up_vector = Vec3(0, 1, 0)

# Camera basis vectors
forward = (look_at - camera_origin).normalize()
right = forward.cross(up_vector).normalize()
up = right.cross(forward).normalize() # Recompute up to be orthogonal

# Materials
mat_red_shiny = Material(Vec3(1, 0, 0), ambient=0.1, diffuse=0.7, specular=0.8, shininess=100)
mat_green_matte = Material(Vec3(0, 1, 0), ambient=0.15, diffuse=0.8, specular=0.2, shininess=20)
mat_blue_reflective = Material(Vec3(0.2, 0.2, 1), ambient=0.1, diffuse=0.6, specular=0.9, shininess=500)
mat_yellow_sphere = Material(Vec3(1, 1, 0.2), ambient=0.1, diffuse=0.8, specular=0.5, shininess=80)
mat_purple_sphere = Material(Vec3(0.8, 0.1, 0.8), ambient=0.1, diffuse=0.7, specular=0.7, shininess=150)
mat_cyan_sphere = Material(Vec3(0.1, 0.8, 0.8), ambient=0.1, diffuse=0.7, specular=0.6, shininess=120)
mat_floor = Material(Vec3(0.7, 0.7, 0.7), ambient=0.2, diffuse=0.8, specular=0.1, shininess=10)
mat_back_wall = Material(Vec3(0.5, 0.5, 0.6), ambient=0.2, diffuse=0.8, specular=0.1, shininess=10)


objects = [
    Sphere(Vec3(-1.5, 0.5, -1), 0.5, mat_red_shiny),
    Sphere(Vec3(0, 0.75, -2), 0.75, mat_green_matte),
    Sphere(Vec3(1.5, 0.3, -0.8), 0.3, mat_blue_reflective),
    Sphere(Vec3(-0.5, -0.1, 0.5), 0.4, mat_yellow_sphere),
    Sphere(Vec3(0.8, 0.2, 0.2), 0.2, mat_purple_sphere),
    Sphere(Vec3(1.8, 0.6, -2.5), 0.6, mat_cyan_sphere),
    Plane(Vec3(0, -0.5, 0), Vec3(0, 1, 0), mat_floor), # Floor
    # Plane(Vec3(0,0,-5), Vec3(0,0,1), mat_back_wall) # Back wall (optional)
]

lights = [
    Light(Vec3(-5, 5, 0), Vec3(1, 0.8, 0.8), intensity=0.8), # Soft white-ish from left-up
    Light(Vec3(5, 3, 2), Vec3(0.8, 0.8, 1), intensity=0.7),  # Cool light from right-up
    Light(Vec3(0, 10, -2), Vec3(0.5, 1, 0.5), intensity=0.5), # Greenish from top
    Light(Vec3(1, 0.5, 3), Vec3(1, 0.5, 0.5), intensity=0.6)  # Reddish from front-right (fill)
]

background_color = Vec3(0.1, 0.1, 0.2) # Dark blueish background

# --- Ray Tracing Functions ---
def find_closest_intersection(ray, objects_list):
    closest_t = float('inf')
    hit_object = None
    for obj in objects_list:
        t = obj.intersect(ray)
        if t is not None and t < closest_t:
            closest_t = t
            hit_object = obj
    if hit_object:
        return hit_object, closest_t
    return None, None

def trace_ray(ray, objects_list, lights_list, depth=0):
    hit_object, t = find_closest_intersection(ray, objects_list)

    if not hit_object:
        return background_color

    hit_point = ray.origin + ray.direction * t
    normal = hit_object.normal_at(hit_point)
    material = hit_object.material

    # Ambient component
    final_color = material.color * material.ambient

    view_dir = (ray.origin - hit_point).normalize() # From hit point to camera

    for light in lights_list:
        light_dir = (light.position - hit_point)
        light_distance = light_dir.length()
        light_dir = light_dir.normalize()

        # Shadow check
        shadow_ray_origin = hit_point + normal * EPSILON # Offset to avoid self-intersection
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        
        in_shadow = False
        # Check intersection with objects *between* hit_point and light source
        # Don't check intersection with the hit_object itself for the shadow ray
        # A simpler way is to check if any object is hit *before* the light
        # This means the distance to intersection (shadow_t) must be less than light_distance
        
        # Check against all objects for shadow
        # Optimization: could pass a list of objects excluding the current one for complex scenes
        # but for simplicity, we check all. Shadow ray will start slightly off the surface.
        
        # Iterate through objects to check for shadow
        shadow_obj, shadow_t = find_closest_intersection(shadow_ray, objects_list)
        if shadow_obj is not None and shadow_t < light_distance:
            in_shadow = True
            
        if not in_shadow:
            # Diffuse component (Lambertian)
            # Angle between normal and light direction
            diffuse_intensity = max(0.0, normal.dot(light_dir))
            diffuse_color = material.color * material.diffuse * diffuse_intensity * light.color * light.intensity
            final_color += diffuse_color

            # Specular component (Phong)
            # Reflection of light_dir around normal
            reflect_dir = (normal * (2 * normal.dot(light_dir))) - light_dir 
            reflect_dir = reflect_dir.normalize()
            
            specular_intensity = max(0.0, view_dir.dot(reflect_dir)) ** material.shininess
            specular_color = Vec3(1,1,1) * material.specular * specular_intensity * light.color * light.intensity # Specular highlights are often whiteish
            final_color += specular_color
            
    # Clamp color components
    final_color.x = max(0.0, min(1.0, final_color.x))
    final_color.y = max(0.0, min(1.0, final_color.y))
    final_color.z = max(0.0, min(1.0, final_color.z))
    
    return final_color


# --- Main Rendering Loop ---
def render():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    inv_width = 1.0 / WIDTH
    inv_height = 1.0 / HEIGHT
    tan_fov_half = math.tan(FOV * 0.5)

    print("Rendering...")
    start_time = time.time()

    for y in tqdm(range(HEIGHT)):
        for x in range(WIDTH):
            # Convert pixel coordinates to screen space (-1 to 1)
            # (0,0) at top-left, so invert y
            # Add 0.5 to sample center of pixel
            px_ndc_x = (x + 0.5) * inv_width
            px_ndc_y = (y + 0.5) * inv_height
            
            # Convert to screen space taking FOV and aspect ratio into account
            # Image plane is 1 unit away from camera in `forward` direction
            px_camera_x = (2 * px_ndc_x - 1) * ASPECT_RATIO * tan_fov_half
            px_camera_y = (1 - 2 * px_ndc_y) * tan_fov_half # Y is inverted in screen space

            # Ray direction from camera eye through this pixel on the image plane
            ray_direction = (forward + right * px_camera_x + up * px_camera_y).normalize()
            ray = Ray(camera_origin, ray_direction)
            
            color_vec = trace_ray(ray, objects, lights)
            
            # Convert color from [0,1] to [0,255]
            r = int(color_vec.x * 255)
            g = int(color_vec.y * 255)
            b = int(color_vec.z * 255)
            pixels[x, y] = (r, g, b)

    end_time = time.time()
    print(f"Rendering complete in {end_time - start_time:.2f} seconds.")
    image.save("raytraced_scene2.png")
    image.show()

if __name__ == "__main__":
    render()