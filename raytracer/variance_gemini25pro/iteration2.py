import numpy as np
from PIL import Image
import math
from tqdm import tqdm
import time # To measure render time

# --- Vector Class ---
class Vec3:
    """Represents a 3D vector or color."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.data = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self): return self.data[0]
    @property
    def y(self): return self.data[1]
    @property
    def z(self): return self.data[2]

    def __add__(self, other): return Vec3(* (self.data + other.data))
    def __sub__(self, other): return Vec3(* (self.data - other.data))
    def __mul__(self, other): # Element-wise or scalar multiplication
        if isinstance(other, Vec3):
            return Vec3(* (self.data * other.data))
        return Vec3(* (self.data * other))
    def __rmul__(self, other): return self * other # Handle scalar * Vec3
    def __truediv__(self, scalar): return Vec3(* (self.data / scalar))
    def __neg__(self): return Vec3(* -self.data)
    def length_squared(self): return np.dot(self.data, self.data)
    def length(self): return math.sqrt(self.length_squared())
    def normalize(self):
        l = self.length()
        if l == 0: return Vec3() # Avoid division by zero
        return self / l
    def dot(self, other): return np.dot(self.data, other.data)
    def cross(self, other): return Vec3(* np.cross(self.data, other.data))
    def __repr__(self): return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# --- Ray Class ---
class Ray:
    """Represents a ray with an origin and direction."""
    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalize() # Ensure direction is normalized

    def at(self, t: float) -> Vec3:
        """Calculate the point along the ray at distance t."""
        return self.origin + self.direction * t

# --- Hit Record ---
class HitRecord:
    """Stores information about a ray-object intersection."""
    def __init__(self, point: Vec3, normal: Vec3, t: float, material_color: Vec3):
        self.point = point
        self.normal = normal # Should be normalized outward
        self.t = t
        self.material_color = material_color

# --- Hittable Objects ---
class Sphere:
    """A sphere object."""
    def __init__(self, center: Vec3, radius: float, color: Vec3):
        self.center = center
        self.radius = max(0.01, radius) # Ensure radius is positive
        self.color = color

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        """Checks if the ray intersects the sphere within the interval [t_min, t_max]."""
        oc = ray.origin - self.center
        a = ray.direction.length_squared()
        half_b = oc.dot(ray.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrt_discriminant = math.sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range
        root = (-half_b - sqrt_discriminant) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrt_discriminant) / a
            if root < t_min or t_max < root:
                return None

        # Intersection found
        t = root
        point = ray.at(t)
        outward_normal = (point - self.center) / self.radius
        # Ensure normal points against the ray
        if ray.direction.dot(outward_normal) > 0.0:
             normal = -outward_normal # Ray is inside the sphere
        else:
             normal = outward_normal # Ray is outside the sphere

        return HitRecord(point, normal, t, self.color)

class HittableList:
    """A list of hittable objects."""
    def __init__(self, objects: list = None):
        self.objects = objects if objects is not None else []

    def add(self, obj):
        self.objects.append(obj)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        """Finds the closest hit among all objects in the list."""
        closest_hit = None
        closest_so_far = t_max

        for obj in self.objects:
            hit_record = obj.hit(ray, t_min, closest_so_far)
            if hit_record:
                closest_so_far = hit_record.t
                closest_hit = hit_record

        return closest_hit

# --- Light Sources ---
class PointLight:
    """A simple point light source."""
    def __init__(self, position: Vec3, color: Vec3, intensity: float = 1.0):
        self.position = position
        self.color = color * intensity # Store intensity directly in color

# --- Scene Setup ---
def setup_scene():
    """Creates the scene objects and lights."""
    world = HittableList()

    # Ground sphere
    world.add(Sphere(center=Vec3(0, -100.5, -1), radius=100, color=Vec3(0.2, 0.8, 0.2))) # Greenish ground

    # Main spheres
    world.add(Sphere(center=Vec3(0, 0, -1.5), radius=0.5, color=Vec3(0.8, 0.1, 0.1)))   # Red sphere center
    world.add(Sphere(center=Vec3(-1.2, 0, -1.2), radius=0.5, color=Vec3(0.1, 0.1, 0.8))) # Blue sphere left
    world.add(Sphere(center=Vec3(1.2, 0, -1.2), radius=0.5, color=Vec3(0.9, 0.9, 0.1))) # Yellow sphere right

    # Smaller decorative spheres
    world.add(Sphere(center=Vec3(-0.5, -0.3, -0.5), radius=0.2, color=Vec3(0.8, 0.1, 0.8))) # Magenta small
    world.add(Sphere(center=Vec3(0.5, -0.3, -0.5), radius=0.2, color=Vec3(0.1, 0.8, 0.8)))  # Cyan small
    world.add(Sphere(center=Vec3(0.0, 0.6, -0.8), radius=0.3, color=Vec3(1.0, 1.0, 1.0)))  # White sphere top


    # Lights
    lights = [
        PointLight(position=Vec3(-5, 5, 1), color=Vec3(1.0, 0.5, 0.5), intensity=100.0), # Reddish light from left-top-front
        PointLight(position=Vec3(5, 3, -3), color=Vec3(0.5, 1.0, 0.5), intensity=80.0),  # Greenish light from right-top-back
        PointLight(position=Vec3(0, 8, -1), color=Vec3(0.5, 0.5, 1.0), intensity=120.0), # Bluish light from top
        PointLight(position=Vec3(2, -2, 0), color=Vec3(1.0, 1.0, 0.0), intensity=50.0) # Yellowish fill light from right-bottom-front
    ]

    return world, lights

# --- Ray Coloring / Shading ---
def ray_color(ray: Ray, world: HittableList, lights: list[PointLight], depth: int) -> Vec3:
    """Calculates the color seen by a ray."""
    # Simple sky gradient background
    unit_direction = ray.direction.normalize()
    t = 0.5 * (unit_direction.y + 1.0)
    background_color = (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0) # White to light blue

    # If we exceed the recursion depth, no more light is gathered.
    if depth <= 0:
        return Vec3(0, 0, 0) # Black

    # Check for hits
    hit_record = world.hit(ray, 0.001, float('inf')) # t_min=0.001 to avoid shadow acne

    if hit_record:
        final_color = Vec3(0, 0, 0)
        hit_point = hit_record.point
        normal = hit_record.normal
        mat_color = hit_record.material_color

        # Add contribution from each light source
        for light in lights:
            light_dir = light.position - hit_point
            distance_to_light_sq = light_dir.length_squared()
            distance_to_light = math.sqrt(distance_to_light_sq)
            light_dir_norm = light_dir / distance_to_light

            # Shadow check: Cast a ray from the hit point towards the light
            shadow_ray = Ray(hit_point + normal * 1e-4, light_dir_norm) # Offset origin slightly
            shadow_hit = world.hit(shadow_ray, 0.001, distance_to_light)

            # If the shadow ray doesn't hit anything *before* the light source
            if not shadow_hit:
                # Lambertian diffuse reflection
                # Intensity falls off with square of distance
                intensity_factor = 1.0 / max(distance_to_light_sq, 1.0) # Avoid division by zero/huge numbers close up

                # Diffuse term (N dot L)
                cos_theta = max(0.0, normal.dot(light_dir_norm))

                # Light color * Material color * (N dot L) * Intensity Falloff
                diffuse_color = light.color * mat_color * cos_theta * intensity_factor
                final_color += diffuse_color

        # Add a small ambient term to avoid pitch black areas
        ambient_light = Vec3(0.05, 0.05, 0.05)
        final_color += ambient_light * mat_color

        return Vec3(min(final_color.x, 1.0), min(final_color.y, 1.0), min(final_color.z, 1.0)) # Clamp color
    else:
        # No hit, return background color
        return background_color

# --- Main Rendering Function ---
def main():
    # Image settings
    aspect_ratio = 16.0 / 9.0
    image_width = 800
    image_height = int(image_width / aspect_ratio)
    if image_height < 1: image_height = 1
    # Ensure height is 600 as requested, adjust width accordingly
    image_height = 600
    image_width = int(image_height * aspect_ratio)
    if image_width < 1: image_width = 1
    print(f"Rendering image: {image_width}x{image_height}")

    samples_per_pixel = 10 # Basic anti-aliasing (more samples = smoother but slower)
    max_depth = 5      # Maximum number of ray bounces (not really used in this simple version)

    # Camera settings
    lookfrom = Vec3(0, 1, 2.5)   # Camera position
    lookat   = Vec3(0, 0, -1)   # Point camera looks at
    vup      = Vec3(0, 1, 0)    # Camera 'up' direction

    vfov_degrees = 60.0         # Vertical field-of-view in degrees
    vfov_radians = math.radians(vfov_degrees)
    h = math.tan(vfov_radians / 2.0)
    viewport_height = 2.0 * h
    viewport_width = aspect_ratio * viewport_height

    # Camera coordinate system
    w = (lookfrom - lookat).normalize() # Forward vector (negative Z)
    u = vup.cross(w).normalize()      # Right vector (X)
    v = w.cross(u).normalize()        # Up vector (Y)

    camera_origin = lookfrom
    horizontal = u * viewport_width
    vertical = v * viewport_height
    lower_left_corner = camera_origin - horizontal / 2 - vertical / 2 - w

    # Scene
    world, lights = setup_scene()

    # Rendering
    img = Image.new('RGB', (image_width, image_height), 'black')
    pixels = img.load()

    start_time = time.time()

    for j in tqdm(range(image_height - 1, -1, -1), desc="Rendering Scanlines"): # Iterate height backwards for typical image coords
        for i in range(image_width):
            pixel_color = Vec3(0, 0, 0)
            # Anti-aliasing loop
            for s in range(samples_per_pixel):
                # Add random offset within the pixel for antialiasing
                u_offset = (i + np.random.rand()) / (image_width - 1)
                v_offset = (j + np.random.rand()) / (image_height - 1)

                ray_direction = lower_left_corner + horizontal * u_offset + vertical * v_offset - camera_origin
                ray = Ray(camera_origin, ray_direction)
                pixel_color += ray_color(ray, world, lights, max_depth)

            # Average color and apply gamma correction (simple sqrt)
            scale = 1.0 / samples_per_pixel
            r = math.sqrt(pixel_color.x * scale)
            g = math.sqrt(pixel_color.y * scale)
            b = math.sqrt(pixel_color.z * scale)

            # Convert float [0,1] to int [0,255]
            ir = int(255.999 * np.clip(r, 0.0, 0.999))
            ig = int(255.999 * np.clip(g, 0.0, 0.999))
            ib = int(255.999 * np.clip(b, 0.0, 0.999))

            pixels[i, image_height - 1 - j] = (ir, ig, ib) # Invert j for PIL coordinates

    end_time = time.time()
    print(f"\nRendering finished in {end_time - start_time:.2f} seconds.")

    # Save image
    output_filename = "raytraced_scene2.png"
    img.save(output_filename)
    print(f"Image saved as {output_filename}")

if __name__ == "__main__":
    main()