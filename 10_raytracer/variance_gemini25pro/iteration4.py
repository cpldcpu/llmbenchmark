import numpy as np
from PIL import Image
import math
import random
from tqdm import tqdm  # Optional: for progress bar

# --- Vector Class ---
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.data = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self): return self.data[0]
    @property
    def y(self): return self.data[1]
    @property
    def z(self): return self.data[2]

    def __neg__(self): return Vec3(-self.x, -self.y, -self.z)
    def __getitem__(self, i): return self.data[i]
    def __setitem__(self, i, value): self.data[i] = value

    def __add__(self, other): return Vec3(*(self.data + other.data))
    def __sub__(self, other): return Vec3(*(self.data - other.data))
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec3(*(self.data * other))
        return Vec3(*(self.data * other.data)) # Element-wise multiplication for colors/attenuation
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec3(*(self.data / other))
        raise NotImplementedError("Division by Vec3 not typically defined")

    def length_squared(self): return np.dot(self.data, self.data)
    def length(self): return np.sqrt(self.length_squared())
    def normalize(self):
        mag = self.length()
        if mag == 0: return Vec3(0,0,0)
        return Vec3(*(self.data / mag))

    def dot(self, other): return np.dot(self.data, other.data)
    def cross(self, other): return Vec3(*np.cross(self.data, other.data))

    @staticmethod
    def random(min_val=0.0, max_val=1.0):
        return Vec3(random.uniform(min_val, max_val),
                    random.uniform(min_val, max_val),
                    random.uniform(min_val, max_val))

    @staticmethod
    def random_in_unit_sphere():
        while True:
            p = Vec3.random(-1, 1)
            if p.length_squared() < 1:
                return p

    @staticmethod
    def random_unit_vector():
        return Vec3.random_in_unit_sphere().normalize()

    @staticmethod
    def reflect(v, n):
        return v - 2 * v.dot(n) * n

    def near_zero(self):
        s = 1e-8
        return (abs(self.x) < s) and (abs(self.y) < s) and (abs(self.z) < s)

    def __str__(self): return f"Vec3({self.x}, {self.y}, {self.z})"

# Type aliases for clarity
Point3 = Vec3
Color = Vec3

# --- Ray Class ---
class Ray:
    def __init__(self, origin: Point3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalize() # Ensure direction is normalized

    def at(self, t: float) -> Point3:
        return self.origin + t * self.direction

# --- Hit Record ---
class HitRecord:
    def __init__(self):
        self.p: Point3 = Point3()
        self.normal: Vec3 = Vec3()
        self.material = None # Will be assigned Material instance
        self.t: float = 0.0
        self.front_face: bool = False

    def set_face_normal(self, r: Ray, outward_normal: Vec3):
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

# --- Materials ---
class Material:
    def scatter(self, r_in: Ray, rec: HitRecord):
        """
        Returns a tuple: (did_scatter, attenuation, scattered_ray)
        did_scatter (bool): True if the ray was scattered.
        attenuation (Color): How much the light is reduced/colored.
        scattered_ray (Ray): The new ray resulting from the scatter.
        Return (False, None, None) if the ray is absorbed.
        """
        raise NotImplementedError

    def emitted(self, u: float, v: float, p: Point3) -> Color:
        """Return emitted light color. Default is black (non-emissive)."""
        return Color(0, 0, 0)

class Lambertian(Material):
    def __init__(self, albedo: Color):
        self.albedo = albedo

    def scatter(self, r_in: Ray, rec: HitRecord):
        scatter_direction = rec.normal + Vec3.random_unit_vector()
        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = rec.normal

        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return True, attenuation, scattered

class Metal(Material):
    def __init__(self, albedo: Color, fuzz: float):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0) # Clamp fuzziness between 0 and 1

    def scatter(self, r_in: Ray, rec: HitRecord):
        reflected = Vec3.reflect(r_in.direction.normalize(), rec.normal)
        scattered = Ray(rec.p, reflected + self.fuzz * Vec3.random_in_unit_sphere())
        attenuation = self.albedo
        # Check if scattered ray goes below the surface
        did_scatter = scattered.direction.dot(rec.normal) > 0
        return did_scatter, attenuation, scattered

class DiffuseLight(Material):
    def __init__(self, emit_color: Color):
        self.emit_color = emit_color

    def scatter(self, r_in: Ray, rec: HitRecord):
        # Light sources don't scatter incoming light in this model
        return False, None, None

    def emitted(self, u: float, v: float, p: Point3) -> Color:
        # Simple uniform emission
        return self.emit_color

# --- Hittable Objects ---
class Hittable:
    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        """
        Checks if the ray `r` hits the object within the interval [t_min, t_max].
        Returns a HitRecord if hit, otherwise None.
        """
        raise NotImplementedError

class Sphere(Hittable):
    def __init__(self, center: Point3, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        oc = r.origin - self.center
        a = r.direction.length_squared()
        half_b = oc.dot(r.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrtd = math.sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range
        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return None

        rec = HitRecord()
        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(r, outward_normal)
        rec.material = self.material

        return rec

class HittableList(Hittable):
    def __init__(self):
        self.objects = []

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        closest_so_far = t_max
        hit_anything = None

        for obj in self.objects:
            temp_rec = obj.hit(r, t_min, closest_so_far)
            if temp_rec:
                hit_anything = temp_rec
                closest_so_far = temp_rec.t

        return hit_anything

# --- Camera ---
class Camera:
    def __init__(self, lookfrom: Point3, lookat: Point3, vup: Vec3,
                 vfov: float, aspect_ratio: float):
        theta = math.radians(vfov)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = (lookfrom - lookat).normalize()
        self.u = vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)

        self.origin = lookfrom
        self.horizontal = viewport_width * self.u
        self.vertical = viewport_height * self.v
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - self.w

    def get_ray(self, s: float, t: float) -> Ray:
        direction = self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin
        return Ray(self.origin, direction)


# --- Ray Tracing Function ---
def ray_color(r: Ray, world: Hittable, depth: int) -> Color:
    # If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0:
        return Color(0, 0, 0)

    # Check for hits, use small t_min to avoid shadow acne
    hit_rec = world.hit(r, 0.001, float('inf'))

    if hit_rec:
        emitted = hit_rec.material.emitted(0, 0, hit_rec.p) # u,v coords not used here
        scatter_result = hit_rec.material.scatter(r, hit_rec)

        if scatter_result[0]: # If scatter occurred
            did_scatter, attenuation, scattered_ray = scatter_result
            return emitted + attenuation * ray_color(scattered_ray, world, depth - 1)
        else: # Hit an emissive object, or absorbed
            return emitted
    else:
        # Background: simple blueish gradient
        unit_direction = r.direction.normalize()
        t = 0.5 * (unit_direction.y + 1.0)
        # return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0) # Gradient
        return Color(0.0, 0.0, 0.0) # Black background for light sources scene


# --- Image Rendering ---
def write_color(pixel_color: Color, samples_per_pixel: int) -> tuple:
    r, g, b = pixel_color.x, pixel_color.y, pixel_color.z

    # Divide the color by the number of samples and gamma-correct for gamma=2.0.
    scale = 1.0 / samples_per_pixel
    r = math.sqrt(scale * r)
    g = math.sqrt(scale * g)
    b = math.sqrt(scale * b)

    # Clamp and convert to integer RGB values
    ir = int(255.999 * np.clip(r, 0.0, 0.999))
    ig = int(255.999 * np.clip(g, 0.0, 0.999))
    ib = int(255.999 * np.clip(b, 0.0, 0.999))
    return (ir, ig, ib)

def main():
    # --- Image ---
    aspect_ratio = 16.0 / 9.0
    image_width = 800
    image_height = int(image_width / aspect_ratio)
    if image_height < 1: image_height = 1
    samples_per_pixel = 4 # Increase for better quality, decrease for faster render
    max_depth = 50          # Max ray bounces

    # --- World ---
    world = HittableList()

    # Materials
    material_ground = Lambertian(Color(0.5, 0.5, 0.5))
    material_center = Lambertian(Color(0.1, 0.2, 0.5))
    material_left = Metal(Color(0.8, 0.8, 0.8), 0.1)
    material_right = Metal(Color(0.8, 0.6, 0.2), 0.6)
    material_bubble = Lambertian(Color(0.9, 0.3, 0.8))

    # Light Source Materials
    light_red = DiffuseLight(Color(4, 0.5, 0.5)) # Intensity > 1 for brightness
    light_green = DiffuseLight(Color(0.5, 4, 0.5))
    light_blue = DiffuseLight(Color(0.5, 0.5, 4))
    light_yellow = DiffuseLight(Color(4, 4, 0.5))
    light_white_top = DiffuseLight(Color(7, 7, 7))

    # Objects
    world.add(Sphere(Point3(0, -1000, 0), 1000, material_ground)) # Ground sphere

    world.add(Sphere(Point3(0, 1, 0), 1.0, material_center))
    world.add(Sphere(Point3(-4, 1, 0), 1.0, material_left))
    world.add(Sphere(Point3(4, 1, 0), 1.0, material_right))
    world.add(Sphere(Point3(2, 0.5, -1), 0.5, material_bubble))

    # Light Sources (Spheres with DiffuseLight material)
    world.add(Sphere(Point3(-2, 0.5, 2), 0.3, light_red))
    world.add(Sphere(Point3(0, 2.5, -1), 0.4, light_green))
    world.add(Sphere(Point3(2, 0.5, 2), 0.3, light_blue))
    world.add(Sphere(Point3(0, 0.5, 3), 0.4, light_yellow))
    world.add(Sphere(Point3(0, 5, 0), 1.5, light_white_top)) # Larger light source overhead

    # Add some smaller random spheres for visual interest
    for a in range(-3, 3):
        for b in range(-3, 3):
            choose_mat = random.random()
            center = Point3(a + 0.9*random.random(), 0.2, b + 0.9*random.random())

            if (center - Point3(4, 0.2, 0)).length() > 0.9 and \
               (center - Point3(-4, 0.2, 0)).length() > 0.9 and \
               (center - Point3(0, 0.2, 0)).length() > 0.9 :
                if choose_mat < 0.6: # diffuse
                    albedo = Color.random() * Color.random()
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.9: # metal
                    albedo = Color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.4)
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                # else: glass (not implemented here)

    # --- Camera ---
    lookfrom = Point3(13, 2, 3)
    lookat = Point3(0, 0.5, 0)
    vup = Vec3(0, 1, 0)
    vfov = 25.0 # Field of view
    cam = Camera(lookfrom, lookat, vup, vfov, aspect_ratio)

    # --- Render ---
    img = Image.new('RGB', (image_width, image_height))
    pixels = img.load()

    # Use tqdm for progress bar if available
    for j in tqdm(range(image_height -1, -1, -1), desc="Rendering Scanlines"):
    # for j in range(image_height - 1, -1, -1): # Without tqdm
        # print(f"\rScanlines remaining: {j}", end=' ', flush=True) # Simple progress
        for i in range(image_width):
            pixel_color_sum = Color(0, 0, 0)
            for s in range(samples_per_pixel):
                u = (i + random.random()) / (image_width - 1)
                v = (j + random.random()) / (image_height - 1)
                r = cam.get_ray(u, v)
                pixel_color_sum += ray_color(r, world, max_depth)

            pixels[i, image_height - 1 - j] = write_color(pixel_color_sum, samples_per_pixel)

    # print("\nDone.")
    img.save("raytraced_scene4.png")
    print("Image saved as raytraced_lights_scene.png")

if __name__ == "__main__":
    main()