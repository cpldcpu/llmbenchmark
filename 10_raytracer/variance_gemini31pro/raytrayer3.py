import math
from PIL import Image
import sys

# --- Basic Vector Math ---
class Vec3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __add__(self, o): return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, c): return Vec3(self.x * c, self.y * c, self.z * c)
    def __rmul__(self, c): return self.__mul__(c)
    def __truediv__(self, c): return Vec3(self.x / c, self.y / c, self.z / c)
    def dot(self, o): return self.x * o.x + self.y * o.y + self.z * o.z
    def length(self): return math.sqrt(self.dot(self))
    def normalize(self): 
        l = self.length()
        return self / l if l > 0 else Vec3(0,0,0)
    
    # Element-wise multiplication for colors
    def multiply(self, o): return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
    def clip(self): return Vec3(min(1., max(0., self.x)), min(1., max(0., self.y)), min(1., max(0., self.z)))

# --- Scene Objects ---
class Sphere:
    def __init__(self, center, radius, color, reflectivity=0.0, specular=50):
        self.c = center
        self.r = radius
        self.color = color
        self.reflectivity = reflectivity
        self.specular = specular

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.c
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.r * self.r
        disc = b * b - 4 * c
        if disc < 0: return float('inf')
        t0 = (-b - math.sqrt(disc)) / 2.0
        t1 = (-b + math.sqrt(disc)) / 2.0
        if t0 > 0.001: return t0
        if t1 > 0.001: return t1
        return float('inf')

    def normal(self, p):
        return (p - self.c).normalize()

class Plane:
    def __init__(self, y_pos, color1, color2, reflectivity=0.2):
        self.y = y_pos
        self.color1 = color1
        self.color2 = color2
        self.reflectivity = reflectivity

    def intersect(self, ray_origin, ray_dir):
        if abs(ray_dir.y) < 1e-6: return float('inf')
        t = (self.y - ray_origin.y) / ray_dir.y
        return t if t > 0.001 else float('inf')

    def normal(self, p):
        return Vec3(0, 1, 0)

    def get_color(self, p):
        # Checkerboard pattern
        checker = int(math.floor(p.x * 2) + math.floor(p.z * 2)) % 2
        return self.color1 if checker == 0 else self.color2

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.pos = position
        self.color = color * intensity

# --- Scene Setup ---
objects = [
    Plane(-1, Vec3(1, 1, 1), Vec3(0.1, 0.1, 0.1), reflectivity=0.3),
    Sphere(Vec3(0, 0.5, 4), 1.5, Vec3(0.1, 0.1, 0.1), reflectivity=0.8, specular=100),   # Central Mirror
    Sphere(Vec3(-2.5, 0, 4.5), 1.0, Vec3(0.9, 0.9, 0.9), reflectivity=0.1, specular=10), # Left Matte White
    Sphere(Vec3(2.5, 0, 3.5), 1.0, Vec3(0.1, 0.1, 0.1), reflectivity=0.1, specular=10)   # Right Matte Dark
]

lights = [
    Light(Vec3(-3, 3, 2), Vec3(1.0, 0.1, 0.1), 1.5),  # Red
    Light(Vec3(0, 3, 1), Vec3(0.1, 1.0, 0.1), 1.5),   # Green
    Light(Vec3(3, 3, 2), Vec3(0.1, 0.1, 1.0), 1.5),   # Blue
    Light(Vec3(-2, 1, 5), Vec3(1.0, 0.1, 1.0), 1.0),  # Magenta
    Light(Vec3(2, 4, 5), Vec3(1.0, 1.0, 0.1), 1.0),   # Yellow
    Light(Vec3(0, -0.5, 2.5), Vec3(0.1, 1.0, 1.0), 0.8) # Cyan (uplight)
]

ambient = Vec3(0.02, 0.02, 0.02)

# --- Raytracing Engine ---
def trace(ray_origin, ray_dir, depth=0):
    t_min = float('inf')
    obj_hit = None
    
    # Find nearest intersection
    for obj in objects:
        t = obj.intersect(ray_origin, ray_dir)
        if t < t_min:
            t_min = t
            obj_hit = obj
            
    if obj_hit is None:
        return Vec3(0, 0, 0) # Background color (black)
        
    hit_pos = ray_origin + ray_dir * t_min
    normal = obj_hit.normal(hit_pos)
    
    # Determine base color
    base_color = obj_hit.get_color(hit_pos) if isinstance(obj_hit, Plane) else obj_hit.color
    
    color = ambient.multiply(base_color)
    
    # Calculate lighting
    for light in lights:
        light_dir = (light.pos - hit_pos).normalize()
        
        # Shadows
        shadow_ray_orig = hit_pos + normal * 0.001
        in_shadow = False
        light_dist = (light.pos - hit_pos).length()
        
        for shadow_obj in objects:
            t = shadow_obj.intersect(shadow_ray_orig, light_dir)
            if t < light_dist:
                in_shadow = True
                break
                
        if not in_shadow:
            # Diffuse
            diffuse_intensity = max(0.0, normal.dot(light_dir))
            color = color + base_color.multiply(light.color) * diffuse_intensity
            
            # Specular
            if hasattr(obj_hit, 'specular'):
                view_dir = (ray_origin - hit_pos).normalize()
                half_vec = (light_dir + view_dir).normalize()
                spec_intensity = math.pow(max(0.0, normal.dot(half_vec)), obj_hit.specular)
                color = color + light.color * spec_intensity

    # Reflection
    if depth < 3 and obj_hit.reflectivity > 0:
        refl_dir = ray_dir - normal * 2.0 * ray_dir.dot(normal)
        refl_color = trace(hit_pos + normal * 0.001, refl_dir, depth + 1)
        color = color * (1.0 - obj_hit.reflectivity) + refl_color * obj_hit.reflectivity
        
    return color.clip()

# --- Image Generation ---
WIDTH, HEIGHT = 800, 600
camera_pos = Vec3(0, 1.5, -2)
fov = math.pi / 3.0
aspect_ratio = WIDTH / float(HEIGHT)

img = Image.new('RGB', (WIDTH, HEIGHT))
pixels = img.load()

print("Rendering scene... this may take a minute or two.")
for y in range(HEIGHT):
    # Progress bar
    sys.stdout.write(f"\rProgress: {y/HEIGHT * 100:.1f}%")
    sys.stdout.flush()
    
    for x in range(WIDTH):
        # Convert pixel to screen coordinates
        screen_x = (2.0 * (x + 0.5) / WIDTH - 1.0) * aspect_ratio * math.tan(fov / 2.0)
        screen_y = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * math.tan(fov / 2.0)
        
        ray_dir = Vec3(screen_x, screen_y, 1.0).normalize()
        
        pixel_color = trace(camera_pos, ray_dir)
        
        # Gamma correction and 8-bit conversion
        r = int(math.pow(pixel_color.x, 1/2.2) * 255)
        g = int(math.pow(pixel_color.y, 1/2.2) * 255)
        b = int(math.pow(pixel_color.z, 1/2.2) * 255)
        
        pixels[x, y] = (r, g, b)

print("\rProgress: 100.0%  ")
img.save("colorful_raytraced_scene.png")
print("Render complete! Saved as 'colorful_raytraced_scene.png'.")