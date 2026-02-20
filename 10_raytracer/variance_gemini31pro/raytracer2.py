import math
import multiprocessing
from PIL import Image

class Vector:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)
        
    def __add__(self, v): return Vector(self.x + v.x, self.y + v.y, self.z + v.z)
    def __sub__(self, v): return Vector(self.x - v.x, self.y - v.y, self.z - v.z)
    def __mul__(self, s): return Vector(self.x * s, self.y * s, self.z * s)
    def __rmul__(self, s): return self * s
    def __truediv__(self, s): return Vector(self.x / s, self.y / s, self.z / s)
    
    def dot(self, v): return self.x * v.x + self.y * v.y + self.z * v.z
    def norm(self): return math.sqrt(self.dot(self))
    def normalize(self):
        n = self.norm()
        return self / n if n > 0 else Vector(0, 0, 0)
    
    def reflect(self, normal):
        # r = d - 2(d \cdot n)n
        return self - normal * 2 * self.dot(normal)

class Sphere:
    def __init__(self, center, radius, color, reflectivity=0.0):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflectivity = reflectivity

    def intersect(self, ray_origin, ray_dir):
        # (o + td - c)^2 = r^2
        oc = ray_origin - self.center
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - 4*c
        
        if discriminant > 0:
            dist = (-b - math.sqrt(discriminant)) / 2.0
            if dist > 0.001:
                return dist
        return None

    def normal(self, point):
        return (point - self.center).normalize()

    def get_color(self, point):
        return self.color

class Plane:
    def __init__(self, y, color1, color2, reflectivity=0.5):
        self.y = y
        self.color1 = color1
        self.color2 = color2
        self.reflectivity = reflectivity

    def intersect(self, ray_origin, ray_dir):
        if abs(ray_dir.y) < 1e-6:
            return None
        dist = (self.y - ray_origin.y) / ray_dir.y
        return dist if dist > 0.001 else None

    def normal(self, point):
        return Vector(0, 1, 0)

    def get_color(self, point):
        # Checkered pattern
        scale = 2.0
        checker = int(math.floor(point.x * scale) + math.floor(point.z * scale)) % 2
        return self.color1 if checker == 0 else self.color2

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# --- Scene Setup ---
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 3

objects = [
    # Floor
    Plane(-1.0, Vector(1, 1, 1), Vector(0.1, 0.1, 0.1), reflectivity=0.4),
    # Spheres
    Sphere(Vector(0, 0, 4), 1.0, Vector(0.1, 0.1, 0.1), reflectivity=0.8), # Center mirror sphere
    Sphere(Vector(-2, 0, 3), 1.0, Vector(0.8, 0.8, 0.8), reflectivity=0.1), # Left matte sphere
    Sphere(Vector(2, -0.2, 5), 0.8, Vector(0.2, 0.2, 0.2), reflectivity=0.6) # Right glossy sphere
]

# Many colourful lightsources
lights = [
    Light(Vector(-3, 3, 2), Vector(1.0, 0.2, 0.2), 0.8),  # Red
    Light(Vector(3, 3, 2), Vector(0.2, 1.0, 0.2), 0.8),   # Green
    Light(Vector(0, 4, 1), Vector(0.2, 0.2, 1.0), 0.8),   # Blue
    Light(Vector(-2, 1, 6), Vector(1.0, 1.0, 0.2), 0.6),  # Yellow
    Light(Vector(2, 2, 6), Vector(1.0, 0.2, 1.0), 0.6),   # Magenta
    Light(Vector(0, -0.5, 2), Vector(0.2, 1.0, 1.0), 0.5) # Cyan (uplight)
]

def trace(ray_origin, ray_dir, depth):
    if depth > MAX_DEPTH:
        return Vector(0, 0, 0)

    # Find nearest intersection
    closest_dist = float('inf')
    hit_obj = None
    
    for obj in objects:
        dist = obj.intersect(ray_origin, ray_dir)
        if dist is not None and dist < closest_dist:
            closest_dist = dist
            hit_obj = obj

    if hit_obj is None:
        return Vector(0.05, 0.05, 0.05) # Ambient background

    hit_point = ray_origin + ray_dir * closest_dist
    normal = hit_obj.normal(hit_point)
    base_color = hit_obj.get_color(hit_point)
    
    final_color = Vector(0, 0, 0)
    
    # Lighting calculation
    for light in lights:
        light_dir = (light.position - hit_point).normalize()
        light_dist = (light.position - hit_point).norm()
        
        # Shadow check
        in_shadow = False
        for obj in objects:
            shadow_dist = obj.intersect(hit_point, light_dir)
            if shadow_dist is not None and shadow_dist < light_dist:
                in_shadow = True
                break
                
        if not in_shadow:
            # Diffuse (Lambert)
            diffuse_intensity = max(0, normal.dot(light_dir))
            # Specular (Blinn-Phong)
            view_dir = (ray_origin - hit_point).normalize()
            half_dir = (light_dir + view_dir).normalize()
            specular_intensity = max(0, normal.dot(half_dir)) ** 50
            
            diffuse_color = Vector(base_color.x * light.color.x, 
                                   base_color.y * light.color.y, 
                                   base_color.z * light.color.z) * diffuse_intensity * light.intensity
                                   
            specular_color = light.color * specular_intensity * light.intensity
            
            final_color = final_color + diffuse_color + specular_color

    # Reflection
    if hit_obj.reflectivity > 0:
        reflect_dir = ray_dir.reflect(normal).normalize()
        reflect_color = trace(hit_point, reflect_dir, depth + 1)
        final_color = final_color * (1 - hit_obj.reflectivity) + reflect_color * hit_obj.reflectivity
        
    return final_color

def render_row(y):
    row_pixels = []
    # Camera setup
    fov = math.pi / 3.0
    aspect_ratio = WIDTH / HEIGHT
    camera_origin = Vector(0, 1, -2)
    
    for x in range(WIDTH):
        # Map pixel to normalized device coordinates
        px = (2.0 * (x + 0.5) / WIDTH - 1.0) * math.tan(fov / 2.0) * aspect_ratio
        py = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * math.tan(fov / 2.0)
        
        ray_dir = Vector(px, py, 1.0).normalize()
        pixel_color = trace(camera_origin, ray_dir, 0)
        
        # Simple Tone Mapping (Clamp to 0-1)
        r = min(1.0, max(0.0, pixel_color.x))
        g = min(1.0, max(0.0, pixel_color.y))
        b = min(1.0, max(0.0, pixel_color.z))
        
        row_pixels.append((int(r * 255), int(g * 255), int(b * 255)))
        
    return y, row_pixels

if __name__ == '__main__':
    print(f"Rendering {WIDTH}x{HEIGHT} scene...")
    
    # Initialize blank image
    img = Image.new('RGB', (WIDTH, HEIGHT))
    pixels = img.load()
    
    # Multiprocessing for speed
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(render_row, range(HEIGHT))
    
    # Map calculated rows back to the image
    for y, row in results:
        for x, color in enumerate(row):
            pixels[x, y] = color
            
    img.save('output.png')
    print("Render complete! Saved as output.png.")