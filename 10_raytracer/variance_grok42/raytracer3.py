import math
import random
from PIL import Image
import numpy as np

# Simple Vec3 class for 3D math
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * float(other), self.y * float(other), self.z * float(other))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (1.0 / float(other))
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        len = self.length()
        if len > 0.0001:
            return self * (1.0 / len)
        return Vec3(0,0,1)
    
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def cross(self, other):
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

    def at(self, t):
        return self.origin + self.direction * t

# Sphere with material info
class Sphere:
    def __init__(self, center, radius, albedo, is_emissive=False, emission=Vec3(0,0,0)):
        self.center = center
        self.radius = float(radius)
        self.albedo = albedo  # base color
        self.is_emissive = is_emissive
        self.emission = emission if is_emissive else Vec3(0,0,0)

    def hit(self, ray, t_min = 0.001, t_max = 1e30):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        sqrt_d = math.sqrt(discriminant)
        for sign in [-1, 1]:
            t = (-b + sign * sqrt_d) / (2*a)
            if t_min < t < t_max:
                p = ray.at(t)
                normal = (p - self.center) / self.radius
                return {
                    't': t,
                    'p': p,
                    'normal': normal,
                    'albedo': self.albedo,
                    'is_emissive': self.is_emissive,
                    'emission': self.emission
                }
        return None

# Hit the world (list of spheres)
def hit_world(ray, world, t_min=0.001, t_max=1e30):
    closest = None
    closest_t = t_max
    for obj in world:
        hit = obj.hit(ray, t_min, closest_t)
        if hit:
            closest_t = hit['t']
            closest = hit
    return closest

# Main render function
def render():
    WIDTH = 800
    HEIGHT = 600
    print(f"Rendering {WIDTH}x{HEIGHT} image... This may take 20-60 seconds.")

    # Create world
    world = []
    point_lights = []

    random.seed(123)

    # Floor
    world.append(Sphere(Vec3(0, -100.5, -1), 100, Vec3(0.85, 0.8, 0.75)))

    # Cluster of colorful spheres
    world.append(Sphere(Vec3(0, 0, -2), 0.5, Vec3(0.9, 0.1, 0.1)))
    world.append(Sphere(Vec3(-1.0, 0.2, -1.8), 0.5, Vec3(0.1, 0.9, 0.2)))
    world.append(Sphere(Vec3(1.2, -0.1, -2.3), 0.5, Vec3(0.1, 0.2, 0.9)))
    world.append(Sphere(Vec3(-0.5, 0.8, -1.6), 0.35, Vec3(0.9, 0.9, 0.1)))
    world.append(Sphere(Vec3(0.6, 0.6, -1.2), 0.4, Vec3(0.9, 0.4, 0.8)))

    # More objects for interest
    world.append(Sphere(Vec3(-2.2, 0.4, -3.5), 0.6, Vec3(0.6, 0.7, 0.9)))
    world.append(Sphere(Vec3(2.5, 1.0, -3.0), 0.55, Vec3(0.95, 0.6, 0.3)))

    # Create many colourful lights
    for i in range(18):
        # Place them in an arc above the scene
        a = i / 18 * math.pi * 1.6 - 0.3
        dist = 4.5 + random.uniform(-0.5, 1.5)
        x = math.cos(a) * dist
        z = -2 + math.sin(a) * dist * 0.7
        y = 2.5 + random.uniform(0.5, 2.5)
        
        pos = Vec3(x, y, z)
        
        # Vibrant colors
        r = 0.6 + 0.4 * random.random()
        g = 0.6 + 0.4 * random.random()
        b = 0.6 + 0.4 * random.random()
        light_col = Vec3(r, g, b) * 1.8
        
        point_lights.append({'pos': pos, 'color': light_col})
        
        # Add small glowing sphere
        world.append(Sphere(pos, 0.09, Vec3(1,1,1), True, light_col * 0.7))

    # Camera setup (looking towards origin from positive z a bit up)
    cam_pos = Vec3(0.0, 0.8, 3.5)
    look_at = Vec3(0.0, 0.3, -1.5)
    up = Vec3(0,1,0)
    
    # Simple pinhole camera
    w = (cam_pos - look_at).normalize()
    u = up.cross(w).normalize()
    v = w.cross(u)
    
    # Viewport
    vfov = 50.0
    aspect = WIDTH / HEIGHT
    theta = math.radians(vfov)
    half_height = math.tan(theta/2)
    half_width = aspect * half_height
    
    lower_left = cam_pos - w - u*half_width - v*half_height
    horizontal = u * (2 * half_width)
    vertical = v * (2 * half_height)

    # Render to numpy array
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for y in range(HEIGHT):
        if y % 100 == 0:
            print(f"Progress: {y/HEIGHT*100:.1f}%")
        for x in range(WIDTH):
            u = x / (WIDTH - 1)
            v = y / (HEIGHT - 1)
            
            ray_dir = lower_left + horizontal * u + vertical * v - cam_pos
            ray = Ray(cam_pos, ray_dir)
            
            hit = hit_world(ray, world)
            
            if hit and hit['is_emissive']:
                col = hit['emission']
            elif hit:
                # Diffuse shading with multiple lights
                p = hit['p']
                n = hit['normal'].normalize()
                albedo = hit['albedo']
                col = Vec3(0.07, 0.07, 0.1)  # ambient
                
                for light in point_lights:
                    lvec = light['pos'] - p
                    dist = lvec.length()
                    ldir = lvec.normalize()
                    
                    # Shadow test
                    shadow_ray = Ray(p + n * 0.003, ldir)
                    if hit_world(shadow_ray, world, 0.001, dist - 0.05) is None:
                        diff = max(0.0, n.dot(ldir))
                        att = 1.0 / (0.5 + 0.1*dist + 0.05*dist*dist)
                        col = col + albedo * light['color'] * (diff * att)
            else:
                # Sky gradient
                unit_dir = ray.direction.normalize()
                t = 0.5 * (unit_dir.y + 1.0)
                col = Vec3(0.6, 0.75, 1.0) * (1.0 - t) + Vec3(0.9, 0.95, 1.0) * t
            
            # To sRGB approx with gamma
            r = int(255 * min(1.0, math.sqrt(col.x)))
            g = int(255 * min(1.0, math.sqrt(col.y)))
            b = int(255 * min(1.0, math.sqrt(col.z)))
            
            pixels[HEIGHT - 1 - y, x] = [r, g, b]  # Flip vertically for image coord

    im = Image.fromarray(pixels)
    im.save("colorful_lights_raytrace.png")
    print("Done! Image saved as 'colorful_lights_raytrace.png'")

if __name__ == "__main__":
    render()