# ...existing code...
import math
from PIL import Image

class Vector:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def __add__(self, other): return Vector(self.x+other.x, self.y+other.y, self.z+other.z)
    def __sub__(self, other): return Vector(self.x-other.x, self.y-other.y, self.z-other.z)
    def __mul__(self, k):     return Vector(self.x*k, self.y*k, self.z*k)
    def dot(self, other):     return self.x*other.x + self.y*other.y + self.z*other.z
    def norm(self):           return math.sqrt(self.dot(self))
    def normalize(self):      
        l = self.norm()
        return Vector(self.x/l, self.y/l, self.z/l)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Sphere:
    def __init__(self, center, radius, color, reflectivity=0.0):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflectivity = reflectivity
    def intersect(self, ray):
        oc = ray.origin - self.center
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius*self.radius
        disc = b*b - 4*c
        if disc < 0: return None
        t1 = (-b - math.sqrt(disc)) / 2
        t2 = (-b + math.sqrt(disc)) / 2
        return min(t for t in [t1,t2] if t>1e-3) if any(t>1e-3 for t in [t1,t2]) else None

    def normal(self, hit_point):
        return (hit_point - self.center).normalize()

class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

def trace_ray(ray, spheres, lights, depth=0, max_depth=3):
    hit_obj, min_t = None, float('inf')
    for obj in spheres:
        t = obj.intersect(ray)
        if t and t<min_t:
            min_t, hit_obj = t, obj
    if not hit_obj:
        return (0,0,0)

    hit_pos = ray.origin + ray.direction * min_t
    normal = hit_obj.normal(hit_pos)
    surface_color = [0, 0, 0]
    for light in lights:
        to_light = (light.position - hit_pos).normalize()
        # Check shadow
        shadow_ray = Ray(hit_pos + to_light*1e-3, to_light)
        if any(s.intersect(shadow_ray) for s in spheres if s != hit_obj):
            continue
        diff = max(normal.dot(to_light), 0)
        surface_color[0] += hit_obj.color[0] * light.color[0] * diff
        surface_color[1] += hit_obj.color[1] * light.color[1] * diff
        surface_color[2] += hit_obj.color[2] * light.color[2] * diff

    # Reflection
    if depth < max_depth and hit_obj.reflectivity > 0:
        r = ray.direction - normal * (2*ray.direction.dot(normal))
        reflect_ray = Ray(hit_pos + r*1e-3, r)
        reflected_color = trace_ray(reflect_ray, spheres, lights, depth+1, max_depth)
        surface_color = [
            (1-hit_obj.reflectivity)*surface_color[i] + hit_obj.reflectivity*reflected_color[i]
            for i in range(3)
        ]
    return tuple(int(min(c, 255)) for c in surface_color)

def render_scene():
    width, height = 800, 600
    camera = Vector(0, 0, -1)
    spheres = [
        Sphere(Vector(0, -1, 3), 1, (255,0,0), 0.3),
        Sphere(Vector(2, 0, 4), 1, (0,255,0), 0.5),
        Sphere(Vector(-2, 0, 4), 1, (0,0,255), 0.5),
        Sphere(Vector(0, -5001, 0), 5000, (255,255,0)),
    ]
    lights = [
        Light(Vector(2,5,-2), (255,255,255)),
        Light(Vector(-3,5,2), (255,128,128)),
        Light(Vector(0,5,5), (128,255,128)),
        Light(Vector(0,-5,10), (128,128,255)),
    ]
    img = Image.new("RGB", (width, height), "black")
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            # NDC to camera space
            i = (x - width/2) / height
            j = (y - height/2) / height
            direction = Vector(i, -j, 1)
            ray = Ray(camera, direction)
            color = trace_ray(ray, spheres, lights)
            pixels[x, y] = color
    img.save("render.png")

if __name__ == "__main__":
    render_scene()