import math
import numpy as np
import matplotlib.pyplot as plt

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    def __neg__(self):
        return self * -1.0

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        return self / l if l > 0 else Vec3(0, 0, 1)

def reflect(incident, normal):
    return incident - normal * (2.0 * incident.dot(normal))

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Material:
    def __init__(self, color, ambient=0.15, diffuse=0.8, specular=0.5, shininess=32, reflective=0.0):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflective = reflective

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return float('inf')
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        if t1 > 0.001:
            return t1
        if t2 > 0.001:
            return t2
        return float('inf')

    def get_normal(self, p):
        return (p - self.center).normalize()

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material

    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 0.0001:
            return float('inf')
        t = self.normal.dot(self.point - ray.origin) / denom
        if t > 0.001:
            return t
        return float('inf')

    def get_normal(self, p):
        return self.normal

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

def shade(hit, normal, view_dir, material, lights, scene):
    color = material.color * material.ambient
    for light in lights:
        light_dir = (light.position - hit).normalize()
        # Shadow check
        shadow_ray = Ray(hit + normal * 0.001, light_dir)
        shadowed = False
        for obj in scene:
            if obj.intersect(shadow_ray) < float('inf'):
                shadowed = True
                break
        if not shadowed:
            # Diffuse
            diff = max(0.0, normal.dot(light_dir))
            color = color + material.color * light.color * (diff * material.diffuse * light.intensity)
            # Specular (Blinn-Phong like)
            half_dir = (light_dir + view_dir).normalize()
            spec = max(0.0, normal.dot(half_dir)) ** material.shininess
            color = color + Vec3(1.0,1.0,1.0) * (spec * material.specular * light.intensity)
    return color

def trace(ray, scene, lights, depth=0, max_depth=4):
    if depth > max_depth:
        return Vec3(0.05, 0.05, 0.1)  # dark background

    closest_t = float('inf')
    closest_obj = None
    for obj in scene:
        t = obj.intersect(ray)
        if t < closest_t:
            closest_t = t
            closest_obj = obj

    if closest_obj is None or closest_t == float('inf'):
        return Vec3(0.1, 0.15, 0.3)  # sky

    hit = ray.origin + ray.direction * closest_t
    normal = closest_obj.get_normal(hit)
    view_dir = (-ray.direction).normalize()  # to viewer

    # Checkerboard for floor
    if isinstance(closest_obj, Plane):
        check = (int(hit.x * 1.5) + int(hit.z * 1.5)) % 2
        mat_color = Vec3(0.95, 0.95, 0.95) if check == 0 else Vec3(0.25, 0.25, 0.25)
        material = Material(mat_color, reflective=0.08, shininess=8)
    else:
        material = closest_obj.material

    local_color = shade(hit, normal, view_dir, material, lights, scene)

    # Reflection
    reflected = Vec3(0,0,0)
    if material.reflective > 0.0 and depth < max_depth:
        refl_dir = reflect(ray.direction, normal)
        refl_ray = Ray(hit + normal * 0.002, refl_dir)
        reflected = trace(refl_ray, scene, lights, depth + 1, max_depth)

    return local_color * (1.0 - material.reflective) + reflected * material.reflective

# Scene setup
scene = []
lights = []

# Floor
floor_mat = Material(Vec3(0.6, 0.6, 0.6), reflective=0.1)
scene.append(Plane(Vec3(0, -1.5, 0), Vec3(0, 1, 0), floor_mat))

# Colorful reflective spheres
scene.append(Sphere(Vec3(-2.5, 0, 3), 1.2, Material(Vec3(0.95, 0.2, 0.2), reflective=0.75, shininess=90)))
scene.append(Sphere(Vec3(2.5, 0, 4), 1.3, Material(Vec3(0.2, 0.95, 0.3), reflective=0.65, shininess=60)))
scene.append(Sphere(Vec3(0, 1.8, 1.5), 0.9, Material(Vec3(0.2, 0.6, 0.95), reflective=0.85, shininess=120)))
scene.append(Sphere(Vec3(-1.2, -0.3, 6), 1.0, Material(Vec3(0.9, 0.8, 0.1), reflective=0.4, shininess=40)))
scene.append(Sphere(Vec3(3.5, 1.0, 2.5), 0.8, Material(Vec3(0.8, 0.3, 0.9), reflective=0.55)))

# Many colorful lights
for i in range(18):
    angle = i * (2 * math.pi / 18)
    lx = 7 * math.cos(angle)
    lz = 4 + 6 * math.sin(angle)
    ly = 5.5 + 1.5 * math.sin(i * 1.1)
    r = 0.5 + 0.5 * math.sin(i * 0.7)
    g = 0.5 + 0.5 * math.cos(i * 0.9)
    b = 0.5 + 0.5 * math.sin(i * 1.3 + 1)
    lights.append(Light(Vec3(lx, ly, lz), Vec3(r, g, b), intensity=1.8))

# Render parameters
WIDTH = 800
HEIGHT = 600
camera_pos = Vec3(0.0, 2.5, -11.0)
fov = math.radians(55)
aspect = WIDTH / HEIGHT
scale = math.tan(fov / 2)

img = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

print("Starting render (800x600 with reflections and 18 colorful lights)... This may take 30-90 seconds.")

for y in range(HEIGHT):
    for x in range(WIDTH):
        px = (2 * (x + 0.5) / WIDTH - 1) * aspect * scale
        py = (1 - 2 * (y + 0.5) / HEIGHT) * scale
        direction = Vec3(px, py, 1.0).normalize()
        ray = Ray(camera_pos, direction)
        col = trace(ray, scene, lights)
        img[y, x] = [col.x, col.y, col.z]

    if y % 80 == 0:
        print(f"Progress: {int(100 * y / HEIGHT)}%")

# Post-process: gamma correction and clamp
img = np.clip(img, 0.0, 1.0)
img = np.power(img, 1.0 / 2.2)

plt.imsave("colorful_lights_raytrace.png", img)
print("Render complete! Image saved as 'colorful_lights_raytrace.png'")