import math
from PIL import Image

# Vector operations
def add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

def sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def scale(v, scalar):
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar]

def normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / length, v[1] / length, v[2] / length]

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

# Sphere class
class Sphere:
    def __init__(self, center, radius, color, specular):
        self.center = center
        self.radius = radius
        self.color = color
        self.specular = specular

# Scene definition
spheres = [
    Sphere([0, -1, 3], 1, [255, 0, 0], 500),       # Red sphere
    Sphere([2, 0, 4], 1, [0, 255, 0], 500),       # Green sphere
    Sphere([-2, 0, 4], 1, [0, 0, 255], 500),      # Blue sphere
    Sphere([0, -5001, 0], 5000, [255, 255, 0], 1000) # Yellow floor
]

lights = [
    {"position": [5, 5, 5], "color": [255, 255, 255]},  # White light
    {"position": [-3, 2, 1], "color": [255, 0, 255]}, # Magenta light
    {"position": [0, 5, 10], "color": [0, 255, 255]}  # Cyan light
]

# Image dimensions
width = 800
height = 600

# Background color
background_color = [0, 0, 0]

# Ray-sphere intersection
def intersect_sphere(ray, sphere):
    oc = sub(ray.origin, sphere.center)
    k1 = dot(ray.direction, ray.direction)
    k2 = 2 * dot(oc, ray.direction)
    k3 = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = k2 * k2 - 4 * k1 * k3
    if discriminant < 0:
        return None
    t1 = (-k2 + math.sqrt(discriminant)) / (2 * k1)
    t2 = (-k2 - math.sqrt(discriminant)) / (2 * k1)
    t = min(t1, t2)
    if t > 0:
        return t
    return None

# Color calculation
def calculate_color(point, normal, sphere):
    color = [0, 0, 0]
    for light in lights:
        light_direction = normalize(sub(light["position"], point))
        
        # Diffuse lighting
        diffuse_intensity = max(0, dot(normal, light_direction))
        diffuse_color = scale(light["color"], diffuse_intensity)
        color = add(color, scale(sphere.color, diffuse_intensity))

        # Specular lighting
        reflection_direction = reflect(light_direction, normal)
        specular_intensity = max(0, dot(reflection_direction, normalize(sub([0,0,0], point)))) ** sphere.specular
        specular_color = scale(light["color"], specular_intensity)
        color = add(color, scale(specular_color, 0.5)) # Reduce specular intensity
        
    return [min(255, c) for c in color]

def reflect(light_direction, normal):
    return sub(scale(normal, 2 * dot(light_direction, normal)), light_direction)

# Ray tracing loop
image = Image.new("RGB", (width, height), (0, 0, 0))
pixels = image.load()

for x in range(width):
    for y in range(height):
        # Create ray
        px = (x - width / 2) / width
        py = -(y - height / 2) / height
        ray = Ray([0, 0, 0], [px, py, 1])
        
        # Find closest intersection
        closest_t = float('inf')
        closest_sphere = None
        for sphere in spheres:
            t = intersect_sphere(ray, sphere)
            if t is not None and t < closest_t:
                closest_t = t
                closest_sphere = sphere
        
        # Calculate color
        if closest_sphere is not None:
            point = add(ray.origin, scale(ray.direction, closest_t))
            normal = normalize(sub(point, closest_sphere.center))
            color = calculate_color(point, normal, closest_sphere)
            pixels[x, y] = tuple(int(c) for c in color)
        else:
            pixels[x, y] = tuple(background_color)

# Save image
image.save("output.png", "PNG")