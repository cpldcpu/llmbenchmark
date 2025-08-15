import math
from PIL import Image

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        length = self.length()
        return Vec3(self.x/length, self.y/length, self.z/length)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Sphere:
    def __init__(self, center, radius, color, ambient=0.1, diffuse=1.0, specular=0.5, reflectivity=0.5, exponent=50):
        self.center = center
        self.radius = radius
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.reflectivity = reflectivity
        self.exponent = exponent

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

def reflect(vector, normal):
    return vector - normal * 2 * vector.dot(normal)

def intersect_sphere(ray, sphere):
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius**2
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return (False, float('inf'), None)
    
    t1 = (-b - math.sqrt(discriminant)) / (2*a)
    t2 = (-b + math.sqrt(discriminant)) / (2*a)
    t = min(t1, t2)
    if t < 0:
        t = max(t1, t2)
        if t < 0:
            return (False, float('inf'), None)
    
    point = ray.origin + ray.direction * t
    normal = (point - sphere.center).normalize()
    return (True, t, normal)

def trace_ray(ray, spheres, lights, depth=3):
    if depth <= 0:
        return Vec3(0, 0, 0)
    
    closest_t = float('inf')
    closest_sphere = None
    closest_normal = None

    for sphere in spheres:
        hit, t, normal = intersect_sphere(ray, sphere)
        if hit and t < closest_t:
            closest_t = t
            closest_sphere = sphere
            closest_normal = normal

    if not closest_sphere:
        return Vec3(0, 0, 0)  # Background color

    # Calculate hit point and material
    hit_point = ray.origin + ray.direction * closest_t
    material = closest_sphere
    normal = closest_normal
    view_dir = (ray.origin - hit_point).normalize()

    # Ambient lighting
    color = material.color * material.ambient

    # Calculate lighting for each light source
    for light in lights:
        light_vec = light.position - hit_point
        light_dist = light_vec.length()
        light_dir = light_vec.normalize()

        # Shadow check
        shadow_ray = Ray(hit_point + normal * 1e-4, light_dir)
        shadow_hit = False
        for sphere in spheres:
            hit, t, _ = intersect_sphere(shadow_ray, sphere)
            if hit and t < light_dist:
                shadow_hit = True
                break
        
        if not shadow_hit:
            # Diffuse component
            diff = max(normal.dot(light_dir), 0.0)
            diffuse = material.diffuse * diff * light.intensity
            color += material.color * light.color * diffuse

            # Specular component
            reflect_dir = reflect(-light_dir, normal)
            spec = max(view_dir.dot(reflect_dir), 0.0) ** material.exponent
            specular = material.specular * spec * light.intensity
            color += light.color * specular

    # Recursive reflection
    if material.reflectivity > 0:
        reflect_dir = reflect(ray.direction, normal).normalize()
        reflect_ray = Ray(hit_point + normal * 1e-4, reflect_dir)
        reflected_color = trace_ray(reflect_ray, spheres, lights, depth-1)
        color += reflected_color * material.reflectivity

    # Clamp color values
    color.x = min(max(color.x, 0), 1)
    color.y = min(max(color.y, 0), 1)
    color.z = min(max(color.z, 0), 1)
    
    return color

# Scene setup
image_width = 400
image_height = 300
viewport_width = 1.0
viewport_height = image_height / image_width * viewport_width
camera = Vec3(0, 0, -1)

spheres = [
    Sphere(Vec3(0, -1000, 0), 990, Vec3(0.5, 0.5, 0.5), ambient=0.2, reflectivity=0.2),
    Sphere(Vec3(-0.8, 0.2, 0), 0.4, Vec3(1, 0.2, 0.2), reflectivity=0.5, exponent=50),
    Sphere(Vec3(0.8, 0.2, 0), 0.4, Vec3(0.2, 1, 0.2), reflectivity=0.5, exponent=50),
    Sphere(Vec3(0, 0.8, 1), 0.4, Vec3(0.2, 0.2, 1), reflectivity=0.5, exponent=50)
]

lights = [
    Light(Vec3(2, 2, -1), Vec3(1, 1, 1), 0.8),
    Light(Vec3(-2, 2, 0), Vec3(1, 0, 0), 0.6),
    Light(Vec3(0, 2, 2), Vec3(0, 0, 1), 0.6)
]

# Render image
image = Image.new('RGB', (image_width, image_height))
pixels = image.load()

for y in range(image_height):
    for x in range(image_width):
        u = (x + 0.5) / image_width
        v = (y + 0.5) / image_height
        ray_x = -viewport_width/2 + u * viewport_width
        ray_y = -viewport_height/2 + v * viewport_height
        ray_dir = Vec3(ray_x, ray_y, 1).normalize()
        ray = Ray(camera, ray_dir)
        
        color = trace_ray(ray, spheres, lights, depth=3)
        
        r = int(color.x * 255)
        g = int(color.y * 255)
        b = int(color.z * 255)
        pixels[x, y] = (r, g, b)

image.save('render.png')
print("Rendering complete! Saved as render.png")