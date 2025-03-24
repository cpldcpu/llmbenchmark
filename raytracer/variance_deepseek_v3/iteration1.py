import math
from PIL import Image

# Vector operations
def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def normalize(v):
    length = math.sqrt(dot(v, v))
    return (v[0] / length, v[1] / length, v[2] / length)

def subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)

# Ray-sphere intersection
def intersect_sphere(ray_origin, ray_dir, sphere):
    center, radius = sphere
    oc = subtract(ray_origin, center)
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None
    else:
        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t > 0:
            return t
        else:
            return None

# Ray tracing function
def trace_ray(ray_origin, ray_dir, spheres, lights):
    closest_t = float('inf')
    closest_sphere = None

    for sphere in spheres:
        t = intersect_sphere(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if closest_sphere is None:
        return (0, 0, 0)  # Background color

    # Compute intersection point and normal
    sphere_center, radius = closest_sphere
    intersection = (ray_origin[0] + closest_t * ray_dir[0],
                    ray_origin[1] + closest_t * ray_dir[1],
                    ray_origin[2] + closest_t * ray_dir[2])
    normal = normalize(subtract(intersection, sphere_center))

    # Shading
    color = (0, 0, 0)
    for light_pos, light_color in lights:
        light_dir = normalize(subtract(light_pos, intersection))
        diffuse = max(0, dot(normal, light_dir))
        color = (color[0] + light_color[0] * diffuse,
                 color[1] + light_color[1] * diffuse,
                 color[2] + light_color[2] * diffuse)

    # Clamp color values to [0, 255]
    color = (min(255, max(0, int(color[0]))),
             min(255, max(0, int(color[1]))),
             min(255, max(0, int(color[2]))))

    return color

# Render the scene
def render(width, height, spheres, lights):
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            # Normalize pixel coordinates to [-1, 1]
            px = (x - width / 2) / (width / 2)
            py = -(y - height / 2) / (height / 2)

            # Ray direction
            ray_dir = normalize((px, py, -1))

            # Trace the ray
            color = trace_ray((0, 0, 0), ray_dir, spheres, lights)

            # Set pixel color
            pixels[x, y] = color

    return image

# Define the scene
spheres = [
    ((0, -1, -5), 1),  # Ground sphere
    ((-2, 0, -10), 2), # Left sphere
    ((2, 0, -10), 2),  # Right sphere
]

lights = [
    ((-5, 5, 0), (255, 0, 0)),    # Red light
    ((5, 5, 0), (0, 255, 0)),     # Green light
    ((0, 5, 5), (0, 0, 255)),     # Blue light
]

# Render and save the image
width, height = 800, 600
image = render(width, height, spheres, lights)
image.save("raytraced_scene1.png")
