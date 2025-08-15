import numpy as np
from PIL import Image

# Define constants
WIDTH, HEIGHT = 800, 600
FOV = np.pi / 2.0
MAX_DEPTH = 3

# Define vector math functions
def normalize(v):
    return v / np.linalg.norm(v)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    oc = ray_origin - center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    else:
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        else:
            return None

# Define a simple scene with spheres and light sources
scene = [
    {'type': 'sphere', 'center': np.array([0, -1, 3]), 'radius': 1, 'color': np.array([1, 0, 0]), 'specular': 500},
    {'type': 'sphere', 'center': np.array([2, 0, 4]), 'radius': 1, 'color': np.array([0, 0, 1]), 'specular': 500},
    {'type': 'sphere', 'center': np.array([-2, 0, 4]), 'radius': 1, 'color': np.array([0, 1, 0]), 'specular': 500},
    {'type': 'sphere', 'center': np.array([0, -5001, 0]), 'radius': 5000, 'color': np.array([1, 1, 0]), 'specular': 1000}
]

lights = [
    {'position': np.array([5, 5, -5]), 'color': np.array([1, 1, 1])},
    {'position': np.array([-5, 5, -5]), 'color': np.array([0.5, 0.5, 1])},
    {'position': np.array([0, 5, -5]), 'color': np.array([1, 0.5, 0.5])}
]

# Raytracing function
def trace_ray(ray_origin, ray_direction, depth=0):
    closest_t = np.inf
    closest_sphere = None

    for sphere in scene:
        t = sphere_intersect(sphere['center'], sphere['radius'], ray_origin, ray_direction)
        if t and t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if not closest_sphere:
        return np.array([0, 0, 0])

    # Compute intersection point and normal
    point = ray_origin + closest_t * ray_direction
    normal = normalize(point - closest_sphere['center'])

    # Compute lighting
    color = np.zeros(3)
    for light in lights:
        light_dir = normalize(light['position'] - point)
        light_distance = np.linalg.norm(light['position'] - point)

        # Shadow check
        shadow_intersect = False
        for sphere in scene:
            t = sphere_intersect(sphere['center'], sphere['radius'], point, light_dir)
            if t and t < light_distance:
                shadow_intersect = True
                break

        if not shadow_intersect:
            # Diffuse
            diffuse = np.dot(normal, light_dir)
            if diffuse > 0:
                color += closest_sphere['color'] * light['color'] * diffuse

            # Specular
            if closest_sphere['specular'] != -1:
                reflected_dir = reflected(light_dir, normal)
                specular = np.dot(reflected_dir, -ray_direction)
                if specular > 0:
                    specular = specular ** closest_sphere['specular']
                    color += light['color'] * specular

    # Reflection
    if depth < MAX_DEPTH:
        reflection_dir = reflected(ray_direction, normal)
        reflection_color = trace_ray(point, reflection_dir, depth + 1)
        color += reflection_color * 0.5

    return color

# Render the scene
camera = np.array([0, 0, 0])
image = np.zeros((HEIGHT, WIDTH, 3))

for y in range(HEIGHT):
    for x in range(WIDTH):
        # Compute ray direction
        screen_x = (2 * (x + 0.5) / WIDTH - 1) * np.tan(FOV / 2.0) * (WIDTH / HEIGHT)
        screen_y = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(FOV / 2.0)
        ray_direction = normalize(np.array([screen_x, screen_y, 1]))

        # Trace the ray and set the pixel color
        color = trace_ray(camera, ray_direction)
        image[y, x] = np.clip(color, 0, 1)

# Save the image
Image.fromarray((image * 255).astype(np.uint8)).save('raytraced_scene3.png')
