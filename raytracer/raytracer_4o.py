import numpy as np
from PIL import Image

# Define vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

# Scene setup
width, height = 800, 600
aspect_ratio = width / height
fov = np.pi / 3

# Colors
BLACK = np.array([0, 0, 0])
WHITE = np.array([1, 1, 1])

# Light and Object Classes
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color) * intensity

class Sphere:
    def __init__(self, center, radius, color, specular=50, reflectivity=0.3):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflectivity = reflectivity

    def intersect(self, origin, direction):
        """Ray-sphere intersection using quadratic formula"""
        OC = origin - self.center
        b = 2 * np.dot(OC, direction)
        c = np.dot(OC, OC) - self.radius**2
        delta = b**2 - 4 * c
        if delta < 0:
            return None
        t1, t2 = (-b - np.sqrt(delta)) / 2, (-b + np.sqrt(delta)) / 2
        return t1 if t1 > 0.001 else (t2 if t2 > 0.001 else None)

# Scene Objects
spheres = [
    Sphere(center=[0, -1, 3], radius=1, color=[1, 0, 0], reflectivity=0.5),
    Sphere(center=[2, 0, 4], radius=1, color=[0, 0, 1], reflectivity=0.4),
    Sphere(center=[-2, 0, 4], radius=1, color=[0, 1, 0], reflectivity=0.4),
    Sphere(center=[0, -5001, 0], radius=5000, color=[1, 1, 0], reflectivity=0.2),
]

lights = [
    Light(position=[-2, 5, 0], color=[1, 0, 0], intensity=1.2),
    Light(position=[2, 5, 0], color=[0, 0, 1], intensity=1.2),
    Light(position=[0, 5, 5], color=[1, 1, 1], intensity=1.5),
]

def trace_ray(origin, direction, depth=3):
    """Trace a ray and compute color with Phong shading & reflections"""
    nearest_t, nearest_obj = float('inf'), None

    # Find closest intersection
    for obj in spheres:
        t = obj.intersect(origin, direction)
        if t and t < nearest_t:
            nearest_t, nearest_obj = t, obj

    if nearest_obj is None:
        return BLACK

    # Compute intersection point & normal
    hit_point = origin + nearest_t * direction
    normal = normalize(hit_point - nearest_obj.center)
    view_dir = -direction

    # Compute lighting
    color = np.zeros(3)
    for light in lights:
        light_dir = normalize(light.position - hit_point)
        shadow_orig = hit_point + normal * 0.001  # Offset to prevent self-shadowing

        # Shadow check
        in_shadow = any(
            obj.intersect(shadow_orig, light_dir) is not None for obj in spheres
        )
        if in_shadow:
            continue

        # Diffuse shading
        diffuse_intensity = max(np.dot(normal, light_dir), 0)
        diffuse = diffuse_intensity * nearest_obj.color * light.color

        # Specular shading
        reflect_dir = reflect(-light_dir, normal)
        specular_intensity = max(np.dot(reflect_dir, view_dir), 0) ** nearest_obj.specular
        specular = specular_intensity * light.color

        # Combine lighting
        color += diffuse + specular

    # Reflection
    if depth > 0 and nearest_obj.reflectivity > 0:
        reflect_dir = normalize(reflect(direction, normal))
        reflect_orig = hit_point + normal * 0.001  # Offset to avoid self-intersection
        reflected_color = trace_ray(reflect_orig, reflect_dir, depth - 1)
        color = color * (1 - nearest_obj.reflectivity) + reflected_color * nearest_obj.reflectivity

    return np.clip(color, 0, 1)

# Render the scene
image = np.zeros((height, width, 3))

camera = np.array([0, 0, 0])
for y in range(height):
    for x in range(width):
        px = (2 * (x + 0.5) / width - 1) * np.tan(fov / 2) * aspect_ratio
        py = (1 - 2 * (y + 0.5) / height) * np.tan(fov / 2)
        direction = normalize(np.array([px, py, 1]))
        image[y, x] = trace_ray(camera, direction)

# Save the image
image = (255 * image).astype(np.uint8)
img = Image.fromarray(image, "RGB")
img.save("raytraced_scene.png")

print("Rendering complete. Saved as 'raytraced_scene.png'")
