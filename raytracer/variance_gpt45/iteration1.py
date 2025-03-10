import numpy as np
from PIL import Image

# Scene setup
WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3

# Simple sphere class
class Sphere:
    def __init__(self, center, radius, color, reflection=0.0):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = reflection
    
    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return np.inf
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        if t1 > 1e-4:
            return t1
        if t2 > 1e-4:
            return t2
        return np.inf

# Scene objects
objects = [
    Sphere([0, -10004, 20], 10000, [0.2, 0.2, 0.2]),  # Ground
    Sphere([0, 0, 20], 4, [1.0, 0.32, 0.36], 0.5),    # Red sphere
    Sphere([5, -1, 15], 2, [0.9, 0.76, 0.46], 0.2),   # Yellow sphere
    Sphere([-5.5, 0, 15], 3, [0.65, 0.77, 0.97], 0.3),# Blue sphere
]

# Light sources: position, color
lights = [
    (np.array([0, 20, 0]), np.array([0.7, 0.7, 0.7])),  # White top light
    (np.array([30, 50, -25]), np.array([1.0, 0.0, 0.0])), # Red light
    (np.array([-30, 50, -25]), np.array([0.0, 0.0, 1.0])), # Blue light
    (np.array([0, 40, 20]), np.array([0.0, 1.0, 0.0])), # Green light
]

# Normalize function
def normalize(vec):
    return vec / np.linalg.norm(vec)

# Reflect function
def reflect(I, N):
    return I - 2 * np.dot(N, I) * N

def trace(ray_origin, ray_dir, depth):
    obj_hit = None
    nearest_dist = np.inf
    for obj in objects:
        dist = obj.intersect(ray_origin, ray_dir)
        if dist < nearest_dist:
            nearest_dist = dist
            obj_hit = obj
    
    if obj_hit is None:
        return np.array([0.0, 0.0, 0.0]) # Background color
    
    # Compute hit point and normal
    phit = ray_origin + ray_dir * nearest_dist
    nhit = normalize(phit - obj_hit.center)
    bias = 1e-4
    pixel_color = np.zeros(3)
    
    # Lighting calculation
    for light_pos, light_color in lights:
        # Direction from point to light
        light_dir = normalize(light_pos - phit)
        
        # Shadow check
        shadow = False
        for obj in objects:
            if obj == obj_hit:
                continue
            if obj.intersect(phit + nhit * bias, light_dir) < np.linalg.norm(light_pos - phit):
                shadow = True
                break
        if not shadow:
            illumination = max(0, np.dot(nhit, light_dir))
            pixel_color += obj_hit.color * light_color * illumination
    
    # Reflection (recursive)
    if depth < MAX_DEPTH and obj_hit.reflection > 0:
        reflection_dir = normalize(reflect(ray_dir, nhit))
        reflection_color = trace(phit + nhit * bias, reflection_dir, depth + 1)
        pixel_color = pixel_color * (1 - obj_hit.reflection) + reflection_color * obj_hit.reflection
    
    return np.clip(pixel_color, 0, 1)

def render():
    aspect_ratio = WIDTH / HEIGHT
    fov = np.pi / 3.0 # 60 degrees field of view
    screen = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Compute primary ray
            xx = (2 * (x + 0.5) / WIDTH - 1) * np.tan(fov / 2) * aspect_ratio
            yy = -(2 * (y + 0.5) / HEIGHT - 1) * np.tan(fov / 2)
            ray_dir = normalize(np.array([xx, yy, 1]))
            color = trace(np.array([0, 0, 0]), ray_dir, 0)
            screen[y, x] = color
        print(f"Rendering progress: {100*y//HEIGHT}%", end="\r")

    # Convert to image
    img = np.clip(screen * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(img, 'RGB')
    image.save('raytraced_scene1.png')
    print("Rendering completed!")

if __name__ == "__main__":
    render()