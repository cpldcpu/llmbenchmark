import numpy as np
from PIL import Image
import math

# --- Vector & Math Helpers ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def intersect_sphere(O, D, S, R):
    # Solves t^2 + 2t(D . OS) + |OS|^2 - R^2 = 0
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        t0 = (-b - distSqrt) / 2.0
        t1 = (-b + distSqrt) / 2.0
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect_plane(O, D, P, N):
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    return d if d > 0 else np.inf

# --- Scene Objects ---
class Sphere:
    def __init__(self, position, radius, color, reflectivity=0.0):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.color = np.array(color, dtype=float)
        self.reflectivity = reflectivity

    def intersect(self, O, D):
        return intersect_sphere(O, D, self.position, self.radius)

    def normal(self, M):
        return normalize(M - self.position)

class Plane:
    def __init__(self, position, normal, reflectivity=0.0):
        self.position = np.array(position, dtype=float)
        self.n = normalize(np.array(normal, dtype=float))
        self.reflectivity = reflectivity

    def intersect(self, O, D):
        return intersect_plane(O, D, self.position, self.n)

    def normal(self, M):
        return self.n

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)
        self.intensity = intensity

# --- Raytracing Engine ---
def get_intersect(O, D, scene):
    t_min = np.inf
    obj_hit = None
    for obj in scene:
        t = obj.intersect(O, D)
        if t < t_min:
            t_min = t
            obj_hit = obj
    return t_min, obj_hit

def trace_ray(O, D, scene, lights, depth=0):
    t, obj = get_intersect(O, D, scene)
    
    # Background color (dark grey)
    if obj is None:
        return np.array([0.05, 0.05, 0.05])

    M = O + D * t          # Point of intersection
    N = obj.normal(M)      # Surface normal
    
    # Nudge intersection point to prevent self-shadowing (acne)
    M_nudge = M + N * 1e-4

    # Determine base color (Checkerboard for Plane)
    if isinstance(obj, Plane):
        if (int(math.floor(M[0] * 2)) + int(math.floor(M[2] * 2))) % 2 == 0:
            obj_color = np.array([1.0, 1.0, 1.0])
        else:
            obj_color = np.array([0.1, 0.1, 0.1])
    else:
        obj_color = obj.color

    # Initialize pixel color with ambient light
    color = obj_color * 0.05

    # Calculate Lighting (Diffuse & Specular) + Shadows
    for light in lights:
        L = light.position - M_nudge
        dist_to_light = np.linalg.norm(L)
        L = normalize(L)
        
        # Shadow check
        t_shadow, obj_shadow = get_intersect(M_nudge, L, scene)
        if t_shadow < dist_to_light:
            continue # In shadow
        
        # Diffuse reflection
        dot_ln = max(np.dot(L, N), 0.0)
        diffuse = dot_ln * light.color * light.intensity * obj_color
        color += diffuse
        
        # Specular reflection (Blinn-Phong)
        V = normalize(O - M)
        H = normalize(L + V)
        specular_intensity = max(np.dot(N, H), 0.0) ** 50
        specular = specular_intensity * light.color * light.intensity
        color += specular

    # Calculate reflections
    if depth < 3 and obj.reflectivity > 0:
        V = normalize(D)
        R = V - 2 * np.dot(V, N) * N
        reflect_color = trace_ray(M_nudge, R, scene, lights, depth + 1)
        color = color * (1 - obj.reflectivity) + reflect_color * obj.reflectivity

    # Ensure RGB values don't exceed 1.0
    return np.clip(color, 0, 1)

# --- Scene Configuration ---
def main():
    w, h = 800, 600

    # Primitives
    scene = [
        Plane([0, -1, 0], [0, 1, 0], reflectivity=0.2),                     # Floor
        Sphere([0, 0, 2.5], 1.0, [0.1, 0.1, 0.1], reflectivity=0.9),        # Center mirror sphere
        Sphere([-2.2, 0, 3], 1.0, [0.9, 0.9, 0.9], reflectivity=0.05),      # Left matte sphere
        Sphere([2.2, 0, 3], 1.0, [0.9, 0.9, 0.9], reflectivity=0.05)        # Right matte sphere
    ]

    # Many colourful lights
    lights = [
        Light([-3, 3, 1],   [1.0, 0.2, 0.2], 1.2), # Red
        Light([3, 3, 1],    [0.2, 0.2, 1.0], 1.2), # Blue
        Light([0, 4, 1.5],  [0.2, 1.0, 0.2], 1.2), # Green
        Light([-2, 1, 4],   [1.0, 1.0, 0.2], 0.8), # Yellow
        Light([2, 1, 4],    [1.0, 0.2, 1.0], 0.8), # Magenta
        Light([0, 0.5, -1], [0.2, 1.0, 1.0], 0.6)  # Cyan
    ]

    # Render setup
    camera = np.array([0., 1., -3.])
    img = np.zeros((h, w, 3))
    aspect_ratio = w / h

    print(f"Starting render of {w}x{h} image...")

    # Main Rendering Loop
    for i in range(h):
        y = 1 - 2 * (i + 0.5) / h # Flipped Y for screen space
        for j in range(w):
            x = (2 * (j + 0.5) / w - 1) * aspect_ratio
            
            # Ray direction
            D = normalize(np.array([x, y, 1.0]))
            
            # Trace
            img[i, j] = trace_ray(camera, D, scene, lights)
        
        # Progress indicator
        if i % 60 == 0 and i > 0:
            print(f"Progress: {i/h*100:.0f}%")

    print("Render complete! Saving image...")

    # Convert to 8-bit RGB and save as PNG
    img_8bit = np.uint8(img * 255)
    image = Image.fromarray(img_8bit)
    image.save("colorful_raytraced_scene.png")
    print("Saved as 'colorful_raytraced_scene.png'")

if __name__ == "__main__":
    main()