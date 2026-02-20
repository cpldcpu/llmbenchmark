import numpy as np
from PIL import Image

# Image dimensions
w = 800
h = 600

def normalize(x):
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

def intersect_sphere(O, D, S, R):
    # Solves the quadratic equation for ray-sphere intersection
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

# Scene Definition: (center, radius, color, specular_exponent, reflectivity)
spheres = [
    (np.array([0., -10004., -20.]), 10000., np.array([0.2, 0.2, 0.2]), 0., 0.1), # Floor
    (np.array([0., 0., -20.]), 4., np.array([0.05, 0.05, 0.05]), 100., 0.9),     # Central Mirror ball
    (np.array([5., -1., -15.]), 2., np.array([0.9, 0.1, 0.1]), 50., 0.2),        # Red ball
    (np.array([-5., -1., -15.]), 2., np.array([0.1, 0.9, 0.1]), 50., 0.2),       # Green ball
    (np.array([0., -3., -10.]), 1., np.array([0.1, 0.1, 0.9]), 50., 0.2)         # Blue ball
]

# Lights Definition: (position, color)
lights = [
    (np.array([10., 10., -10.]), np.array([1.0, 0.2, 0.2])),  # Red light
    (np.array([-10., 10., -10.]), np.array([0.2, 1.0, 0.2])), # Green light
    (np.array([0., 10., -5.]), np.array([0.2, 0.2, 1.0])),    # Blue light
    (np.array([0., 5., -25.]), np.array([1.0, 0.0, 1.0]))     # Magenta backlight
]

camera = np.array([0., 0., 1.])
img = np.zeros((h, w, 3))

def trace(O, D, depth):
    # Cap recursion depth for reflections
    if depth > 3:
        return np.array([0., 0., 0.])
    
    t = np.inf
    obj_idx = -1
    for i, obj in enumerate(spheres):
        t_obj = intersect_sphere(O, D, obj[0], obj[1])
        if t_obj < t:
            t = t_obj
            obj_idx = i
            
    if t == np.inf:
        return np.array([0.02, 0.02, 0.02]) # Dark background
        
    obj = spheres[obj_idx]
    P = O + D * t
    N = normalize(P - obj[0])
    color = np.zeros(3)
    
    # Offset intersection point slightly to avoid self-shadowing acne
    P_shifted = P + N * 1e-4
    
    # Ambient light
    color += obj[2] * 0.1
    
    for light in lights:
        light_dir = light[0] - P_shifted
        light_dist = np.linalg.norm(light_dir)
        L = normalize(light_dir)
        
        # Shadow check by casting a ray toward the light source
        shadow_t = np.inf
        for s_obj in spheres:
            t_shadow = intersect_sphere(P_shifted, L, s_obj[0], s_obj[1])
            if t_shadow < shadow_t:
                shadow_t = t_shadow
        
        # If no object is blocking the light (or the blocker is behind the light)
        if shadow_t >= light_dist:
            # Diffuse Shading (Lambertian)
            diffuse = max(np.dot(N, L), 0)
            color += obj[2] * light[1] * diffuse
            
            # Specular Highlights (Phong)
            if obj[3] > 0:
                V = -D
                R_light = normalize(2 * np.dot(N, L) * N - L)
                specular = max(np.dot(R_light, V), 0) ** obj[3]
                color += light[1] * specular
                
    # Calculate Reflections recursively
    if obj[4] > 0:
        R_ray = normalize(D - 2 * np.dot(D, N) * N)
        refl_color = trace(P_shifted, R_ray, depth + 1)
        color = color * (1 - obj[4]) + refl_color * obj[4]
        
    return color

# Coordinate mapping for perspective
ratio = float(w) / h
screen = (-1, 1 / ratio, 1, -1 / ratio)

print("Starting render...")
for i in range(h):
    y = screen[1] - i * (screen[1] - screen[3]) / (h - 1)
    for j in range(w):
        x = screen[0] + j * (screen[2] - screen[0]) / (w - 1)
        # Create a ray from the camera passing through the pixel
        D = normalize(np.array([x, y, -1.]) - camera)
        img[i, j] = np.clip(trace(camera, D, 0), 0, 1)
        
    # Print progress
    if i % 10 == 0:
        print(f"Rendering: {int(i / h * 100)}%", end="\r")

print("\nSaving image to output.png...")
# Convert from 0-1 float to 0-255 uint8 format for PIL
img = (img * 255).astype(np.uint8)
image = Image.fromarray(img)
image.save("output.png")
print("Done!")