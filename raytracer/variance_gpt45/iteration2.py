import numpy as np
from PIL import Image

# Scene setup
WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3

# Utility functions
def normalize(v):
    return v / np.linalg.norm(v)

# Ray-sphere intersection
def intersect_sphere(ray_origin, ray_dir, sphere):
    center, radius, material = sphere
    oc = ray_origin - center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return np.inf, None
    sqrtd = np.sqrt(discriminant)
    t1 = (-b - sqrtd) / (2 * a)
    t2 = (-b + sqrtd) / (2 * a)
    t = min(t1, t2)
    if t < 0:
        t = max(t1, t2)
    if t < 0:
        return np.inf, None
    point = ray_origin + t * ray_dir
    normal = normalize(point - center)
    return t, (point, normal, material)

# Compute lighting and shading
def compute_lighting(point, normal, view_dir, lights, materials, sphere_material):
    diffuse_color = materials[sphere_material]["color"]
    ambient = 0.05 * diffuse_color
    color = ambient.copy()
    
    for light_pos, light_color in lights:
        light_dir = normalize(light_pos - point)
        diffuse_intensity = max(np.dot(normal, light_dir), 0)
        
        # Simple Lambertian shading
        diffuse = diffuse_intensity * diffuse_color * light_color
        
        # Specular highlight (Phong reflection)
        reflect_dir = 2 * normal * np.dot(normal,light_dir) - light_dir
        spec_intensity = max(np.dot(reflect_dir, view_dir), 0) ** 32
        specular = spec_intensity * light_color
        
        color += diffuse + specular
    
    color = np.clip(color, 0, 1)
    return color

# Render function
def render(scene, lights, camera, materials):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    fov = np.pi/3
    aspect_ratio = WIDTH / HEIGHT

    for y in range(HEIGHT):
        for x in range(WIDTH):
            px = (2 * ((x + 0.5) / WIDTH) - 1) * np.tan(fov / 2) * aspect_ratio
            py = -(2 * ((y + 0.5) / HEIGHT) - 1) * np.tan(fov / 2)
            dir = normalize(np.array([px, py, -1]))
            color = cast_ray(camera, dir, scene, lights, materials)
            img[y, x] = np.clip(color * 255, 0, 255)

        if y % 50 == 0:
            print(f"Rendering... {(y / HEIGHT) * 100:.1f}%")

    return img

def cast_ray(origin, direction, scene, lights, materials, depth=0):
    nearest_t = np.inf
    hit_info = None
    
    for sphere in scene:
        t, info = intersect_sphere(origin, direction, sphere)
        if t < nearest_t:
            nearest_t = t
            hit_info = info

    if hit_info is None:
        return np.array([0.0, 0.0, 0.0])  # Black background
    
    hit_point, normal, material = hit_info
    view_dir = normalize(-direction)
    color = compute_lighting(hit_point, normal, view_dir, lights, materials, material)
    
    return color

if __name__ == '__main__':
    camera = np.array([0.0, 0.0, 1.0])
    
    # Materials dictionary
    materials = {
        "red": {"color": np.array([1.0, 0.0, 0.0])},
        "blue": {"color": np.array([0.0, 0.0, 1.0])},
        "green": {"color": np.array([0.0, 1.0, 0.0])},
        "yellow": {"color": np.array([1.0, 1.0, 0.0])},
        "mirror": {"color": np.array([1.0, 1.0, 1.0])},
    }

    # Scene definition (spheres)
    scene = [
        (np.array([0.0, -0.5, -3.0]), 0.5, "red"),
        (np.array([2.0, 0.0, -4.0]), 1.0, "blue"),
        (np.array([-2.0, 0.0, -4.5]), 1.0, "green"),
        (np.array([0.0, -5001.0, 0.0]), 5000.0, "yellow"), # Floor
    ]

    # Multiple colorful lights in scene
    lights = [
        (np.array([2.0, 5.0, 0.0]), np.array([1.0, 0.4, 0.4])), # red-ish light
        (np.array([-2.0, 3.0, -5.0]), np.array([0.4, 0.4, 1.0])), # blue-ish light
        (np.array([0.0, 5.0, -3.0]), np.array([0.4, 1.0, 0.4])), # green-ish light
        (np.array([0.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.8])), # warm bottom light
    ]

    final_img = render(scene, lights, camera, materials)
    image = Image.fromarray(final_img, 'RGB')
    image.save('raytraced_scene2.png')
    print("Rendering completed. Saved as raytraced_scene2.png.")