import numpy as np
from PIL import Image
import math

# Vector operations
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalize(np.array(direction))

# Sphere class
class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.material = material  # dict with 'ambient', 'diffuse', 'specular', 'shininess', 'reflective'
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t > 0.001:  # Avoid self-intersection
            return t
        
        t = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t > 0.001:
            return t
        
        return None

# Light class
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Scene setup
def create_scene():
    spheres = [
        # Large central sphere - metallic blue
        Sphere([0, 0, 0], 1.5, [0.1, 0.3, 0.8], 
               {'ambient': 0.1, 'diffuse': 0.3, 'specular': 0.9, 'shininess': 128, 'reflective': 0.7}),
        
        # Red sphere - matte
        Sphere([-3, -1, 1], 1.0, [0.9, 0.1, 0.1],
               {'ambient': 0.2, 'diffuse': 0.8, 'specular': 0.2, 'shininess': 32, 'reflective': 0.1}),
        
        # Green sphere - glossy
        Sphere([3, -1, -1], 1.0, [0.1, 0.9, 0.1],
               {'ambient': 0.1, 'diffuse': 0.6, 'specular': 0.7, 'shininess': 64, 'reflective': 0.4}),
        
        # Purple sphere - small
        Sphere([-1.5, 2, -2], 0.7, [0.7, 0.1, 0.9],
               {'ambient': 0.15, 'diffuse': 0.7, 'specular': 0.5, 'shininess': 48, 'reflective': 0.3}),
        
        # Yellow sphere
        Sphere([2, 1.5, 2], 0.8, [0.9, 0.8, 0.1],
               {'ambient': 0.2, 'diffuse': 0.7, 'specular': 0.4, 'shininess': 32, 'reflective': 0.2}),
        
        # Ground plane (large sphere to simulate ground)
        Sphere([0, -101.5, 0], 100, [0.4, 0.4, 0.4],
               {'ambient': 0.2, 'diffuse': 0.6, 'specular': 0.1, 'shininess': 16, 'reflective': 0.2}),
        
        # Floating cyan sphere
        Sphere([0, 2.5, -3], 0.6, [0.1, 0.8, 0.9],
               {'ambient': 0.1, 'diffuse': 0.5, 'specular': 0.8, 'shininess': 96, 'reflective': 0.5}),
    ]
    
    lights = [
        # Main white light from above
        Light([0, 8, 3], [1.0, 1.0, 1.0], 0.8),
        
        # Colored accent lights
        Light([-5, 3, 5], [1.0, 0.3, 0.3], 0.6),  # Red light from left
        Light([5, 3, 5], [0.3, 0.3, 1.0], 0.6),   # Blue light from right
        Light([0, -0.5, 8], [0.3, 1.0, 0.3], 0.4), # Green light from front
        Light([-3, 5, -3], [1.0, 0.0, 1.0], 0.5),  # Magenta light
        Light([3, 1, -5], [1.0, 1.0, 0.0], 0.4),   # Yellow light from back
        Light([0, 3, 0], [0.0, 1.0, 1.0], 0.3),    # Cyan light from center
    ]
    
    return spheres, lights

# Ray-sphere intersection
def find_nearest_intersection(ray, spheres):
    nearest_t = float('inf')
    nearest_sphere = None
    
    for sphere in spheres:
        t = sphere.intersect(ray)
        if t and t < nearest_t:
            nearest_t = t
            nearest_sphere = sphere
    
    if nearest_sphere:
        return nearest_t, nearest_sphere
    return None, None

# Calculate lighting
def calculate_lighting(point, normal, view_dir, sphere, spheres, lights):
    color = np.zeros(3)
    
    # Ambient component
    ambient = sphere.material['ambient'] * sphere.color
    color += ambient
    
    for light in lights:
        light_dir = normalize(light.position - point)
        
        # Check for shadows
        shadow_ray = Ray(point, light_dir)
        shadow_t, _ = find_nearest_intersection(shadow_ray, spheres)
        
        if shadow_t and shadow_t < np.linalg.norm(light.position - point):
            continue  # Point is in shadow for this light
        
        # Diffuse component
        diff = max(0, np.dot(normal, light_dir))
        diffuse = sphere.material['diffuse'] * diff * sphere.color * light.color * light.intensity
        
        # Specular component
        reflect_dir = reflect(-light_dir, normal)
        spec = max(0, np.dot(view_dir, reflect_dir))
        spec = pow(spec, sphere.material['shininess'])
        specular = sphere.material['specular'] * spec * light.color * light.intensity
        
        color += diffuse + specular
    
    return np.clip(color, 0, 1)

# Trace a ray
def trace_ray(ray, spheres, lights, depth=0, max_depth=3):
    if depth > max_depth:
        return np.array([0.1, 0.1, 0.15])  # Dark blue background
    
    t, sphere = find_nearest_intersection(ray, spheres)
    
    if not sphere:
        # Background gradient
        y = ray.direction[1]
        t = 0.5 * (y + 1.0)
        return (1.0 - t) * np.array([0.1, 0.1, 0.15]) + t * np.array([0.3, 0.4, 0.6])
    
    # Calculate intersection point and normal
    point = ray.origin + t * ray.direction
    normal = normalize(point - sphere.center)
    view_dir = -ray.direction
    
    # Calculate local color
    local_color = calculate_lighting(point, normal, view_dir, sphere, spheres, lights)
    
    # Add reflection
    if sphere.material['reflective'] > 0 and depth < max_depth:
        reflect_dir = reflect(ray.direction, normal)
        reflect_ray = Ray(point, reflect_dir)
        reflect_color = trace_ray(reflect_ray, spheres, lights, depth + 1, max_depth)
        local_color = (1 - sphere.material['reflective']) * local_color + \
                      sphere.material['reflective'] * reflect_color
    
    return np.clip(local_color, 0, 1)

# Main rendering function
def render(width=800, height=600):
    # Setup
    spheres, lights = create_scene()
    
    # Camera parameters
    camera_pos = np.array([0, 2, 10])
    camera_target = np.array([0, 0, 0])
    camera_up = np.array([0, 1, 0])
    
    # Calculate camera basis vectors
    camera_dir = normalize(camera_target - camera_pos)
    camera_right = normalize(np.cross(camera_dir, camera_up))
    camera_up = np.cross(camera_right, camera_dir)
    
    # Field of view
    fov = math.radians(60)
    aspect_ratio = width / height
    
    # Create image
    image = np.zeros((height, width, 3))
    
    # Render each pixel
    for y in range(height):
        if y % 50 == 0:
            print(f"Rendering line {y}/{height}")
        
        for x in range(width):
            # Convert pixel coordinates to normalized device coordinates
            u = (2.0 * x / width - 1.0) * aspect_ratio * math.tan(fov / 2)
            v = (1.0 - 2.0 * y / height) * math.tan(fov / 2)
            
            # Calculate ray direction
            ray_dir = normalize(camera_dir + u * camera_right + v * camera_up)
            ray = Ray(camera_pos, ray_dir)
            
            # Trace ray and get color
            color = trace_ray(ray, spheres, lights)
            image[y, x] = color
    
    return image

# Main execution
if __name__ == "__main__":
    print("Starting raytracer...")
    print("This will take a few moments to render...")
    
    # Render the scene
    image = render(800, 600)
    
    # Convert to 8-bit and save
    image_8bit = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_8bit)
    img.save("raytraced_scene.png")
    
    print("Rendering complete! Image saved as 'raytraced_scene.png'")
    print("\nScene features:")
    print("- 7 spheres with different materials (metallic, matte, glossy)")
    print("- 7 colored light sources creating complex lighting")
    print("- Reflections up to 3 bounces")
    print("- Shadows and specular highlights")
    print("- Background gradient")