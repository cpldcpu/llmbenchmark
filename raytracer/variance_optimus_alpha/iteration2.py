import numpy as np
from PIL import Image

# ---- Raytracer Classes ----

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(direction, normal):
    return direction - 2 * np.dot(direction, normal) * normal

class Sphere:
    def __init__(self, center, radius, color, specular=0.5, reflection=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflection = reflection

    def intersect(self, orig, dir):
        # Ray-sphere intersection
        L = self.center - orig
        tca = np.dot(L, dir)
        d2 = np.dot(L, L) - tca * tca
        r2 = self.radius * self.radius
        if d2 > r2:
            return None
        thc = np.sqrt(r2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t1 < 0:
            return None
        t = t0 if t0 > 0 else t1
        hit = orig + dir * t
        normal = normalize(hit - self.center)
        return (t, hit, normal)

class Plane:
    def __init__(self, point, normal, color, specular=0, reflection=0):
        self.point = np.array(point)
        self.normal = normalize(normal)
        self.color = np.array(color)
        self.specular = specular
        self.reflection = reflection

    def intersect(self, orig, dir):
        denom = np.dot(self.normal, dir)
        if np.abs(denom) < 1e-6:
            return None
        t = np.dot(self.point - orig, self.normal) / denom
        if t < 0:
            return None
        hit = orig + t * dir
        normal = self.normal
        return (t, hit, normal)
        
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# ---- Rendering ----

WIDTH, HEIGHT = 800, 600
CAMERA_POS = np.array([0, 1, 2.5])

def raytrace(scene, lights, depth=0, max_depth=3):
    # Precompute:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    
    aspect = WIDTH / HEIGHT
    fov = np.pi / 3
    for y in range(HEIGHT):
        sy = 1 - 2 * (y + 0.5) / HEIGHT
        for x in range(WIDTH):
            sx = 2 * (x + 0.5) / WIDTH - 1
            sx *= aspect
            # Build ray
            d = normalize(np.array([sx * np.tan(fov/2), sy * np.tan(fov/2), -1]))
            color = cast_ray(CAMERA_POS, d, scene, lights, depth, max_depth)
            image[y, x] = np.clip(color, 0, 1)
        if y % 40 == 0:
            print(f"Progress: {100*y/HEIGHT:.1f}%")
    return image

def cast_ray(orig, dir, scene, lights, depth, max_depth):
    nearest_t = float('inf')
    nearest_obj = None
    hit = None
    normal = None

    # Find intersection
    for obj in scene:
        result = obj.intersect(orig, dir)
        if result:
            t, h, n = result
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                hit = h
                normal = n
    if nearest_obj is None:
        return np.array([0.03, 0.05, 0.10])  # Sky color

    # Local shading
    local_col = ambient = 0.09 * nearest_obj.color
    for light in lights:
        to_light = light.position - hit
        to_light_dir = normalize(to_light)
        # Shadow check
        shadow = False
        for obj in scene:
            if obj is nearest_obj:
                continue
            shad_result = obj.intersect(hit + normal*1e-4, to_light_dir)
            if shad_result:
                t, _, _ = shad_result
                if t < np.linalg.norm(to_light):
                    shadow = True
                    break
        if not shadow:
            # Diffuse shading
            diff = max(0, np.dot(normal, to_light_dir))
            # Specular shading
            view_dir = normalize(CAMERA_POS - hit)
            reflect_dir = reflect(-to_light_dir, normal)
            spec = max(0, np.dot(view_dir, reflect_dir)) ** (32 + 32*nearest_obj.specular)
            local_col += (diff * 0.8 * light.intensity * nearest_obj.color * light.color +
                          0.2 * spec * nearest_obj.specular * light.color)
    # Reflection
    reflected = np.zeros(3)
    if depth < max_depth and nearest_obj.reflection > 0:
        reflect_dir = reflect(dir, normal)
        reflected = cast_ray(hit + normal*1e-4, reflect_dir, scene, lights, depth+1, max_depth)
    return (1 - nearest_obj.reflection) * local_col + nearest_obj.reflection * reflected

def save_img(image, filename):
    img8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(filename)
    print(f"Image saved to '{filename}'")

# ---- Scene ----

def make_scene():
    scene = [
        # Glossy spheres
        Sphere(center=[-1.2, 0.6, -2.2], radius=0.6, color=[0.95, 0.2, 0.1], specular=0.8, reflection=0.32),
        Sphere(center=[0.95, 0.5, -2.0], radius=0.5, color=[0.16, 0.8, 0.34], specular=0.5, reflection=0.20),
        Sphere(center=[0.0, 0.33, -1.3], radius=0.33, color=[0.1, 0.5, 1.0], specular=0.9, reflection=0.55),
        Sphere(center=[0.7, 0.12, -1.1], radius=0.12, color=[0.98, 0.9, 0.21], specular=1.0, reflection=0.31),
        Sphere(center=[-0.5, 0.185, -1.0], radius=0.18, color=[1.0, 0.11, 0.71], specular=0.75, reflection=0.25),
        # Floor
        Plane(point=[0, 0, 0], normal=[0, 1, 0], color=[0.15, 0.13, 0.14], specular=0.35, reflection=0.08),
    ]
    # Many colored lights
    radius = 4
    h = 5.2
    lights = [
        Light(position=[ np.cos(np.pi*2*i/6)*radius, h, np.sin(np.pi*2*i/6)*radius-2], 
              color=[(np.sin(i)+1)/2, (np.cos(i)+1)/2, (np.sin(i*3)+1)/2], 
              intensity=2.8)
        for i in range(6)
    ]
    # Add a white light overhead
    lights.append(Light(position=[0, 8, -3], color=[1.0, 1.0, 1.0], intensity=1.05))
    return scene, lights

# ---- Main ----

if __name__ == '__main__':
    scene, lights = make_scene()
    image = raytrace(scene, lights)
    save_img(image, "raytraced_scene2.png")
