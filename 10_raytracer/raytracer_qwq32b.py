import math
import random
from PIL import Image

# Vector operations
def add(v1, v2):
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2])

def sub(v1, v2):
    return (v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2])

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def length(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def normalize(v):
    l = length(v)
    if l == 0:
        return (0, 0, 0)
    return (v[0]/l, v[1]/l, v[2]/l)

def scale(s, v):
    return (s*v[0], s*v[1], s*v[2])

class Sphere:
    def __init__(self, center, radius, color, emissive_color):
        self.center = center
        self.radius = radius
        self.color = color
        self.emissive_color = emissive_color

class Plane:
    def __init__(self, point, normal, color):
        self.point = point
        self.normal = normalize(normal)
        self.color = color

    def intersect(self, ray):
        denom = dot(ray[1], self.normal)
        if abs(denom) < 1e-6:
            return None
        t = dot(sub(self.point, ray[0]), self.normal) / denom
        if t < 1e-6:
            return None
        hit_point = add(ray[0], scale(t, ray[1]))
        return (t, hit_point, self.normal)

def intersect_sphere(ray, sphere):
    oc = sub(ray[0], sphere.center)
    a = dot(ray[1], ray[1])
    b = 2 * dot(oc, ray[1])
    c = dot(oc, oc) - sphere.radius**2
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d)/(2*a)
    t2 = (-b + sqrt_d)/(2*a)
    t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)
    if t < 0.0001:
        return None
    hit_point = add(ray[0], scale(t, ray[1]))
    normal = normalize(sub(hit_point, sphere.center))
    return (t, hit_point, normal)

def trace_ray(ray, scene, max_depth):
    closest_t = float('inf')
    closest_obj = None
    hit_point = None
    normal = None

    for obj in scene:
        if isinstance(obj, Sphere):
            result = intersect_sphere(ray, obj)
            if result is not None:
                t, p, n = result
                if t < closest_t and t > 0.0001:
                    closest_t, closest_obj, hit_point, normal = t, obj, p, n
        elif isinstance(obj, Plane):
            result = obj.intersect(ray)
            if result is not None:
                t, p, n = result
                if t < closest_t and t > 0.0001:
                    closest_t, closest_obj, hit_point, normal = t, obj, p, n

    if closest_obj is None:
        return (0.2, 0.7, 0.8)  # Sky color

    point = hit_point
    eye_dir = normalize(sub(ray[0], point))
    ambient_light = 0.1
    total_color = [0.0, 0.0, 0.0]

    lights = [obj for obj in scene if isinstance(obj, Sphere) and obj.emissive_color != (0, 0, 0)]

    for light in lights:
        light_center = light.center
        light_dir = sub(light_center, point)
        light_dist_sq = dot(light_dir, light_dir)

        if light_dist_sq < 1e-6:
            light_dir_normalized = (1, 0, 0)
        else:
            light_dir_normalized = normalize(light_dir)
            light_dist = math.sqrt(light_dist_sq)

        shadow_hit = False
        shadow_ray = (point, light_dir_normalized)

        for obj_in_scene in scene:
            if obj_in_scene is closest_obj or obj_in_scene is light:
                continue
            if isinstance(obj_in_scene, Sphere):
                result = intersect_sphere(shadow_ray, obj_in_scene)
                if result is not None:
                    t_shad, _, _ = result
                    if t_shad < light_dist - 1e-6:
                        shadow_hit = True
                        break
            elif isinstance(obj_in_scene, Plane):
                result = obj_in_scene.intersect(shadow_ray)
                if result is not None:
                    t_shad, _, _ = result
                    if t_shad < light_dist - 1e-6:
                        shadow_hit = True
                        break

        if not shadow_hit:
            diffuse_factor = max(0.0, dot(normal, light_dir_normalized))
            if diffuse_factor > 0:
                inv_dist_sq = 1.0 / light_dist_sq
                contribution = (
                    light.emissive_color[0] * diffuse_factor * inv_dist_sq,
                    light.emissive_color[1] * diffuse_factor * inv_dist_sq,
                    light.emissive_color[2] * diffuse_factor * inv_dist_sq
                )
                total_color[0] += contribution[0]
                total_color[1] += contribution[1]
                total_color[2] += contribution[2]

    ambient = (
        ambient_light * closest_obj.color[0],
        ambient_light * closest_obj.color[1],
        ambient_light * closest_obj.color[2]
    )

    total_color[0] += ambient[0]
    total_color[1] += ambient[1]
    total_color[2] += ambient[2]

    total_color = (
        min(total_color[0], 1.0),
        min(total_color[1], 1.0),
        min(total_color[2], 1.0)
    )

    return tuple(total_color)

def main():
    image_width = 800
    image_height = 600
    scene = []

    # Ground plane
    ground_plane = Plane((0, 0, 0), (0, 0, 1), (1.0, 1.0, 1.0))
    scene.append(ground_plane)

    # Add light sources (spheres)
    num_lights = 20
    for _ in range(num_lights):
        x = (random.random() - 0.5) * 5.0
        y = 1.0 + random.random() * 2.0
        z = (random.random() - 0.5) * 5.0
        radius = 0.2
        r = random.random() * 3.0
        g = random.random() * 3.0
        b = random.random() * 3.0
        emissive = (min(r, 1.0), min(g, 1.0), min(b, 1.0))
        sphere = Sphere((x, y, z), radius, (0, 0, 0), emissive)
        scene.append(sphere)

    # Add non-light spheres
    for _ in range(5):
        x = (random.random() - 0.5) * 5.0
        y = 0.5 + random.random() * 1.5
        z = (random.random() - 0.5) * 5.0
        radius = 0.5 + random.random()
        color = (random.random(), random.random(), random.random())
        sphere = Sphere((x, y, z), radius, color, (0, 0, 0))
        scene.append(sphere)

    # Camera parameters
    camera_pos = (0, 0, 0)
    fov_y = math.radians(90)
    aspect_ratio = image_width / image_height
    height = 2.0 * math.tan(fov_y * 0.5)
    width = height * aspect_ratio

    image = Image.new('RGB', (image_width, image_height), (0, 0, 0))
    pixels = image.load()

    for j in range(image_height):
        for i in range(image_width):
            x = -width/2 + (i / (image_width - 1)) * width
            y = height/2 - (j / (image_height - 1)) * height
            direction = normalize((x, y, -1.0))
            ray = (camera_pos, direction)
            color = trace_ray(ray, scene, max_depth=0)
            r = int(color[0] * 255)
            g = int(color[1] * 255)
            b = int(color[2] * 255)
            pixels[i, j] = (r, g, b)

    image.save('output.png')

if __name__ == "__main__":
    main()