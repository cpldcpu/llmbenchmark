import numpy as np
from PIL import Image
import time
import random

class Vector:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return np.sqrt(self.dot(self))

    def normalize(self):
        magnitude = self.norm()
        if magnitude > 0:  # Avoid division by zero
            return self / magnitude
        return self #Return the zero vector
    
    def to_tuple(self):
        return (self.x, self.y, self.z)



class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()  # Always normalized


class Sphere:
    def __init__(self, center, radius, color, reflectivity=0.0, emission=(0,0,0)):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflectivity = reflectivity
        self.emission = emission  # Add emission property

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        if t1 < 0 and t2 < 0:
            return None   #both intersections behind ray

        t = min(t for t in (t1, t2) if t > 0)  # Choose the closest positive intersection

        return t  # Distance to the intersection point


class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.ambient_light = (0.1, 0.1, 0.1)

    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)


    def intersect(self, ray):
        closest_t = float('inf')
        closest_obj = None

        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_t = t
                closest_obj = obj

        return closest_t, closest_obj

def trace_ray(scene, ray, depth=0, max_depth=5):
    if depth > max_depth:
        return (0, 0, 0) #Black if beyond max depth

    t, obj = scene.intersect(ray)

    if obj is None:
        return (0, 0, 0)  # Background color (black)

    #If hit a light source, return emission
    if obj.emission != (0, 0, 0):
        return obj.emission

    intersection_point = ray.origin + ray.direction * t
    normal = (intersection_point - obj.center).normalize()

     # Make sure normal is facing the viewer (important for correct shading)
    if normal.dot(ray.direction) > 0:
      normal = normal * -1

    color = [0, 0, 0]

    #Ambient lighting
    for i in range(3): #For R, G, and B
      color[i] += scene.ambient_light[i] * obj.color[i]


    #Diffuse and specular
    for light in scene.lights:
      light_dir = (light.center - intersection_point).normalize()
      light_distance = (light.center - intersection_point).norm()
      shadow_ray = Ray(intersection_point + normal * 0.001, light_dir)  # Offset to avoid self-intersection
      shadow_t, shadow_obj = scene.intersect(shadow_ray)


      #Check if in shadow
      if shadow_obj is not None and shadow_t < light_distance:
          continue #In shadow, skip diffuse and specular

      #Diffuse
      diffuse_intensity = max(0, normal.dot(light_dir)) #Clamp at zero
      for i in range(3):
        color[i] += light.emission[i] * obj.color[i] * diffuse_intensity / (light_distance ** 2)  # Inverse square law


      #Specular
      view_dir = (ray.origin - intersection_point).normalize()
      reflect_dir = light_dir - normal * 2 * light_dir.dot(normal)
      specular_intensity = max(0, view_dir.dot(reflect_dir)) ** 50  # Phong exponent = 50
      for i in range(3):
          color[i] += light.emission[i] * specular_intensity / (light_distance**2) * 0.5 # 0.5 is specular coefficient


    #Reflection
    if obj.reflectivity > 0:
        reflect_dir = ray.direction - normal * 2 * ray.direction.dot(normal)
        reflected_ray = Ray(intersection_point + normal * 0.001, reflect_dir) # Offset to avoid self-intersection
        reflected_color = trace_ray(scene, reflected_ray, depth + 1, max_depth)
        for i in range(3):
            color[i] += reflected_color[i] * obj.reflectivity


    return tuple(color)

def render(scene, width, height):
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    aspect_ratio = width / height
    scale = np.tan(np.radians(90 / 2))  # 90 degrees FOV

    camera_pos = Vector(0, 0, 0)

    start_time = time.time()

    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to normalized device coordinates (-1 to 1)
            px = (2 * ((x + 0.5) / width) - 1) * aspect_ratio * scale
            py = (1 - 2 * ((y + 0.5) / height)) * scale

            direction = Vector(px, py, -1)  # -1 for Z-axis pointing into the scene
            ray = Ray(camera_pos, direction)

            color = trace_ray(scene, ray)

            # Clamp color values
            color = tuple(int(min(c, 255)) for c in color)

            pixels[x, y] = color
        
        if (y % 50 == 0):
          print(f"Progress {100*y/height:.0f}%")

    end_time = time.time()
    print(f"Render time: {end_time - start_time:.2f} seconds")
    return image


# --- Scene setup ---
scene = Scene()

# Spheres
scene.add_object(Sphere(Vector(0, -1, -5), 1, (255, 0, 0), reflectivity=0.2))      # Red
scene.add_object(Sphere(Vector(2, 0, -4), 0.8, (0, 255, 0), reflectivity=0.5))     # Green, more reflective
scene.add_object(Sphere(Vector(-2, 0, -6), 1.2, (0, 0, 255), reflectivity=0.1))    # Blue
scene.add_object(Sphere(Vector(0, -1001, -5), 1000, (200, 200, 200), reflectivity = 0.3))  #Large grey "floor"
scene.add_object(Sphere(Vector(-4, 1, -7), 1, (255, 255, 0), reflectivity = 0.2))   # Yellow in back
scene.add_object(Sphere(Vector(3, 1.5, -8), 1.3, (200,0,200), reflectivity = 0.6))  # reflective purple sphere
scene.add_object(Sphere(Vector(0, 1, -3), 0.5, (0,255,255), reflectivity = 0.8)) #small highly reflective cyan sphere

# Add some interesting, colorful lights!
scene.add_light(Sphere(Vector(5,5,5), 0.2, (0,0,0), emission=(255*1000, 255*1000, 255*1000))) # white light
scene.add_light(Sphere(Vector(-5,5,-2), 0.2, (0,0,0), emission=(0,0, 255 * 1000)))      # Blue light
scene.add_light(Sphere(Vector(0, 5, -8), 0.3, (0, 0, 0), emission=(0, 255 * 500, 0)))  # Green light behind
scene.add_light(Sphere(Vector(3, 8, -5), 0.4, (0,0,0), emission=(255 * 1500, 0, 0)))    # Bright red light
scene.add_light(Sphere(Vector(-3, 5, 2), 0.1, (0,0,0), emission=(255 * 800, 255 * 800, 0)))   #Yellowish light




# --- Render ---
width = 800
height = 600
image = render(scene, width, height)
image.save("output.png")

print("Image saved as output.png")