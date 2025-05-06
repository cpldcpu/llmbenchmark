import math
import random
from PIL import Image
from tqdm import tqdm # For a progress bar

# --- Configuration ---
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 3  # Max reflection bounces
EPSILON = 1e-4 # To avoid self-intersection

# --- Vector Class ---
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, Vec3): # Element-wise multiplication for colors
            return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0: # Avoid division by zero
            # Decide on behavior: return zero vector, raise error, or return large values
            # For colors, returning black or a very dark color might be okay.
            # For directions, this is problematic.
            # Let's return a zero vector for now, assuming it's mostly color scaling
            return Vec3(0,0,0) if self.x == 0 and self.y == 0 and self.z == 0 else Vec3(float('inf'), float('inf'), float('inf'))
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self): # <<< --- ADDED THIS METHOD ---
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.length_squared())

    def normalize(self):
        l = self.length()
        if l == 0: return Vec3() # Avoid division by zero
        return self / l

    def clamp(self, min_val=0.0, max_val=1.0):
        return Vec3(
            max(min_val, min(self.x, max_val)),
            max(min_val, min(self.y, max_val)),
            max(min_val, min(self.z, max_val)),
        )

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# --- Ray Class ---
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize() # Ensure direction is normalized

    def at(self, t):
        return self.origin + self.direction * t

# --- Material Class ---
class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.3, shininess=32, reflection=0.0):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection # 0 to 1

# --- Object Base Class (for potential extension) ---
class Hittable:
    def __init__(self, material):
        self.material = material

    def intersect(self, ray):
        # Returns distance t if hit, else None
        raise NotImplementedError

    def normal_at(self, point):
        raise NotImplementedError

# --- Sphere Class ---
class Sphere(Hittable):
    def __init__(self, center, radius, material):
        super().__init__(material)
        self.center = center
        self.radius = radius
        self.radius2 = radius * radius

    def intersect(self, ray):
        oc = ray.origin - self.center
        # a = ray.direction.dot(ray.direction) # Should be 1 if direction is normalized, so we can optimize
        a = 1.0 # Since ray.direction is normalized
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius2
        discriminant = b * b - 4 * a * c # Note: 4*a*c becomes 4*c if a=1

        if discriminant < 0:
            return None
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            # t0 = (-b - sqrt_discriminant) / (2.0 * a)
            # t1 = (-b + sqrt_discriminant) / (2.0 * a)
            t0 = (-b - sqrt_discriminant) / 2.0 # Since a=1
            t1 = (-b + sqrt_discriminant) / 2.0 # Since a=1


            # We want the smallest positive t
            if t0 > EPSILON and t1 > EPSILON:
                return min(t0, t1)
            elif t0 > EPSILON:
                return t0
            elif t1 > EPSILON:
                return t1
            return None


    def normal_at(self, point):
        return (point - self.center).normalize()

# --- Light Class ---
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# --- Scene Class ---
class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.background_color = Vec3(0.05, 0.05, 0.1) # Dark blueish sky
        self.ambient_light_color = Vec3(1,1,1) # White ambient light

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def find_closest_intersection(self, ray):
        closest_t = float('inf')
        hit_object = None
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t: # Ensure t is positive (already handled by intersect > EPSILON)
                closest_t = t
                hit_object = obj
        if hit_object:
            return hit_object, closest_t
        return None, None

# --- Ray Tracing Core ---
def shade(scene, hit_object, hit_point, normal, view_dir, depth):
    mat = hit_object.material
    effective_color = mat.color * scene.ambient_light_color * mat.ambient

    for light in scene.lights:
        light_dir = (light.position - hit_point).normalize()
        light_distance = (light.position - hit_point).length()

        # Shadow check
        shadow_ray_origin = hit_point + normal * EPSILON # Offset to avoid self-shadowing
        shadow_ray = Ray(shadow_ray_origin, light_dir)
        in_shadow = False
        # Check intersection with all objects except the current one for shadows
        # (though technically, current one should not cause issues if shadow ray starts slightly off surface)
        for obj_idx, obj in enumerate(scene.objects):
            # if obj == hit_object: continue # This comparison might be slow for many objects.
            # A simple check if it's the same object instance is fine.
            t_shadow = obj.intersect(shadow_ray)
            if t_shadow is not None and t_shadow < light_distance:
                in_shadow = True
                break
        
        if not in_shadow:
            # Diffuse
            diffuse_intensity = max(0.0, normal.dot(light_dir))
            diffuse_component = mat.color * light.color * diffuse_intensity * mat.diffuse * light.intensity
            effective_color += diffuse_component

            # Specular (Blinn-Phong)
            # view_dir is from hit_point to camera
            half_vector = (light_dir + view_dir).normalize()
            specular_intensity = max(0.0, normal.dot(half_vector)) ** mat.shininess
            specular_component = light.color * specular_intensity * mat.specular * light.intensity
            effective_color += specular_component
            
    # Reflection
    if depth < MAX_DEPTH and mat.reflection > 0:
        # R = I - 2 * N * dot(N, I) where I is incoming (view_dir from camera, so -view_dir for reflection formula)
        # Or simply: R = V - 2 * N * dot(N, V) where V is view_dir from hit point to eye
        # Here, view_dir is already from hit_point to camera, so we reflect -view_dir (which is ray.direction)
        # reflect_dir = (ray.direction - normal * 2 * ray.direction.dot(normal)).normalize() # This is for ray.direction
        # If view_dir is from hit_point to eye (as used in Blinn-Phong), reflection formula is:
        # reflect_dir = view_dir - normal * 2 * view_dir.dot(normal)
        # Let's stick to the view_dir definition: from hit point to camera.
        # The incoming light vector for reflection is -view_dir.
        # Reflected_vector = Incident_vector - 2 * Normal * dot(Normal, Incident_vector)
        # Incident_vector = -view_dir (the vector from the eye TO the surface point)
        # This is a bit confusing with naming. Let's use V = view_dir (vector from surface to eye)
        # Reflected ray direction R = V - 2N(N.V)
        # But our view_dir is -ray.direction. So, it's already pointing "out" from the surface towards the eye.
        # For reflection, the ray an observer sees is as if it came from reflect_dir.
        # The incident ray direction on the surface was ray.direction.
        # Reflected direction = D - 2 * N * (D dot N), where D is incident ray.direction
        
        incident_dir = -view_dir # This is original ray.direction
        reflect_dir = (incident_dir - normal * 2 * incident_dir.dot(normal)).normalize()
        
        reflect_ray_origin = hit_point + normal * EPSILON # Offset
        reflect_ray = Ray(reflect_ray_origin, reflect_dir)
        reflected_color = trace_ray(scene, reflect_ray, depth + 1)
        effective_color = effective_color * (1 - mat.reflection) + reflected_color * mat.reflection

    return effective_color.clamp()


def trace_ray(scene, ray, depth):
    hit_object, t = scene.find_closest_intersection(ray)

    if hit_object:
        hit_point = ray.at(t)
        normal = hit_object.normal_at(hit_point)
        view_dir = -ray.direction # Direction from hit point to camera/eye
        return shade(scene, hit_object, hit_point, normal, view_dir, depth)
    else:
        return scene.background_color


# --- Camera and Rendering ---
def render(scene, width, height):
    aspect_ratio = width / height
    fov_degrees = 60.0
    fov_rad = math.radians(fov_degrees)
    
    # Camera setup (simple, looking along -Z axis from a positive Z position)
    eye = Vec3(0, 1, 4) # Camera position (moved back a bit)
    target = Vec3(0, 0.2, 0) # Look-at point (slightly down)
    up_vector = Vec3(0, 1, 0)

    # Camera coordinate system
    forward = (target - eye).normalize() # z-axis (points from eye to target)
    
    # Handle up_vector parallel to forward (gimbal lock scenario)
    if abs(forward.dot(up_vector)) > 0.999: # If nearly parallel or anti-parallel
        # If forward is (0,1,0) or (0,-1,0), right would be (1,0,0)
        # A more general solution if forward is not perfectly aligned with Y:
        if abs(forward.y) > 0.999 : # forward is mostly along Y
            right = Vec3(1, 0, 0) # Choose X as right
        else: # forward is mostly in XZ plane
            right = Vec3(0,1,0).cross(forward).normalize() # Standard cross product
    else:
         right = up_vector.cross(forward).normalize() # x-axis (points to the right of the camera)

    # Recompute true 'up' to ensure orthogonality
    up = forward.cross(right).normalize() # y-axis (points upwards relative to camera)

    viewport_height = 2.0 * math.tan(fov_rad / 2.0) # * distance_to_viewport (assume 1 for now)
    viewport_width = aspect_ratio * viewport_height

    # Vectors for traversing the viewport
    # Note: forward is pointing *from* the eye *to* the scene.
    # The viewport plane is 'in front' of the eye along this forward direction.
    # lower_left_corner = eye + (forward * focal_length) - (right * viewport_width / 2) - (up * viewport_height / 2)
    # If focal_length is 1:
    # We need to be careful here. 'forward' is the direction the camera *looks*.
    # The viewport center is eye + forward.
    # The image plane is perpendicular to 'forward'.
    # Let's define 'w' as -forward (points from scene to eye, common in ray tracing literature for view direction)
    
    # Simpler approach for screen plane 1 unit away in -forward direction from eye:
    # Origin for rays is 'eye'.
    # The direction vector points from 'eye' to a point on the viewport.
    # Viewport center: eye + forward (if forward is normalized and represents distance 1)
    # Let's use the common Peter Shirley's setup:
    # focal_length = 1.0 (distance from eye to viewport plane)
    # horizontal_span = right * viewport_width
    # vertical_span = up * viewport_height
    # viewport_lower_left = eye + (forward * focal_length) - (horizontal_span / 2) - (vertical_span / 2)

    # Alternative: camera at origin, looking down -Z. Then transform rays.
    # Or, more directly:
    # Center of the viewport is 'eye + forward' (assuming 'forward' direction and viewport distance of 1)
    # Let the viewport be at distance 'd' along the 'forward' vector.
    # For simplicity, let d=1. The center of the viewport is eye + forward.
    
    # Corrected viewport calculation:
    # 'forward' is direction camera is looking.
    # 'right' is camera's right.
    # 'up' is camera's up.
    # The viewport plane is some distance 'd' in front of the 'eye' along 'forward'.
    # For simplicity, let this distance be 1.0 (focal length).
    viewport_center = eye + forward # Assuming focal length of 1
    
    horizontal_vec = right * viewport_width
    vertical_vec = up * viewport_height
    
    # Point on the viewport plane:
    # lower_left_corner = viewport_center - (horizontal_vec / 2.0) - (vertical_vec / 2.0)
    # This is a point in 3D space. Ray direction is (point_on_viewport - eye).

    # Pixel loop
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for j in tqdm(range(height), desc="Rendering"):
        for i in range(width):
            # u, v are viewport coordinates from [0,1]
            # Map to [-0.5, 0.5] then scale by viewport width/height
            u_screen = (i + 0.5) / width  # Add 0.5 for pixel center
            v_screen = (j + 0.5) / height # Add 0.5 for pixel center

            # Direction from eye to the point on the viewport
            # Point on viewport: lower_left + horizontal_vec * u_screen + vertical_vec * v_screen
            # Where lower_left is relative to eye if we set it up like that.
            # A common way:
            # origin = eye
            # ray_dir = lower_left_corner + u*horizontal_span + v*vertical_span - eye
            # (where lower_left_corner, horizontal_span, vertical_span are defined correctly)

            # Let's use the ray generation from "Ray Tracing in One Weekend"
            # Camera origin is `eye`
            # `look_from` = eye
            # `look_at` = target
            # `vup` = up_vector
            # `vfov` = fov_degrees
            # `aspect_ratio`
            
            # `w_cam = (look_from - look_at).normalize()`  => This is -forward
            # `u_cam = vup.cross(w_cam).normalize()`      => This is right
            # `v_cam = w_cam.cross(u_cam)`                => This is -up (if w_cam points from scene)
            
            # Our forward, right, up are already camera space axes.
            # forward: camera's Z axis (points into scene)
            # right:   camera's X axis
            # up:      camera's Y axis

            # Point on image plane (distance 1.0 along 'forward' from 'eye'):
            # P(u,v) = eye + forward + right * (u_screen - 0.5) * viewport_width + up * (v_screen - 0.5) * viewport_height
            # We want v_screen to go from bottom to top for positive up.
            # Image coordinates (j) go from top to bottom.
            
            px = (i + 0.5) / width  # 0 to 1
            py = (j + 0.5) / height # 0 to 1

            # Map px, py from [0,1] to [-1,1] for screen space like coordinates
            # then scale by half viewport dimensions
            # This maps to a screen centered at (0,0) in the camera's XY plane,
            # then placed 'focal_length' units away along camera's Z.
            
            # x = (2 * px - 1) * aspect_ratio * math.tan(fov_rad / 2)
            # y = (1 - 2 * py) * math.tan(fov_rad / 2) # 1 - 2*py to flip Y
            
            # Correct ray direction calculation:
            # `u` goes from 0 (left) to 1 (right)
            # `v` goes from 0 (bottom) to 1 (top) for viewport calculation
            # Image `j` goes from 0 (top) to `height-1` (bottom)
            
            # u_vp = i / (width - 1.0)
            # v_vp = (height - 1.0 - j) / (height - 1.0) # Flipped j for viewport (bottom to top)
            # For pixel center:
            u_vp = (i + 0.5) / width
            v_vp = (height - (j + 0.5)) / height # Flipped j for viewport (bottom to top)

            # Direction relative to viewport center
            # Viewport is at distance 1 along 'forward'
            dir_x_comp = (u_vp - 0.5) * viewport_width
            dir_y_comp = (v_vp - 0.5) * viewport_height

            # Ray direction in world space
            # ray_direction = (forward + right * dir_x_comp + up * dir_y_comp).normalize()
            # More robust: Point on view plane - eye
            point_on_view_plane = eye + forward + (right * dir_x_comp) + (up * dir_y_comp)
            ray_direction = (point_on_view_plane - eye).normalize()

            ray = Ray(eye, ray_direction)
            
            color_vec = trace_ray(scene, ray, 0)
            
            r = int(max(0, min(255, color_vec.x * 255.999)))
            g = int(max(0, min(255, color_vec.y * 255.999)))
            b = int(max(0, min(255, color_vec.z * 255.999)))
            pixels[i, j] = (r, g, b)
            
    return image

# --- Scene Definition ---
def create_scene():
    scene = Scene()

    # Materials
    mat_ground = Material(color=Vec3(0.3, 0.3, 0.3), ambient=0.2, diffuse=0.8, specular=0.1, shininess=10, reflection=0.1)
    mat_red_shiny = Material(color=Vec3(0.9, 0.1, 0.1), ambient=0.1, diffuse=0.7, specular=0.8, shininess=100, reflection=0.2)
    mat_green_matte = Material(color=Vec3(0.1, 0.7, 0.1), ambient=0.1, diffuse=0.9, specular=0.1, shininess=5, reflection=0.05)
    mat_blue_reflective = Material(color=Vec3(0.2, 0.2, 0.8), ambient=0.1, diffuse=0.5, specular=0.9, shininess=200, reflection=0.6)
    mat_yellow_bright = Material(color=Vec3(0.9,0.9,0.2), ambient=0.15, diffuse=0.8, specular=0.5, shininess=50, reflection=0.1)
    mat_purple_mirror = Material(color=Vec3(0.6, 0.1, 0.9), ambient=0.05, diffuse=0.3, specular=0.9, shininess=500, reflection=0.8)
    mat_cyan = Material(color=Vec3(0.1, 0.8, 0.8), ambient=0.1, diffuse=0.7, specular=0.4, shininess=60, reflection=0.15)
    mat_orange_reflective = Material(color=Vec3(1.0, 0.5, 0.0), ambient=0.1, diffuse=0.6, specular=0.7, shininess=150, reflection=0.4)


    # Objects
    scene.add_object(Sphere(Vec3(0, -1000.5, -1), 1000, mat_ground)) # Ground plane (large sphere)
    
    scene.add_object(Sphere(Vec3(0, 0, -1), 0.5, mat_red_shiny))
    scene.add_object(Sphere(Vec3(-1.2, 0.2, -1.5), 0.7, mat_blue_reflective))
    scene.add_object(Sphere(Vec3(1.5, -0.1, -2.0), 0.4, mat_green_matte))
    scene.add_object(Sphere(Vec3(0.8, 0.8, -0.8), 0.3, mat_yellow_bright))
    scene.add_object(Sphere(Vec3(-0.7, -0.2, -0.5), 0.3, mat_purple_mirror))
    scene.add_object(Sphere(Vec3(2.0, 0.5, -1.2), 0.2, mat_cyan))
    scene.add_object(Sphere(Vec3(-2.2, 0.0, -0.7), 0.5, mat_orange_reflective))


    # Lights (many colorful lights!)
    # Stronger lights for more vivid colors
    scene.add_light(Light(Vec3(-5, 5, 2), Vec3(1, 0.2, 0.2), intensity=1.2))  # Reddish Stronger
    scene.add_light(Light(Vec3(5, 5, 1), Vec3(0.2, 0.2, 1), intensity=1.2))   # Bluish Stronger
    scene.add_light(Light(Vec3(0, 8, -3), Vec3(0.2, 1, 0.2), intensity=1.5)) # Greenish top Stronger
    scene.add_light(Light(Vec3(3, 3, 4), Vec3(1, 1, 0.3), intensity=1.0))  # Yellowish front-right
    scene.add_light(Light(Vec3(-3, 2, 4), Vec3(0.8, 0.3, 1), intensity=1.0)) # Purplish front-left
    scene.add_light(Light(Vec3(0, 1, 6), Vec3(1,1,1), intensity=0.5)) # White fill from camera direction (further back)
    scene.add_light(Light(Vec3(1, -2, -0.5), Vec3(1.0, 0.5, 0.0), intensity=0.8)) # Orange from below/side

    scene.background_color = Vec3(0.01, 0.01, 0.02) # Even Darker sky
    scene.ambient_light_color = Vec3(0.1,0.1,0.1) # Dimmer global ambient

    return scene

# --- Main Execution ---
if __name__ == "__main__":
    print("Setting up scene...")
    my_scene = create_scene()
    
    print(f"Rendering {WIDTH}x{HEIGHT} image with MAX_DEPTH={MAX_DEPTH}...")
    image = render(my_scene, WIDTH, HEIGHT)
    
    output_filename = "raytraced_scene2.png"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    try:
        image.show() # Tries to open with default image viewer
    except Exception as e:
        print(f"Could not show image automatically: {e}")