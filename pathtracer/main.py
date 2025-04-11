import glfw
from OpenGL.GL import *
import sys
from shaders import create_program
from controls import Controls
from scene import Scene
import numpy as np
import glm
import time
import math
import os
from PIL import Image
from datetime import datetime

# Vertex and fragment shader sources for a fullscreen quad (placeholder)
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 uv;
void main() {
    uv = (aPos + 1.0) * 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

# Improved fragment shader with multiple materials and better quality
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 uv;
out vec4 FragColor;
uniform vec3 camPos;
uniform vec3 camDir;
uniform int frame;
uniform float randSeed;
uniform int screenWidth;
uniform int screenHeight;
uniform vec3 light1Pos;
uniform vec3 light1Color;
uniform vec3 light2Pos;
uniform vec3 light2Color;
uniform float time;
uniform float aperture;      // 0.0 = pinhole camera, larger = more blur
uniform float focalDistance; // Distance to focus plane
uniform int samplesPerFrame; // Dynamic samples per frame

#define MAX_SPHERES 8
#define MAX_BOUNCES 8

// Improved hash for better randomness
float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float rand(vec2 co, int sample, int frame) {
    // Mix in sample and frame for better decorrelation
    return hash12(co * 1.37 + float(sample) * 17.17 + float(frame) * 0.618);
}

struct Material {
    vec3 color;
    float emission;
    float reflectivity; // 0 = diffuse, 1 = perfect mirror
    float roughness;    // 0 = perfect, 1 = fully diffuse
    float refractivity; // 0 = opaque, 1 = transparent
    float ior;          // Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass)
};

struct Sphere {
    vec3 center;
    float radius;
    Material mat;
};

struct Plane {
    vec3 point;
    vec3 normal;
    Material mat;
};

Sphere spheres[MAX_SPHERES];
Plane ground;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float rand(vec2 co) {
    return hash(co + randSeed);
}

// Simple procedural bump map
vec3 bumpNormal(vec3 p, vec3 n, float strength) {
    float bump = sin(p.x * 6.0 + sin(p.z * 6.0)) * sin(p.z * 6.0 + sin(p.x * 6.0));
    vec3 grad = vec3(
        cos(p.x * 6.0 + sin(p.z * 6.0)) * 6.0 * sin(p.z * 6.0 + sin(p.x * 6.0)),
        0.0,
        cos(p.z * 6.0 + sin(p.x * 6.0)) * 6.0 * sin(p.x * 6.0 + sin(p.z * 6.0))
    );
    grad = normalize(grad);
    return normalize(n + grad * bump * strength);
}

bool intersectSphere(vec3 ro, vec3 rd, Sphere sph, out float t, out vec3 n, out Material mat) {
    vec3 oc = ro - sph.center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.radius * sph.radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    t = -b - h;
    if (t < 0.0) t = -b + h;
    if (t < 0.0) return false;
    vec3 hit = ro + rd * t;
    n = normalize(hit - sph.center);
    // Apply bump mapping to spheres except for the light
    if (sph.mat.emission < 1.0) {
        n = bumpNormal(hit, n, 0.25);
    }
    mat = sph.mat;
    return true;
}

// Checkerboard function
vec3 checkerboard(vec3 p) {
    float scale = 2.0;
    float check = mod(floor(p.x * scale) + floor(p.z * scale), 2.0);
    return mix(vec3(0.9,0.9,0.9), vec3(0.1,0.1,0.1), check);
}

bool intersectPlane(vec3 ro, vec3 rd, Plane pl, out float t, out vec3 n, out Material mat) {
    float denom = dot(rd, pl.normal);
    if (abs(denom) < 1e-4) return false;
    t = dot(pl.point - ro, pl.normal) / denom;
    if (t < 0.0) return false;
    n = pl.normal;
    // Apply bump mapping to ground
    vec3 hit = ro + rd * t;
    n = bumpNormal(hit, n, 0.15);
    mat = pl.mat;
    // Checkerboard color
    mat.color = checkerboard(hit);
    return true;
}

vec3 randomHemisphere(vec3 n, vec2 uv) {
    float phi = 2.0 * 3.1415926 * rand(uv);
    float cosTheta = pow(rand(uv + 0.1), 0.7); // more uniform
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 tangent = normalize(cross(n, abs(n.x) < 0.5 ? vec3(1,0,0) : vec3(0,1,0)));
    vec3 bitangent = cross(n, tangent);
    return normalize(sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * n);
}

vec3 lerp(vec3 a, vec3 b, float t) { return a + t * (b - a); }

// Sample a point on a disk with radius 1 using concentric mapping (better distribution)
vec2 disk_sample(vec2 u) {
    float r = sqrt(u.x);
    float theta = 2.0 * 3.14159265 * u.y;
    return vec2(r * cos(theta), r * sin(theta));
}

// ACES tone mapping function
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

// Calculate Fresnel reflectance (Schlick's approximation)
float fresnel(vec3 I, vec3 N, float ior) {
    float cosi = clamp(dot(-I, N), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    if (cosi > 0.0) { etai = ior; etat = 1.0; }
    cosi = abs(cosi);
    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1.0) return 1.0;
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    return (Rs * Rs + Rp * Rp) / 2.0;
}

// Refract function - reimplementation of GLSL's built-in refract
vec3 refract_ray(vec3 I, vec3 N, float eta) {
    float k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));
    if (k < 0.0) {
        // Total internal reflection
        return reflect(I, N);
    }
    return eta * I - (eta * dot(N, I) + sqrt(k)) * N;
}

// Chromatic dispersion for glass
vec3 refract_dispersion(vec3 I, vec3 N, float iorR, float iorG, float iorB) {
    return vec3(
        refract_ray(I, N, iorR).x,
        refract_ray(I, N, iorG).y,
        refract_ray(I, N, iorB).z
    );
}

void setupScene() {
    float t = time;
    Material diffuseRed;    diffuseRed.color = vec3(1,0.2,0.2);   diffuseRed.emission = 0.0; diffuseRed.reflectivity = 0.0; diffuseRed.roughness = 1.0; diffuseRed.refractivity = 0.0; diffuseRed.ior = 1.0;
    Material diffuseGreen;  diffuseGreen.color = vec3(0.2,1,0.2); diffuseGreen.emission = 0.0; diffuseGreen.reflectivity = 0.0; diffuseGreen.roughness = 1.0; diffuseGreen.refractivity = 0.0; diffuseGreen.ior = 1.0;
    Material mirror;        mirror.color = vec3(0.95,0.95,0.95);  mirror.emission = 0.0; mirror.reflectivity = 1.0; mirror.roughness = 0.0; mirror.refractivity = 0.0; mirror.ior = 1.0;
    Material glossy;        glossy.color = vec3(0.8,0.8,1.0);     glossy.emission = 0.0; glossy.reflectivity = 0.7; glossy.roughness = 0.3; glossy.refractivity = 0.0; glossy.ior = 1.0;
    Material light;         light.color = vec3(1,1,1);            light.emission = 10.0; light.reflectivity = 0.0; light.roughness = 1.0; light.refractivity = 0.0; light.ior = 1.0;
    Material blue;          blue.color = vec3(0.2,0.4,1.0);       blue.emission = 0.0; blue.reflectivity = 0.0; blue.roughness = 1.0; blue.refractivity = 0.0; blue.ior = 1.0;
    Material yellow;        yellow.color = vec3(1.0,1.0,0.2);     yellow.emission = 0.0; yellow.reflectivity = 0.0; yellow.roughness = 1.0; yellow.refractivity = 0.0; yellow.ior = 1.0;
    Material metallic;      metallic.color = vec3(0.9,0.9,0.7);   metallic.emission = 0.0; metallic.reflectivity = 0.8; metallic.roughness = 0.15; metallic.refractivity = 0.0; metallic.ior = 1.0;
    
    // Add glass material
    Material glass;         glass.color = vec3(0.9,0.9,1.0); glass.emission = 0.0; glass.reflectivity = 0.1; glass.roughness = 0.0; glass.refractivity = 0.9; glass.ior = 1.5;

    spheres[0].center = vec3(0,1.0 + 0.2 * sin(t + 0.0),0);    spheres[0].radius = 1.0;  spheres[0].mat = diffuseRed;
    spheres[1].center = vec3(-2,1.0 + 0.2 * sin(t + 1.0),2);   spheres[1].radius = 1.0;  spheres[1].mat = mirror;
    spheres[2].center = vec3(2,1.0 + 0.2 * sin(t + 2.0),2);    spheres[2].radius = 1.0;  spheres[2].mat = glossy;
    spheres[3].center = vec3(0,5.0 + 0.2 * sin(t + 3.0),0);    spheres[3].radius = 0.5;  spheres[3].mat = light;
    spheres[4].center = vec3(-3,0.7 + 0.2 * sin(t + 4.0),-2);  spheres[4].radius = 0.7;  spheres[4].mat = blue;
    // Replace yellow sphere with glass
    spheres[5].center = vec3(3,0.5 + 0.2 * sin(t + 5.0),-2);   spheres[5].radius = 0.5;  spheres[5].mat = glass;
    spheres[6].center = vec3(1.5,0.4 + 0.2 * sin(t + 6.0),-3); spheres[6].radius = 0.4;  spheres[6].mat = metallic;
    spheres[7].center = vec3(-1.5,0.6 + 0.2 * sin(t + 7.0),-3);spheres[7].radius = 0.6;  spheres[7].mat = diffuseGreen;

    ground.point = vec3(0,0,0); ground.normal = vec3(0,1,0);
    ground.mat = diffuseRed; ground.mat.color = vec3(0.8,0.8,0.8); ground.mat.emission = 0.0; ground.mat.reflectivity = 0.0; ground.mat.roughness = 1.0; ground.mat.refractivity = 0.0; ground.mat.ior = 1.0;
}

// Soft shadow ray
float softShadow(vec3 ro, vec3 rd, float maxDist) {
    float res = 1.0;
    for (int i = 0; i < MAX_SPHERES; ++i) {
        float t; vec3 n; Material m;
        if (intersectSphere(ro, rd, spheres[i], t, n, m)) {
            if (t > 0.01 && t < maxDist) res *= 0.0;
        }
    }
    float tP; vec3 nP; Material mP;
    if (intersectPlane(ro, rd, ground, tP, nP, mP)) {
        if (tP > 0.01 && tP < maxDist) res *= 0.0;
    }
    return res;
}

// Atmospheric scattering for sun
vec3 sunLight(vec3 ray) {
    float sun = clamp(dot(ray, normalize(vec3(0.5, 0.5, 0.5))), 0.0, 1.0);
    vec3 sunColor = vec3(1.0, 0.9, 0.7);
    return sunColor * pow(sun, 8.0);
}

vec3 skyColor(vec3 rd) {
    float t = 0.5 * (rd.y + 1.0);
    vec3 skyCol = lerp(vec3(0.7,0.8,1.0), vec3(0.2,0.3,0.5), t);
    // Add sun halo
    skyCol += sunLight(rd);
    return skyCol;
}

void main() {
    setupScene();
    vec3 color = vec3(0);
    int spp = samplesPerFrame;
    float aspect = float(screenWidth) / float(screenHeight);
    for (int sample = 0; sample < spp; ++sample) {
        // Improved jitter using new rand
        vec2 jitter = vec2(rand(uv + float(sample), sample, frame), rand(uv + float(sample) + 0.5, sample, frame));
        vec2 xy = (uv + (jitter - 0.5) / vec2(float(screenWidth), float(screenHeight))) * 2.0 - 1.0;
        xy.x *= aspect;
        
        vec3 forward = normalize(camDir);
        vec3 right = normalize(cross(forward, vec3(0,1,0)));
        vec3 up = cross(right, forward);
        float fov = 1.0;

        // Calculate ray direction with depth of field
        vec3 pixelPos = camPos + forward * focalDistance + xy.x * right * focalDistance * fov + xy.y * up * focalDistance * fov;
        
        // Apply lens effect if aperture > 0
        vec3 ro = camPos;
        vec3 rd;
        
        if (aperture > 0.0) {
            // Generate sample on lens
            vec2 lensPos = disk_sample(vec2(
                rand(uv + vec2(0.13, 0.27), sample, frame),
                rand(uv + vec2(0.31, 0.17), sample, frame)
            )) * aperture;
            
            // Offset ray origin by lens sample position
            ro = camPos + right * lensPos.x + up * lensPos.y;
            rd = normalize(pixelPos - ro);
        } else {
            // No depth of field, use standard ray direction
            rd = normalize(forward + xy.x * right * fov + xy.y * up * fov);
        }

        vec3 throughput = vec3(1);
        bool hitAnything = false;
        vec3 initialRd = rd;  // Store initial ray for later
        
        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
            float tMin = 1e20;
            vec3 n;
            Material mat;
            bool hit = false;
            for (int i = 0; i < MAX_SPHERES; ++i) {
                float t; vec3 ns; Material ms;
                if (intersectSphere(ro, rd, spheres[i], t, ns, ms)) {
                    if (t < tMin) { tMin = t; n = ns; mat = ms; hit = true; }
                }
            }
            float tP; vec3 nP; Material mP;
            if (intersectPlane(ro, rd, ground, tP, nP, mP)) {
                if (tP < tMin) { tMin = tP; n = nP; mat = mP; hit = true; }
            }
            if (hit) {
                hitAnything = true;
                ro = ro + rd * tMin + n * 0.001;
                color += throughput * mat.color * mat.emission / float(spp);
                if (mat.emission > 0.0 && bounce == 0) {
                    break;
                }
                
                float refl = mat.reflectivity;
                float refr = mat.refractivity;
                float rough = mat.roughness;
                
                // Handle glass/refraction
                if (refr > 0.0) {
                    // Calculate Fresnel term to blend between reflection and refraction
                    float F = fresnel(rd, n, mat.ior);
                    
                    // Probabilistically choose reflection or refraction
                    if (rand(uv + vec2(bounce, sample), sample, frame) < F) {
                        // Reflection
                        vec3 perfect = reflect(rd, n);
                        rd = perfect; // Glass is never rough in this demo
                    } else {
                        // Refraction - make sure normal faces correct direction
                        vec3 normal = dot(rd, n) < 0.0 ? n : -n;
                        float etaR = dot(rd, n) < 0.0 ? (1.0 / 1.515) : 1.515;
                        float etaG = dot(rd, n) < 0.0 ? (1.0 / 1.505) : 1.505;
                        float etaB = dot(rd, n) < 0.0 ? (1.0 / 1.495) : 1.495;
                        rd = normalize(vec3(
                            refract_ray(rd, normal, etaR).x,
                            refract_ray(rd, normal, etaG).y,
                            refract_ray(rd, normal, etaB).z
                        ));
                    }
                    throughput *= mat.color;
                }
                // Handle reflection & diffuse (existing code)
                else if (refl > 0.0) {
                    vec3 perfect = reflect(rd, n);
                    // Use improved randomness for glossy reflection
                    vec3 glossy = normalize(lerp(perfect, randomHemisphere(n, uv + float(bounce) + randSeed + float(sample) + float(frame)), rough));
                    rd = normalize(lerp(perfect, glossy, rough));
                    throughput *= mat.color;
                } else {
                    rd = randomHemisphere(n, uv + float(bounce) + randSeed + float(sample) + float(frame));
                    throughput *= mat.color;
                }
                // Direct lighting from moving lights
                for (int li = 0; li < 2; ++li) {
                    vec3 lpos = li == 0 ? light1Pos : light2Pos;
                    vec3 lcol = li == 0 ? light1Color : light2Color;
                    vec3 toLight = normalize(lpos - ro);
                    float distToLight = length(lpos - ro);
                    float shadow = softShadow(ro, toLight, distToLight);
                    float nDotL = max(dot(n, toLight), 0.0);
                    color += throughput * lcol * 8.0 * nDotL * shadow * 0.05 / float(spp);
                }
            } else {
                color += throughput * skyColor(rd) / float(spp);
                break;
            }
        }
        
        // No fog calculation needed
    }
    
    // Vignette effect
    float vignette = smoothstep(1.0, 0.7, length(uv - 0.5) * 1.4);
    color *= vignette;
    
    // Apply tone mapping for better visual quality
    color = ACESFilm(color * 1.2); // 1.2 = exposure adjustment
    
    FragColor = vec4(color, 1.0);
}
"""

def create_fullscreen_quad():
    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao

def get_camera_uniforms(camera):
    pos = camera.position
    dir = camera.get_direction()
    return pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]

def save_screenshot(width, height, program, quad_vao, current_samples=8, samples=256):
    """
    Capture the current framebuffer and save it as a PNG file with enhanced quality
    
    Args:
        width: Screen width
        height: Screen height
        program: Shader program
        quad_vao: Vertex Array Object for the fullscreen quad
        current_samples: Current samples value (passed from main loop)
        samples: Number of samples for high quality rendering (default: 256)
    """
    # Make sure the correct program is active
    glUseProgram(program)
    
    # Store current samples location
    samples_loc = glGetUniformLocation(program, "samplesPerFrame")
    
    # Set high quality samples for screenshot
    print(f"Rendering high-quality screenshot with {samples} samples...")
    glUniform1i(samples_loc, samples)
    
    # Render one frame with high quality
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(quad_vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glBindVertexArray(0)
    
    # Create a buffer to store the pixel data
    buffer = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    
    # Convert buffer to numpy array
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    
    # OpenGL returns the image flipped vertically, so we need to flip it back
    image = np.flipud(image)
    
    # Create a copy of the array since the buffer from glReadPixels is read-only
    image = image.copy()
    
    # Set alpha channel to fully opaque to avoid transparency issues
    image[:, :, 3] = 255
    
    # Convert to PIL Image
    image = Image.fromarray(image, 'RGBA')
    
    # Create screenshots directory if it doesn't exist
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshots_dir, f"render_{timestamp}_{samples}spp.png")
    
    # Save the image
    image.save(filename)
    
    # Restore original sample count
    glUniform1i(samples_loc, current_samples)
    
    print(f"High-quality screenshot saved to {filename}")
    return filename

def main():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    # Get primary monitor resolution
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    width, height = mode.size.width, mode.size.height

    # Create window
    window = glfw.create_window(width, height, "Monte Carlo Path Tracing Demo", monitor, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        sys.exit(1)

    glfw.make_context_current(window)

    scene = Scene()
    controls = Controls(window)
    program = create_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    quad_vao = create_fullscreen_quad()
    camPosLoc = glGetUniformLocation(program, "camPos")
    camDirLoc = glGetUniformLocation(program, "camDir")
    screenWidthLoc = glGetUniformLocation(program, "screenWidth")
    screenHeightLoc = glGetUniformLocation(program, "screenHeight")
    frame = 0
    prev_cam_pos = np.copy(scene.camera.position)
    prev_cam_yaw = scene.camera.yaw
    prev_cam_pitch = scene.camera.pitch
    start_time = time.time()
    time_loc = glGetUniformLocation(program, "time")
    aperture_loc = glGetUniformLocation(program, "aperture")
    focal_distance_loc = glGetUniformLocation(program, "focalDistance")
    samples_loc = glGetUniformLocation(program, "samplesPerFrame")
    
    # Default DoF settings - can be adjusted
    aperture = 0.05
    focal_distance = 5.0

    # Adaptive sampling parameters
    min_samples = 8    # Minimum samples during movement
    max_samples = 64   # Maximum samples when stationary
    stable_frames = 0  # Count of frames with no movement
    current_samples = min_samples

    # FPS counter variables
    prev_time = time.time()
    frame_count = 0
    fps = 0

    # Camera reset hotkey (R)
    default_cam_pos = np.array([0, 1, 5], dtype=np.float32)
    default_cam_yaw = 0.0
    default_cam_pitch = 0.0

    # Main loop
    while not glfw.window_should_close(window):
        # Update camera from controls
        yaw, pitch, fwd, back, left, right = controls.get_camera_delta()
        scene.camera.yaw = yaw
        scene.camera.pitch = pitch
        move_dir = glm.vec3(0)
        cam_dir = glm.normalize(glm.vec3(*scene.camera.get_direction()))
        cam_right = glm.normalize(glm.cross(cam_dir, glm.vec3(0,1,0)))
        if fwd:
            move_dir += cam_dir
        if back:
            move_dir -= cam_dir
        if left:
            move_dir -= cam_right
        if right:
            move_dir += cam_right
        if glm.length(move_dir) > 0:
            move_dir = glm.normalize(move_dir)
            scene.camera.position += move_dir * 0.1
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        # Pass camera uniforms
        pos = scene.camera.position
        dir = scene.camera.get_direction()
        glUniform3f(camPosLoc, pos[0], pos[1], pos[2])
        glUniform3f(camDirLoc, dir[0], dir[1], dir[2])
        # Pass frame and random seed
        frame_loc = glGetUniformLocation(program, "frame")
        rand_seed_loc = glGetUniformLocation(program, "randSeed")
        glUniform1i(frame_loc, frame)
        glUniform1f(rand_seed_loc, float(time.time() % 1000))
        glUniform1i(screenWidthLoc, width)
        glUniform1i(screenHeightLoc, height)
        cam_moved = not np.allclose(scene.camera.position, prev_cam_pos) or scene.camera.yaw != prev_cam_yaw or scene.camera.pitch != prev_cam_pitch
        if cam_moved:
            frame = 0
            stable_frames = 0
            current_samples = min_samples
            prev_cam_pos = np.copy(scene.camera.position)
            prev_cam_yaw = scene.camera.yaw
            prev_cam_pitch = scene.camera.pitch
        else:
            # Gradually increase samples as camera remains still
            stable_frames += 1
            if stable_frames % 10 == 0 and current_samples < max_samples:
                current_samples = min(max_samples, current_samples + 4)
        
        # Pass the adaptive sample count to shader
        glUniform1i(samples_loc, current_samples)
        
        now = time.time()
        t = now - start_time
        glUniform1f(time_loc, t)
        
        # Calculate FPS
        frame_count += 1
        if now - prev_time >= 1.0:  # Update every second
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
            # Print status to console
            print(f"FPS: {fps:.1f}, Aperture: {aperture:.3f}, Focal Distance: {focal_distance:.1f}, Samples: {current_samples} | Controls: WASD/mouse=move, Z/X=aperture, C/V=focal dist, R=reset, F10=screenshot", end="\r")
            
        # Animate colorful moving lights
        # Light 1: orbiting, magenta
        light1_pos = [2.5 * math.cos(t), 2.5 + 1.0 * math.sin(t * 1.2), 2.5 * math.sin(t)]
        light1_color = [1.0, 0.2 + 0.8 * abs(math.sin(t * 0.7)), 1.0]
        # Light 2: orbiting, cyan
        light2_pos = [2.5 * math.cos(t + math.pi), 2.5 + 1.0 * math.cos(t * 1.1), 2.5 * math.sin(t + math.pi)]
        light2_color = [0.2 + 0.8 * abs(math.cos(t * 0.9)), 1.0, 1.0]
        # Pass animated light positions/colors as uniforms
        light1_pos_loc = glGetUniformLocation(program, "light1Pos")
        light1_col_loc = glGetUniformLocation(program, "light1Color")
        light2_pos_loc = glGetUniformLocation(program, "light2Pos")
        light2_col_loc = glGetUniformLocation(program, "light2Color")
        glUniform3f(light1_pos_loc, *light1_pos)
        glUniform3f(light1_col_loc, *light1_color)
        glUniform3f(light2_pos_loc, *light2_pos)
        glUniform3f(light2_col_loc, *light2_color)
        
        # Handle keys for aperture and focal distance adjustment
        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
            aperture = max(0.0, aperture - 0.005)
        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
            aperture += 0.005
        if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS:
            focal_distance = max(0.1, focal_distance - 0.2)
        if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS:
            focal_distance += 0.2
            
        # Pass DoF parameters to shader
        glUniform1f(aperture_loc, aperture)
        glUniform1f(focal_distance_loc, focal_distance)
        
        # Camera reset
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            scene.camera.position = np.copy(default_cam_pos)
            scene.camera.yaw = default_cam_yaw
            scene.camera.pitch = default_cam_pitch
        
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        # Screenshot functionality (must be before swap_buffers)
        if glfw.get_key(window, glfw.KEY_F10) == glfw.PRESS:
            screenshot_path = save_screenshot(width, height, program, quad_vao, current_samples)
            print(f"Screenshot saved to: {screenshot_path}")
        glfw.swap_buffers(window)
        glfw.poll_events()
        frame += 1

    glfw.terminate()


if __name__ == "__main__":
    main()
