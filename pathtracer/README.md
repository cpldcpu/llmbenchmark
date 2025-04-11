# Monte Carlo Path Tracer 

A real-time interactive Monte Carlo path tracer created as a vibe coding experiment generated with "Optimus Alpha" on openrouter, with additional assistance from Sonnet-3.7.


<p align="center">
  <img src="screenshot.png" width="70%" alt="Path Tracer Screenshot">
</p>

## Description

This project implements a Monte Carlo path tracer using OpenGL shaders to achieve real-time performance. Path tracing is a rendering technique that simulates realistic lighting by tracing the path of light rays as they bounce through a scene. 

The renderer handles:
- Multiple material types with varying reflectivity and refraction
- Soft shadows
- Fresnel effects for realistic glass
- Procedural textures and bump mapping
- Tone mapping for improved visual quality

## How to Use

### Controls

- **WASD**: Move camera position
- **Mouse**: Look around
- **Z/X**: Decrease/Increase aperture (depth of field effect)
- **C/V**: Decrease/Increase focal distance
- **R**: Reset camera to default position
- **F10**: Take high-quality screenshot
- **ESC**: Exit application

### Running the Project

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the main script:
```
python main.py
```

## Building from Source

### Prerequisites

- Python 3.6 or higher
- Dependencies (listed in requirements.txt):
  - PyOpenGL
  - PyOpenGL_accelerate
  - glfw
  - numpy
  - PIL (Pillow)

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/mcpt_shade.git
cd mcpt_shade
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
python main.py
```

## Technical Details

The path tracer uses a fragment shader to perform all ray tracing calculations on the GPU. Each frame:
1. A full-screen quad is rendered
2. For each pixel, multiple sample rays are cast
3. Each ray is traced through the scene with multiple light bounces
4. Results are accumulated and tone-mapped for display

The implementation uses adaptive sampling to maintain interactive frame rates - when the camera is moving, fewer samples are used per pixel, and when stationary, the sample count gradually increases for better quality.

## Screenshot

Take high-quality screenshots at any time by pressing F10. Screenshots are saved to the `screenshots` folder with a timestamp and sample count in the filename.

## License

This project is open source and available for educational and personal use.

## Acknowledgments

This project was generated as a vibe coding experiment with "Optimus Alpha" on openrouter, with some help from Sonnet-3.7. This readme was written mostly by Sonnet-3.7.