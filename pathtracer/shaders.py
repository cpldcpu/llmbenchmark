from OpenGL.GL import *


def load_shader(shader_type, shader_source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    # Check compilation status
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {error}")
    return shader


def create_program(vertex_src, fragment_src):
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_src)
    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_src)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # Check link status
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    # Clean up shaders (no longer needed after linking)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program
