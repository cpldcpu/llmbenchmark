import glfw

class Controls:
    def __init__(self, window):
        self.window = window
        self.last_x, self.last_y = glfw.get_cursor_pos(window)
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_forward = False
        self.move_backward = False
        self.move_left = False
        self.move_right = False
        self.mouse_look = False
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_cursor_pos_callback(window, self.mouse_callback)
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if key == glfw.KEY_W:
            self.move_forward = action != glfw.RELEASE
        if key == glfw.KEY_S:
            self.move_backward = action != glfw.RELEASE
        if key == glfw.KEY_A:
            self.move_left = action != glfw.RELEASE
        if key == glfw.KEY_D:
            self.move_right = action != glfw.RELEASE

    def mouse_callback(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        self.yaw += dx * 0.1
        self.pitch -= dy * 0.1
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def get_camera_delta(self):
        return self.yaw, self.pitch, self.move_forward, self.move_backward, self.move_left, self.move_right
