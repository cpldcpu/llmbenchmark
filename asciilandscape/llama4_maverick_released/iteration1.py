import random

def generate_landscape(width, height):
    """
    Generates a landscape as ASCII art.

    Args:
        width (int): The width of the landscape.
        height (int): The height of the landscape.

    Returns:
        str: The ASCII art landscape.
    """
    landscape = ""

    # Sky
    for _ in range(height // 3):
        landscape += "*" * width + "\n"

    # Mountains
    mountain_height = random.randint(height // 3, 2 * height // 3)
    for i in range(height // 3, mountain_height):
        mountain_line = ""
        for _ in range(width):
            if random.random() < 0.7:
                mountain_line += "^"
            else:
                mountain_line += "*"
        landscape += mountain_line + "\n"

    # Ground
    for _ in range(mountain_height, height):
        landscape += "~" * width + "\n"

    return landscape

def main():
    width = 80
    height = 20
    print(generate_landscape(width, height))

if __name__ == "__main__":
    main()
