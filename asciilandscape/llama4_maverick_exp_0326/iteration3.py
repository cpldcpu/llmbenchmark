import random
import os

# Constants
WIDTH = 80
HEIGHT = 20
TREE_CHARS = ['^', 'Y', 'T']
SUN_CHAR = '*'

def generate_hills(height):
    """Generate hill heights across the landscape"""
    hills = [random.randint(0, height // 2) for _ in range(WIDTH)]
    # Smooth out the hills by averaging neighboring points
    for i in range(1, WIDTH - 1):
        hills[i] = (hills[i - 1] + hills[i] + hills[i + 1]) // 3
    return hills

def add_trees(landscape):
    """Randomly place trees on the landscape"""
    for _ in range(WIDTH // 5):  # 20% of the width will have trees
        x = random.randint(0, WIDTH - 1)
        tree_height = random.randint(2, 5)
        for y in range(tree_height):
            if landscape[HEIGHT - y - 1][x] == ' ':
                landscape[HEIGHT - y - 1][x] = random.choice(TREE_CHARS)

def draw_sun(landscape):
    """Draw the sun in the top left corner"""
    landscape[0][0] = SUN_CHAR
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = 0 + dx, 0 + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            landscape[ny][nx] = SUN_CHAR

def generate_landscape(with_sun=False):
    """Generate the ASCII art landscape"""
    landscape = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    hills = generate_hills(HEIGHT)

    # Draw the hills
    for x, hill_height in enumerate(hills):
        for y in range(HEIGHT - hill_height, HEIGHT):
            landscape[y][x] = '~' if y == HEIGHT - 1 else '.'

    # Add trees
    add_trees(landscape)

    # Draw the sun
    if with_sun:
        draw_sun(landscape)

    # Convert the landscape to a string
    return '\n'.join([''.join(row) for row in landscape])

def main():
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
    print(generate_landscape(with_sun=True))

if __name__ == "__main__":
    main()