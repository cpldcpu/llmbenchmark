import random

def draw_landscape(width=80, height=20):
    # Initialize the landscape with empty space
    landscape = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw the horizon line
    horizon_line = height // 2
    for x in range(width):
        landscape[horizon_line][x] = '-'

    # Draw mountains
    mountain_height = random.randint(horizon_line // 2, horizon_line)
    mountain_width = random.randint(width // 4, width // 2)
    mountain_x = random.randint(0, width - mountain_width)
    for x in range(mountain_x, mountain_x + mountain_width):
        y = int(horizon_line - (x - mountain_x) * (horizon_line - mountain_height) / (mountain_width // 2))
        if y < horizon_line:
            landscape[y][x] = '^'

    # Draw trees
    num_trees = random.randint(3, 6)
    for _ in range(num_trees):
        tree_x = random.randint(0, width - 1)
        tree_y = horizon_line + 1
        tree_height = random.randint(2, 5)
        for y in range(tree_y, tree_y + tree_height):
            if y < height:
                landscape[y][tree_x] = '|'
        landscape[tree_y + tree_height - 1][tree_x] = '+'

    # Draw the sun
    sun_x = random.randint(0, width // 2)
    sun_y = random.randint(0, horizon_line // 2)
    landscape[sun_y][sun_x] = '*'

    # Print the landscape
    for row in landscape:
        print(''.join(row))

def main():
    draw_landscape()

if __name__ == "__main__":
    main()
