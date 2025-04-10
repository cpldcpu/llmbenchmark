import random

def draw_landscape(width=80, height=20):
    """
    Generate a landscape as ASCII art.

    Args:
        width (int): Width of the landscape (default: 80)
        height (int): Height of the landscape (default: 20)

    Returns:
        str: ASCII art landscape as a string
    """
    landscape = []

    # Sky
    for y in range(height // 3):
        line = ''
        for x in range(width):
            if x == width // 2 and y == 2:  # Sun position
                line += '@'
            else:
                line += ' '
        landscape.append(line)

    # Hills
    hill_height = height // 3
    for y in range(height // 3, height):
        line = ''
        for x in range(width):
            hill_threshold = int(hill_height + 2 * hill_height * (1 - (x - width // 2) ** 2 / (width // 2) ** 2) ** 0.5)
            if y > hill_threshold:
                line += '_'
            elif y == hill_threshold:
                if random.random() < 0.2:  # Add some randomness to hill tops
                    line += '/'
                elif random.random() < 0.2:
                    line += '\\'
                else:
                    line += '_'
            else:
                line += ' '
        landscape.append(line)

    # Trees
    tree_positions = random.sample(range(5, width - 5), 5)  # Randomly select tree positions
    for tree_x in tree_positions:
        tree_height = random.randint(3, 6)  # Random tree height
        for y in range(height - tree_height, height):
            landscape[y] = landscape[y][:tree_x] + '|' + landscape[y][tree_x + 1:]
        landscape[height - tree_height - 1] = landscape[height - tree_height - 1][:tree_x] + '^' + landscape[height - tree_height - 1][tree_x + 1:]

    return '\n'.join(landscape)

def main():
    print(draw_landscape())

if __name__ == "__main__":
    main()