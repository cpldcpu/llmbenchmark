import random

def generate_landscape(width=80, height=20):
    """
    Generates a landscape as ASCII art.

    Args:
        width (int): The width of the landscape (default: 80)
        height (int): The height of the landscape (default: 20)

    Returns:
        str: The ASCII art landscape as a string
    """
    landscape = []

    # Sky
    sky_height = int(height * 0.3)
    for _ in range(sky_height):
        sky_line = ''
        for _ in range(width):
            if random.random() < 0.05:  # 5% chance of a star
                sky_line += '*'
            else:
                sky_line += ' '
        landscape.append(sky_line)

    # Mountains
    mountain_height = int(height * 0.4)
    mountain_peak = random.randint(int(width * 0.3), int(width * 0.7))
    for i in range(mountain_height):
        mountain_line = ''
        for j in range(width):
            if j < mountain_peak - i or j > mountain_peak + i:
                mountain_line += ' '
            else:
                if random.random() < 0.5:
                    mountain_line += '/'
                else:
                    mountain_line += '\\'
        landscape.append(mountain_line)

    # Ground
    ground_height = height - sky_height - mountain_height
    tree_line = int(ground_height * 0.7)  # Line where trees start
    for i in range(ground_height):
        ground_line = ''
        for j in range(width):
            if i < tree_line:
                ground_line += '~'  # Grass
            else:
                if random.random() < 0.1:  # 10% chance of a tree
                    if j > 0 and ground_line[-1] == '|':
                        ground_line += '_'
                    else:
                        ground_line += '|'
                else:
                    ground_line += '~'
        landscape.append(ground_line)

    return '\n'.join(landscape)

def main():
    print(generate_landscape())

if __name__ == "__main__":
    main()