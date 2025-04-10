import random

def generate_landscape(width=80, height=20):
    """
    Generate a landscape as ASCII art.

    Args:
        width (int): The width of the landscape. Defaults to 80.
        height (int): The height of the landscape. Defaults to 20.

    Returns:
        str: The ASCII art landscape as a string.
    """
    landscape = []

    # Sky
    sky_height = height // 3
    for _ in range(sky_height):
        row = ''
        for _ in range(width):
            if random.random() < 0.05:  # 5% chance of a star
                row += '*'
            else:
                row += ' '
        landscape.append(row)

    # Mountains
    mountain_height = height // 3
    mountain_peaks = [random.randint(0, width - 1) for _ in range(3)]  # 3 mountain peaks
    for y in range(mountain_height):
        row = ''
        for x in range(width):
            for peak in mountain_peaks:
                if x == peak - y or x == peak + y:
                    row += '/'
                    break
                elif x == peak:
                    row += '^'  # peak of the mountain
                    break
            else:
                if y == mountain_height - 1:  # base of the mountains
                    row += '_'
                else:
                    row += ' '
        landscape.append(row)

    # Ground
    ground_height = height - sky_height - mountain_height
    tree_line = height - ground_height // 2
    for y in range(ground_height):
        row = ''
        for x in range(width):
            if y == 0 and random.random() < 0.2:  # 20% chance of a tree
                row += '|'
            elif y < ground_height - 1:
                row += ' '
            else:
                if random.random() < 0.1:  # 10% chance of a rock
                    row += '^'
                else:
                    row += '~'  # grass
            # Add some tree trunks and foliage
            if y > 0 and y < tree_line and random.random() < 0.05:
                if row[-1] == ' ':
                    row = row[:-1] + '|'
                elif row[-1] == '|':
                    row = row[:-1] + '_'
        landscape.append(row)

    return '\n'.join(landscape)

def main():
    print(generate_landscape())

if __name__ == "__main__":
    main()