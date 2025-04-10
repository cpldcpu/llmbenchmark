import random

def generate_landscape(width=80, height=20):
    """
    Generate a simple landscape as ASCII art.

    Args:
        width (int): The width of the landscape.
        height (int): The height of the landscape.

    Returns:
        str: The ASCII art landscape as a string.
    """
    landscape = ""
    for y in range(height):
        for x in range(width):
            if y == height - 1:
                # Ground level
                landscape += "~"
            elif y > height - 3 - random.randint(0, 2):
                # Hills
                if random.random() < 0.5:
                    landscape += "*"
                else:
                    landscape += "~"
            elif y == height - random.randint(3, 6):
                # Mountain peaks
                landscape += "^"
            elif random.random() < 0.01:
                # Trees
                landscape += "T"
            else:
                # Sky
                landscape += " "
        landscape += "\n"
    return landscape

def main():
    print(generate_landscape())

if __name__ == "__main__":
    main()
