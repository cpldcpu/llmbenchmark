import random

def generate_landscape(width=80, height=20):
    """
    Generates a landscape as ASCII art.

    Args:
        width (int, optional): The width of the landscape. Defaults to 80.
        height (int, optional): The height of the landscape. Defaults to 20.

    Returns:
        str: The ASCII art landscape as a string.
    """
    landscape = ""

    # Sky
    for _ in range(height // 3):
        row = ""
        for _ in range(width):
            if random.random() < 0.05:
                row += "*"
            else:
                row += " "
        landscape += row + "\n"

    # Mountains
    for _ in range(height // 3):
        row = ""
        for _ in range(width):
            if random.random() < 0.2:
                row += "^"
            else:
                row += " "
        landscape += row + "\n"

    # Trees and Ground
    for i in range(height - (height // 3) * 2):
        row = ""
        for j in range(width):
            if i == height - (height // 3) * 2 - 1:
                row += "~"
            elif random.random() < 0.05 and i > 0:
                row += "T"
            else:
                row += " "
        landscape += row + "\n"

    return landscape

def main():
    print(generate_landscape())

if __name__ == "__main__":
    main()
