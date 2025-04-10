def draw_landscape():
    # Set the dimensions
    width = 80
    height = 25
    
    # Initialize the canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw sky
    for y in range(10):
        for x in range(width):
            if x % 20 == 0 and y < 3:
                canvas[y][x] = '*'  # stars
    
    # Draw mountains
    mountain_points = [(0, 15), (15, 8), (30, 12), (45, 6), (60, 10), (width-1, 14)]
    for i in range(len(mountain_points) - 1):
        x1, y1 = mountain_points[i]
        x2, y2 = mountain_points[i+1]
        
        # Draw mountain lines
        for x in range(x1, x2 + 1):
            # Linear interpolation
            progress = (x - x1) / (x2 - x1) if x2 != x1 else 0
            y = int(y1 + progress * (y2 - y1))
            
            # Fill mountains
            for fy in range(y, height):
                if fy < height - 10:  # Leave space for lake
                    canvas[fy][x] = '^' if fy == y else '#'
    
    # Draw lake
    for y in range(height - 10, height - 5):
        for x in range(10, width - 10):
            canvas[y][x] = '~'
    
    # Draw trees
    tree_positions = [5, 12, 20, 35, 50, 65, 75]
    for x in tree_positions:
        # Tree top
        if x < width - 2 and height - 11 < height:
            canvas[height - 11][x] = '^'
            canvas[height - 12][x] = '^'
            canvas[height - 11][x - 1] = '^'
            canvas[height - 11][x + 1] = '^'
        
        # Tree trunk
        if height - 10 < height:
            canvas[height - 10][x] = '|'
    
    # Draw cabin
    cabin_x, cabin_y = 25, height - 8
    
    # Cabin walls
    for y in range(cabin_y, cabin_y + 3):
        for x in range(cabin_x, cabin_x + 6):
            canvas[y][x] = '#'
    
    # Cabin door
    canvas[cabin_y + 1][cabin_x + 2] = '|'
    canvas[cabin_y + 2][cabin_x + 2] = '|'
    
    # Cabin window
    canvas[cabin_y + 1][cabin_x + 4] = '□'
    
    # Cabin roof
    for i in range(6):
        if 0 <= cabin_y - 1 < height and 0 <= cabin_x + i < width:
            canvas[cabin_y - 1][cabin_x + i] = '^'
    
    # Draw ground
    for x in range(width):
        if height - 5 < height:
            canvas[height - 5][x] = '_'
    
    # Convert canvas to string and print
    landscape = '\n'.join([''.join(row) for row in canvas])
    print(landscape)

if __name__ == "__main__":
    print("Mountain Landscape ASCII Art:")
    draw_landscape()
    print("\nEnjoy the view!")