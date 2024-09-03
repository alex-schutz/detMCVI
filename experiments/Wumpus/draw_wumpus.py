def ascii_to_svg(ascii_grid, square_size=50):
    # Function to read and return the content of an SVG file
    def read_svg_file(filename):
        with open(filename, "r") as file:
            return file.read()

    # Function to check if a position is within grid bounds
    def in_bounds(x, y, width, height):
        return 0 <= x < width and 0 <= y < height

    # Define the SVG file paths for different characters
    images = {
        "G": "experiments/Wumpus/gold.svg",  # Gold image
        "W": "experiments/Wumpus/wumpus.svg",  # Wumpus image
        "P": "experiments/Wumpus/pit.svg",  # Pit image
        ">": "experiments/Wumpus/player.svg",  # Player image facing right
        "q": "experiments/Wumpus/arrow.svg",  # Arrow image
    }

    # Read and store the SVG content for each character
    svg_images = {char: read_svg_file(filename) for char, filename in images.items()}

    rows = ascii_grid.strip().split("\n")
    grid_height = len(rows) // 4  # Number of grid rows
    grid_width = len(rows[0]) // 4  # Number of grid columns

    # Calculate the dimensions of the SVG canvas
    svg_width = grid_width * square_size
    svg_height = grid_height * square_size
    subgrid_size = square_size / 3

    # SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">\n'
    svg += (
        f'<rect width="100%" height="100%" fill="white" />\n'  # Set background to white
    )

    # Draw the grid lines
    for i in range(grid_width + 1):
        svg += f'<line x1="{i * square_size}" y1="0" x2="{i * square_size}" y2="{svg_height}" stroke="lightgrey" stroke-width="2" />\n'
    for i in range(grid_height + 1):
        svg += f'<line x1="0" y1="{i * square_size}" x2="{svg_width}" y2="{i * square_size}" stroke="lightgrey" stroke-width="2" />\n'

    # Process each cell and add the corresponding embedded SVG content
    obs = [["" for _ in range(grid_height)] for _ in range(grid_width)]
    for y in range(grid_height):
        for x in range(grid_width):
            # Process each 3x3 subgrid position
            for sub_y in range(3):
                for sub_x in range(3):
                    # Calculate the character position in the ASCII grid
                    cell_char = rows[y * 4 + sub_y + 1][x * 4 + sub_x + 1]

                    if cell_char in svg_images:
                        # Embed the SVG content corresponding to the character
                        embedded_svg = svg_images[cell_char]
                        grid_pos_x, grid_pos_y = (sub_x, sub_y)
                        translate_x = x * square_size + grid_pos_x * subgrid_size
                        translate_y = y * square_size + grid_pos_y * subgrid_size
                        transformed_svg = f'<g transform="translate({translate_x}, {translate_y}) scale({subgrid_size/50})">{embedded_svg}</g>\n'
                        svg += transformed_svg

                    if cell_char == "G":
                        obs[x][y] += "glitter\n"
                    if cell_char == "W":
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if in_bounds(nx, ny, grid_width, grid_height):
                                obs[nx][ny] += "stench\n"
                    if cell_char == "P":
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if in_bounds(nx, ny, grid_width, grid_height):
                                obs[nx][ny] += "breeze\n"

    for y in range(grid_height):
        for x in range(grid_width):
            o = obs[x][y]
            if o:
                elems = o.splitlines()
                t = f'</tspan><tspan x="{x * square_size + square_size / 2}" dy="6">'.join(
                    elems
                )
                svg += f'<text x="{x * square_size + square_size / 2}" y="{y * square_size + 4*square_size/5}" font-size="6" text-anchor="middle" dominant-baseline="central"><tspan x="{x * square_size + square_size / 2}">{t}</tspan></text>\n'

    # SVG footer
    svg += "</svg>"

    return svg


# Example usage:
ascii_grid = """
+---+---+---+---+
|   |   |   |   |
| G | W |   | P |
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
|   |   | P |   |
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
| >q|   |   |   |
|   |   |   |   |
+---+---+---+---+
"""

svg_output = ascii_to_svg(ascii_grid)
with open("wumpus.svg", "w") as f:
    print(svg_output, file=f)
