def ascii_maze_to_svg(ascii_maze, square_size=20):
    # Split the input into rows
    rows = ascii_maze.strip().split("\n")
    height = len(rows)
    width = len(rows[0])

    # SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * square_size}" height="{height * square_size}" viewBox="0 0 {width * square_size} {height * square_size}">'

    # Process each character in the maze
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            color = "#000000"  # Default color is black for walls
            if char == " ":
                color = "#FFFFFF"  # White for corridors
            elif char == "G":
                color = "#2ca02c"  # Green for the goal
            elif char == "O":
                color = "#ff7f0e"  # Orange for origin

            # Draw the rectangle
            svg += f'<rect x="{x * square_size}" y="{y * square_size}" width="{square_size}" height="{square_size}" fill="{color}" />'

    # SVG footer
    svg += "</svg>"

    return svg


maze = """
#####################
# # # #      O      #
# # # # ########### #
#     #         #   #
##### ####### # # ###
# #       # # # #   #
# ##### ### # ### ###
# #     #     # # # #
# # # ### ##### ### #
#   # #     # #   # #
##### ##### # # ### #
# #       # #     # #
# ### # ### # ### # #
#     # #       #   #
# ### ##### ### ### #
#   #     # # # # # #
##### ##### # ### # #
#       # # #   # # #
# ####### # ### # # #
#                 #G#
#####################
"""
svg_output = ascii_maze_to_svg(maze)
with open("maze.svg", "w") as f:
    print(svg_output, file=f)
