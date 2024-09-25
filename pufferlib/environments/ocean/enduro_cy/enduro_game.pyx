cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t  # Correct import for uint8_t

# Declare memset to clear the frame buffer
cdef extern from "string.h":
    void memset(void *s, int c, size_t n)

# Constants for the game dimensions
cdef int SCREEN_WIDTH = 160
cdef int SCREEN_HEIGHT = 210
cdef int PIXEL_DEPTH = 3  # RGB

# Game rendering area
cdef int GAME_AREA_WIDTH = 152
cdef int GAME_AREA_HEIGHT = 155
cdef int GAME_START_X = 8
cdef int GAME_END_Y = 155

# Scoreboard
cdef int SCOREBOARD_X = 49
cdef int SCOREBOARD_Y = 161
cdef int SCOREBOARD_WIDTH = 64
cdef int SCOREBOARD_HEIGHT = 30

# Initialize the frame buffer using a typed memory view
cdef class EnduroGame:
    cdef:
        cnp.uint8_t[:, :, :] frame_buffer  # Typed memory view for fast pixel manipulation

    def __init__(self):
        # Allocate memory for the frame buffer (full screen: 160x210)
        self.frame_buffer = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, PIXEL_DEPTH), dtype=np.uint8)

    def init_frame_buffer(self):
        """Clear the frame buffer by setting all values to 0."""
        memset(&self.frame_buffer[0, 0, 0], 0, SCREEN_WIDTH * SCREEN_HEIGHT * PIXEL_DEPTH)

    def render_game(self):
        """Render the game screen with sky, green background, road, and player's car."""
        self.init_frame_buffer()  # Clear the buffer for the new frame

        # Render the sky
        self.render_sky()

        # Render the green background (below the sky)
        self.render_background()

        # Render the road with the proper middle sections
        self.render_road()

        # Render the player's car
        self.render_car()

        # Render the black borders
        self.render_borders()

        # Render the scoreboard
        self.render_scoreboard()

    def render_sky(self):
        """Render the sky (blue) from y=0 to y=51."""
        cdef int i, j

        for i in range(52):  # y=0 to y=51
            for j in range(SCREEN_WIDTH):
                self.frame_buffer[i, j, 0] = 0   # Blue
                self.frame_buffer[i, j, 1] = 0
                self.frame_buffer[i, j, 2] = 255

    def render_background(self):
        """Render the green background for the game area below the sky (y=52 to y=155)."""
        cdef int i, j
        for i in range(52, GAME_END_Y):  # Start from y=52 (just below the sky) to y=155 (end of game area)
            for j in range(GAME_START_X, SCREEN_WIDTH):  # The game area starts at x=8 and spans to x=160
                self.frame_buffer[i, j, 0] = 0   # Green background
                self.frame_buffer[i, j, 1] = 128
                self.frame_buffer[i, j, 2] = 0

    def render_road(self):
        """Render the road, starting with the taper at y=54 and following the exact pixel positions provided."""
        
        # Left side of the road
        left_road_px = [
            (87, 52), (87, 53), (87, 54),
            (86, 55), (86, 56), (86, 57),
            (85, 58), (85, 59), (85, 60),
            (84, 61), (84, 62),
            (83, 63), (83, 64),
            (82, 65),
            (81, 66), (81, 67),
            (80, 68),
            (79, 69), (79, 70),
            (78, 71), (78, 72),
            (77, 73), (77, 74),
            (76, 75), (76, 76),
            (75, 77), (75, 78),
            (74, 79), (74, 80),
            (73, 81), (73, 82),
            (72, 83), (72, 84),
            (71, 85), (71, 86),
            (70, 87), (70, 88),
            (69, 89), (69, 90),
            (68, 91), (68, 92),
            (67, 93), (67, 94),
            (66, 95), (66, 96),
            (65, 97), (65, 98),
            (64, 99), (64, 100),
            (63, 101), (63, 102),
            (62, 103), (62, 104),
            (61, 105), (61, 106),
            (60, 107), (60, 108),
            (59, 109), (59, 110),
            (58, 111), (58, 112),
            (57, 113), (57, 114),
            (56, 115), (56, 116),
            (55, 117), (55, 118),
            (54, 119), (54, 120),
            (53, 121), (53, 122),
            (52, 123), (52, 124),
            (51, 125), (51, 126),
            (50, 127), (50, 128),
            (49, 129), (49, 130),
            (48, 131), (48, 132),
            (47, 133), (47, 134),
            (46, 135), (46, 136),
            (45, 137), (45, 138),
            (44, 139), (44, 140),
            (43, 141), (43, 142),
            (42, 143), (42, 144),
            (41, 145), (41, 146),
            (40, 147), (40, 148),
            (39, 149), (39, 150),
            (38, 151), (38, 152),
            (37, 153), (37, 154)
        ]

        # Right side of the road
        right_road_px = [
            (88, 54), (88, 55),
            (89, 56), (89, 57),
            (90, 58), (90, 59), (90, 60),
            (91, 61), (91, 62),
            (92, 63), (92, 64),
            (93, 65),
            (94, 66), (94, 67),
            (95, 68), (95, 69),
            (96, 70), (96, 71),
            (97, 72), (97, 73),
            (98, 74), (98, 75),
            (99, 76), (99, 77),
            (100, 78), (100, 79),
            (101, 80), (101, 81),
            (102, 82), (102, 83),
            (103, 84), (103, 85),
            (104, 86), (104, 87),
            (105, 88), (105, 89),
            (106, 90), (106, 91),
            (107, 92), (107, 93),
            (108, 94), (108, 95),
            (109, 96), (109, 97),
            (110, 98), (110, 99),
            (111, 100), (111, 101),
            (112, 102), (112, 103),
            (113, 104), (113, 105),
            (114, 106), (114, 107),
            (115, 108), (115, 109),
            (116, 110), (116, 111),
            (117, 112), (117, 113),
            (118, 114), (118, 115),
            (119, 116), (119, 117),
            (120, 118), (120, 119),
            (121, 120), (121, 121),
            (122, 122), (122, 123),
            (123, 124), (123, 125),
            (124, 126), (124, 127),
            (125, 128), (125, 129),
            (126, 130), (126, 131),
            (127, 132), (127, 133),
            (128, 134), (128, 135),
            (129, 136), (129, 137),
            (130, 138), (130, 139),
            (131, 140), (131, 141),
            (132, 142), (132, 143),
            (133, 144), (133, 145),
            (134, 146), (134, 147),
            (135, 148), (135, 149),
            (136, 150), (136, 151),
            (137, 152), (137, 153),
            (138, 154)
        ]

        # Draw the left side of the road
        for x, y in left_road_px:
            self.frame_buffer[y, x, 0] = 200  # Gray/white road
            self.frame_buffer[y, x, 1] = 200
            self.frame_buffer[y, x, 2] = 200

        # Draw the right side of the road
        for x, y in right_road_px:
            self.frame_buffer[y, x, 0] = 200  # Gray/white road
            self.frame_buffer[y, x, 1] = 200
            self.frame_buffer[y, x, 2] = 200


    def render_borders(self):
        """Render the black borders (left bar and bottom bar)."""
        cdef int i, j

        # Left vertical black bar (8 pixels wide)
        for i in range(SCREEN_HEIGHT):
            for j in range(GAME_START_X):
                self.frame_buffer[i, j, 0] = 0   # Black
                self.frame_buffer[i, j, 1] = 0
                self.frame_buffer[i, j, 2] = 0

        # Bottom horizontal black bar (55 pixels high)
        for i in range(GAME_END_Y, SCREEN_HEIGHT):
            for j in range(SCREEN_WIDTH):
                self.frame_buffer[i, j, 0] = 0   # Black
                self.frame_buffer[i, j, 1] = 0
                self.frame_buffer[i, j, 2] = 0

    def render_car(self):
        """Render the player's car at the starting position with white pixels."""
        car_pixels = {
            77: [147, 149, 151, 153],
            78: [147, 149, 151, 153],
            79: [144, 145, 146, 148, 150, 152, 154],
            80: [144, 145, 146, 148, 150, 152, 154],
            81: [145, 146, 148, 149, 150, 151, 152, 153],
            82: [145, 146, 148, 149, 150, 151, 152, 153],
            83: list(range(144, 155)),
            84: list(range(144, 155)),
            85: list(range(144, 155)),
            86: list(range(144, 155)),
            87: [145, 146, 148, 149, 150, 151, 152, 153],
            88: [145, 146, 148, 149, 150, 151, 152, 153],
            89: [144, 145, 146, 147, 149, 151, 153],
            90: [144, 145, 146, 147, 149, 151, 153],
            91: [148, 150, 152, 154],
            92: [148, 150, 152, 154]
        }

        # Render the car pixels in white
        for x, y_list in car_pixels.items():
            for y in y_list:
                self.frame_buffer[y, x, 0] = 255  # White
                self.frame_buffer[y, x, 1] = 255
                self.frame_buffer[y, x, 2] = 255

    def render_scoreboard(self):
        """Render the scoreboard as a red rectangle."""
        cdef int i, j
        for i in range(SCOREBOARD_Y, SCOREBOARD_Y + SCOREBOARD_HEIGHT):
            for j in range(SCOREBOARD_X, SCOREBOARD_X + SCOREBOARD_WIDTH):
                self.frame_buffer[i, j, 0] = 255  # Red
                self.frame_buffer[i, j, 1] = 0
                self.frame_buffer[i, j, 2] = 0

    def get_frame_buffer(self):
        """Return the frame buffer."""
        return self.frame_buffer
