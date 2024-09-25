from pufferlib.environments.ocean.enduro_cy import enduro_game
import matplotlib.pyplot as plt
import time

# Create an instance of the EnduroGame class
game = enduro_game.EnduroGame()

# Loop to render the game frame-by-frame
for _ in range(50):  # 50 frames to see some progress
    # Render the game (including the player and road)
    game.render_game()

    # Get the frame buffer and display it
    frame = game.get_frame_buffer()
    print("Frame buffer shape:", frame.shape)

    # Show the frame in matplotlib
    plt.imshow(frame)
    plt.draw()
    plt.pause(0)  # Pause to simulate frame rate
    plt.clf()
