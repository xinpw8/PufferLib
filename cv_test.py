import numpy as np
import cv2
from pokegym import checkpoint_tracker

def create_data_image(width, height):
    data_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Initialize the formatted table string
    formatted_table = ""

    with open("experiments/run_stats.txt", "r") as file:
        epoch_sps = file.read()
    
    # Create a table header
    formatted_table += f"Epoch SPS: {epoch_sps}\n"
    formatted_table += f"\n\n\n\n\n"
    formatted_table += f"{' ' * 5}{'Milestone':<20} {' Time (hours)':<15}\n"
    formatted_table += f"{'_' * 5}{'________'}{'_' * 9} {'______________':<15}\n"

    # Iterate over each checkpoint and extract milestone and time achieved
    for milestone, (_, time_achieved) in checkpoints.items():
        time_achieved = str(time_achieved) if time_achieved is not None else '-'
        # Calculate the position of the pipe symbol dynamically
        pipe_position = 25 + 1 + (20 - len(milestone))
        formatted_table += f"{milestone}{' ' * (25 - len(milestone))}| {time_achieved:<15}\n"
    
    font_scale = 5
    font_thickness = 10
    font_color = (255, 255, 255)

    x, y = 40, 40 # 440 with no epoch sps printout
    i = 0
    for line in formatted_table.split('\n'):
        i += font_scale * 35
        # Find the position of the pipe symbol
        pipe_position = line.find('|')
        # Mandatory to prevent sps number from wrapping onto same line
        if line.startswith("Epoch SPS:"):
            cv2.putText(data_image, line, (x, i + y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        else:
            cv2.putText(data_image, line[:pipe_position], (x, i + y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            cv2.putText(data_image, line[pipe_position:], (x + pipe_position * 80, i + y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    return data_image

checkpoints = checkpoint_tracker.checkpoint_met(40, 40)

bg = cv2.imread('kanto_map_dsv.png')
di_height = 6976 * 1.5
data_image = create_data_image(int(di_height), 7104)

# Define the position where the data image will be rendered
x_position = 6976
y_position = 7104

# Calculate the starting position for rendering the data image
start_x = x_position - data_image.shape[1]
start_y = y_position - data_image.shape[0]

# Ensure the data image fits within the bounds of the background image
end_x = min(start_x + data_image.shape[1], bg.shape[1])
end_y = min(start_y + data_image.shape[0], bg.shape[0])

# Calculate the region of interest on the background image
roi_start_x = max(start_x, 0)
roi_end_x = min(end_x, bg.shape[1])
roi_start_y = max(start_y, 0)
roi_end_y = min(end_y, bg.shape[0])

# Calculate the corresponding region on the data image
data_roi_start_x = roi_start_x - start_x
data_roi_end_x = data_roi_start_x + (roi_end_x - roi_start_x)
data_roi_start_y = roi_start_y - start_y
data_roi_end_y = data_roi_start_y + (roi_end_y - roi_start_y)

# Overlay the data image onto the background image
# bg[roi_start_y:roi_end_y, roi_start_x:roi_end_x] += data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x]
data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x] += bg[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

# For testing - uncomment if you want to see what it looks like before wandb
# cv2.imwrite('image_test.png',data_image)
