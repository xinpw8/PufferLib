import numpy as np
import torch
import cv2
from pokegym import checkpoint_tracker

def create_data_image(data, width, height):
    text = '\n'.join([f'{k},{v}\n' for k, v in data.items()])

    data_image = cv2.imread('your_image_path.jpg')

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)

    x, y = 440, 440

    for line in text.split('\n'):
        cv2.putText(data_image, line, (x, y), font, font_scale, font_color, font_thickness)

    cv2.imsave('Image with Text', data_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data_image

def make_pokemon_red_overlay(bg, counts):
    nonzero = np.where(counts > 0, 1, 0)
    scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.zeros((*counts.shape, 3))
    hsv[..., 0] = (240.0 / 360) - scaled * (240.0 / 360.0) # bad heatmap with too much icky light green 2*(1-scaled)/3
    hsv[..., 1] = nonzero
    hsv[..., 2] = nonzero

    # Convert the HSV image to RGB
    import matplotlib.colors as mcolors
    overlay = 255*mcolors.hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ones((16, 16, 1), dtype=np.uint8)
    overlay = np.kron(overlay, kernel).astype(np.uint8)
    mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=-1).astype(bool)

    # Combine with background
    render = bg.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render

def rollout(env_creator, env_kwargs, agent_creator, agent_kwargs, model_path=None, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True

    import cv2
    bg = cv2.imread('kanto_map_dsv.png')
    
    while True:
        if terminal or truncated:
            if verbose:
                print('---  Reset  ---')
                
            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        counts_map = env.env.counts_map
        if np.sum(counts_map) > 0 and step % 500 == 0:
            overlay = make_pokemon_red_overlay(bg, counts_map)
            
            # BET 
            checkpoints = checkpoint_tracker.checkpoint_met(40, 40)
            di_height = 6976 * 1.5
            data_image = create_data_image(checkpoints, int(di_height), 7104)

            # Define the position where the data image will be rendered
            x_position =6976
            y_position =7104

            # Calculate the starting position for rendering the data image
            start_x = x_position - data_image.shape[1]
            start_y = y_position - data_image.shape[0]

            # Ensure the data image fits within the bounds of the background image
            end_x = min(start_x + data_image.shape[1], overlay.shape[1])
            end_y = min(start_y + data_image.shape[0], overlay.shape[0])

            # Calculate the region of interest on the background image
            roi_start_x = max(start_x, 0)
            roi_end_x = min(end_x, overlay.shape[1])
            roi_start_y = max(start_y, 0)
            roi_end_y = min(end_y, overlay.shape[0])

            # Calculate the corresponding region on the data image
            data_roi_start_x = roi_start_x - start_x
            data_roi_end_x = data_roi_start_x + (roi_end_x - roi_start_x)
            data_roi_start_y = roi_start_y - start_y
            data_roi_end_y = data_roi_start_y + (roi_end_y - roi_start_y)

            # Overlay the data image onto the background image
            # bg[roi_start_y:roi_end_y, roi_start_x:roi_end_x] += data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x]
            data_image[data_roi_start_y:data_roi_end_y, data_roi_start_x:data_roi_end_x] += overlay[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

            cv2.imshow('Pokemon Red', data_image)
            # cv2.imshow('Pokemon Red', overlay[1000:][::4, ::4])
            cv2.waitKey(1)

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        if not env_kwargs['headless']:
            env.render()

        step += 1
