#!/usr/bin/env python3

import os
import sys

###############################################################################
# Terminal Utilities
###############################################################################

def clear_screen():
    """Clears the terminal screen (Unix-like). Adjust if needed for Windows."""
    os.system("clear")  # On Windows, you might do "cls" instead.

def pufferfish_header():
    """
    Prints a line of pufferfish with "PUFFER OCEAN UI" centered, replacing
    that many fish. We assume 30 total 'slots' for the line.
    """
    total_slots = 30
    header_text = "PUFFER OCEAN UI"
    text_len = len(header_text)

    # How many fish do we need on each side?
    # We'll do a simple integer division approach
    left_fish_count = max(0, (total_slots - text_len) // 2)
    right_fish_count = max(0, total_slots - left_fish_count - text_len)

    left_fish = "🐡" * left_fish_count
    right_fish = "🐡" * right_fish_count

    print(left_fish + header_text + right_fish)

def pufferfish_divider():
    """A shorter pufferfish line for dividing sections."""
    print("🐡" * 30)

def get_keypress():
    """
    Reads a single key from stdin, immediately, without enter.
    - 'x' => exit
    - Otherwise returns the character as-is (lower/upper not forced here).
    """
    try:
        # Windows approach
        import msvcrt
        ch = msvcrt.getch()
        c = ch.decode('utf-8', errors='ignore')
        if c.lower() == 'x':
            sys.exit("Exited.")
        return c
    except ImportError:
        # Unix-like approach
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch.lower() == 'x':
                sys.exit("Exited.")
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

###############################################################################
# Helper Functions
###############################################################################

def show_summary(selected):
    """Clear screen, print the pufferfish header, then show current selections."""
    clear_screen()
    pufferfish_header()
    pufferfish_divider()
    print("Current Selections:")
    print(f"  Mode: {selected['mode'] or '-'}")
    print(f"  Environment: {selected['env'] or '-'}")
    print(f"  Extract Weights: {selected['extract'] or '-'}")
    print(f"  Runtime: {selected['runtime'] or '-'}")
    pufferfish_divider()

def load_env_names(config_subdir="config/ocean"):
    """
    Looks for .ini files inside config/ocean/.
    Returns a list of environment names (with .ini removed).
    """
    env_names = []
    if not os.path.isdir(config_subdir):
        return env_names

    for filename in os.listdir(config_subdir):
        if filename.endswith(".ini"):
            env_names.append(filename[:-4])  # remove ".ini"
    return env_names

def decimal_to_hex_label(n):
    """
    Convert a 1-based integer (1..16) to single-hex digit (1..9, a..f).
    Returns None if n > 16 (we only label up to 16 per page).
    """
    if n <= 9:
        return str(n)
    elif 10 <= n <= 15:
        return chr(ord('a') + (n - 10))
    elif n == 16:
        return 'f'
    else:
        return None

def chunk_envs(env_names, page_size=16):
    """Break the environment list into pages of size 16."""
    return [env_names[i:i+page_size] for i in range(0, len(env_names), page_size)]

def pick_env_menu(env_names, selected):
    """
    Presents a paginated menu of environment names and returns the chosen name.
    Up to 16 envs per page (labeled 1..9,a..f).

    Navigation Pane:
    [ p : previous | n : next | 0 : main menu | x : exit ]

    'p' => previous page
    'n' => next page
    '0' => return to main menu
    'x' => exit
    '1..9,a..f' => pick environment
    """
    pages = chunk_envs(env_names, page_size=16)
    if not pages:
        return None

    current_page = 0

    while True:
        show_summary(selected)
        print("Environments (Page {}/{}):".format(current_page+1, len(pages)))
        pufferfish_divider()

        this_page_envs = pages[current_page]
        for i, env in enumerate(this_page_envs, start=1):
            label = decimal_to_hex_label(i)
            if label:
                print(f"[{label}] {env}")

        pufferfish_divider()
        print("[ p : previous | n : next | 0 : main menu | x : exit ]")
        print("Press single key to select environment or navigate.")
        print("Choice: ", end='', flush=True)

        ch = get_keypress().lower()
        print(ch)  # echo

        if ch == 'p':
            if current_page > 0:
                current_page -= 1
            continue
        elif ch == 'n':
            if current_page < len(pages) - 1:
                current_page += 1
            continue
        elif ch == '0':
            return None

        # Check if it's a valid environment label
        valid_labels = []
        for i, env in enumerate(this_page_envs, start=1):
            lbl = decimal_to_hex_label(i)
            if lbl:
                valid_labels.append(lbl)

        if ch in valid_labels:
            if ch.isdigit():
                idx = int(ch)
            else:
                idx = 10 + (ord(ch) - ord('a'))
            chosen_env = this_page_envs[idx - 1]
            return chosen_env
        # If not recognized, loop again

def pick_yes_no(prompt):
    """
    Ask the user a yes/no question: (y/n/x).
    Returns 'y' or 'n'.
    'x' => exit immediately.
    """
    while True:
        print(f"{prompt} (y/n/x): ", end='', flush=True)
        ch = get_keypress().lower()
        print(ch)  # echo
        if ch in ['y', 'n']:
            return ch
        elif ch == 'x':
            sys.exit("Exited.")

def pick_train_runtime(selected):
    """
    Prompt for [1] multiprocessing, [2] native, [3] serial.
    'x' => exit immediately.
    """
    while True:
        show_summary(selected)
        print("Choose a runtime option for training:")
        pufferfish_divider()
        print("[1] multiprocessing")
        print("[2] native")
        print("[3] serial")
        pufferfish_divider()
        print("Choice (1/2/3, x=exit): ", end='', flush=True)

        ch = get_keypress().lower()
        print(ch)
        if ch == '1':
            return "multiprocessing"
        elif ch == '2':
            return "native"
        elif ch == '3':
            return "serial"
        elif ch == 'x':
            sys.exit("Exited.")
        # Otherwise invalid => loop

###############################################################################
# Main Flow
###############################################################################

def main():
    # Keep track of user selections
    selected = {
        "mode": None,
        "env": None,
        "extract": None,  # y/n for eval -w
        "runtime": None   # train runtime
    }

    env_names = load_env_names("config/ocean")

    while True:
        show_summary(selected)
        print("Main Menu - Select Mode:")
        pufferfish_divider()
        print("[1] eval\n[2] train\n[3] compile\n(x to exit)")
        print("Choice: ", end='', flush=True)

        ch = get_keypress().lower()
        print(ch)  # echo
        if ch == '1':
            selected["mode"] = "eval"
        elif ch == '2':
            selected["mode"] = "train"
        elif ch == '3':
            selected["mode"] = "compile"
        elif ch == 'x':
            sys.exit("Exited.")
        else:
            continue  # invalid => loop again

        # Clear after mode selection
        show_summary(selected)

        # Pick environment
        env = pick_env_menu(env_names, selected)
        if not env:
            # user pressed 0 => back to main menu
            selected["env"] = None
            continue
        selected["env"] = env

        # If eval => ask about extracting weights
        if selected["mode"] == "eval":
            ans = pick_yes_no("Extract latest model weights before eval?")
            selected["extract"] = ans

        # If train => pick runtime
        if selected["mode"] == "train":
            runtime = pick_train_runtime(selected)
            selected["runtime"] = runtime

        # Build the final command
        mode = selected["mode"]
        command_parts = ["python", "run.py", selected["env"], mode]

        # If eval + y => -w
        if mode == "eval" and selected["extract"] == 'y':
            command_parts.append("-w")

        # If train => --vec ...
        if mode == "train" and selected["runtime"]:
            command_parts.extend(["--vec", selected["runtime"]])

        cmd_str = " ".join(command_parts)

        show_summary(selected)
        pufferfish_divider()
        print(f"Constructed command: {cmd_str}")
        pufferfish_divider()

        # Confirm run
        ans = pick_yes_no("Run this command now?")
        if ans == 'y':
            print("Running the command...")
            os.system(cmd_str)
        else:
            print("Command not executed.")

        # Return to main menu loop (keeping user selections if you like)
        # If you want to reset everything each time, you could do so here.

if __name__ == "__main__":
    main()
