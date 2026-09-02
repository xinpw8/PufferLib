"""Keep REK in a private-arena AI match by pressing space at the lobby prompt.

In a private area, a match is two or three rounds and then the client returns to
a lobby showing:

    no other players here
    space to fight ai

If nobody presses space, the client eventually drops out of the private lobby.
That is the thing currently costing capture sessions: the recorder is armed, the
lobby sits idle, and no round ever starts.

    python keep_in_ai_match.py --dry-run          # watch and log, press nothing
    python keep_in_ai_match.py                    # watch and press space
    python keep_in_ai_match.py --calibrate        # save screenshots to look at

Detection is deliberately conservative, because a stray space during a live
round is an input the recorder will faithfully capture as if a pilot meant it.
Space is pressed only when the prompt is actually on screen, only once per
appearance, only while REK is the foreground window, and never faster than the
cooldown. Every press is logged with its trigger and its confidence.

Two detectors:

    ocr       pytesseract over the window. Preferred: the prompt is known text,
              and matching text tolerates resolution and layout changes.
    template  a reference crop matched by normalised cross-correlation. Needs
              numpy. Use when OCR is unavailable or unreliable.

What is verified and what is not, stated plainly. The phrase matching and the
press policy are pure logic and are covered by test_keep_in_ai_match.py, which
runs anywhere. The screen capture and the keystroke are Win32 glue that cannot
be exercised off Windows; they are kept as thin as possible for that reason, and
--dry-run exists so the detector can be checked against the real screen before
anything is ever sent.
"""

import argparse
import ctypes
import json
import re
import sys
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

# The lobby text. Matched fuzzily: OCR reads "pIayers" for "players" often
# enough that exact matching would simply never fire.
PROMPT_PHRASES = ('no other players here', 'space to fight ai')

# Similarity at which a phrase counts as present. 0.8 tolerates a few
# mis-read characters without matching unrelated UI text.
DEFAULT_THRESHOLD = 0.8

VK_SPACE = 0x20
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101


# ----------------------------------------------------------------- pure logic

def normalise(text):
    """Lowercase, strip punctuation and collapse whitespace.

    OCR output is full of stray punctuation and line breaks; comparing raw
    strings would make the threshold meaningless.
    """
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower())).strip()


def phrase_score(text, phrase):
    """Best similarity between `phrase` and any similar-length window of `text`."""
    hay, needle = normalise(text), normalise(phrase)
    if not hay or not needle:
        return 0.0
    if needle in hay:
        return 1.0
    n = len(needle)
    best = 0.0
    # Windows a little shorter and longer than the phrase, so a dropped or
    # doubled character does not push the true match out of range.
    for width in (n, int(n * 1.25) + 1):
        for start in range(0, max(1, len(hay) - width + 1)):
            window = hay[start:start + width]
            best = max(best, SequenceMatcher(None, needle, window).ratio())
            if best >= 0.999:
                return best
    return best


def detect_prompt(text, phrases=PROMPT_PHRASES, threshold=DEFAULT_THRESHOLD):
    """Is the lobby prompt on screen? Returns (visible, per-phrase scores).

    Any one phrase is enough. The two lines are rendered separately and OCR
    regularly recovers one cleanly and mangles the other.
    """
    scores = {p: phrase_score(text, p) for p in phrases}
    return (max(scores.values(), default=0.0) >= threshold), scores


class PressPolicy:
    """Decides when to actually send space.

    The rules exist to make two failures impossible rather than unlikely:
    pressing during a round, and pressing repeatedly because one press did not
    register. Time is injected so this is testable without waiting.
    """

    def __init__(self, cooldown_s=3.0, retry_after_s=6.0, max_retries=3,
                 max_per_minute=12):
        self.cooldown_s = cooldown_s
        self.retry_after_s = retry_after_s
        self.max_retries = max_retries
        self.max_per_minute = max_per_minute
        self.recent = deque()
        self.last_press = None
        self.presses_this_prompt = 0
        self.prompt_up = False

    def _rate_limited(self, now):
        while self.recent and now - self.recent[0] > 60.0:
            self.recent.popleft()
        return len(self.recent) >= self.max_per_minute

    def decide(self, now, prompt_visible):
        """Return (should_press, reason)."""
        if not prompt_visible:
            if self.prompt_up:
                # Prompt cleared: the match started, so re-arm for next time.
                self.prompt_up = False
                self.presses_this_prompt = 0
            return False, 'prompt not visible'

        if not self.prompt_up:
            self.prompt_up = True
            self.presses_this_prompt = 0

        if self.last_press is not None and now - self.last_press < self.cooldown_s:
            return False, 'cooldown'

        if self.presses_this_prompt == 0:
            if self._rate_limited(now):
                return False, 'rate limited'
            return True, 'prompt appeared'

        # Still showing after a press. Either the keystroke did not land or the
        # client is slow; retry a bounded number of times, then stop and let a
        # human look rather than hammering it.
        if now - self.last_press < self.retry_after_s:
            return False, 'waiting to see if the press took'
        if self.presses_this_prompt >= self.max_retries:
            return False, f'gave up after {self.presses_this_prompt} presses'
        if self._rate_limited(now):
            return False, 'rate limited'
        return True, f'retry {self.presses_this_prompt}'

    def record_press(self, now):
        self.last_press = now
        self.presses_this_prompt += 1
        self.recent.append(now)


# -------------------------------------------------------------- windows glue
# Everything below talks to Win32 and cannot be exercised off Windows.

def _require_windows():
    if not sys.platform.startswith('win'):
        sys.exit('This runs on the Windows machine with REK on it. '
                 'The phrase matching and press policy are testable anywhere: '
                 'python test_keep_in_ai_match.py')


def find_window(process_name='REK'):
    """(hwnd, (l, t, r, b), pid) for the REK window, or None.

    Matched by the owning process, never by title alone: sending input to a
    window that merely mentions REK would be exactly the interference this is
    supposed to avoid.
    """
    import ctypes.wintypes as wt
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def enum(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel32.OpenProcess(0x0410, False, pid)   # QUERY_INFO|VM_READ
        if not handle:
            return True
        try:
            buf = ctypes.create_unicode_buffer(512)
            if psapi.GetModuleBaseNameW(handle, None, buf, 512):
                if process_name.lower() in buf.value.lower():
                    rect = wt.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    if rect.right - rect.left > 200:
                        found.append((hwnd,
                                      (rect.left, rect.top, rect.right, rect.bottom),
                                      pid.value))
        finally:
            kernel32.CloseHandle(handle)
        return True

    user32.EnumWindows(enum, 0)
    return found[0] if found else None


def is_foreground(hwnd):
    return ctypes.windll.user32.GetForegroundWindow() == hwnd


def grab(rect):
    from PIL import ImageGrab
    return ImageGrab.grab(bbox=rect, all_screens=True)


def read_text(image, scale=1.0):
    """OCR the image. Returns '' when pytesseract is unavailable."""
    try:
        import pytesseract
    except ImportError:
        return None
    if scale != 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return pytesseract.image_to_string(image.convert('L'))


def template_score(image, template_path):
    """Normalised cross-correlation of a reference crop against the image."""
    import numpy as np
    from PIL import Image
    hay = np.asarray(image.convert('L'), dtype=np.float64)
    needle = np.asarray(Image.open(template_path).convert('L'), dtype=np.float64)
    th, tw = needle.shape
    if hay.shape[0] < th or hay.shape[1] < tw:
        return 0.0
    needle = needle - needle.mean()
    nnorm = np.sqrt((needle ** 2).sum()) or 1.0
    best = 0.0
    # Coarse stride first: the prompt does not move between frames, so exact
    # alignment is not needed to decide it is there.
    for y in range(0, hay.shape[0] - th + 1, 2):
        for x in range(0, hay.shape[1] - tw + 1, 2):
            patch = hay[y:y + th, x:x + tw]
            patch = patch - patch.mean()
            denom = (np.sqrt((patch ** 2).sum()) or 1.0) * nnorm
            best = max(best, float((patch * needle).sum() / denom))
    return best


def press_space(hwnd, background=False):
    user32 = ctypes.windll.user32
    if background:
        # Non-interfering, but Unity reads raw input and usually ignores this.
        user32.PostMessageW(hwnd, WM_KEYDOWN, VK_SPACE, 0)
        user32.PostMessageW(hwnd, WM_KEYUP, VK_SPACE, 0)
        return 'PostMessage'
    user32.keybd_event(VK_SPACE, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(VK_SPACE, 0, 2, 0)     # KEYEVENTF_KEYUP
    return 'keybd_event'


# --------------------------------------------------------------------- driver

def watch(args):
    _require_windows()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    policy = PressPolicy(args.cooldown, args.retry_after, args.max_retries,
                         args.max_per_minute)
    win = find_window(args.process)
    if not win:
        sys.exit(f'no visible window belonging to a process named {args.process!r}')
    hwnd, rect, pid = win
    print(f'REK window {hwnd} pid {pid} at {rect}')
    print(f'detector={args.detector}  dry_run={args.dry_run}  log={log_path}')

    pressed = 0
    with log_path.open('a', encoding='utf-8') as log:
        while True:
            now = time.time()
            try:
                win = find_window(args.process) or win
                hwnd, rect, pid = win
                image = grab(rect)
            except Exception as e:
                json.dump({'t': now, 'event': 'capture_error', 'error': str(e)}, log)
                log.write('\n'); log.flush()
                time.sleep(args.interval)
                continue

            evidence = {}
            if args.detector == 'template':
                score = template_score(image, args.template)
                visible = score >= args.template_threshold
                evidence = {'template_score': round(score, 4)}
            else:
                text = read_text(image, args.ocr_scale)
                if text is None:
                    sys.exit('pytesseract is not installed. pip install pytesseract '
                             'and install the Tesseract binary, or use '
                             '--detector template --template crop.png')
                visible, scores = detect_prompt(text, threshold=args.threshold)
                evidence = {'scores': {k: round(v, 3) for k, v in scores.items()}}

            should, reason = policy.decide(now, visible)
            fg = is_foreground(hwnd)
            if should and not fg and not args.allow_background:
                should, reason = False, 'REK is not the foreground window'

            entry = {'t': now, 'prompt_visible': visible, 'reason': reason,
                     'foreground': fg, **evidence}

            if should:
                if args.dry_run:
                    entry['event'] = 'would_press'
                else:
                    entry['event'] = 'press'
                    entry['method'] = press_space(hwnd, args.allow_background
                                                  and not fg)
                    policy.record_press(now)
                    pressed += 1
                if shots and pressed <= args.keep_shots:
                    path = shots / f'press-{int(now)}.png'
                    image.save(path)
                    entry['screenshot'] = str(path)
                print(f'{time.strftime("%H:%M:%S")}  {entry["event"]}  {reason}')
            elif visible:
                print(f'{time.strftime("%H:%M:%S")}  prompt up, holding: {reason}')

            json.dump(entry, log)
            log.write('\n')
            log.flush()
            time.sleep(args.interval)


def calibrate(args):
    _require_windows()
    win = find_window(args.process)
    if not win:
        sys.exit(f'no window for process {args.process!r}')
    hwnd, rect, pid = win
    out = Path(args.shots or 'calibration')
    out.mkdir(parents=True, exist_ok=True)
    print(f'window {rect}. Capturing every {args.interval}s. Leave the lobby '
          f'prompt on screen for some of it, then Ctrl-C.')
    print('Crop one frame down to just the prompt text and pass it as '
          '--template for the template detector.')
    n = 0
    try:
        while True:
            image = grab(rect)
            path = out / f'frame-{int(time.time())}.png'
            image.save(path)
            text = read_text(image, args.ocr_scale)
            if text is not None:
                visible, scores = detect_prompt(text, threshold=args.threshold)
                print(f'{path.name}  prompt={visible}  '
                      f'{ {k: round(v, 2) for k, v in scores.items()} }')
            else:
                print(f'{path.name}  (no OCR available)')
            n += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f'\n{n} frames in {out}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--process', default='REK')
    ap.add_argument('--detector', default='ocr', choices=('ocr', 'template'))
    ap.add_argument('--template', help='reference crop for --detector template')
    ap.add_argument('--template-threshold', type=float, default=0.9)
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--ocr-scale', type=float, default=1.0)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--cooldown', type=float, default=3.0)
    ap.add_argument('--retry-after', type=float, default=6.0)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--max-per-minute', type=int, default=12)
    ap.add_argument('--allow-background', action='store_true',
        help='send input even when REK is not focused. Unity usually ignores '
             'PostMessage, so this often does nothing; off by default so this '
             'never types into whatever else you are doing')
    ap.add_argument('--dry-run', action='store_true',
        help='detect and log, press nothing. Run this first')
    ap.add_argument('--log', default='keep_in_ai_match.jsonl')
    ap.add_argument('--shots', help='directory to save screenshots into')
    ap.add_argument('--keep-shots', type=int, default=10)
    ap.add_argument('--calibrate', action='store_true',
        help='save frames and report detection, press nothing')
    args = ap.parse_args()

    if args.detector == 'template' and not args.template:
        ap.error('--detector template needs --template CROP.png')
    if args.calibrate:
        calibrate(args)
        return 0
    watch(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
