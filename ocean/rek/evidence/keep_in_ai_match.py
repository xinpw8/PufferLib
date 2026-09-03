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

This is the zero-install fallback. The RekUiBridgeAgent plugin under
evidence/windows reads the client's actual scene and lobby_screen state and
handles post-fight continue properly, which is strictly better whenever it is
built and injected. This needs nothing installed and no plugin, which is what
makes it worth keeping for the case where the lobby is about to time out.

Detection is deliberately conservative, because a stray space during a live
round is an input the recorder will faithfully capture as if a pilot meant it.
Space is pressed only when the prompt is actually on screen, only once per
appearance, only while REK is the foreground window, and never faster than the
cooldown. Every press is logged with its trigger and its confidence.

Three detectors, and `auto` picks the best one available, so this runs with no
installs at all:

    static    the lobby screen is frozen; a fight is not. Captures the window
              through GDI and presses once it has been unchanged for a few
              seconds. Needs nothing beyond the standard library, which is why
              it is the fallback.
    ocr       pytesseract over the window. Better when available: the prompt is
              known text, and matching text tolerates layout changes.
    template  a reference crop matched by normalised cross-correlation. Needs
              numpy and Pillow.

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

# Downscale the window to this before comparing frames. Small enough that a
# pure-Python difference over every pixel is trivial, large enough that two
# robots moving in an arena change it obviously.
SAMPLE_W, SAMPLE_H = 64, 36


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


class StaticDetector:
    """Treats a frozen screen as the lobby.

    A fight moves: two robots, a camera, a timer. The lobby prompt does not. So
    "the window has not changed for a few seconds" stands in for "the prompt is
    up" when there is no OCR to read it with, and it needs no calibration and
    nothing installed.

    It is a heuristic, not a reading of the text. A KO freeze or a pause is also
    static, which is why the dwell is seconds rather than one frame, and why the
    press policy still fires only once per episode. Prefer --detector ocr where
    pytesseract is available.
    """

    def __init__(self, dwell_s=4.0, threshold=2.0):
        self.dwell_s = dwell_s
        self.threshold = threshold
        self.last_change = None
        self.last_diff = None

    def update(self, now, diff):
        """Feed one frame difference. Returns whether the screen looks static."""
        self.last_diff = diff
        if self.last_change is None or diff > self.threshold:
            self.last_change = now
            return False
        return (now - self.last_change) >= self.dwell_s


def frame_diff(a, b):
    """Mean absolute luma difference between two downscaled BGRA frames."""
    if a is None or b is None or len(a) != len(b) or not a:
        return 255.0
    total = 0
    n = 0
    for i in range(0, len(a) - 3, 4):
        # Green alone tracks luma closely enough to decide whether this moved.
        total += abs(a[i + 1] - b[i + 1])
        n += 1
    return (total / n) if n else 255.0


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


def grab_sample(hwnd, w=SAMPLE_W, h=SAMPLE_H):
    """Downscaled BGRA bytes of a window, through GDI only.

    Deliberately dependency-free: this has to work on a machine where nothing
    can be installed right now. StretchBlt does the scaling in the driver, so
    the pure-Python comparison afterwards is over a few thousand pixels.
    """
    import ctypes.wintypes as wt
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32

    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    sw, sh = rect.right - rect.left, rect.bottom - rect.top
    if sw <= 0 or sh <= 0:
        return None

    src = user32.GetDC(hwnd)
    if not src:
        return None
    dst = gdi32.CreateCompatibleDC(src)
    bmp = gdi32.CreateCompatibleBitmap(src, w, h)
    old = gdi32.SelectObject(dst, bmp)
    try:
        gdi32.SetStretchBltMode(dst, 4)                  # HALFTONE
        if not gdi32.StretchBlt(dst, 0, 0, w, h, src, 0, 0, sw, sh, 0x00CC0020):
            return None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [('biSize', wt.DWORD), ('biWidth', ctypes.c_long),
                        ('biHeight', ctypes.c_long), ('biPlanes', wt.WORD),
                        ('biBitCount', wt.WORD), ('biCompression', wt.DWORD),
                        ('biSizeImage', wt.DWORD),
                        ('biXPelsPerMeter', ctypes.c_long),
                        ('biYPelsPerMeter', ctypes.c_long),
                        ('biClrUsed', wt.DWORD), ('biClrImportant', wt.DWORD)]

        info = BITMAPINFOHEADER()
        info.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.biWidth, info.biHeight = w, -h              # negative: top-down
        info.biPlanes, info.biBitCount = 1, 32
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(dst, bmp, 0, h, buf, ctypes.byref(info), 0):
            return None
        return buf.raw
    finally:
        gdi32.SelectObject(dst, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(dst)
        user32.ReleaseDC(hwnd, src)


def choose_detector(requested, want=None):
    """Resolve 'auto' to whatever this machine can actually do.

    `want` is an injection point so the resolution order is testable off
    Windows; it maps a detector name to whether its dependencies import.
    """
    if requested != 'auto':
        return requested
    if want is None:
        def want(name):
            try:
                import pytesseract
                from PIL import ImageGrab                   # noqa: F401
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                return False
    return 'ocr' if want('ocr') else 'static'


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

    detector = choose_detector(args.detector)
    policy = PressPolicy(args.cooldown, args.retry_after, args.max_retries,
                         args.max_per_minute)
    static = StaticDetector(args.dwell, args.diff_threshold)
    win = find_window(args.process)
    if not win:
        sys.exit(f'no visible window belonging to a process named {args.process!r}')
    hwnd, rect, pid = win
    print(f'REK window {hwnd} pid {pid} at {rect}')
    print(f'detector={detector} (asked for {args.detector})  '
          f'dry_run={args.dry_run}  log={log_path}')
    if detector == 'static':
        print(f'watching for the window to stay unchanged for {args.dwell}s. '
              f'A fight moves; the lobby does not.')

    pressed = 0
    previous = None
    with log_path.open('a', encoding='utf-8') as log:
        while True:
            now = time.time()
            image = None
            try:
                win = find_window(args.process) or win
                hwnd, rect, pid = win
                if detector == 'static':
                    sample = grab_sample(hwnd)
                    diff = frame_diff(sample, previous)
                    previous = sample
                    visible = static.update(now, diff)
                    evidence = {'frame_diff': round(diff, 3),
                                'still_for': round(now - (static.last_change or now), 2)}
                else:
                    image = grab(rect)
                    if detector == 'template':
                        score = template_score(image, args.template)
                        visible = score >= args.template_threshold
                        evidence = {'template_score': round(score, 4)}
                    else:
                        text = read_text(image, args.ocr_scale)
                        if text is None:
                            sys.exit('pytesseract is not importable. Use '
                                     '--detector static, which needs nothing.')
                        visible, scores = detect_prompt(text, threshold=args.threshold)
                        evidence = {'scores': {k: round(v, 3)
                                               for k, v in scores.items()}}
            except Exception as e:
                json.dump({'t': now, 'event': 'capture_error',
                           'error': f'{type(e).__name__}: {e}'}, log)
                log.write('\n')
                log.flush()
                time.sleep(args.interval)
                continue

            should, reason = policy.decide(now, visible)
            fg = is_foreground(hwnd)
            if should and not fg and not args.allow_background:
                should, reason = False, 'REK is not the foreground window'

            entry = {'t': now, 'detector': detector, 'prompt_visible': visible,
                     'reason': reason, 'foreground': fg, **evidence}

            if should:
                if args.dry_run:
                    entry['event'] = 'would_press'
                else:
                    entry['event'] = 'press'
                    entry['method'] = press_space(
                        hwnd, args.allow_background and not fg)
                    policy.record_press(now)
                    pressed += 1
                if shots and image is not None and pressed <= args.keep_shots:
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
    ap.add_argument('--detector', default='auto',
        choices=('auto', 'static', 'ocr', 'template'),
        help='auto uses ocr when pytesseract works and static otherwise, so '
             'this runs with nothing installed')
    ap.add_argument('--dwell', type=float, default=4.0,
        help='seconds the window must stay unchanged for --detector static')
    ap.add_argument('--diff-threshold', type=float, default=2.0,
        help='mean pixel change below which a frame counts as unchanged')
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
