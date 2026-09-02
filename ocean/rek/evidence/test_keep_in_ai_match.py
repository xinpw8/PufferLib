"""Checks for the parts of keep_in_ai_match.py that can be wrong quietly.

The screen capture and the keystroke are Win32 glue and cannot run here. What
can, and what carries the actual risk, is the decision logic: recognising the
lobby prompt through OCR noise without also recognising something else, and
deciding when to press. A false positive sends a space into a live round, which
the recorder will capture as if a pilot meant it.

    python ocean/rek/evidence/test_keep_in_ai_match.py
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('keep', HERE / 'keep_in_ai_match.py')
keep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keep)

checks = []


def check(fn):
    checks.append(fn)
    return fn


@check
def normalise_strips_what_ocr_adds():
    assert keep.normalise('  NO OTHER\n PLAYERS   here! ') == 'no other players here'
    assert keep.normalise('space  to|fight_ai') == 'space to fight ai'
    assert keep.normalise('') == ''


@check
def the_prompt_is_recognised_through_ocr_noise():
    # Real OCR output for light text over a dark 3D scene: swapped letters,
    # dropped spacing, stray punctuation, inconsistent case.
    for text in (
            'no other players here\nspace to fight ai',
            'N0 OTHER PLAYERS HERE\nSPACE T0 FIGHT AI',
            'no other pIayers here',
            'no othor players hero',
            'space to fight ai.',
            'nootherplayers here  space to fight ai',
            'PRIVATE AREA\n\nno other players here\n\nspace to fight ai\n\n[esc] menu',
    ):
        visible, scores = keep.detect_prompt(text)
        assert visible, (text, scores)


@check
def other_screens_do_not_trigger_it():
    # A false positive puts a space into a live round. These are the strings
    # most likely to be on screen when that would matter.
    for text in (
            '',
            'round 2 of 3',
            'sparring bot 1',
            'KO',
            'you win',
            'press escape to leave the arena',
            'connecting to server',
            'waiting for other players',
            'select your robot  l100  h100',
            'fight',
            'space',
    ):
        visible, scores = keep.detect_prompt(text)
        assert not visible, (text, scores)


@check
def one_recovered_line_is_enough():
    # The two lines render separately and OCR often mangles one of them.
    visible, scores = keep.detect_prompt('n0 0th3r pI4y3rs h3r3\nspace to fight ai')
    assert visible and scores['space to fight ai'] > scores['no other players here']


@check
def it_presses_once_when_the_prompt_appears():
    p = keep.PressPolicy()
    assert p.decide(100.0, False) == (False, 'prompt not visible')

    should, why = p.decide(101.0, True)
    assert should and why == 'prompt appeared'
    p.record_press(101.0)

    # Still up a moment later: do not hammer it.
    assert p.decide(102.0, True)[0] is False
    assert p.decide(104.0, True) == (False, 'waiting to see if the press took')


@check
def it_retries_a_bounded_number_of_times_then_stops():
    p = keep.PressPolicy(cooldown_s=1.0, retry_after_s=5.0, max_retries=3)
    t = 0.0
    presses = 0
    for _ in range(200):
        t += 1.0
        should, why = p.decide(t, True)          # prompt never clears
        if should:
            p.record_press(t)
            presses += 1
    assert presses == 3, presses
    assert 'gave up' in p.decide(t + 1, True)[1]


@check
def a_started_match_rearms_it_for_the_next_one():
    p = keep.PressPolicy(cooldown_s=1.0)
    assert p.decide(10.0, True)[0]
    p.record_press(10.0)

    # Prompt clears: the match started.
    assert p.decide(11.0, False)[0] is False
    # Match ends, prompt returns. This is the whole point — one press per match,
    # indefinitely.
    should, why = p.decide(300.0, True)
    assert should and why == 'prompt appeared'


@check
def the_cooldown_holds_across_prompt_flicker():
    # OCR will occasionally miss a frame and report the prompt gone, which would
    # otherwise re-arm the policy and press again immediately.
    p = keep.PressPolicy(cooldown_s=5.0)
    assert p.decide(0.0, True)[0]
    p.record_press(0.0)
    p.decide(1.0, False)                  # dropped frame
    assert p.decide(1.5, True) == (False, 'cooldown')
    assert p.decide(6.0, True)[0] is True


@check
def the_rate_limit_is_a_backstop():
    # If detection goes wrong in a way the other rules do not catch, this caps
    # the damage at something a human will notice rather than a stream of input.
    p = keep.PressPolicy(cooldown_s=0.0, retry_after_s=0.0, max_retries=10**6,
                         max_per_minute=5)
    t, presses = 0.0, 0
    for _ in range(100):
        t += 0.5
        if p.decide(t, True)[0]:
            p.record_press(t)
            presses += 1
    assert presses == 5, presses
    assert p.decide(t, True)[1] == 'rate limited'
    # It recovers once the window rolls over.
    assert p.decide(t + 61, True)[0] is True


def main() -> int:
    failed = 0
    for fn in checks:
        try:
            fn()
            print(f'  ok    {fn.__name__.replace("_", " ")}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__.replace("_", " ")}: {e}')
        except Exception as e:
            failed += 1
            print(f'  ERROR {fn.__name__.replace("_", " ")}: {type(e).__name__}: {e}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
