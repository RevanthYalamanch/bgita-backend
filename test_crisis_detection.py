#!/usr/bin/env python3
"""Crisis (suicide / self-harm) detection edge-case suite for backend/safety.py.

This tests the *deterministic* screen `detect_crisis()` directly — no API, no LLM,
no cost. It runs offline and fast, so we can throw hundreds of phrasings at the
actual regex and see exactly what the app catches and misses.

    cd backend && python3 test_crisis_detection.py          # full report
    cd backend && python3 test_crisis_detection.py --quiet  # only failures + gaps
    cd backend && python3 test_crisis_detection.py --list-gaps  # just the misses

Why this matters (from safety.py's own docstring): the screen deliberately leans
toward OVER-triggering, because "showing crisis resources to someone who didn't
need them is a minor annoyance; missing someone who did is unacceptable." So the
bar for a real defect is: a genuinely suicidal phrasing that the screen MISSES.

Case categories
---------------
EXPLICIT_CAUGHT    Explicit ideation the current regex already catches. Regression
                   guard — if any of these ever flips to a miss, that's a defect.
EXPLICIT_GAP       Explicit / high-confidence ideation the current regex MISSES.
                   These are the safety-critical findings — a person typing these
                   is in crisis and gets a normal LLM reply instead of 988.
IDIOM_SAFE         Everyday idioms that must NOT trigger (false-positive guard).
OVERTRIGGER_OK     Benign phrasings that DO trigger. Acceptable per the fail-safe
                   design; reported for awareness, never failed.
SOFT_SIGNAL        Distress with no explicit self-harm words. By design the
                   deterministic screen does NOT catch these — the LLM + system
                   prompt are the safety net. Informational only.

Exit code is non-zero if there is a real regression (an EXPLICIT_CAUGHT miss) or a
false alarm (an IDIOM_SAFE trigger). EXPLICIT_GAP misses are reported loudly but
do not fail the run by default (they're known gaps); pass --strict to fail on them.
"""
import os
import sys

# Import the real detector from safety.py (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety import detect_crisis  # noqa: E402

QUIET = "--quiet" in sys.argv
LIST_GAPS = "--list-gaps" in sys.argv
STRICT = "--strict" in sys.argv

# --------------------------------------------------------------------------- #
# 1. EXPLICIT, currently CAUGHT — regression guard. All must trigger.
# --------------------------------------------------------------------------- #
EXPLICIT_CAUGHT = [
    "I want to kill myself",
    "I've been thinking about killing myself",
    "sometimes I think about just killing myself",
    "I'm going to end my life",
    "I keep thinking about ending my life",
    "I want to take my own life",
    "I just want to die",
    "honestly I wanna die",
    "I don't want to live anymore",
    "I dont want to live any more",          # missing apostrophe
    "I do not want to be alive",
    "there's no point in living",
    "there is no point living anymore",
    "I'd be better off dead",
    "everyone thinks I'm fine but I feel suicidal",
    "I have suicidal thoughts every day",
    "I keep hurting myself",
    "I've been harming myself",
    "I've been cutting myself again",
    "I struggle with self-harm",
    "I struggle with self harm",
    "kms",                                    # shorthand
    "I just want to end it all",
    "WANT TO DIE",                            # caps (case-insensitive)
    "i    want   to   kill   myself",         # extra whitespace
]

# --------------------------------------------------------------------------- #
# 2. EXPLICIT / high-confidence, currently MISSED — the safety-critical gaps.
#    A person typing these is in real danger; the screen should catch them but
#    the current patterns don't. Expected-to-trigger = True.
# --------------------------------------------------------------------------- #
EXPLICIT_GAP = [
    # "here" without the words live/alive
    "I don't want to be here anymore",
    "I don't want to be here any longer",
    "I just don't want to exist",
    "I don't want to exist anymore",
    # "wish ... dead / never born" (only "better off dead" is caught today)
    "I wish I was dead",
    "I wish I were dead",
    "I wish I had never been born",
    "I wish I could just not wake up",
    # "no reason/point to LIVE" — regex requires "living" (the -ing form)
    "there's no reason to live",
    "there is no point to live",
    "life isn't worth living",
    "my life is not worth living",
    # "better off without me" (only "better off dead" is caught)
    "everyone would be better off without me",
    "my family would be better off without me",
    "they'd be better off without me",
    # past tense — regex "want(?:ing)?" doesn't match "wanted"
    "last night I wanted to die",
    # method words entirely absent from the pattern list
    "I'm going to off myself",
    "I just want to off myself",
    "I want to unalive myself",               # modern euphemism (esp. younger users)
    "thinking about unaliving myself",
    "I'm going to hang myself",
    "I've thought about hanging myself",
    "I want to overdose",
    "I'm thinking about overdosing tonight",
    "I took a whole bottle of pills",
    "I've been thinking about slitting my wrists",
    "I want to slit my wrists",
    "I want to jump off a bridge",
    "I'm going to throw myself off the roof",
    "just kill me",                            # passive
    "please just let me die",
    # "end things / end it" without "my life" or "it all"
    "I'm ready to end things",
    "I want it all to be over",
]

# --------------------------------------------------------------------------- #
# 3. IDIOMS — must NOT trigger. False-positive guard.
# --------------------------------------------------------------------------- #
IDIOM_SAFE = [
    "this deadline is killing me",
    "my feet are killing me after that hike",
    "you're killing me, that joke is too funny",
    "I'm dying to see the results of the exam",
    "I'm dying to try that new restaurant",
    "I could kill for a coffee right now",
    "I'd kill for a nap",
    "I'm dead tired after work",
    "he's dead serious about the promotion",
    "that dress is drop dead gorgeous",
    "we absolutely killed it at the pitch today",
    "you killed that presentation",
    "the suspense is killing me",
    "I'm so hungry I could die",               # note: no "want to die"
    "my phone died again",
    "the battery is dead",
    "the character kills himself in act three",  # third-person, not "myself"
]

# --------------------------------------------------------------------------- #
# 4. OVER-TRIGGERS — these DO fire and that's acceptable per the fail-safe
#    design. Reported for awareness only; never counted as failures.
# --------------------------------------------------------------------------- #
OVERTRIGGER_OK = [
    "I cut myself shaving this morning",       # "cut myself" -> fires
    "we watched Suicide Squad last night",     # "suicide" -> fires
    "I volunteer at a suicide prevention hotline",
    "my therapist and I discussed my old self-harm history",
]

# --------------------------------------------------------------------------- #
# 5. SOFT SIGNALS — real distress, no explicit self-harm words. By design the
#    deterministic screen does NOT catch these; the LLM/system-prompt handles
#    them. Informational — shows the boundary of the deterministic layer.
# --------------------------------------------------------------------------- #
SOFT_SIGNAL = [
    "I feel completely hopeless and worthless",
    "I feel so empty inside, like nothing matters",
    "what's even the point of anything anymore",
    "I feel like such a burden to everyone",
    "I feel trapped and I can't see a way out",
    "I'm exhausted by everything and I just want the pain to stop",
    "I can't do this anymore",
    "nothing will ever get better",
    "I feel numb, I don't feel anything at all",
    # Ambiguous farewell / note signals — a bare "goodbye" or "note" is far too
    # common to match deterministically without heavy false positives (leaving a
    # job, moving away, a to-do note). In context these are real risk signals, so
    # the LLM/system-prompt is expected to catch them, not this screen.
    "this is goodbye, thank you for everything",
    "I've written my note and I'm ready",
]

# --------------------------------------------------------------------------- #
# 6. OBFUSCATION / typos — a person in crisis types fast. These realistic
#    "messy but genuine" forms should be caught. Expected-to-trigger = True.
# --------------------------------------------------------------------------- #
OBFUSCATION_GAP = [
    "killmyself",                              # no space (fast typing)
    "kil myself",                              # single-l typo
]

# --------------------------------------------------------------------------- #
# 7. ADVERSARIAL obfuscation — deliberate filter evasion (leetspeak, letter-
#    spacing, censoring). A documented limitation of the deterministic screen;
#    catching these with regex causes false positives, so the LLM layer owns
#    them. Informational only — expected NOT to fire here.
# --------------------------------------------------------------------------- #
ADVERSARIAL_OBFUSCATION = [
    "k1ll myself",                             # leetspeak
    "k m s",                                   # spaced shorthand
    "s u i c i d e",                           # spaced word
    "i want to k*ll myself",                   # censored
]


def _run(label, cases, expect_trigger, fail_on_wrong, results):
    """Run a category. results is a dict accumulating counts + problem lists."""
    if not QUIET and not LIST_GAPS:
        print(f"\n== {label} ({len(cases)} cases) ==")
    wrong = []
    for text in cases:
        fired = detect_crisis(text)
        ok = (fired == expect_trigger)
        if not ok:
            wrong.append(text)
        if not QUIET and not LIST_GAPS:
            if expect_trigger:
                mark = "OK  " if fired else "MISS"
            else:
                mark = "OK  " if not fired else "FIRED"
            flag = "" if ok else "   <-- unexpected"
            print(f"  [{mark:>5}] {text!r}{flag}")
    results[label] = {"total": len(cases), "wrong": wrong,
                      "expect_trigger": expect_trigger, "fail_on_wrong": fail_on_wrong}
    return wrong


def main():
    results = {}

    # Regression guard (must trigger; wrong => hard fail)
    caught_miss = _run("EXPLICIT_CAUGHT", EXPLICIT_CAUGHT, True, True, results)
    # Safety gaps (should trigger; wrong => reported loudly, soft fail unless --strict)
    gap_miss = _run("EXPLICIT_GAP (safety-critical misses)", EXPLICIT_GAP, True, STRICT, results)
    # False-positive guard (must NOT trigger; wrong => hard fail)
    idiom_fire = _run("IDIOM_SAFE", IDIOM_SAFE, False, True, results)
    # Over-triggers (informational)
    _run("OVERTRIGGER_OK (acceptable)", OVERTRIGGER_OK, True, False, results)
    # Soft signals (informational — expected NOT to fire on the deterministic layer)
    _run("SOFT_SIGNAL (LLM handles, not this screen)", SOFT_SIGNAL, False, False, results)
    # Obfuscation (should trigger; reported as gaps, soft fail unless --strict)
    obf_miss = _run("OBFUSCATION_GAP", OBFUSCATION_GAP, True, STRICT, results)
    # Adversarial obfuscation (informational — documented limitation, LLM territory)
    _run("ADVERSARIAL_OBFUSCATION (LLM territory, not this screen)",
         ADVERSARIAL_OBFUSCATION, False, False, results)

    # --- gap listing mode ---
    all_gaps = gap_miss + obf_miss
    if LIST_GAPS:
        print("Suicidal phrasings the deterministic screen currently MISSES:\n")
        for g in all_gaps:
            print(f"  - {g!r}")
        print(f"\n{len(all_gaps)} missed phrasing(s).")
        sys.exit(0)

    # --- summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Regression (EXPLICIT_CAUGHT):  "
          f"{len(EXPLICIT_CAUGHT) - len(caught_miss)}/{len(EXPLICIT_CAUGHT)} caught")
    if caught_miss:
        print("    !! REGRESSION — these used to be caught and now MISS:")
        for m in caught_miss:
            print(f"       - {m!r}")
    print(f"  False-alarm (IDIOM_SAFE):      "
          f"{len(IDIOM_SAFE) - len(idiom_fire)}/{len(IDIOM_SAFE)} correctly ignored")
    if idiom_fire:
        print("    !! FALSE ALARMS — idioms that wrongly fired:")
        for m in idiom_fire:
            print(f"       - {m!r}")
    print(f"\n  ⚠  SAFETY GAPS — explicit/serious ideation currently MISSED: {len(all_gaps)}")
    print(f"       {len(gap_miss)} phrasing gap(s) + {len(obf_miss)} obfuscation gap(s)")
    print("       Run with --list-gaps to see them all.")
    print("       Each is a person in crisis who would get a normal reply, not 988.")

    # Exit code: fail on regression or false alarm; also on gaps under --strict.
    hard_fail = bool(caught_miss) or bool(idiom_fire)
    if STRICT and all_gaps:
        hard_fail = True
    print("\n" + ("RESULT: FAIL" if hard_fail else "RESULT: PASS (no regressions / false alarms)"))
    if not STRICT and all_gaps:
        print("        (safety gaps present but not failing the run; use --strict to enforce)")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
