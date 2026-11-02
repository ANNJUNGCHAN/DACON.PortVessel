"""Rule-based Latin -> Hangul transliteration for synthetic training pairs.

Two readings are produced per word:
  - loanword reading (외래어 표기 style): Glovis -> 글로비스, Morning -> 모닝
  - Korean romanization read back:       Daesan -> 대산, Keoyoung -> 거영
Ambiguous English choices (a: 아/애, u: 우/어, post-vocalic r: dropped/르,
initial s: ㅅ/ㅆ, ...) yield a small set of variants. Uses only the catalog
spelling itself, no external data.
"""

from __future__ import annotations

import itertools
import re

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

W_GLIDE = {"ㅏ": "ㅘ", "ㅓ": "ㅝ", "ㅔ": "ㅞ", "ㅣ": "ㅟ", "ㅐ": "ㅙ", "ㅗ": "ㅝ", "ㅜ": "ㅜ"}
Y_GLIDE = {"ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ", "ㅔ": "ㅖ", "ㅐ": "ㅒ", "ㅣ": "ㅣ"}
# Epenthetic vowel after an unsyllabified consonant (default ㅡ).
EPENTHESIS = {"ㅈ": "ㅣ", "ㅊ": "ㅣ"}


def compose(cho: str, jung: str, jong: str = "") -> str:
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


# ---------------------------------------------------------------------------
# Loanword reading.
# Tokens: ("C", sym) consonant, ("V", jamo, long) vowel, ("W",)/("Y",) glides.
# Consonant symbols are onset jamo plus F, V (never codas), L, R, NG, SH.
# ---------------------------------------------------------------------------

VOWELS = set("aeiouAEIOU@")  # "@" = unstressed initial a read 어 (Alliance 얼라이언스)
# Long-vowel placeholders written by _preprocess: A=에이 I=아이 O=오 U=유 E=이
LONG = {"A": [("ㅔ", True), ("ㅣ", True)], "I": [("ㅏ", True), ("ㅣ", True)],
        "O": [("ㅗ", True)], "U": [("Y",), ("ㅜ", True)], "E": [("ㅣ", True)]}
SHORT = {"a": "ㅏ", "e": "ㅔ", "i": "ㅣ", "o": "ㅗ", "u": "ㅜ"}
CONS = {"b": "ㅂ", "d": "ㄷ", "f": "F", "j": "ㅈ", "k": "ㅋ", "l": "L", "m": "ㅁ",
        "n": "ㄴ", "p": "ㅍ", "q": "ㅋ", "r": "R", "t": "ㅌ", "v": "V", "z": "ㅈ"}
ONSET = {"F": "ㅍ", "V": "ㅂ", "R": "ㄹ", "L": "ㄹ", "SH": "ㅅ", "NG": "ㅇ"}


def _preprocess(w: str, opt: dict) -> str:
    """Spelling-level rewrites: silent e, long vowels, -le endings."""
    if len(w) > 5:
        w = re.sub(r"(?<=[il])ance$", "@ns", w)                           # Alliance 얼라이언스, Balance
    w = re.sub(r"cean", "sh@n", w)                                       # Ocean 오션
    w = re.sub(r"(e[ea]|(?<=[^aeiou])io)r$", "Eer", w)                   # Pioneer, Warrior
    w = re.sub(r"(?<![aeiou])([aiou])re$", lambda m: m.group(1).upper() + "er", w)  # Sapphire
    w = re.sub(r"^([^aeiou]*)i(?=[ao])", r"\1I", w)                      # Pioneer, Giant, Diamond
    w = re.sub(r"^([^aeiouy]+)y$", r"\1I", w)                            # Sky, My
    w = re.sub(r"^u(?=[bcdfgklmnpstvz][aeiou])", "U", w)                 # Unique, Union
    if len(w) > 4:
        w = re.sub(r"(?<=[^aeiou])ea$", "ia", w)                         # Korea, Azalea (not Sea)
    if len(w) > 3:
        w = re.sub(r"(?<=[^aeiou])ey$", "y", w)                          # Journey
    if len(w) >= 3:
        # Magic e: single vowel + single consonant + final e (Ace, Cape, Rose, Smile, Wave).
        m = re.search(r"(?<![aeiouAEIOU])([aeiou])([bcdfgklmnpstvz])e$", w)
        if m:
            cons = {"c": "s", "g": "j"}.get(m.group(2), m.group(2))
            w = w[: m.start()] + m.group(1).upper() + cons
        # Consonant + le: Maple, Eagle, Humble (a single consonant lengthens the vowel).
        w = re.sub(r"(?<![aeiouAEIOU])([aiou])([bcdfgkpstvz])le$",
                   lambda m: m.group(1).upper() + m.group(2) + "l", w)
        w = re.sub(r"([^aeiouAEIOU])le$", r"\1l", w)
        if opt["final_e"]:
            w = re.sub(r"ce$", "s", w)
            w = re.sub(r"ge$", "j", w)
            w = re.sub(r"(?<=[^aeiouAEIOU])e$", "", w)
    if opt["a_eo"]:
        w = re.sub(r"^a(?=(ll|dm|cc|pp|tt|ss|rr|bb|mm|nn))", "@", w)
    if opt["i_ai_mid"]:
        w = re.sub(r"(?<=[^aeiouAEIOU])i(?=[a@])", "I", w)               # Alliance, Reliance
    if opt["open_long"]:
        # Open-syllable a/i read long: Navi 네이비, Safety 세이프티, Tiger 타이거.
        w = re.sub(r"(?<![aeiouAEIOU])([ai])(?=[bcdfgkmnpstvz][aeiouy])",
                   lambda m: m.group(1).upper(), w, count=1)
    return w


def _loanword_tokens(w: str, opt: dict) -> list[tuple]:
    w = _preprocess(w, opt)
    toks: list[tuple] = []
    i, n = 0, len(w)

    def V(j, long=False):
        return ("V", j, long)

    while i < n:
        rest = w[i:]
        nxt = w[i + 1] if i + 1 < n else ""
        prev = w[i - 1] if i else ""
        soft = nxt != "" and nxt in "eiyEIY"
        k, items = 1, []
        if rest[0] == "@":
            items = [V("ㅓ")]
        elif rest[0] in LONG:
            items = [V(*t) if len(t) == 2 else t for t in LONG[rest[0]]]
        elif rest.startswith("igh"):
            k, items = 3, [V("ㅏ", True), V("ㅣ", True)]
        elif rest.startswith(("tion", "ssion")):
            k = 5 if rest.startswith("ssion") else 4
            items = [("C", "SH"), V("ㅓ"), ("C", "ㄴ")]
        elif rest.startswith("sion"):
            k, items = 4, [("C", "ㅈ"), V("ㅓ"), ("C", "ㄴ")]
        elif rest.startswith(("ai", "ay")):
            k = 2
            items = [V("ㅔ", True), V("ㅣ", True)] if opt["ai_ei"] else [V("ㅏ", True), V("ㅣ", True)]
        elif rest.startswith(("ea", "ee", "ie")):
            k, items = 2, [V("ㅣ", True)]
        elif rest.startswith(("ei", "ey")):
            k, items = 2, [V("ㅔ", True), V("ㅣ", True)]
        elif rest.startswith("oo"):
            k, items = 2, [V("ㅜ", True)]
        elif rest.startswith("our"):
            k, items = 3, [V("ㅓ"), ("C", "R")]
        elif rest.startswith("ou"):
            k, items = 2, [V("ㅏ", True), V("ㅜ", True)]
        elif rest.startswith("ow"):
            k = 2
            items = [V("ㅗ", True), V("ㅜ", True)] if i + 2 >= n else [V("ㅏ", True), V("ㅜ", True)]
        elif rest.startswith("oa"):
            k, items = 2, [V("ㅗ", True)]
        elif rest.startswith(("oi", "oy")):
            k, items = 2, [V("ㅗ", True), V("ㅣ", True)]
        elif rest.startswith(("au", "aw")):
            k, items = 2, [V("ㅗ", True)]
        elif rest.startswith(("eu", "ew")):
            k, items = 2, [("Y",), V("ㅜ", True)]
        elif rest.startswith(("er", "ir", "ur")) and not (i + 2 < n and w[i + 2] in VOWELS):
            k, items = 2, [V("ㅓ"), ("C", "R")]
        elif rest[0] == "y":
            items = [("Y",)] if (nxt and nxt in VOWELS and not prev) else [V("ㅣ")]
        elif rest.startswith(("wh", "w")):
            k = 2 if rest.startswith("wh") else 1
            nx = w[i + k] if i + k < n else ""
            if nx and nx in VOWELS:
                items = [("W",)]
            elif not (prev and prev in VOWELS):
                items = [V("ㅜ")]
        elif rest[0] in SHORT:
            ch = rest[0]
            after = w[i + 1: i + 3]
            closed = (len(after) >= 1 and after[0] not in VOWELS and after[0] not in "rwy"
                      and (len(after) == 1 or after[1] not in VOWELS))
            if ch == "a":
                items = [V("ㅐ" if closed and opt["a_ae"] else "ㅏ")]
            elif ch == "u":
                items = [V("ㅓ" if closed and opt["u_eo"] else "ㅜ")]
            else:
                items = [V(SHORT[ch])]
        elif rest.startswith(("tch", "ch")):
            k, items = (3 if rest.startswith("tch") else 2), [("C", "ㅊ")]
        elif rest.startswith("sh"):
            k, items = 2, [("C", "SH")]
        elif rest.startswith("th"):
            k, items = 2, [("C", "ㅅ")]
        elif rest.startswith(("pph", "ph")):
            k, items = (3 if rest.startswith("pph") else 2), [("C", "F")]
        elif rest.startswith("ck"):
            k, items = 2, [("C", "ㅋ")]
        elif rest.startswith("qu"):
            k, items = 2, [("C", "ㅋ"), ("W",)]
        elif rest.startswith("dge"):
            k, items = 3, [("C", "ㅈ")]
        elif rest.startswith("gg") and i + 2 < n and w[i + 2] in "eiyEIY":
            k, items = 2, [("C", "ㅈ")]                                     # Melaggio 멜라지오
        elif rest.startswith("cc") and i + 2 < n and w[i + 2] in "eiyEIY":
            k, items = 2, [("C", "ㄱ"), ("C", "ㅅ")]
        elif rest.startswith("ng"):
            k, items = 2, [("C", "NG")]
        elif rest.startswith("gh"):
            k = 2
        elif rest[0] == "x":
            items = [("C", "ㅅ")] if not prev else [("C", "ㄱ"), ("C", "ㅅ")]  # Xin 신, Max 맥스
        elif rest[0] == "c":
            items = [("C", "ㅅ" if soft else "ㅋ")]
        elif rest[0] == "g":
            # Soft before i/y; before e only after a/n (Angela, Voyager) - Tiger, Rigel stay hard.
            g_soft = (nxt != "" and nxt in "iyIY") or (nxt == "e" and prev in ("a", "n"))
            # Toggle flips only the ambiguous cases: soft g read hard, or hard ge read soft.
            soft = g_soft if opt["g_soft"] else (not g_soft and nxt == "e")
            items = [("C", "ㅈ" if soft else "ㄱ")]
        elif rest[0] == "s":
            tense = not prev and nxt and nxt in VOWELS and opt["s_tense"]
            items = [("C", "ㅆ" if tense else "ㅅ")]
        elif rest[0] == "h":
            if (not prev or prev in VOWELS) and nxt and nxt in VOWELS:
                items = [("C", "ㅎ")]
        elif rest[0] == "z" and opt["z_s"]:
            items = [("C", "ㅅ")]                                         # Luzon 루손
        elif rest[0] in CONS:
            items = [("C", CONS[rest[0]])]
        # Doubled consonants collapse, except n/m/l which split across syllables.
        if (len(items) == 1 and items[0][0] == "C" and toks and toks[-1] == items[0]
                and items[0][1] not in ("ㄴ", "ㅁ", "L")):
            items = []
        toks.extend(items)
        i += k
    return toks


def _compose_loanword(toks: list[tuple], opt: dict) -> str:
    out: list[list] = []  # [cho, jung, jong, long]
    i, n = 0, len(toks)

    def vowel_at(j):
        """(jamo, long, next index) for a vowel at j, optionally glide-prefixed."""
        glide = None
        if j < n and toks[j][0] in ("W", "Y"):
            glide, j = toks[j][0], j + 1
        if j < n and toks[j][0] == "V":
            v, long = toks[j][1], toks[j][2]
            if glide:
                v = (W_GLIDE if glide == "W" else Y_GLIDE).get(v, v)
            return v, long, j + 1
        return None

    while i < n:
        t = toks[i]
        if t[0] != "C":
            hit = vowel_at(i)
            if hit:
                out.append(["ㅇ", hit[0], "", hit[1]])
                i = hit[2]
            else:  # dangling glide
                out.append(["ㅇ", "ㅜ" if t[0] == "W" else "ㅣ", "", False])
                i += 1
            continue
        c = t[1]
        hit = vowel_at(i + 1)
        if hit:
            v, long, nxt_i = hit
            if c == "SH":
                v = Y_GLIDE.get(v, v)
            if c == "L" and out and not out[-1][2]:
                out[-1][2] = "ㄹ"
            if c == "NG" and out and not out[-1][2]:
                out[-1][2] = "ㅇ"
            out.append([ONSET.get(c, c), v, "", long])
            i = nxt_i
            continue
        nt = toks[i + 1] if i + 1 < n else None
        final = nt is None
        prev_open = bool(out) and not out[-1][2]
        if c == "R":
            if not opt["r_drop"]:
                out.append(["ㄹ", "ㅡ", "", False])
        elif c in ("L", "NG", "ㄴ", "ㅁ") and prev_open:
            out[-1][2] = {"L": "ㄹ", "NG": "ㅇ"}.get(c, c)
        elif (c in ("ㅋ", "ㅍ", "ㄱ", "ㅂ") and prev_open and not out[-1][3] and opt["stop_coda"]
              and (final or (nt[0] == "C" and nt[1] in ("ㅌ", "ㅅ", "ㅊ")))):
            # Stop as coda after a short vowel: Pacific 퍼시픽, Captain 캡틴.
            out[-1][2] = "ㄱ" if c in ("ㅋ", "ㄱ") else "ㅂ"
        elif c == "SH":
            out.append(["ㅅ", "ㅣ", "", False])
        else:
            out.append([ONSET.get(c, c), EPENTHESIS.get(c, "ㅡ"), "", False])
        i += 1
    return "".join(compose(*s[:3]) for s in out)


# ---------------------------------------------------------------------------
# Korean romanization (RR / McCune-Reischauer-ish) read back.
# ---------------------------------------------------------------------------
_KO_VOWELS = [("yeo", "ㅕ"), ("yae", "ㅒ"), ("wae", "ㅙ"), ("woo", "ㅜ"), ("ae", "ㅐ"), ("eo", "ㅓ"),
              ("eu", "ㅡ"), ("oe", "ㅚ"), ("ui", "ㅢ"), ("ya", "ㅑ"), ("yo", "ㅛ"), ("yu", "ㅠ"),
              ("ye", "ㅖ"), ("wa", "ㅘ"), ("wo", "ㅝ"), ("we", "ㅞ"), ("wi", "ㅟ"), ("oo", "ㅜ"),
              ("ou", "ㅗ"), ("a", "ㅏ"), ("e", "ㅔ"), ("i", "ㅣ"), ("o", "ㅗ"), ("u", "ㅜ")]
_KO_CONS = [("ng", "ㅇ"), ("kk", "ㄲ"), ("tt", "ㄸ"), ("pp", "ㅃ"), ("ss", "ㅆ"), ("jj", "ㅉ"),
            ("ch", "ㅊ"), ("sh", "ㅅ"), ("ck", "ㄱ"), ("g", "ㄱ"), ("k", "ㄱ"), ("n", "ㄴ"),
            ("d", "ㄷ"), ("t", "ㄷ"), ("r", "ㄹ"), ("l", "ㄹ"), ("m", "ㅁ"), ("b", "ㅂ"),
            ("p", "ㅂ"), ("s", "ㅅ"), ("j", "ㅈ"), ("h", "ㅎ")]
_KO_CODA = {"ㅇ": "ㅇ", "ㄴ": "ㄴ", "ㄹ": "ㄹ", "ㅁ": "ㅁ", "ㄱ": "ㄱ", "ㅂ": "ㅂ", "ㄷ": "ㅅ", "ㅅ": "ㅅ"}


def korean_romanization(w: str, split_ng: bool = False) -> str | None:
    """Read a Korean romanization back; split_ng reads ng+vowel as n|g (Singan 신간)."""
    toks: list[tuple[str, str]] = []
    i = 0
    while i < len(w):
        for table, kind in ((_KO_VOWELS, "V"), (_KO_CONS, "C")):
            hit = next(((s, j) for s, j in table if w.startswith(s, i)
                        and not (split_ng and s == "ng" and i + 2 < len(w) and w[i + 2] in "aeiouy")), None)
            if hit:
                toks.append((kind, hit[1]))
                i += len(hit[0])
                break
        else:
            if w[i] in "-'":
                i += 1
                continue
            return None
    out: list[list[str]] = []
    k = 0
    while k < len(toks):
        kind, j = toks[k]
        if kind == "V":
            out.append(["ㅇ", j, ""])
            k += 1
        elif k + 1 < len(toks) and toks[k + 1][0] == "V" and j != "ㅇ":
            out.append([j, toks[k + 1][1], ""])
            k += 2
        elif out and not out[-1][2] and j in _KO_CODA:
            out[-1][2] = _KO_CODA[j]
            k += 1
        else:
            return None  # consonant cluster: not a Korean romanization
    return "".join(compose(*s) for s in out) if out else None


_KO_HINT = re.compile(r"(eo|ae|eu(?!r)|(eong|ung|ong)$|^(kim|lee|park|han|dae|dong|hyun|seong|sung"
                      r"|jin|hae|kwang|heung|hyundai|sam|woo))")


def _looks_korean(w: str) -> bool:
    return bool(_KO_HINT.search(w))


# ---------------------------------------------------------------------------
LETTER_NAMES = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    ["에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠",
     "엔", "오", "피", "큐", "알", "에스", "티", "유", "브이", "더블유", "엑스", "와이", "지"]))

DEFAULT = dict(a_ae=False, u_eo=True, r_drop=True, s_tense=False, g_soft=True, ai_ei=True,
               stop_coda=True, final_e=True, open_long=False, z_s=False, a_eo=False, i_ai_mid=False)
TOGGLES = ("a_ae", "open_long", "r_drop", "u_eo", "z_s", "a_eo", "i_ai_mid", "s_tense", "final_e",
           "g_soft", "ai_ei")


def loanword(w: str, **flips) -> str:
    opt = dict(DEFAULT, **flips)
    return _compose_loanword(_loanword_tokens(w, opt), opt)


# ---------------------------------------------------------------------------
# Chinese pinyin reading (중국어 표기 style, with common loose variants).
# ---------------------------------------------------------------------------
_PY_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
                "j", "q", "x", "r", "z", "c", "s", "y", "w", ""]
_PY_ONSET = {"zh": "ㅈ", "ch": "ㅊ", "sh": "ㅅ", "b": "ㅂ", "p": "ㅍ", "m": "ㅁ", "f": "ㅍ", "d": "ㄷ",
             "t": "ㅌ", "n": "ㄴ", "l": "ㄹ", "g": "ㄱ", "k": "ㅋ", "h": "ㅎ", "j": "ㅈ", "q": "ㅊ",
             "x": "ㅅ", "r": "ㄹ", "z": "ㅈ", "c": "ㅊ", "s": "ㅅ", "y": "ㅇ", "w": "ㅇ", "": "ㅇ"}
# final -> list of (vowel, coda) syllables; the first takes the initial.
_PY_FINALS = {
    "iang": [("ㅑ", "ㅇ")], "iong": [("ㅠ", "ㅇ")], "uang": [("ㅘ", "ㅇ")], "ueng": [("ㅝ", "ㅇ")],
    "ang": [("ㅏ", "ㅇ")], "eng": [("ㅓ", "ㅇ")], "ing": [("ㅣ", "ㅇ")], "ong": [("ㅗ", "ㅇ")],
    "ian": [("ㅖ", "ㄴ")], "iao": [("ㅑ", ""), ("ㅗ", "")], "uai": [("ㅘ", ""), ("ㅣ", "")],
    "uan": [("ㅘ", "ㄴ")], "ai": [("ㅏ", ""), ("ㅣ", "")], "ao": [("ㅏ", ""), ("ㅗ", "")],
    "an": [("ㅏ", "ㄴ")], "ei": [("ㅔ", ""), ("ㅣ", "")], "en": [("ㅓ", "ㄴ")], "er": [("ㅓ", "ㄹ")],
    "ia": [("ㅑ", "")], "ie": [("ㅖ", "")], "in": [("ㅣ", "ㄴ")], "iu": [("ㅠ", "")],
    "ou": [("ㅓ", ""), ("ㅜ", "")], "ua": [("ㅘ", "")], "ui": [("ㅜ", ""), ("ㅣ", "")],
    "un": [("ㅜ", "ㄴ")], "uo": [("ㅝ", "")], "a": [("ㅏ", "")], "e": [("ㅓ", "")],
    "i": [("ㅣ", "")], "o": [("ㅗ", "")], "u": [("ㅜ", "")], "v": [("ㅟ", "")],
}
_PY_FINAL_KEYS = sorted(_PY_FINALS, key=len, reverse=True)
# Strong pinyin evidence only; plain English words (Reliance, Ocean) must not qualify.
_PY_MARKERS = re.compile(r"(x[iu]|q[iu]|zh|iu$|uo|ao|iang|iong|(eng|ong|ang)$|^(feng|hui|fu|hai|shan|dong"
                         r"|jiang|xin|zhen|ning|bao|yang|tian|hua|guo|ming|cheng|wei|jin|lian|xiang)$|[tjqxnlbmd]ian)")
_UNGLIDE = {"ㅑ": "ㅏ", "ㅕ": "ㅓ", "ㅛ": "ㅗ", "ㅠ": "ㅜ", "ㅖ": "ㅔ"}


def _pinyin_syllables(w: str) -> list[tuple[str, str]] | None:
    """Segment a word into (initial, final) pinyin syllables, fewest syllables first."""
    best: dict[int, list] = {0: []}
    for i in range(len(w)):
        if i not in best:
            continue
        for ini in _PY_INITIALS:
            if not w.startswith(ini, i):
                continue
            for fin in _PY_FINAL_KEYS:
                j = i + len(ini) + len(fin)
                if w.startswith(fin, i + len(ini)) and (j not in best or len(best[j]) > len(best[i]) + 1):
                    best[j] = best[i] + [(ini, fin)]
    return best.get(len(w))


def pinyin_reading(w: str, english_eng: bool = False) -> str | None:
    syl = _pinyin_syllables(w)
    if not syl:
        return None
    out = []
    for ini, fin in syl:
        parts = _PY_FINALS[fin]
        if ini in ("y", "w") and fin in ("i", "u", "in", "ing", "un"):
            pass  # yi, wu, yin: the glide letter is silent
        onset = _PY_ONSET[ini]
        for k, (v, coda) in enumerate(parts):
            if k == 0:
                if ini == "w":
                    v = W_GLIDE.get(v, v)
                elif ini == "y":
                    v = Y_GLIDE.get(v, v)
                if ini in ("j", "q", "zh", "ch", "z", "c"):
                    v = _UNGLIDE.get(v, v)                                 # jiang 장, qian 첸
                if english_eng and fin == "eng":
                    v = "ㅔ"                                               # Feng 펭
                if ini in ("zh", "ch", "sh", "r", "z", "c", "s") and fin == "i":
                    v = "ㅡ"                                               # shi 스, zi 즈
                out.append(compose(onset, v, coda))
            else:
                out.append(compose("ㅇ", v, coda))
    return "".join(out)


def _looks_pinyin(w: str) -> bool:
    return bool(_PY_MARKERS.search(w)) and _pinyin_syllables(w) is not None


def word_variants(word: str, max_variants: int = 8) -> list[str]:
    """Candidate Hangul spellings of one Latin word, most likely first."""
    raw = word
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return [word]
    out: list[str] = []
    acronym = raw.isupper() and len(raw) <= 4
    if acronym:
        out.append("".join(LETTER_NAMES[ch] for ch in w))
    out.append(loanword(w))
    ko = korean_romanization(w)
    if ko:
        out.insert(int(acronym), ko) if _looks_korean(w) else out.append(ko)
    ko_split = korean_romanization(w, split_ng=True)
    if ko_split and ko_split != ko:
        out.append(ko_split)
    if _looks_pinyin(w):
        out[1:1] = [pinyin_reading(w), pinyin_reading(w, english_eng=True)]
    for flip in itertools.chain.from_iterable(itertools.combinations(TOGGLES, r) for r in (1, 2)):
        if len(set(out)) >= max_variants:
            break
        out.append(loanword(w, **{f: not DEFAULT[f] for f in flip}))
    return list(dict.fromkeys(x for x in out if x))[:max_variants]


def name_variants(name: str, max_variants: int = 4) -> list[str]:
    """Whole-name Hangul variants: primary reading, then one word varied at a time."""
    words = [w for w in re.findall(r"[A-Za-z]+|\d+", name) if w.upper() != "NO"]
    per_word = [[w] if w.isdigit() else word_variants(w) for w in words]
    primary = [v[0] for v in per_word]
    results = [" ".join(primary)]
    for idx, vs in enumerate(per_word):
        for alt in vs[1:]:
            results.append(" ".join(primary[:idx] + [alt] + primary[idx + 1:]))
    return list(dict.fromkeys(results))[:max_variants]
