"""Rule-based phonetic keys shared by Hangul transliterations and Latin names.

Both sides are reduced to a consonant skeleton over a small alphabet
(K N T R M P S J C); H is dropped on both sides so that 모닝랜턴 and MORNING LANTERN land close:
    모닝랜턴        -> MNRNTN
    MORNING LANTERN -> MNRNTN
Vowels are dropped because Korean loanword vowels map to English letters
too loosely; consonants carry most of the identity.
"""

from __future__ import annotations

import re
import unicodedata

# Hangul syllable decomposition (Unicode block AC00-D7A3).
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG_COUNT = 21
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
         "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_CHO_KEY = {
    "ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㄴ": "N", "ㄷ": "T", "ㄸ": "T", "ㅌ": "T",
    "ㄹ": "R", "ㅁ": "M", "ㅂ": "P", "ㅃ": "P", "ㅍ": "P", "ㅅ": "S", "ㅆ": "S",
    "ㅇ": "", "ㅈ": "J", "ㅉ": "J", "ㅊ": "C", "ㅎ": "",
}
_JONG_KEY = {
    "": "", "ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㄳ": "K", "ㄺ": "K", "ㄴ": "N", "ㄵ": "N",
    "ㄶ": "N", "ㄷ": "T", "ㅌ": "T", "ㄹ": "R", "ㄼ": "R", "ㄽ": "R", "ㄾ": "R", "ㅀ": "R",
    "ㅁ": "M", "ㄻ": "M", "ㅂ": "P", "ㅍ": "P", "ㄿ": "P", "ㅄ": "P", "ㅅ": "S", "ㅆ": "S",
    "ㅇ": "N", "ㅈ": "J", "ㅊ": "C", "ㅎ": "",
}

_HO_SUFFIX = re.compile(r"호$")
_JE_PREFIX = re.compile(r"^제(?=\d)")


def _collapse(key: str) -> str:
    out = []
    for ch in key:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def hangul_word_key(word: str) -> str:
    out = []
    for ch in word:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            cho, rest = divmod(code, _JUNG_COUNT * 28)
            jong = rest % 28
            out.append(_CHO_KEY[_CHO[cho]])
            out.append(_JONG_KEY[_JONG[jong]])
        elif ch.isascii() and ch.isalpha():
            out.append(latin_word_key(ch))
    return _collapse("".join(out))


# Ordered Latin rewrite rules approximating how Korean renders English sounds.
_LATIN_RULES = [
    (r"^WR", "R"), (r"^PS", "S"), (r"CH(?=R|EM)", "K"),
    (r"TCH", "#"), (r"DGE", "J"), (r"SCH", "S"), (r"CH", "#"), (r"SH", "S"),
    (r"PH", "P"), (r"TH", "S"), (r"GH", ""), (r"CK", "K"), (r"QU", "K"),
    (r"NG", "N"), (r"X", "KS"), (r"C(?=[EIY])", "S"), (r"G(?=[EIY])", "J"),
    (r"(?<=[AEIOU])R(?![AEIOUY])", ""),   # non-rhotic: Morning -> 모닝
    (r"H", ""),
]
_LATIN_MAP = str.maketrans({
    "C": "K", "Q": "K", "G": "K", "D": "T", "B": "P", "F": "P", "V": "P", "L": "R",
    "Z": "J", "W": "", "Y": "", "A": "", "E": "", "I": "", "O": "", "U": "",
})


def latin_word_key(word: str) -> str:
    w = "".join(ch for ch in word.upper() if "A" <= ch <= "Z")
    for pattern, repl in _LATIN_RULES:
        w = re.sub(pattern, repl, w)
    # CH/TCH became "#" so the C->K map leaves the affricate as C.
    w = w.translate(_LATIN_MAP).replace("#", "C")
    return _collapse(w)


# How Korean spells Latin letter names, for acronyms such as HC -> 에이치씨.
_LETTER_NAMES = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ["에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠",
     "엔", "오", "피", "큐", "알", "에스", "티", "유", "브이", "더블유", "엑스", "와이", "지"],
))


def _words(name: str) -> list[str]:
    name = unicodedata.normalize("NFKC", name)
    return [w for w in re.split(r"[^0-9A-Za-z가-힣]+", name) if w]


# Standalone number words (never the first word of a name): AB Marine II <-> 에이비 마린 투.
_NUMBER_WORDS = {
    **{w: str(i) for i, w in enumerate(
        ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]) if w},
    **{w: str(i) for i, w in enumerate(
        ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]) if w},
    **{w: str(i) for i, ws in enumerate(
        [[], ["원"], ["투"], ["쓰리", "스리"], ["포"], ["파이브"], ["식스"], ["세븐"], ["에잇", "에이트"],
         ["나인"], ["텐"]]) for w in ws},
}


def name_digits(name: object) -> str:
    """Digits of a name, with standalone number words and Roman numerals read as digits."""
    if not isinstance(name, str):
        return ""
    out = []
    for i, w in enumerate(_words(name)):
        if i > 0 and w.upper() in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w.upper()])
        else:
            out.extend(re.findall(r"\d+", w))
    return "".join(out)


# C5 (reports/diag/action_plan.md): multi-syllable English number words glued to the end of a Hangul word
# (에이스나인 -> 9, 아볼로에이트호 -> 8). Single-syllable 원/투/포/텐 are common name endings (해원, 대원) and
# are never split off; Sino-Korean 일/이/삼/사/오/구 and spelled letters (이 = E, 오 = O) are never read.
_GLUED_NUMBER = {"쓰리": "3", "스리": "3", "파이브": "5", "식스": "6", "세븐": "7", "에잇": "8", "에이트": "8",
                 "나인": "9"}
_GLUED_NUMBER_RE = re.compile(r"^(.{2,}?)(" + "|".join(sorted(_GLUED_NUMBER, key=len, reverse=True)) + r")호?$")
_HANGUL_NUMBER_WORDS = {w for w in _NUMBER_WORDS if any("가" <= ch <= "힣" for ch in w)}


def name_digits_ext(name: object, mv_fix: bool = False, num_words: bool = False, num_forms: bool = False) -> str:
    """name_digits with the S4 options (all off -> exactly name_digits).

    mv_fix:    'M/V X', 'M.V. X', 'M V X': the V of the vessel prefix is not the Roman numeral 5.
    num_words: C5 restricted reader. Besides name_digits' standalone words, read (a) a standalone
               Hangul number word with a trailing 호 (투호 -> 2, 파이브호 -> 5; never the first word) and
               (b) a multi-syllable English number word glued to the end of a Hangul word (_GLUED_NUMBER).
               Only Hangul text is affected, so Latin catalog names keep their digits.
    num_forms: S5c number-form reader (name_digit_groups_forms with NUM_FORM_PARTS), HANGUL names only; names
               without a Hangul syllable keep the mv_fix / num_words reading. For Hangul names it REPLACES
               num_words: the standalone 투호 reading (part nw_a) is in NUM_FORM_PARTS, but num_words' glued
               쓰리..나인 reading (part nw_b, 에이스나인 -> 9) is not, so num_forms=True with num_words=True reads
               에이스나인 as ''.
    """
    if num_forms and isinstance(name, str) and any("가" <= ch <= "힣" for ch in name):
        return "".join(name_digit_groups_forms(name))
    if not (mv_fix or num_words):
        return name_digits(name)
    if not isinstance(name, str):
        return ""
    words = _words(name)
    skip = 1 if mv_fix and len(words) >= 2 and words[0].upper() == "M" and words[1].upper() == "V" else -1
    out = []
    for i, w in enumerate(words):
        if i == skip:
            continue
        if i > 0 and w.upper() in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w.upper()])
            continue
        digits = re.findall(r"\d+", w)
        if digits or not num_words:
            out.extend(digits)
            continue
        if i > 0 and len(w) >= 2 and w.endswith("호") and w[:-1] in _HANGUL_NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w[:-1]])
            continue
        m = _GLUED_NUMBER_RE.match(w)
        if m and any("가" <= ch <= "힣" for ch in m.group(1)):
            out.append(_GLUED_NUMBER[m.group(2)])
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# S5c num_forms (experiments/s5_robust/s5c/REPORT.md): number forms of HANGUL record names, read only as a standalone
# (non-first) token, at the end of the LAST word, or right before a 호 -- never inside a word. Every form is a named
# part; NUM_FORM_PARTS (the flag) holds only the parts that change no digit_state on the 12,222 train rows nor on the
# v5 synthetic rows the shipped ranker was fit with, and that rarely misfire on en2ko readings of the catalog
# (experiments/s5_robust/s5c/s5c_audit_digit_state.py). Spelled letters (아이아이 for II) are never read.
_ENG_KO_NUMBER = {"원": "1", "투": "2", "쓰리": "3", "스리": "3", "포": "4", "파이브": "5", "식스": "6", "세븐": "7",
                  "에잇": "8", "에이트": "8", "나인": "9", "텐": "10", "일레븐": "11", "트웰브": "12"}
_ENG_KO_ORDER = sorted(_ENG_KO_NUMBER, key=len, reverse=True)
_ENG_11_12 = {"일레븐", "트웰브"}
_SINO_RE = re.compile(r"^(?:([이삼사오육칠팔구])?(천))?(?:([이삼사오육칠팔구])?(백))?(?:([이삼사오육칠팔구])?(십))?"
                      r"([일이삼사오육칠팔구])?$")
_SINO_DIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_SINGLE = ("일", "이", "삼", "사", "오", "육", "칠", "팔", "구", "십")
# S5c fix (experiments/s5_robust/gpu/GATES.md, 2026-10-18), conservative Sino-Korean reading, always right before 호:
#  - a multi-syllable Sino numeral is read only as a two-digit number 11..99 written with 십 (삼십오 = 35, 십일 = 11);
#    numerals with 백 / 천 are never read: their short forms are common words and place names (천사 'angel' -> 1004,
#    천일, 사천, 오천, 이천, 이백, 백사, 백일) and no train record writes a hull number in Sino reading;
#  - glued to a name, the numeral is the LONGEST Sino suffix of the word (삼십오호 = 35, never 십오 = 15); if that
#    longest numeral is not readable, nothing is read (no fallback to a shorter suffix);
#  - a glued multi-syllable numeral needs >= 2 Hangul syllables before it and must start with 삼/육/칠/팔/구 (the
#    syllables the audit trusts as a glued boundary): 이/사/오/십 are common name endings (하이, 라사, 칼리오,
#    프렌드십), so 하이십호 (= 하이 10 or 하 20) is ambiguous and not read.
_SINO_TWO_DIGIT = re.compile(r"^[이삼사오육칠팔구]?십[일이삼사오육칠팔구]?$")
_SINO_GLUED_HEAD = frozenset("삼육칠팔구")
_ROMAN_RE = re.compile(r"^(X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VAL = {"I": 1, "V": 5, "X": 10}
# All parts (for the audit). Glued forms need >= 2 Hangul syllables before a one-syllable number (옥포호, 해원).
NUM_FORM_ALL_PARTS = frozenset(
    {"nw_a", "nw_b", "eng_ho", "eng_11_12", "eng_glued_11_12", "eng_glued_multi", "eng_glued_1syl",
     "roman_standalone", "roman_glued", "sino_ho_multi"}
    | {f"sino_ho_1:{s}" for s in _SINO_SINGLE} | {f"sino_glued_1:{s}" for s in _SINO_SINGLE})
# The flag's parts, selected by experiments/s5_robust/s5c/s5c_audit_digit_state.py with the rule fixed before any stress
# evaluation: a part changes 0 digit_state pairs on train + the v5 synthetic sets, and misfires on <= 100 of the
# 949,408 en2ko catalog reading forms (as read / no spaces / + 호 / no spaces + 호). Left out: glued 원/투/포/텐 and
# glued 쓰리..나인 (en2ko readings: 사투, 키스텐, 아볼로에이트), Sino 이/일/사/오 glued and 이 standalone (하이호, 게일호),
# Sino 십 glued (프렌드십호).
NUM_FORM_PARTS = frozenset({"nw_a", "eng_ho", "eng_11_12", "eng_glued_11_12", "roman_standalone", "roman_glued",
                            "sino_ho_multi"}
                           | {f"sino_ho_1:{s}" for s in ("일", "삼", "사", "오", "육", "칠", "팔", "구", "십")}
                           | {f"sino_glued_1:{s}" for s in ("삼", "육", "칠", "팔", "구")})


def _sino_value(s: str) -> str | None:
    """Canonical Sino-Korean numeral 1..9999 (삼 = 3, 십일 = 11, 삼십오 = 35, 사백이십일 = 421, 육천일 = 6001)."""
    m = _SINO_RE.match(s) if s else None
    if not m:
        return None
    v = 0
    for d, unit, scale in ((1, 2, 1000), (3, 4, 100), (5, 6, 10)):
        if m.group(unit):
            v += _SINO_DIGIT[m.group(d)] * scale if m.group(d) else scale
    if m.group(7):
        v += _SINO_DIGIT[m.group(7)]
    return str(v) if v else None


def _roman_value(s: str) -> str | None:
    """Upper-case Roman numeral 1..39 (II = 2, XII = 12)."""
    if not s or not _ROMAN_RE.match(s):
        return None
    vals = [_ROMAN_VAL[ch] for ch in s]
    v = sum(-x if i + 1 < len(vals) and x < vals[i + 1] else x for i, x in enumerate(vals))
    return str(v) if v else None


def _is_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _glued_prefix_ok(prefix: str, number_len: int) -> bool:
    """Text before a glued number: ends in a Hangul syllable; >= 2 syllables before a one-syllable number."""
    if not prefix or not _is_syllable(prefix[-1]):
        return False
    return number_len > 1 or sum(_is_syllable(ch) for ch in prefix) >= 2


def _sino_ok(num: str, parts, kind: str) -> bool:
    """Is this Sino-Korean numeral string enabled? kind = 'ho' (standalone + 호) or 'glued' (glued + 호).
    Multi-syllable numerals: two-digit 11..99 only (see _SINO_TWO_DIGIT)."""
    if len(num) > 1:
        return "sino_ho_multi" in parts and bool(_SINO_TWO_DIGIT.match(num))
    return f"sino_{kind}_1:{num}" in parts


def _glued_sino(core: str, parts) -> str | None:
    """Sino numeral glued to the end of ``core`` (the last word without its 호), maximal munch (see the S5c fix
    comment above): the whole word (삼십오 -> 35, multi-syllable only), else the longest Sino suffix with >= 2
    syllables before it; a multi-syllable suffix must start with 삼/육/칠/팔/구."""
    j = next((j for j in range(len(core)) if _sino_value(core[j:])), None)
    if j is None:
        return None
    num, prefix = core[j:], core[:j]
    if not prefix:
        return _sino_value(num) if len(num) > 1 and _sino_ok(num, parts, "glued") else None
    if len(num) > 1 and num[0] not in _SINO_GLUED_HEAD:
        return None
    if not _glued_prefix_ok(prefix, 1):  # number_len 1: ends in a syllable and >= 2 syllables before the numeral
        return None
    return _sino_value(num) if _sino_ok(num, parts, "glued") else None


def _standalone_form(w: str, parts) -> str | None:
    """A non-first word that is a whole number: 일레븐, 투호 / 일레븐호, XII / II호, 이호 / 삼십오호."""
    core, ho = (w[:-1], True) if len(w) >= 2 and w.endswith("호") else (w, False)
    if core in _ENG_KO_NUMBER and ((ho and ("eng_ho" in parts or ("nw_a" in parts and core in _HANGUL_NUMBER_WORDS)))
                                   or (not ho and core in _ENG_11_12 and "eng_11_12" in parts)):
        return _ENG_KO_NUMBER[core]
    if "roman_standalone" in parts and (v := _roman_value(core)):
        return v
    if ho and (v := _sino_value(core)) and _sino_ok(core, parts, "ho"):
        return v
    return None


def _glued_form(w: str, parts) -> str | None:
    """A number glued to the end of the last word: 에이스일레븐, 하나투(호), 하나II(호), 금양삼호, 하나삼십오호."""
    core, ho = (w[:-1], True) if len(w) >= 2 and w.endswith("호") else (w, False)
    for nw in _ENG_KO_ORDER:
        if core.endswith(nw) and _glued_prefix_ok(core[: -len(nw)], len(nw)):
            part = ("eng_glued_11_12" if nw in _ENG_11_12 else "eng_glued_1syl" if len(nw) == 1
                    else "eng_glued_multi")
            if part in parts:
                return _ENG_KO_NUMBER[nw]
            break
    if "roman_glued" in parts:
        m = re.match(r"^(.*?)([IVX]+)$", core)
        if m and _glued_prefix_ok(m.group(1), 2) and (v := _roman_value(m.group(2))):
            return v
    if ho:
        return _glued_sino(core, parts)
    return None


def name_digit_groups_forms(name: object, parts=NUM_FORM_PARTS) -> list[str]:
    """Digit groups of a HANGUL record name under the S5c num_forms reader (see the block comment above).

    Per word: name_digits' standalone number words (non-first) and Arabic digit runs (제7, 3호, 6001) as before;
    else (nw_b) num_words' glued multi-syllable reading; else a standalone number form (non-first word); else, for
    the LAST word only, a glued number form. ``parts`` (subset of NUM_FORM_ALL_PARTS) exists for the S5c audit."""
    if not isinstance(name, str):
        return []
    words = _words(name)
    out: list[str] = []
    for i, w in enumerate(words):
        if i > 0 and w.upper() in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w.upper()])
            continue
        digits = re.findall(r"\d+", w)
        if digits:
            out.extend(digits)
            continue
        v = _standalone_form(w, parts) if i > 0 else None
        if v is None and "nw_b" in parts:
            m = _GLUED_NUMBER_RE.match(w)
            if m and any("가" <= ch <= "힣" for ch in m.group(1)):
                v = _GLUED_NUMBER[m.group(2)]
        if v is None and i == len(words) - 1:
            v = _glued_form(w, parts)
        if v is not None:
            out.append(v)
    return out


def strip_leading_zeros(digits_groups: list[str]) -> str:
    """'02' == '2' for the num_forms digit-match state (groups joined after stripping; '0' stays '0')."""
    return "".join(g.lstrip("0") or "0" for g in digits_groups)


def name_digit_groups(name: object, mv_fix: bool = False, num_words: bool = False) -> list[str]:
    """The digit groups behind name_digits_ext(name, mv_fix, num_words) (Latin catalog names: \\d+ runs and
    standalone number words), for strip_leading_zeros."""
    if not isinstance(name, str):
        return []
    words = _words(name)
    skip = 1 if mv_fix and len(words) >= 2 and words[0].upper() == "M" and words[1].upper() == "V" else -1
    out = []
    for i, w in enumerate(words):
        if i == skip:
            continue
        if i > 0 and w.upper() in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w.upper()])
            continue
        digits = re.findall(r"\d+", w)
        if digits or not num_words:
            out.extend(digits)
            continue
        if i > 0 and len(w) >= 2 and w.endswith("호") and w[:-1] in _HANGUL_NUMBER_WORDS:
            out.append(_NUMBER_WORDS[w[:-1]])
            continue
        m = _GLUED_NUMBER_RE.match(w)
        if m and any("가" <= ch <= "힣" for ch in m.group(1)):
            out.append(_GLUED_NUMBER[m.group(2)])
    return out


def _word_variants(word: str, is_last_hangul: bool) -> list[str]:
    if any("가" <= ch <= "힣" for ch in word):
        if is_last_hangul and len(word) >= 2:
            word = _HO_SUFFIX.sub("", word)
        return [hangul_word_key(word)]
    if word.upper() == "NO":
        return [""]
    variants = [latin_word_key(word)]
    # Short all-caps tokens are usually read letter by letter (SITC -> 에스아이티씨).
    if word.isupper() and len(word) <= 4:
        spelled = hangul_word_key("".join(_LETTER_NAMES[ch] for ch in word))
        if spelled not in variants:
            variants.append(spelled)
    return variants


def phonetic_keys(name: object, max_variants: int = 8) -> list[str]:
    """Whole-name skeleton variants; word boundaries removed (Korean spacing is unreliable)."""
    if not isinstance(name, str):
        return [""]
    words = [re.sub(r"\d+", "", _JE_PREFIX.sub("", w)) for w in _words(name)]
    words = [w for w in words if w]
    hangul_idx = [i for i, w in enumerate(words) if any("가" <= ch <= "힣" for ch in w)]
    last_hangul = hangul_idx[-1] if hangul_idx else -1
    keys = [""]
    for i, w in enumerate(words):
        keys = [k + v for k in keys for v in _word_variants(w, i == last_hangul)][:max_variants]
    return list(dict.fromkeys(_collapse(k) for k in keys))


def to_jamo(text: str) -> str:
    """Decompose Hangul syllables into compatibility-free conjoining jamo."""
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(chr(0x1100 + code // 588))
            out.append(chr(0x1161 + (code % 588) // 28))
            if code % 28:
                out.append(chr(0x11A7 + code % 28))
        else:
            out.append(ch)
    return "".join(out)


def hangul_core(name: object) -> str:
    """Hangul record name without digits, 제N prefix, trailing 호 and spacing."""
    if not isinstance(name, str):
        return ""
    words = [re.sub(r"\d+", "", _JE_PREFIX.sub("", w)) for w in _words(name)]
    words = [w for w in words if w]
    if words and any("가" <= ch <= "힣" for ch in words[-1]) and len(words[-1]) >= 2:
        words[-1] = _HO_SUFFIX.sub("", words[-1])
    return "".join(words)


def phonetic_key(name: object) -> str:
    return phonetic_keys(name)[0]


# ---------------------------------------------------------------------------
# S9 hangul_num_v2 (rank.candidates option, default off): port of the experiments/diag/hangul_fragility/patches.py
# "digits" + "numcore" prototype (reports/diag/hangul_fragility.md sec. 4). Nothing above calls these.
# ---------------------------------------------------------------------------
_V2_EXTRA_WORDS = {
    **{w: str(i) for i, w in enumerate(
        ["XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"], start=12)},  # XI = pinyin
    **{w: str(i) for i, w in enumerate(
        ["ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN",
         "NINETEEN", "TWENTY"], start=11)},
    "일레븐": "11", "트웰브": "12",
}
_V2_WORDS = {**_NUMBER_WORDS, **_V2_EXTRA_WORDS}
_V2_GLUED = {**_GLUED_NUMBER, "일레븐": "11", "트웰브": "12"}
_V2_GLUED_RE = re.compile(r"^(.{2,}?)(" + "|".join(sorted(_V2_GLUED, key=len, reverse=True)) + r")호?$")
_V2_SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_V2_SINO_RE = re.compile(r"^(?:([이삼사오육칠팔구])?(십))?([일이삼사오육칠팔구])?$")
_V2_HANGUL_NUM = {w for w in _V2_WORDS if any("가" <= ch <= "힣" for ch in w)}


def _sino_value(word: str) -> str | None:
    m = _V2_SINO_RE.match(word)
    if not word or not m:
        return None
    tens = (_V2_SINO[m.group(1)] if m.group(1) else 1) if m.group(2) else 0
    ones = _V2_SINO[m.group(3)] if m.group(3) else 0
    v = tens * 10 + ones
    return str(v) if v else None


def name_digits_v2(name: object) -> str:
    """name_digits + standalone Sino-Korean numerals (i>0), glued multi-syllable English number words, number
    word + 호, ELEVEN..TWENTY / 일레븐 / 트웰브, Roman XII-XX (XI stays pinyin)."""
    if not isinstance(name, str):
        return ""
    out = []
    for i, w in enumerate(_words(name)):
        u = w.upper()
        core = w[:-1] if len(w) >= 2 and w.endswith("호") and not re.search(r"\d", w) else w
        if i > 0 and (u in _V2_WORDS or core in _V2_WORDS):
            out.append(_V2_WORDS.get(u, _V2_WORDS.get(core)))
            continue
        if i > 0 and (v := _sino_value(core)) is not None:
            out.append(v)
            continue
        digits = re.findall(r"\d+", w)
        if digits:
            out.extend(digits)
            continue
        m = _V2_GLUED_RE.match(w)
        if m and any("가" <= ch <= "힣" for ch in m.group(1)):
            out.append(_V2_GLUED[m.group(2)])
    return "".join(out)


def strip_number_words(name: object) -> object:
    """Hangul name without its Hangul number words (i>0: 투, 나인호, 십일 ...; glued 에이스나인 -> 에이스), so the
    phonetic keys / transliteration core compare the name part only. Unchanged when nothing is stripped."""
    if not isinstance(name, str):
        return name
    out = []
    for i, w in enumerate(_words(name)):
        core = w[:-1] if len(w) >= 2 and w.endswith("호") else w
        if i > 0 and any("가" <= ch <= "힣" for ch in w) and (core in _V2_HANGUL_NUM or _sino_value(core)):
            continue
        m = _V2_GLUED_RE.match(w)
        if m and any("가" <= ch <= "힣" for ch in m.group(1)):
            w = m.group(1)
        out.append(w)
    return " ".join(out) if out else name
