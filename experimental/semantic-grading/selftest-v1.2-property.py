"""v1.2 assurance suite: property tests for span offsets + fuzz crash-safety for
validate_record. Deterministic, no model, no pools. Run: python3 selftest-v1.2-property.py"""
import json, os, random, string, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h

fails = []
def check(name, cond, detail=""):
    if not cond: fails.append((name, detail))
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")

# --- property 1: round-trip on every substring window of a mixed AR/EN doc ---
DOC = h.norm("The network ran 40 pages. مصر قالت إنها نفت التقرير المفصل. Citizen Lab said the group used bots in 2024.")
ok = True
for i in range(0, len(DOC), 7):
    sub = DOC[i:i+13]
    if not sub.strip(): continue
    loc = h.locate_span(DOC, sub, len(h.b(DOC[:i])))
    if not isinstance(loc, tuple): ok = False; break
    s, e = loc
    # decode byte offsets back and compare text
    raw = h.b(DOC)
    if raw[s:e].decode("utf-8") != sub: ok = False; break
check("round-trip byte offsets (mixed AR/EN, every 7th window)", ok)

# --- property 2: NFD input equals NFC result ---
nfd = unicodedata.normalize("NFD", "مصر قالت إنها")
loc_nfc = h.locate_span(DOC, nfd, 0)
check("NFD span text resolves (NFC convention)", isinstance(loc_nfc, tuple))

# --- property 3: duplicate text -> nearest occurrence; tie -> ambiguous ---
DUP = "abc X abc Y abc"
near_0 = h.locate_span(DUP, "abc", 0)
r1 = isinstance(near_0, tuple) and near_0[0] == 0
near_end = h.locate_span(DUP, "abc", len(h.b(DUP)))
r2 = isinstance(near_end, tuple) and near_end[0] == len(h.b("abc X abc Y "))
# exact midpoint tie: 'a X a' with near_byte exactly between
TIE = "a...a"
tie = h.locate_span(TIE, "a", 2)
r3 = tie == "ambiguous"
check("duplicate nearest-occurrence", r1 and r2)
check("exact tie -> ambiguous", r3)

# --- property 4: empty/whitespace span never locates ---
check("empty span -> None", h.locate_span(DOC, "", 0) is None)

# --- fuzz: validate_record must never raise ---
rng = random.Random(20261026)
ATOMS = [None, True, False, 0, 1, 1.5, "finding", "x", "", [], {}, ["finding"], {"a":1}, ["x",["y"]]]
def rrec(depth):
    if depth <= 0: return rng.choice(ATOMS)
    t = rng.random()
    if t < 0.4: return {rng.choice(["role","subject_kind","asserted_by","spans","claim","x"]): rrec(depth-1) for _ in range(rng.randint(0,4))}
    if t < 0.7: return [rrec(depth-1) for _ in range(rng.randint(0,4))]
    return rng.choice(ATOMS)
crashes = 0
N = 3000
for _ in range(N):
    rec = rrec(3)
    try:
        v, why, resolved = h.validate_record(rec, DOC, 10)
        assert isinstance(v, bool) and isinstance(why, str) and isinstance(resolved, dict)
    except Exception as e:
        crashes += 1
        if crashes <= 3: print("CRASH on:", repr(rec)[:120], "->", type(e).__name__, e)
check("fuzz validate_record x%d: zero crashes" % N, crashes == 0, "%d crashes" % crashes)

# --- fuzz: grade_unit on random strings must never raise ---
crashes2 = 0
for _ in range(1000):
    s = "".join(rng.choice(string.printable + "مصرقالتإنها") for _ in range(rng.randint(0, 400)))
    try:
        g = h.grade_unit(s, DOC, 0, 10)
        assert g["decision"] in ("abstain","deny","bind")
    except Exception as e:
        crashes2 += 1
        if crashes2 <= 3: print("CRASH grade_unit on:", repr(s[:80]), "->", type(e).__name__, e)
check("fuzz grade_unit x1000: zero crashes", crashes2 == 0, "%d crashes" % crashes2)

print("ASSURANCE SUITE:", "ALL PASS" if not fails else "%d FAILURES" % len(fails))
sys.exit(1 if fails else 0)
