"""Phase-1 semantic grading harness - deterministic core under the frozen contract
(v0.2-final + addendum, bundle sha256 a37095de...). Model side is a stub until
runtime route + labels are authorized. No model calls, no downloads.

Unicode convention v1: NFC normalization, offsets are UTF-8 BYTE offsets into the
normalized document; duplicate span text resolves to the occurrence nearest the
candidate sentence start; ties -> ambiguous (span invalid).
"""
import json, re, unicodedata, hashlib

UCONV = "unicode-v1:nfc+utf8-bytes+nearest-occurrence"
ROLES = {"finding","methodology","background","conditional","caption_description","caption_analysis","furniture","denial","finding_in_methodology","other"}
SUBJECTS = {"analysis_subject","example","quoted_source"}
ASSERTED = {"outlet_analysis","document","quoted_actor","analyst_unnamed"}
FINDING_LABELS = {"finding","caption_analysis","finding_in_methodology"}

def norm(t): return unicodedata.normalize("NFC", t)
def b(s): return s.encode("utf-8")

def locate_span(doc, span_text, near_byte):
    """Return (start,end) byte offsets of span_text in doc (NFC, utf-8), nearest to
    near_byte; 'ambiguous' on tie, None when not found."""
    d, s = norm(doc), norm(span_text)
    if not s: return None
    hits, i = [], d.find(s)
    while i >= 0:
        hits.append((len(b(d[:i])), len(b(d[:i])) + len(b(s))))
        i = d.find(s, i + 1)
    if not hits: return None
    hits.sort(key=lambda se: abs(se[0] - near_byte))
    if len(hits) > 1 and abs(hits[0][0] - near_byte) == abs(hits[1][0] - near_byte):
        return "ambiguous"
    return hits[0]

def verify_span(doc, span_text, near_byte):
    r = locate_span(doc, span_text, near_byte)
    return r if isinstance(r, tuple) else None

# ---------- deterministic context bundle ----------
HEAD = re.compile(r"^(#{1,4})\s+(.*\S)\s*$", re.M)
BLOC_EN = re.compile(r"\b(pro-[A-Za-z][\w-]*|[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+){0,3})\s+(accounts?|network|group|campaign|bots?|pages?)\b")
BLOC_AR = re.compile(r"(حسابات|شبكة|شبكات|صفحات|مجموعة)(?: +[\u0600-\u06FF]{2,}){1,5}")
DATE_R = re.compile(r"\b(20\d\d)(?:[-/](0?[1-9]|1[0-2]))?\b|[\u0600-\u06FF]+\s+20\d\d")

def anchor_pack(doc):
    """Frozen deterministic anchor table v1: title, headings, named account blocs/
    networks (EN+AR), year/date mentions. No model judgment."""
    d = norm(doc)
    pack = {"title": None, "headings": [], "entities": [], "dates": []}
    for m in HEAD.finditer(d):
        txt = m.group(2).strip()
        if pack["title"] is None: pack["title"] = txt
        pack["headings"].append(txt)
    for rx in (BLOC_EN, BLOC_AR):
        for m in rx.finditer(d):
            e = m.group(0).strip()
            if e not in pack["entities"]: pack["entities"].append(e)
    for m in DATE_R.finditer(d):
        if m.group(0) not in pack["dates"]: pack["dates"].append(m.group(0))
    return pack

def context_bundle(doc, cand_start, cand_end, window=2):
    """Heading path + window neighbor sentences + anchor pack. Truncation is the
    CALLER's logged decision against a measured model budget (R6) - never silent."""
    d = norm(doc)
    heads = [m.group(2).strip() for m in HEAD.finditer(d) if len(b(d[:m.start()])) < cand_start]
    sents = re.split(r"(?<=[.!?\u061F])\s+|\n+", d)
    out, pos, ci = [], 0, 0
    for i, s in enumerate(sents):
        sb = len(b(d[:d.find(s, pos)])) if d.find(s, pos) >= 0 else pos
        if sb <= cand_start < sb + len(b(s)): ci = i
        pos = d.find(s, pos) + len(s)
    lo, hi = max(0, ci - window), min(len(sents), ci + window + 1)
    return {"heading_path": heads[-3:], "window": sents[lo:hi], "anchor_pack": anchor_pack(doc)}

# ---------- validity gate (shape only) ----------
def validate_record(rec, doc, near_byte):
    """Return (valid: bool, reason: str, resolved: dict of facet->span offsets)."""
    if not isinstance(rec, dict): return False, "not_object", {}
    roles = rec.get("role")
    if not isinstance(roles, list) or not roles or any(not isinstance(r, str) or r not in ROLES for r in roles):
        return False, "bad_role_set", {}
    sk = rec.get("subject_kind")
    if not isinstance(sk, str) or sk not in SUBJECTS: return False, "bad_subject", {}
    ab = rec.get("asserted_by")
    if ab is not None and (not isinstance(ab, str) or ab not in ASSERTED):
        return False, "bad_asserted_by", {}
    resolved, spans = {}, rec.get("spans") or {}
    if not isinstance(spans, dict): return False, "bad_spans", {}
    for facet, text in spans.items():
        if text is None: continue
        loc = locate_span(doc, str(text), near_byte)
        if not isinstance(loc, tuple): return False, "span_unverified:" + facet, {}
        resolved[facet] = {"text": str(text), "start": loc[0], "end": loc[1]}
    return True, "ok", resolved

# ---------- frozen arbiter (v1.0, pre-measurement freeze) ----------
def arbitrate(rec, resolved, truncated=False):
    """Frozen precedence v1.0. Returns {decision, reason, scope_notes}."""
    if truncated: return {"decision": "abstain", "reason": "truncated"}
    roles = set(rec["role"])
    notes = []
    if rec.get("uncertainty"):
        return {"decision": "abstain", "reason": "grader_uncertain", "scope_notes": notes}
    if rec.get("contradiction"):
        return {"decision": "abstain", "reason": "contradiction_flagged", "scope_notes": notes}
    if "conditional" in roles and (roles & FINDING_LABELS):
        notes.append("conditional_tail_scoped_out")  # C3: conditional dominates only its own scope
    if not (roles & FINDING_LABELS):
        for d in ("denial","methodology","background","caption_description","furniture"):
            if d in roles:
                return {"decision": "deny", "reason": "role:" + d, "scope_notes": notes}
        return {"decision": "deny", "reason": "role:other", "scope_notes": notes}
    if "denial" in roles:  # denial dominates only its own scope; without scope spans, conservative
        if "negation" not in resolved:
            return {"decision": "abstain", "reason": "denial_scope_ungrounded", "scope_notes": notes}
        return {"decision": "deny", "reason": "denial_scoped", "scope_notes": notes}
    ab = rec.get("asserted_by")
    if rec.get("subject_kind") != "analysis_subject":
        return {"decision": "deny", "reason": "subject:" + str(rec.get("subject_kind")), "scope_notes": notes}
    if ab == "analyst_unnamed":
        return {"decision": "abstain", "reason": "analyst_unnamed_uncorroborated", "scope_notes": notes}
    if ab in ("outlet_analysis","document","quoted_actor"):
        need = "attribution" if ab != "outlet_analysis" else None
        if need and need not in resolved:
            return {"decision": "abstain", "reason": "attribution_span_missing", "scope_notes": notes}
        if "claim" not in resolved:  # v1.3: bind requires a non-null VERIFIED claim span
            return {"decision": "abstain", "reason": "claim_span_missing", "scope_notes": notes}
        return {"decision": "bind", "reason": "eligible:" + ab, "scope_notes": notes}
    return {"decision": "abstain", "reason": "asserted_by_missing", "scope_notes": notes}

def grade_unit(model_output, doc, cand_start, cand_end, truncated=False):
    """Full pipeline for one unit: parse -> validity -> arbiter. model_output is the
    RAW untrusted model string (stub source until runtime authorized)."""
    try: rec = json.loads(model_output)
    except Exception: return {"decision": "abstain", "reason": "model_invalid:parse"}
    ok, why, resolved = validate_record(rec, doc, cand_start)
    if not ok: return {"decision": "abstain", "reason": "model_invalid:" + why}
    out = arbitrate(rec, resolved, truncated=truncated)
    out["resolved_spans"] = resolved
    return out

def run_metadata(extra):
    m = {"uconv": UCONV, "arbiter": "v1.3-frozen", "contract_sha256":
         "a37095de65f2aed6c47a11db185546b757dacbcbcd627f22d44e37b57fd83a52"}
    m.update(extra); return m
