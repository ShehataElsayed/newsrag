"""Adversarial semantic eval: comparator scored from gold facets separately
from model extraction; end-to-end scored last. Raw artifacts always saved.
Usage: python3 run_eval.py [--model PATH --binary PATH --grammar PATH]
Without --model only the comparator-from-gold layer runs (no model needed).
"""
import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from newsrag import semantic as sm

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = json.load(open(os.path.join(HERE, "semantic_adversarial.json"), encoding="utf-8"))

def verdict_ok(case, rel, dim):
    if rel not in case.get("acceptable", [case["expected"]]):
        return False
    if rel == case["expected"] and case.get("expected_dimension") and dim != case["expected_dimension"]:
        return False
    return True

def run_comparator_from_gold():
    rows = []
    for c in CASES["cases"]:
        v = sm.compare_facets(c["gold_claim_facets"], c["gold_evidence_facets"])
        ok = verdict_ok(c, v["relation"], v.get("dimension"))
        rows.append({"id": c["id"], "relation": v["relation"], "dimension": v.get("dimension"),
                     "reason": v["reason"], "expected": c["expected"], "ok": ok})
    return rows

def facet_diff(gold, got):
    """Per-facet extraction scoring against gold. Returns dict of facet->bool/None."""
    if got is None:
        return {k: None for k in ("indicator", "period", "polarity", "numbers", "attribution")}
    return {
        "indicator": sm._indicator(got) == sm._indicator(gold),
        "period": sm._period(got.get("time_period")) == sm._period(gold.get("time_period")),
        "polarity": (got.get("polarity") or "none") == (gold.get("polarity") or "none"),
        "numbers": sm._numbers(gold.get("numbers")) <= sm._numbers(got.get("numbers")),
        "attribution": sm._canon(got.get("attribution"), (sm._ATTR_CBE, sm._ATTR_CAPMAS)) ==
                       sm._canon(gold.get("attribution"), (sm._ATTR_CBE, sm._ATTR_CAPMAS)),
    }

CACHE = os.path.join(HERE, "extraction_cache.json")

def run_with_model(model, binary, grammar, only=None):
    from newsrag.qwen import llamacpp_generate
    gen = llamacpp_generate(model, binary=binary, n_ctx=2048, n_predict=220,
                            threads=2, temperature=0.0, grammar_file=grammar,
                            extra_flags=["--single-turn"])
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    for c in CASES["cases"]:
        if only and c["id"] not in only:
            continue
        t0 = time.time()
        cf = sm.extract_facets(c["claim"], gen)
        sf = sm.extract_facets(c["evidence"], gen)
        cache[c["id"]] = {"claim_facets": cf, "evidence_facets": sf,
                          "seconds": round(time.time() - t0, 1)}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"cached {c['id']} ({cache[c['id']]['seconds']}s)", flush=True)
    extraction_rows, e2e_rows = [], []
    for c in CASES["cases"]:
        if c["id"] not in cache:
            continue
        cf, sf = cache[c["id"]]["claim_facets"], cache[c["id"]]["evidence_facets"]
        elapsed = cache[c["id"]]["seconds"]
        extraction_rows.append({"id": c["id"], "claim_facets": cf, "evidence_facets": sf,
                                "claim_score": facet_diff(c["gold_claim_facets"], cf),
                                "evidence_score": facet_diff(c["gold_evidence_facets"], sf),
                                "seconds": elapsed})
        if cf is None or sf is None:
            e2e_rows.append({"id": c["id"], "relation": "insufficient", "dimension": None,
                             "reason": "extraction failed (no usable JSON); abstained", "ok": c["expected"] == "insufficient"})
            continue
        v = sm.compare_facets(cf, sf)
        e2e_rows.append({"id": c["id"], "relation": v["relation"], "dimension": v.get("dimension"),
                         "reason": v["reason"], "ok": verdict_ok(c, v["relation"], v.get("dimension"))})
    return extraction_rows, e2e_rows

def main():
    args = sys.argv[1:]
    out = {"set_provenance": CASES["provenance"], "layers": {}}
    gold_rows = run_comparator_from_gold()
    out["layers"]["comparator_from_gold"] = {
        "score": f"{sum(r['ok'] for r in gold_rows)}/{len(gold_rows)}", "rows": gold_rows}
    print("comparator-from-gold:", out["layers"]["comparator_from_gold"]["score"])
    for r in gold_rows:
        print(f"  {r['id']}: {r['relation']}/{r['dimension']} ok={r['ok']}")
    if "--model" in args:
        m = args[args.index("--model") + 1]
        b = args[args.index("--binary") + 1]
        g = args[args.index("--grammar") + 1]
        only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
        ex, e2e = run_with_model(m, b, g, only)
        def facet_acc(rows, key):
            vals = [v for r in rows for v in r[key].values() if v is not None]
            return f"{sum(vals)}/{len(vals)}"
        out["layers"]["extraction"] = {
            "claim_facet_accuracy": facet_acc(ex, "claim_score"),
            "evidence_facet_accuracy": facet_acc(ex, "evidence_score"), "rows": ex}
        out["layers"]["end_to_end"] = {
            "score": f"{sum(r['ok'] for r in e2e)}/{len(e2e)}", "rows": e2e}
        print("extraction claim facets:", out["layers"]["extraction"]["claim_facet_accuracy"],
              "| evidence facets:", out["layers"]["extraction"]["evidence_facet_accuracy"])
        print("end-to-end:", out["layers"]["end_to_end"]["score"])
        for r in e2e:
            print(f"  {r['id']}: {r['relation']}/{r.get('dimension')} ok={r['ok']}")
    dest = os.path.join(HERE, "adversarial_results.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("raw artifacts ->", dest)

if __name__ == "__main__":
    main()
