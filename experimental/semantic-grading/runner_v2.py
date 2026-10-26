"""probe runner v2 - hardened recorder. Rules learned from probe15:
- abort the whole run on the FIRST non-zero returncode (infra failure, never a retry loop)
- retain FULL stderr per unit (no truncation)
- classify load crashes as infra:load_crash, NOT model_invalid:parse
- raw stdout preserved unchanged alongside the adapter-trimmed copy
- per-unit record: rc, latency, ru_maxrss delta (cumulative child max - labeled as such)
No model logic here; BIN is injected. Tested with fake binaries only."""
import json, subprocess, sys, time, resource

def run_units(bin_path, model_path, units, prompt_fn, grade_fn, out_path, ctx=2048, n_max=220, threads=2, vlim_kb=None, log=print):
    """units: list of dicts with keys k, src, lang, doc, cand_start, cand_end.
    prompt_fn(unit)->str ; grade_fn(trimmed, unit)->dict. Aborts on first rc!=0."""
    def lim():
        if vlim_kb: resource.setrlimit(resource.RLIMIT_AS, (vlim_kb*1024, vlim_kb*1024))
    outf = open(out_path, "a")
    for u in units:
        t0 = time.time()
        try:
            r = subprocess.run([bin_path, "-m", model_path, "-c", str(ctx), "-n", str(n_max), "-t", str(threads),
                                "--no-display-prompt", "-p", prompt_fn(u)],
                               capture_output=True, text=True, timeout=110, preexec_fn=lim)
            lat, out, err, rc = time.time()-t0, r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            lat, out, err, rc = time.time()-t0, "", "TIMEOUT", -9
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        marker = "\n> EOF by user"
        trimmed = out[:out.index(marker)] if marker in out else out
        rec = {"k": u["k"], "src": u["src"], "lang": u["lang"], "lat_s": round(lat,2), "rc": rc,
               "peak_rss_kb_cumulative_childmax": rss,
               "raw_out": out, "adapter_trimmed": trimmed, "stderr_full": err}
        if rc != 0:
            rec["classification"] = "infra:load_crash" if not out.strip() else "infra:runtime_error"
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush()
            log("ABORT on first infra failure at unit", u["k"], "rc=", rc)
            outf.close()
            return {"status": "aborted_infra", "at_unit": u["k"], "rc": rc}
        g = grade_fn(trimmed, u)
        rec.update({"decision": g["decision"], "reason": g.get("reason"), "classification": "graded"})
        outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush()
        log(u["k"], u["src"], round(lat,1), g["decision"], g.get("reason") or "")
    outf.close()
    return {"status": "completed", "units": len(units)}
