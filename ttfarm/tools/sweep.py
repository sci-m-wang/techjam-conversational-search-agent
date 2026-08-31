"""Config sweeps on the TUNE split only (holdout stays untouched)."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run(env_extra, split="tune"):
    env = dict(os.environ, **{k: json.dumps(v) for k, v in env_extra.items()})
    out = subprocess.run([sys.executable, "-m", "ttfarm.tools.run_eval", "--split", split, "--quiet"],
                         capture_output=True, text=True, cwd=ROOT, env=env)
    d = json.loads(out.stdout)
    return d["recommended_technical_score"], d["hit_rate_at_10"], d["mrr"], d["mttc"]

mode = sys.argv[1]
if mode == "k":
    results = []
    for b1 in (0, 1, 3, 5, 10):
        for br1 in (0, 1, 3, 5, 10):
            kp = {"buying": {1: b1}, "browsing": {1: br1}, "override_likely": {}}
            ts, hr, mrr, mttc = run({"COPILOT_K_POLICY": kp})
            results.append((ts, hr, mrr, mttc, kp))
            print(f"buy_t1={b1:>2} browse_t1={br1:>2}  TS={ts:.4f} HR={hr:.3f} MRR={mrr:.4f} MTTC={mttc:.2f}")
    results.sort(key=lambda r: -r[0])
    print("\nTOP 5:")
    for ts, hr, mrr, mttc, kp in results[:5]:
        print(f"  TS={ts:.4f}  {json.dumps(kp)}")
elif mode == "k2":
    base = json.loads(sys.argv[2])
    for b2 in (10, 5, 3, 2, 1):
        for br2 in (10, 5, 3, 2, 1):
            kp = {"buying": {**base["buying"], "2": b2},
                  "browsing": {**base["browsing"], "2": br2}, "override_likely": {}}
            ts, hr, mrr, mttc = run({"COPILOT_K_POLICY": kp})
            print(f"buy_t2={b2:>2} browse_t2={br2:>2}  TS={ts:.4f} HR={hr:.3f} MRR={mrr:.4f} MTTC={mttc:.2f}")
elif mode == "w":
    base = {"phrase": 4.0, "full": 2.0, "cover": 1.5, "pop": 1.0, "price": 0.5}
    for key, values in [("pop", (0.3, 0.6, 1.0, 1.5, 2.5)),
                        ("cover", (0.75, 1.5, 3.0)),
                        ("phrase", (2.0, 4.0, 8.0)),
                        ("price", (0.0, 0.5, 1.5))]:
        for v in values:
            w = dict(base); w[key] = v
            ts, hr, mrr, mttc = run({"COPILOT_WEIGHTS": w})
            print(f"{key}={v:<5} TS={ts:.4f} HR={hr:.3f} MRR={mrr:.4f} MTTC={mttc:.2f}")
