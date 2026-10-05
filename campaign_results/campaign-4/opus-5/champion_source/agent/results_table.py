"""Render the official results table from the official_eval.sh output files.

Usage: uv run python agent/results_table.py [champion.json] [baseline.json]
"""
import json, sys
from pathlib import Path

W = Path(__file__).resolve().parent.parent / "miner_outputs"
ORDER = [("crypto-24h", ["BTC", "ETH", "SOL", "XRP", "HYPE"]),
         ("com-equ-24h", ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]),
         ("crypto-1h", ["BTC", "ETH", "SOL", "XRP", "HYPE"])]


def load(name):
    out = {}
    for line in (W / name).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[(d["cfg"], d["asset"])] = d["s"]
    return out


def main():
    champ = sys.argv[1] if len(sys.argv) > 1 else "off_champ.json"
    base = sys.argv[2] if len(sys.argv) > 2 else "off_base.json"
    c, b = load(champ), load(base)
    lines = []
    for cfg, assets in ORDER:
        have = [a for a in assets if (cfg, a) in c and (cfg, a) in b]
        if not have:
            continue
        lines.append("")
        lines.append(f"**{cfg.upper().replace('-', '_')}**")
        lines.append("")
        lines.append("| asset | prompts | champion CRPS | starting point CRPS | change |")
        lines.append("|---|---:|---:|---:|---:|")
        for a in have:
            cc, bb = c[(cfg, a)], b[(cfg, a)]
            lines.append(f"| {a} | {cc['num_prompts']} | {cc['mean_crps']:.1f} | "
                         f"{bb['mean_crps']:.1f} | {100*(cc['mean_crps']/bb['mean_crps']-1):+.2f}% |")
        mc = sum(c[(cfg, a)]["mean_crps"] for a in have) / len(have)
        mb = sum(b[(cfg, a)]["mean_crps"] for a in have) / len(have)
        lines.append(f"| **mean** | | **{mc:.1f}** | **{mb:.1f}** | **{100*(mc/mb-1):+.2f}%** |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
