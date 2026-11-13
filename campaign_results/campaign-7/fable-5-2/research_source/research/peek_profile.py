import sys, json, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
d = sys.argv[1]
for a in sys.argv[2].split(","):
    p = np.array(json.load(open(f"{d}/profile_{a}.json"))["profile"])
    for label, m0 in [("Sun 17:50-18:20", 6 * 1440 + 1070), ("Tue 17:55-18:10", 1 * 1440 + 1075), ("Sun 19:55-20:10", 6 * 1440 + 1195)]:
        n = 30 if "Sun 17" in label else 15
        print(a, label, "sd mult:", np.sqrt(p[m0:m0 + n]))
