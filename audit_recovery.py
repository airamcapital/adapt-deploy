import pandas as pd
from adapt.data_loader import load_settings
from adapt.allocator.combined_dynamic import run_combined_dynamic

settings = load_settings()
df, _ = run_combined_dynamic(settings)

ret = df["ret"].dropna()
eq = (1 + ret).cumprod()
dd = eq / eq.cummax() - 1

underwater = dd < 0
lengths = []
cur = 0

for flag in underwater:
    if flag:
        cur += 1
    elif cur > 0:
        lengths.append(cur)
        cur = 0

if cur > 0:
    lengths.append(cur)

s = pd.Series(lengths, dtype=float)

print("Median recovery", round(float(s.median()), 2))
print("Average recovery", round(float(s.mean()), 2))
print("Max recovery", round(float(s.max()), 2))
