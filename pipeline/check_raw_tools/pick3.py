import sys, pandas as pd
h = pd.read_csv(sys.argv[1]); img = pd.read_csv(sys.argv[2])
s = h.groupby("nsin").agg(ev=("event", "max"), views=("views", "first"), fmts=("fmts", "first"), kl0max=("kl0", "max"), st=("status10", lambda x: ",".join(x)))
a = s[(s.ev == 1) & (s.views == "O") & (s.fmts == "tif") & (s.kl0max == 2)].index[0]
b = s[(s.ev == 0) & (s.views == "O") & (s.fmts == "dcm") & (s.kl0max == 0) & (s.st == "event-free@10y,event-free@10y")].index[0]
c = s[(s.views == "LR") & (s.ev == 1)].index[0]
print(a, b, c)
for n in (a, b, c): print(n, img[img.nsin == n].path.tolist())
