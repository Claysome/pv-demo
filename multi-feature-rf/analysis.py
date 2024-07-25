import pandas as pd
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


res = pd.read_csv('models/result.csv', index_col=0)
# 根据index排序
res = res.sort_index()
res.reset_index(drop=True, inplace=True)
res_grouped = res.groupby(res.index // 12).apply(lambda x: x.sum())

res_grouped.plot()
plt.show()