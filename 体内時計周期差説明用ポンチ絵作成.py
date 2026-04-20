import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（Windows想定：環境に応じて利用可能なものが自動で使われます）
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. データの準備（画像の数値を再現）
index_values = [23.8, 23.9, 24, 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7]
data = []

# 各行の差分を計算してマトリックスを作成
for i in index_values:
    row = [round(abs(i - j), 1) if abs(i - j) >= 0.1 else 0.01 if i != j else 0 for j in index_values]
    data.append(row)

df = pd.DataFrame(data, index=index_values, columns=index_values)

# 2. ヒートマップの描画
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2g", cbar_kws={'label': 'コスト関数J目標までの距離[h]'})
ax.invert_yaxis()  # 下が23.8[h]になるように縦軸を反転

plt.title("体内時計周期の差によるコスト関数Jへの悪影響 ヒートマップ")
plt.xlabel("体内時計周期 [h]")
plt.ylabel("体内時計周期 [h]")
plt.tight_layout()
plt.show()