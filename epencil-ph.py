
import numpy
from matplotlib.markers import CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN
from matplotlib.colors import ListedColormap
import itertools
from epencil import coefabc_fixed, find_intersect_mu, epoly

import homcloud.interface as hc
import matplotlib.pyplot as plt

# オリジナルの色リストを使ってカラーマップを作成
my_cmap = ListedColormap([
    'red',       # 赤
    'blue',      # 青
    'green',     # 緑
    'cyan',      # シアン
    'magenta',   # マゼンタ
    'orange',    # オレンジ
    'purple',    # 紫
    'black',     # 黒
    ])

# 固定シードでランダムパラメータを生成
numpy.random.seed(42)
N = 100

ellipses = []
coeff0012t = []
for _ in range(N):
    x0 = numpy.random.uniform(-5, 5)
    y0 = numpy.random.uniform(-5, 5)
    r1 = numpy.random.uniform(0.1, 1)
    r2 = numpy.random.uniform(0.1, 1)
    theta = numpy.random.uniform(0, 2 * numpy.pi)
    ellipses.append(coefabc_fixed(x0, y0, r1, r2, theta))
    coeff0012t.append([x0, y0, r1, r2, theta])

coeff0012t = numpy.array(coeff0012t)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(*coeff0012t[:, :2].T, 'ko', markersize=5, label='Input Points')

# カラーマップの取得
cmap = plt.get_cmap('prism')  # 'viridis', 'Set1', 'tab20' など他のカラーマップも利用可能
# cmap = my_cmap

# 楕円をプロット
for i, coeffs in enumerate(ellipses):
    x0, y0, r1, r2, theta = coeff0012t[i]
    color = cmap(i % cmap.N)
    ellipse = plt.matplotlib.patches.Ellipse(
        (x0, y0),
        width=2*r1,
        height=2*r2,
        angle=numpy.degrees(theta),
        edgecolor=color,
        facecolor='none',
        linestyle='--',
        linewidth=3
    )
    ax.add_patch(ellipse)

plt.show()

dist = numpy.zeros((N, N))

# 接点のプロット
for (i, p), (j, q) in itertools.combinations(enumerate(ellipses), 2):
    intersect_mu = find_intersect_mu(p, q, method='brentq+newton')
    q_mu = (1 - intersect_mu) * p + intersect_mu * q
    intersect = -numpy.linalg.inv(numpy.array([[q_mu[0], q_mu[1]], [q_mu[1], q_mu[2]]])).dot(q_mu[3:5])
    intersect_t = numpy.sqrt(epoly(*q_mu, *intersect))
    dist[i, j] = dist[j, i] = intersect_t

plt.matshow(dist)
plt.show()

hc.PDList.from_rips_filtration(dist, maxdim=2, save_to="rips.pdgm")
pdlist = hc.PDList("rips.pdgm")
pd1 = pdlist.dth_diagram(1)
pd1.histogram().plot(colorbar={"type": "log"})
plt.show()

