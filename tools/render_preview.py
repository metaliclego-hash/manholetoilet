"""make_model.py と同じ形状を matplotlib で描いて、ページ用のプレビュー画像を作る。

使い方:  python tools/render_preview.py
  → toilet_tent.png / toilet.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from make_model import Scene, MATERIALS, build_tent, build_toilet

LIGHT = np.array([0.4, 0.8, 0.6]); LIGHT /= np.linalg.norm(LIGHT)


def render(scene, path, elev=14, azim=-62, size=900):
    polys, colors = [], []
    for mat, (P, N, F) in scene.parts.items():
        P = np.array(P); N = np.array(N)
        base = np.array(MATERIALS[mat]['rgb']) / 255
        alpha = MATERIALS[mat].get('opacity', 1.0)
        for f in F:
            n = N[f].mean(0)
            shade = 0.45 + 0.55 * max(np.dot(n, LIGHT), 0)
            # matplotlib は Y 上ではなく Z 上なので (x, z, y) に並べ替え
            polys.append(P[f][:, [0, 2, 1]] * [1, -1, 1])
            colors.append((*np.clip(base * shade, 0, 1), alpha))
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors='none', linewidths=0))
    allp = np.concatenate(polys)
    lo, hi = allp.min(0), allp.max(0)
    c = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print(path)


def main():
    full = Scene(); build_tent(full); build_toilet(full)
    only = Scene(); build_toilet(only)
    render(full, 'toilet_tent.png')
    render(only, 'toilet.png', elev=18, azim=-55)


if __name__ == '__main__':
    main()
