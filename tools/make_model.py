"""マンホールトイレ（洋式タイプ VE100W／PTAM テント Mサイズ付）の AR Quick Look 用 USDZ を作る。

寸法はミドリ安全の商品ページ（商品コード 4082105820）の仕様より:
  トイレ  W670 × D750 × H790〜865 mm
  テント  W1200 × D1200 × H2000 mm
形状は商品写真を参考にプリミティブ（パイプ・箱・面）で組んでいる。単位は cm、Y が上、正面が +Z。

使い方:  python tools/make_model.py
  → toilet_tent.usdz（テント＋トイレ）と toilet.usdz（トイレのみ）を出力
"""
import io, math, zipfile
import numpy as np

# ---------------------------------------------------------------- 材質（sRGB 0-255 で指定）
MATERIALS = {
    'TentBlue':   dict(rgb=(38, 78, 158), rough=0.85),
    'TentLining': dict(rgb=(200, 206, 214), rough=0.6, metal=0.3),
    'TentBeige':  dict(rgb=(236, 224, 204), rough=0.9),
    'Mesh':       dict(rgb=(40, 42, 48), rough=0.9, opacity=0.75),
    'WhitePipe':  dict(rgb=(235, 235, 232), rough=0.5),
    'Aluminum':   dict(rgb=(196, 200, 206), rough=0.3, metal=0.9),
    'GripBlue':   dict(rgb=(30, 70, 165), rough=0.45),
    'WhitePlastic': dict(rgb=(238, 236, 228), rough=0.45),
    'Bag':        dict(rgb=(78, 82, 88), rough=0.75),
    'Plate':      dict(rgb=(170, 174, 180), rough=0.35, metal=0.8),
    'Dark':       dict(rgb=(18, 18, 20), rough=0.9),
    'Paper':      dict(rgb=(248, 248, 245), rough=0.9),
    'Rope':       dict(rgb=(30, 30, 30), rough=0.9),
}


def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ---------------------------------------------------------------- メッシュ組み立て
class Scene:
    def __init__(self):
        self.parts = {}   # material -> (points, normals, faces)

    def add(self, mat, pts, nrm, faces):
        """faces は頂点 index のリスト。頂点ごとの法線と面の向きが一致するよう巻き方向を揃える"""
        P, N, F = self.parts.setdefault(mat, ([], [], []))
        base = len(P)
        pts = [np.asarray(p, float) for p in pts]
        nrm = [np.asarray(n, float) / (np.linalg.norm(n) or 1) for n in nrm]
        for f in faces:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            gn = np.cross(b - a, c - a)
            if len(f) == 4 and np.linalg.norm(gn) < 1e-9:
                gn = np.cross(c - a, pts[f[3]] - a)
            want = sum(nrm[i] for i in f)
            if np.dot(gn, want) < 0:
                f = f[::-1]
            F.append([base + i for i in f])
        P.extend(pts)
        N.extend(nrm)

    # 平面ポリゴン（凸）。double=True で裏面に別材質を貼る
    def poly(self, mat, pts, normal, back=None, off=0.25):
        pts = [np.asarray(p, float) for p in pts]
        n = np.asarray(normal, float); n /= np.linalg.norm(n)
        idx = list(range(len(pts)))
        faces = [[0, i, i + 1] for i in range(1, len(pts) - 1)]
        self.add(mat, pts, [n] * len(pts), faces)
        if back:
            self.add(back, [p - n * off for p in pts], [-n] * len(pts), faces)

    def box(self, mat, center, size):
        c = np.asarray(center, float); h = np.asarray(size, float) / 2
        for axis in range(3):
            for s in (-1, 1):
                n = np.zeros(3); n[axis] = s
                u, v = [a for a in range(3) if a != axis]
                pts = []
                for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = c.copy(); p[axis] += s * h[axis]; p[u] += du * h[u]; p[v] += dv * h[v]
                    pts.append(p)
                self.add(mat, pts, [n] * 4, [[0, 1, 2, 3]])

    def tube(self, mat, path, r, seg=14, closed=False, caps=True):
        """折れ線 path に沿って半径 r のパイプを掃引する（平行移動フレーム）"""
        path = [np.asarray(p, float) for p in path]
        n = len(path)
        tang = []
        for i in range(n):
            if closed:
                t = path[(i + 1) % n] - path[i - 1]
            else:
                t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
            tang.append(t / np.linalg.norm(t))
        ref = np.array([0, 1, 0]) if abs(tang[0][1]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(tang[0], ref); u /= np.linalg.norm(u)
        pts, nrm = [], []
        for i in range(n):
            if i:
                u = u - np.dot(u, tang[i]) * tang[i]; u /= np.linalg.norm(u)
            v = np.cross(tang[i], u)
            for k in range(seg):
                a = 2 * math.pi * k / seg
                d = math.cos(a) * u + math.sin(a) * v
                pts.append(path[i] + r * d); nrm.append(d)
        rings = n if closed else n - 1
        faces = []
        for i in range(rings):
            j = (i + 1) % n
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append([i * seg + k, i * seg + k2, j * seg + k2, j * seg + k])
        self.add(mat, pts, nrm, faces)
        if caps and not closed:
            for i, sgn in ((0, -1), (n - 1, 1)):
                ring = pts[i * seg:(i + 1) * seg]
                self.add(mat, ring, [tang[i] * sgn] * seg, [[0, k, k + 1] for k in range(1, seg - 1)])

    def cylinder(self, mat, a, b, r, seg=16):
        self.tube(mat, [a, b], r, seg)

    def frustum(self, mat, center, axis_h, r0, r1, seg=24):
        """底面中心 center から上へ高さ axis_h、半径 r0→r1 の円錐台（側面のみ）"""
        c = np.asarray(center, float)
        pts, nrm = [], []
        slope = (r0 - r1) / axis_h
        for y, r in ((0, r0), (axis_h, r1)):
            for k in range(seg):
                a = 2 * math.pi * k / seg
                d = np.array([math.cos(a), 0, math.sin(a)])
                pts.append(c + d * r + [0, y, 0]); nrm.append(d + [0, slope, 0])
        faces = [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)]
        self.add(mat, pts, nrm, faces)

    def disk(self, mat, center, r, normal=(0, 1, 0), seg=28):
        c = np.asarray(center, float); n = np.asarray(normal, float)
        ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1])
        u = np.cross(n, ref); u /= np.linalg.norm(u); v = np.cross(n, u)
        pts = [c + r * (math.cos(2 * math.pi * k / seg) * u + math.sin(2 * math.pi * k / seg) * v)
               for k in range(seg)]
        self.add(mat, pts, [n] * seg, [[0, k, k + 1] for k in range(1, seg - 1)])

    def torus(self, mat, center, R, r, seg=36, rseg=12):
        c = np.asarray(center, float)
        path = [c + [R * math.cos(2 * math.pi * k / seg), 0, R * math.sin(2 * math.pi * k / seg)]
                for k in range(seg)]
        self.tube(mat, path, r, rseg, closed=True)

    # ------------------------------------------------------------ 書き出し
    def usda(self):
        out = ['#usda 1.0\n(\n    defaultPrim = "Root"\n    metersPerUnit = 0.01\n    upAxis = "Y"\n)\n',
               'def Xform "Root"\n{']
        for mat, (P, N, F) in self.parts.items():
            pts = ', '.join(f'({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})' for p in P)
            nrm = ', '.join(f'({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})' for n in N)
            counts = ', '.join(str(len(f)) for f in F)
            idx = ', '.join(str(i) for f in F for i in f)
            lo = np.min(P, axis=0); hi = np.max(P, axis=0)
            out.append(f'''
    def Mesh "{mat}_geo" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [({lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f}), ({hi[0]:.2f}, {hi[1]:.2f}, {hi[2]:.2f})]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{idx}]
        point3f[] points = [{pts}]
        normal3f[] normals = [{nrm}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </Root/Materials/{mat}>
    }}''')
        out.append('\n    def Scope "Materials"\n    {')
        for mat in self.parts:
            m = MATERIALS[mat]
            r, g, b = (srgb_to_linear(c) for c in m['rgb'])
            out.append(f'''
        def Material "{mat}"
        {{
            token outputs:surface.connect = </Root/Materials/{mat}/S.outputs:surface>
            def Shader "S"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.4f}, {g:.4f}, {b:.4f})
                float inputs:roughness = {m.get('rough', 0.6)}
                float inputs:metallic = {m.get('metal', 0.0)}
                float inputs:opacity = {m.get('opacity', 1.0)}
                token outputs:surface
            }}
        }}''')
        out.append('    }\n}\n')
        return '\n'.join(out)


def write_usdz(path, files):
    """USDZ仕様どおり無圧縮・各ファイルのデータ先頭を64バイト境界に揃えて書き出す"""
    with open(path, 'wb') as f, zipfile.ZipFile(f, 'w', zipfile.ZIP_STORED) as zf:
        for name, data in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            header_end = f.tell() + 30 + len(name.encode())
            pad = (-(header_end + 4)) % 64
            info.extra = b'\x12\x34' + pad.to_bytes(2, 'little') + b'\0' * pad
            zf.writestr(info, data)


# ---------------------------------------------------------------- トイレ本体
PIPE = 1.25          # アルミパイプ半径(cm)
SEAT_Y = 44          # 便座の高さ（座面調節 H790〜865 の低い側相当）


def build_toilet(s):
    # 床プレート（マンホール上に敷く鋼板）と中央の開口
    s.box('Plate', (0, 0.15, -2), (66, 0.3, 70))
    s.disk('Dark', (0, 0.32, 2), 15)

    xl, xr, zf, zb = -29, 29, 18, -24          # 脚の位置
    for x in (xl, xr):
        # 脚の足元プレート（左右に渡したレール）
        s.box('Aluminum', (x, 0.9, (zf + zb) / 2), (4, 1.2, zf - zb + 10))
        # 前脚：床から肘掛けまで
        s.cylinder('Aluminum', (x, 1.5, zf), (x, 60, zf), PIPE)
        s.cylinder('GripBlue', (x, 1.5, zf), (x, 4, zf), PIPE + 0.25)
        s.cylinder('GripBlue', (x, 20, zf), (x, 22, zf), PIPE + 0.2)
        # 後脚：床から背もたれまで
        s.cylinder('Aluminum', (x, 1.5, zb), (x, 84, zb), PIPE)
        s.cylinder('GripBlue', (x, 1.5, zb), (x, 4, zb), PIPE + 0.25)
        s.cylinder('GripBlue', (x, 20, zb), (x, 22, zb), PIPE + 0.2)
        # 肘掛け：前脚上端から後ろへ曲がって後脚へ（青いグリップ付き）
        arm = [(x, 52, zf + 1), (x, 60, zf + 1), (x, 64, zf - 2), (x, 66, zf - 8), (x, 67, zb + 2)]
        s.tube('Aluminum', arm[3:] + [(x, 67, zb)], PIPE, caps=False)
        s.tube('GripBlue', [(x, 48, zf + 1)] + arm[:4], PIPE + 0.7)
        # 便座フレーム側面
        s.cylinder('Aluminum', (x, SEAT_Y - 3, zf), (x, SEAT_Y - 3, zb), PIPE)
        # 白い樹脂ジョイント
        s.box('WhitePlastic', (x, SEAT_Y - 3, zb), (3.6, 4, 3.6))
        s.box('WhitePlastic', (x, 70, zb), (3.6, 5, 3.6))

    # 便座フレーム：前は U 字に張り出したパイプ、後ろは横棒
    s.tube('Aluminum', [(29 * math.cos(a), SEAT_Y - 3, zf - 6 + 10 * math.sin(a))
                        for a in np.linspace(math.pi, 0, 13)], PIPE)
    s.cylinder('Aluminum', (xl, SEAT_Y - 3, zb + 2), (xr, SEAT_Y - 3, zb + 2), PIPE)
    s.cylinder('Aluminum', (xl, SEAT_Y + 4, zb), (xr, SEAT_Y + 4, zb), PIPE)

    # 便座（白いドーナツ形）と下に吊るす汚物袋（スリーブ）
    seat_c = (0, SEAT_Y, -1)
    s.torus('WhitePlastic', seat_c, 15.5, 3.2)
    s.box('WhitePlastic', (0, SEAT_Y - 2.5, -17.5), (14, 2, 6))     # 便座の後ろの取付部
    s.frustum('Bag', (0, 0.3, 1), SEAT_Y - 3.5, 14, 13, seg=20)
    s.disk('Dark', (0, SEAT_Y + 0.5, -1), 12.4)                      # 便座の穴の奥（袋の口）

    # 背もたれ（白い樹脂板、取っ手穴付き）
    s.box('WhitePlastic', (0, 75, zb - 1.8), (62, 21, 2.2))
    s.box('WhitePlastic', (0, 85.6, zb - 1.8), (58, 0.4, 2.2))      # 上端を少し丸く見せる段
    s.box('Dark', (0, 81, zb - 0.6), (14, 2.8, 0.2))
    s.box('Dark', (0, 81, zb - 3.0), (14, 2.8, 0.2))

    # ペーパーホルダーとトイレットペーパー（右側）
    s.cylinder('Aluminum', (xr + 1, 42, zb + 14), (xr + 7, 42, zb + 14), 0.4)
    s.cylinder('Paper', (xr + 2, 37, zb + 14), (xr + 13, 37, zb + 14), 5.5, seg=20)
    s.disk('Dark', (xr + 13.05, 37, zb + 14), 2, normal=(1, 0, 0), seg=16)


# ---------------------------------------------------------------- テント
TW = 60        # 底辺の半幅
TH = 150       # 青い壁の高さ
TW1 = 42       # 壁上端の半幅
BH = 172       # ベージュ帯の上端
TW2 = 37
RIDGE = 200
RIDGE_HALF = 12
DOOR_W = 50    # 入口の床位置での半幅
DOOR_TOP = 145


def wall_z(y, w0=TW, w1=TW1, h=TH):
    return w0 + (w1 - w0) * y / h


def build_tent(s):
    def face(pts2d, side, mat, back='TentLining', y0=0, y1=TH, w0=TW, w1=TW1):
        """側面の2D座標(u=横, v=高さ)を3Dに。side=0..3 で +Z, +X, -Z, -X の面"""
        ang = side * math.pi / 2
        c, sn = math.cos(ang), math.sin(ang)
        pts = []
        for u, v in pts2d:
            d = w0 + (w1 - w0) * (v - y0) / (y1 - y0)       # 壁面までの距離
            pts.append((u * c + d * sn, v, d * c - u * sn))
        # 外向き法線（傾き込み）
        slope = (w0 - w1) / (y1 - y0)
        n = (sn, slope, c)
        s.poly(mat, pts, n, back=back)

    # 青い壁（正面は入口の V 字開口を残す）
    for side in (1, 2, 3):
        face([(-TW, 0), (TW, 0), (TW1, TH), (-TW1, TH)], side, 'TentBlue')
    hw_top = TW1
    face([(-TW, 0), (-DOOR_W, 0), (0, DOOR_TOP), (0, TH), (-hw_top, TH)], 0, 'TentBlue')
    face([(DOOR_W, 0), (TW, 0), (hw_top, TH), (0, TH), (0, DOOR_TOP)], 0, 'TentBlue')

    # ベージュの上部（壁の続き）
    for side in range(4):
        face([(-TW1, TH), (TW1, TH), (TW2, BH), (-TW2, BH)], side, 'TentBeige',
             y0=TH, y1=BH, w0=TW1, w1=TW2)

    # 屋根：棟（X方向）を持つ寄棟
    r0, r1 = np.array([-RIDGE_HALF, RIDGE, 0.0]), np.array([RIDGE_HALF, RIDGE, 0.0])
    c = [np.array(p, float) for p in ((-TW2, BH, TW2), (TW2, BH, TW2), (TW2, BH, -TW2), (-TW2, BH, -TW2))]

    def roof(pts):
        a, b, d = pts[0], pts[1], pts[-1]
        n = np.cross(b - a, d - a)
        if n[1] < 0:
            n = -n
        s.poly('TentBeige', pts, n, back='TentLining')
    roof([c[0], c[1], r1, r0])      # 正面
    roof([c[2], c[3], r0, r1])      # 背面
    roof([c[1], c[2], r1])          # 右
    roof([c[3], c[0], r0])          # 左
    s.tube('TentBeige', [r0, r1], 1.0)

    # 骨組み：四隅の稜線に沿った縫い目（細い白線）と床の白パイプ
    for side in range(4):
        ang = side * math.pi / 2
        for sg in (-1, 1):
            def p(u, v, w):
                return (u * math.cos(ang) + w * math.sin(ang), v, w * math.cos(ang) - u * math.sin(ang))
            s.tube('TentBeige', [p(sg * TW, 0, TW), p(sg * TW1, TH, TW1), p(sg * TW2, BH, TW2)], 0.5, seg=6)
    base = [(-TW, 1.5, TW), (TW, 1.5, TW), (TW, 1.5, -TW), (-TW, 1.5, -TW)]
    s.tube('WhitePipe', base, 1.5, closed=True, seg=10)

    # 入口の巻き上げたフラップ（V字の縁に沿う青いロール）とひも
    for sg in (-1, 1):
        a = np.array([sg * DOOR_W, 2, wall_z(2) + 1.5])
        b = np.array([0, DOOR_TOP, wall_z(DOOR_TOP) + 1.5])
        s.cylinder('TentBlue', a, b, 2.2, seg=10)
        for t in (0.35, 0.7):
            q = a + (b - a) * t
            s.cylinder('Rope', q + [0, 0, -0.5], q + [0, -6, 2.5], 0.3, seg=6)
        s.cylinder('Rope', (sg * TW, 1.5, TW), (sg * (TW + 4), 0.4, TW + 4), 0.4, seg=6)
        s.cylinder('Rope', (sg * TW, 1.5, -TW), (sg * (TW + 4), 0.4, -TW - 4), 0.4, seg=6)

    # 背面上部のメッシュ窓（内側）
    def zin(y):
        return -wall_z(y) + 0.8
    s.poly('Mesh', [(-20, 108, zin(108)), (20, 108, zin(108)), (20, 142, zin(142)), (-20, 142, zin(142))],
           (0, (TW - TW1) / TH, 1))
    s.box('Dark', (0, 105.5, zin(105) + 0.4), (4, 2, 1))


def main():
    full = Scene(); build_tent(full); build_toilet(full)
    only = Scene(); build_toilet(only)
    for name, sc in (('toilet_tent', full), ('toilet', only)):
        usda = sc.usda()
        write_usdz(f'{name}.usdz', [(f'{name}.usda', usda.encode())])
        allp = np.concatenate([np.array(P) for P, _, _ in sc.parts.values()])
        lo, hi = allp.min(0), allp.max(0)
        print(f'{name}.usdz  {sum(len(F) for _, _, F in sc.parts.values())} faces, '
              f'size W{hi[0]-lo[0]:.0f} x D{hi[2]-lo[2]:.0f} x H{hi[1]-lo[1]:.0f} cm')


if __name__ == '__main__':
    main()
