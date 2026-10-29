# manholetoilet

マンホールトイレ（ミドリ安全 洋式タイプ VE100W／PTAM テント Mサイズ付、商品コード 4082105820）を
iPhone / iPad の AR Quick Look で原寸表示する。

- `index.html` … 「ARで見る」ページ（`<a rel="ar">` で USDZ を開く。テント付き／トイレのみを切替）
- `tools/make_model.py` … 商品ページの寸法・写真をもとに形状を組み立てて `toilet_tent.usdz` / `toilet.usdz` を生成
- `tools/render_preview.py` … ページ用のプレビュー画像 `toilet_tent.png` / `toilet.png` を生成

```
python tools/make_model.py
python tools/render_preview.py
```

寸法（商品ページより）: トイレ W670×D750×H790〜865mm、テント W1200×D1200×H2000mm
