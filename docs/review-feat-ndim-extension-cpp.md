# feat/ndim-extension-cpp レビューレポート

対象ブランチ: `feat/ndim-extension-cpp`  
比較対象: `codex/extend-c++-tangency-backend-for-n-dimensional-support`、`codex/extend-c++-tangency-backend-for-n-dimensional-support-8ifwnf`

## 評価サマリ
- Python バインディングと `solver` が任意次元を C++ にルーティングできるようになり、テストも両バックエンドを網羅する形に拡張されているため、今回の 3 ブランチの中で最も完成度が高い。
- C++ 実装では `SolverContext` を導入し、デコード済みの二次形式や差分を再利用することで導関数計算のコストとコードの見通しを改善している。
- 追加で取り込むべき改善は主に数値的な堅牢性と ABI 安全性に関するもので、他ブランチからの cherry-pick で補える。

## 具体的な強み
1. **前段チェックの徹底**  
   `src/ellphi/_tangency_cpp.py` で係数ベクトルの次元・形状を Python 側で検証し、早期に例外を出せる。加えて `solver.py` では `backend="cpp"`/`"auto"` が次元に依存せず利用できるようになった。
2. **テストカバレッジの拡張**  
   `tests/test_basic.py` の 3D/高次元ケースが `solver_backend` fixture を用いて両バックエンドを実行するため、C++ 実装の回 regresion を早期検知できる。
3. **C++ 側の構造化**  
   `src/ellphi/_tangency_cpp_impl.cpp` に `SolverContext`/`PencilGeometry` を導入し、ターゲット関数と導関数が同じデータを共有。係数差分も一度だけ作って再利用しているため、他ブランチに比べて CPU/GPU キャッシュ効率が良い。

## 他ブランチから取り込むべき改善
| 改善内容 | 出典ブランチ | 推奨取り込み方法 |
| --- | --- | --- |
| **sqrt 入力のクランプ**: 混合二次形式の値が丸め誤差で負になる場合に 0 へ丸めてから平方根を取ることで NaN を防ぐ。 | `codex/extend-c++-tangency-backend-for-n-dimensional-support` (`4b75630`) | `tangency_solve`/`pdist_tangency` で `value = std::max(value, 0.0);` のように処理を追加。 |
| **出力バッファ長の検証**: C++ 側が `point_out` のサイズを受け取り、過小サイズなら即座にエラーにする。 | `codex/extend-c++-tangency-backend-for-n-dimensional-support-8ifwnf` (`cbae84c`) | ABI 変更になるため、Python バインディングから長さを渡す形へ拡張し、呼び出し元の `ctypes` 定義も同期させる。 |
| **`pdist_tangency` 入力形状チェック**: `(m, n)` 形状を Python 側で保証する。 | `codex/extend-c++-tangency-backend-for-n-dimensional-support-8ifwnf` | 既存チェックを流用し、エラーメッセージを統一。 |

## 推奨アクション
1. `feat/ndim-extension-cpp` をベースとしてマージ準備を進める。
2. 上記 3 点を順に cherry-pick / 手動取り込みし、数値安定性と ABI 安全性を補強する。
3. 変更後は必ず AGENTS.md に記載のローカルチェック（`poetry run black`, `flake8`, `mypy`, `pytest`）を一連で実行し、`docs/n_dim_extension.md` など既存ドキュメントに必要な追記があれば後続タスクで対応する。
