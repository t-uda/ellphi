# n 次元拡張準備メモ

本メモは現行 2 次元実装の数式を整理しつつ、n 次元楕円体 (ellipsoid) への拡張方針をまとめたものです。
コード参照は `repo_root/path:line` 形式で付記しています。

---

## 1. 現行 (2D) 手法の数学的整理

### 1.1 楕円の表現 (`src/ellphi/geometry.py`)

- 2 次元楕円は 2×2 正定値行列 `Σ` と中心 `x₀ ∈ ℝ²` から
  \[
  F_p(x) = (x - x₀)^\top Σ^{-1} (x - x₀)
  \]
  を用いて等高線 `F_p(x) = τ²`（τ ≥ 0）の形で表現する。
- 実装では上式を展開した
  \[
  F_p(x) = x^\top A_p x + 2 b_p^\top x + c_p
  \]
  の係数を 6 要素ベクトル `[a, b, c, d, e, f]` として保持している
  (`coef_from_cov`, `coef_from_axes`; `src/ellphi/geometry.py:32-111`)。
  - `A_p = Σ^{-1} / scale²`（`scale` は等高線レベルの再スケーリング）。
  - `b_p = -A_p x₀`, `c_p = x₀^\top A_p x₀`。
  - `quad_eval`（`src/ellphi/_solver_python.py:29-35`）は `F_p(x)` を返す。
- `t`（tangency distance）は `t = √F_p(x_t)` として定義され、
  `t=1` が元の楕円、`t>1` が膨張後の等高線を意味する
  (`src/ellphi/_solver_python.py:111-118`、`tests/test_basic.py:118-205`)。

### 1.2 Tangency 方程式 (`src/ellphi/_solver_python.py`)

1. 2 つの楕円 `F_p(x)`, `F_q(x)` からパラメータ付きペンシル
   \[
   F_\mu(x) = (1-\mu) F_p(x) + \mu F_q(x), \quad \mu ∈ [0,1]
   \]
   を構成する（`pencil`; `src/ellphi/_solver_python.py:38-41`）。
2. 2 次形式 `F_\mu` は
   \[
   A_\mu = (1-\mu)A_p + \mu A_q,\quad
   b_\mu = (1-\mu)b_p + \mu b_q
   \]
   を満たし、極値条件 `∇_x F_\mu(x_c)=0` から
   \[
   x_c(\mu) = -A_\mu^{-1} b_\mu
   \]
   を一意に復元できる (`_center`; `src/ellphi/_solver_python.py:47-60`)。
3. Tangency 条件は
   \[
   Δ(\mu) = F_p(x_c(\mu)) - F_q(x_c(\mu)) = 0
   \]
   に還元され、`solve_mu` で 1 次元根を探索する
   (`solve_mu`; `src/ellphi/_solver_python.py:75-101`)。
4. 得られた `μ*` に対し、接触点 `x_t = x_c(μ*)`、スケーリング `t = √F_p(x_t)` が `TangencyResult` として返される
   (`tangency`; `src/ellphi/_solver_python.py:104-118`)。

上記の `Δ(μ)` は楕円ごとの距離関数が一致するスケールを意味し、
同時に勾配 `∇F_p(x_t)` と `∇F_q(x_t)` が反対向きになるため、接触点を与える。

### 1.3 導関数とペンシル幾何 (`src/ellphi/_tangent_pencil.py`, `differentiable_solver.py`)

- `build_tangent_pencil` は `A_\mu`, `A_\mu^{-1}`, `x_c(μ)` をキャッシュする
  (`src/ellphi/_tangent_pencil.py:34-50`)。
- `Δ'(μ)` は
  \[
  Δ'(μ) = 2\, r^\top A_\mu^{-1} r,\quad
  r = (A_q - A_p) x_c + (b_q - b_p)
  \]
  で計算される (`target_prime_from_pencil`; `src/ellphi/_tangent_pencil.py:53-62`)。
- `center_jacobian` は中心位置の係数微分 `∂x_c/∂r` を返し、
  `differentiable_solver.solve_mu_gradients` が
  `∂μ/∂p`, `∂μ/∂q` を解析的に求めている
  (`src/ellphi/differentiable_solver.py:59-118`)。

---

## 2. n 次元楕円体への一般化

### 2.1 楕円体の表現

任意次元 `n ≥ 2` では楕円体を
\[
F_p(x) = (x - x₀)^\top Σ^{-1} (x - x₀) = x^\top A_p x + 2 b_p^\top x + c_p
\]
として扱える。ここで

- `A_p ∈ ℝ^{n×n}` は対称正定値（`Σ^{-1}/scale²`）。
- `b_p = -A_p x₀ ∈ ℝ^n`。
- `c_p = x₀^\top A_p x₀`。

パラメータ数は
\[
m = \frac{n(n+1)}{2} + n + 1
\]
（対称行列の上三角要素 + 線形項 + 定数項）。
2 次元既存仕様 (`m=6`) は `n=2` の特殊ケースとなる。

#### 係数のシリアライズ案

1. **行列 + ベクトル表現**
   API 内部では `(A, b, c)` を保持し、外部公開の `coef` は `np.ndarray` に平坦化して返す。
2. **構造化データクラス**
   互換性を維持しつつ、`ConicCoefficients` (dataclass) を追加し `__array__` を実装して既存 API を壊さない。

### 2.2 Tangency 条件の一般化

すべての式は行列サイズを `2→n` に置き換えるだけで成り立つ。

- `A_\mu = (1-\mu)A_p + \mu A_q`、`b_\mu = (1-\mu)b_p + \mu b_q`。
- `x_c(\mu) = -A_\mu^{-1} b_\mu` は `A_\mu` が SPD であれば任意次元で求まる。
- ルート関数 `Δ(μ) = F_p(x_c) - F_q(x_c)` は同型。
- `Δ'(μ) = 2\,r^\top A_\mu^{-1} r` も同型で、`r = (A_q - A_p) x_c + (b_q - b_p)`。

### 2.3 スケーリング距離

接触点 `x_t` におけるスケールは
\[
t = \sqrt{F_p(x_t)} = \sqrt{(x_t - x_{0,p})^\top Σ_p^{-1} (x_t - x_{0,p})}
\]
で定義され、任意次元で不変。
`pdist_tangency` も `ellcloud.coef` の長さから `n` を復元すれば同じロジックで計算できる。

### 2.4 数値的留意点

- `A_\mu` の逆行列は `O(n^3)`。高次元では `solve_mu` 内での反復回数や分解再利用を検討する。
- `Δ'(μ)` がゼロに近づくケース（ほぼ同一楕円体）では `solve_mu` のブレンチング戦略を見直す必要がある。
- `LocalCov` による共分散推定は `n` 増加でサンプル不足になりやすいので、`k` の既定値やリッジ項の導入を検討する。

---

## 3. 実装ノート（インターフェース維持方針）

| 領域 | 変更ポイント | 方針 |
| --- | --- | --- |
| `ellphi.geometry` | `_inv_broadcast`, `coef_from_cov`, `coef_from_axes`, `axes_from_cov` | `coef_from_cov` を任意 `n` に拡張（`numpy.linalg.inv` or `np.linalg.solve` をバッチ化）。`coef_from_axes` は 2D 専用関数として残し、nD 向けに `coef_from_eigendecomp(center, eigenvalues, eigenvectors)` を追加する。 |
| `EllipseCloud` (`src/ellphi/ellcloud.py`) | `coef` の shape, docstring, `plot` | `coef` shape を `(N, m)` に一般化し `n_dim` プロパティを導入。`plot` は 2D のみ対応と明記し、nD では例外 or 次元削減を提供。 |
| `LocalCov` | 入出力 shape の一般化 | `X` の shape を `(N, n)` とし、`pdist` 呼び出しのコメント／型ヒントを更新。 |
| `solver` (`_solver_python.py`, `_tangent_pencil.py`) | 行列表現の一般化 | `quad_eval`, `_center`, `build_tangent_pencil`, `target_prime_from_pencil`, `center_jacobian` を `(A, b, c)` ベースに書き換える。2D 互換のため、内部で「6 要素 → (A, b, c)」のアンパックユーティリティを導入する。 |
| `differentiable_solver.py` | `center_jacobian` の利用 | `center_jacobian` を nD 対応へ拡張後、`solve_mu_gradients` は同じ数式で動作。Jacobian の shape は `(m, n)` に変わる点をテストで担保する。 |
| `pdist_tangency` | `coef.shape[1]` 依存 | 係数長 `m` から `n` を解く helper `infer_dim_from_coef_length(m)` を追加し、並列実装には既存ロジックを流用。 |
| C++ バックエンド (`_tangency_cpp_impl.cpp`) | ハードコードされた 2×2 行列 | 行列サイズをテンプレート化する必要がある。段階的には Python 実装を nD 対応 → C++ は 2D fallback とし、nD サポートは将来対応に分離してもよい。 |
| テスト (`tests/`) | 期待値の更新 | 既存 2D テストはそのまま維持し、新たに nD（例: 3D 球体）の解析解ケースを追加。`factories.py` に任意次元楕円体ジェネレータを導入。 |
| ドキュメント／ノートブック | API 説明 | `README` や notebooks で「nD 対応」の前提と、2D でのみ可視化できる点を説明する。 |

### 実装ステップ案

1. `coef` の一般化ユーティリティ（pack/unpack, dim 推定）を追加。
2. `geometry`, `ellcloud`, `LocalCov` を nD 対応化し、既存 2D テストをパスさせる。
3. `solver` 群の線形代数を nD 化、`differentiable_solver` も更新。
4. `pdist_tangency` および `EllipseCloud.pdist_tangency` を新 `coef` 仕様に適合。
5. 3D 球体などで数値検証用テストを追加。
6. C++ バックエンド／notebook／ドキュメントの更新を段階的に行う。

以上を踏まえれば、公開 API（`tangency`, `pdist_tangency`, `EllipseCloud` など）のシグネチャを維持したまま、内部表現のみを拡張して n 次元版を提供できる。

---

## 4. テストドリブンで進めるための具体的な開発順序

既存 2D 実装を壊さずに TDD を回すには、「各層を独立に一般化し、段階ごとにテストを追加→既存テストを緑のまま維持する」ことが重要。以下の順序で進めると、どの段階でも `poetry run pytest` が通る形を保てる。

1. **ユーティリティ層の整備（ノーオプ変更多めの初手）**
   - `ellphi.geometry` に `pack_conic(A, b, c)`, `unpack_conic(coef)`、`infer_dim_from_coef_length(m)` を追加。
   - 新しい単体テストを `tests/test_geometry.py` に書き、2D 係数との往復が恒等になることを確認。
   - まだ既存コードはこれらを使わないので、既存テストは影響なし。

2. **`coef_from_cov` / `_inv_broadcast` の nD 化**
   - 実装を書き換えたら、旧 2D テストに加えて `pytest.mark.parametrize("dim", [2, 3])` のような軽量ケースを追加。
   - ここでも `EllipseCloud` などには手を入れず、`coef_from_cov` 単独のテストのみで緑を確認。

3. **`EllipseCloud` / `LocalCov` の対応**
   - `EllipseCloud` に `n_dim` プロパティと `plot` の 2D 限定ガードを追加。
   - `tests/factories.py` に「任意次元ガウス分布から楕円体を生成する」関数を追加し、`tests/test_ellcloud.py` で `n=3` ケースを新規テスト。
   - ここまでで solver 系は 2D のままなので、`tests/test_basic.py` などは影響せずに実行可能。

4. **Solver へのユーティリティ適用**
   - `_solver_python.py` と `_tangent_pencil.py` を `(A, b, c)` ベースに書き換える。
   - まずは既存 2D テストが通ることを確認し、その後 3D 球体に対する解析解テストを `tests/test_basic.py` へ追加。
   - この段階で `tangency`/`pdist_tangency` の public API は不変なので、インクリメンタルに TDD を継続できる。

5. **`differentiable_solver` の更新**
   - `center_jacobian` を nD 対応にした後、`solve_mu_gradients` 用の 3D 単体テストを追加（例: 2 つの球体で `μ` が既知になるケース）。
   - 2D テストを温存したまま新テストを追加することで、失敗時にどこが退行したかが明確になる。

6. **`pdist_tangency` / `EllipseCloud.pdist_tangency` の仕上げ**
   - 係数長から自動的に次元を判定する helper を導入し、Python 実装を nD 対応にする。
   - `tests/test_solver.py` に 3D 用のランダムクラウドテストを追加するが、必要に応じて `pytest.importorskip("scipy")` などで負荷を抑える。
   - まだ C++ バックエンドは 2D のままなので、`backend="python"` を強制するフラグをテストで設定しておくと CI が安定。

7. **C++ バックエンドを最後に拡張 or ガード**
   - Python 実装が安定し、nD の検証テストが十分緑になった段階で着手。
   - バックエンドが未対応の間は `has_cpp_backend` に「n>2 の場合は False を返す」ガードを入れておけばテストの実行可否が明確になる。

この順序だと、常に「狭い領域のリファクタ → その場でテスト追加 → 既存テストも緑」というサイクルを保てるため、AGENT が docs を参照しながらでも滑らかにテストドリブン開発を進められる。
