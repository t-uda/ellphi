# 数値安定性の調査レポート（Python 3.11 CI 環境）

## 1. CI 環境のバージョン調査
- CI は Python 3.10/3.11 のマトリクスで実行され、Poetry を用いて依存パッケージをインストールする。テストステップは flake8 → black --check → pytest の順で、型チェックは Python 3.11 で mypy と stubtest を回している。【F:.github/workflows/python-app.yml†L4-L83】
- ロックファイルでは NumPy 2.2.6、SciPy 1.15.3 が解決されており、CI でも同一バージョンが入る。【F:poetry.lock†L2111-L2145】【F:poetry.lock†L3436-L3465】
- この作業環境は Python 3.12.12 だが、Poetry 経由でインストールされた NumPy/SciPy は CI と同じ 2.2.6/1.15.3 であることを確認した。【66655e†L1-L3】

## 2. 問題の背景
- 厳しめの数値テスト（`tests/test_numerical_stability.py`）で、Python バージョンや BLAS/LAPACK 実装の違いにより収束挙動が変わるケースが報告された。
- Python 実装では conic の中心計算 `_center` が Cholesky 分解に失敗した場合、`numpy.linalg.lstsq` にフォールバックしていた。LAPACK 依存の疑似逆行列解法は、バージョンやビルドによりピボット選択が変わり得るため、微妙な違いが Newton ステップの発散につながるリスクがあった。
- 一方で C++ バックエンドは Cholesky 失敗時に部分ピボット付きガウス消去を実装しており、Python との差分が環境依存の挙動の原因になり得る。

## 3. 実施した対策
- Python 側でも C++ バックエンドに合わせた部分ピボット付きガウス消去 `_gaussian_elimination` を実装し、Cholesky が失敗したときのフォールバックに採用した。【F:src/ellphi/_solver_python.py†L147-L200】
- `tests/test_solver.py` に Cholesky 失敗ケース（非正定な 2×2 行列）を追加し、フォールバックが期待通りに働くことを確認する回帰テストを追加した。【F:tests/test_solver.py†L1-L72】
- スタブファイル `src/ellphi/_solver_python.pyi` に新規プライベート関数のシグネチャを追加し、型検査ツールとの整合性を取った。【F:src/ellphi/_solver_python.pyi†L33-L35】

## 4. 調査結果の要約
- CI で使用される Python/NumPy/SciPy バージョンは固定されており、今回の発散は依存バージョンのブレよりも「Cholesky → `lstsq`」というフォールバック経路の数値的不安定さが原因になっていた可能性が高い。
- ガウス消去によるフォールバックは C++ 実装と同じピボット戦略を取るため、Python バージョンや LAPACK 実装差による解のブレを抑えられる。

## 5. 今後の提案
- **CI マトリクスの拡張:** 将来的には Python 3.12、NumPy 2.0/2.1 系、SciPy 1.11–1.15 系の組み合わせを増やし、数値回帰を早期検知する。テスト時間増加を抑えるため、週次のスケジュール実行や軽量サンプリングケースを用意するのが現実的。
- **バージョン方針:** 2.2.6/1.15.3 以外で発散が再現した場合には、リリースノートや README に非推奨バージョンを明記し、Poetry の `python`/`numpy`/`scipy` の下限を環境検証済みの組み合わせに揃える。
- **追加の数値防御:** Newton 失敗時のリトライ戦略（例: μ の再初期化やステップ制限）、およびテスト用に C++/Python 双方の中心計算を比較するプロパティベーステストを導入すると、将来の BLAS/LAPACK 更新に対する安全性が高まる。
