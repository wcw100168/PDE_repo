# 高階球面 Summation-by-Parts Discontinuous Galerkin (SBP-DG) 方法專案

本專案實作三維嵌入球面（$\mathbb{S}^2 \subset \mathbb{R}^3$）上的高階 Summation-by-Parts Discontinuous Galerkin (SBP-DG) 球面雙曲型守恆律求解器。採用**細分八面體球面網格 (Subdivided Octahedral Spherical Mesh)** 與 **三維歐氏空間笛卡爾坐標內在投影**，解決極點幾何奇異性與長時間數值穩定性問題。

---

## 🚀 快速開始 (Quick Start)

### 1. 安裝依賴環境
本專案基於 Python 3.10+ 與標準科學計算庫（NumPy, SciPy, Pytest）：
```bash
pip install numpy scipy pytest
```

### 2. 一鍵運行球面平流模擬範例 (Run Advection Demo)
執行 Williamson (1992) 剛體旋轉平流模擬範例：
```bash
python3 examples/run_spherical_advection.py
```
**輸出範例：**
```text
======================================================================
 Spherical SBP-DG Advection Simulation Demo (ndivs=4, order=3)
======================================================================
[1/4] Generating Subdivided Octahedral Spherical Mesh...
      Total spherical triangular elements K = 128
      Quadrature nodes per element Np = 18 (Order N=3)
[2/4] Constructing Dubiner basis & SBP operators...
[3/4] Setting initial Gaussian bell and rigid body velocity field...
      Initial Mass M0   = 7.851427042559e-01
      Initial Energy E0 = 3.927061830972e-01
[4/4] Running 50 LSRK45 time steps...

----------------------------------------------------------------------
 Execution Summary (Elapsed: 0.085 s)
----------------------------------------------------------------------
 Formulation | Relative Mass Drift | Relative Energy Drift | Status
----------------------------------------------------------------------
 Split2 (Two-Term)  | 1.655925e-09    | 1.008215e-10     | Energy Stable
 Split3 (Three-Term)| 1.747570e-09    | 2.506519e-09     | Mass Conserving
======================================================================
```

### 3. 執行單元測試 (Run Pytest Suite)
執行 26 項自動化單元測試與雙重交叉驗證測試：
```bash
python3 -m pytest tests/ -v
```

---

## 📁 專案架構 (Project Architecture)

```
數學專案測試版/
├── context/                   # AI 語境與數學合約層 (README, Notations, Assumptions, Theorems)
├── src/                       # 核心算子與求解器模組
│   ├── geometry/              # 八面體球面網格生成 (sphere_mesh.py) & 度規張量 (metrics.py)
│   ├── operators/             # Dubiner 基底 (basis.py), Cholesky 正交化, SBP 閉式算子 (sbp.py)
│   └── solver/                # 介面通量 (fluxes.py), RHS 三大格式 (formulations.py), LSRK45 推進器
├── tests/                     # 26/26 全部通過之 Pytest 自動化測試集
├── references/                # 外部代碼對照組 (Simplex-DG-solver & Collaborator repos)
├── raw/                       # 原始舊講義與筆記檔
└── examples/                  # 一鍵執行範例腳本
```

---

## 🔬 核心數學公式與驗證指標

### 1. 控制方程 (Scalar Advection on $\mathbb{S}^2$)
$$
\mathcal{J}\frac{\partial q}{\partial t} + \frac{\partial}{\partial \xi}(\mathcal{J} u^\xi q) + \frac{\partial}{\partial \eta}(\mathcal{J} u^\eta q) = 0, \quad (\xi, \eta) \in T
$$

### 2. Modal Cholesky 正交化
$$
\hat{M} = V_{\text{raw}}^T W V_{\text{raw}} = L L^T \implies V = V_{\text{raw}} (L^T)^{-1} \implies V^T W V = I
$$
雙重驗證正交殘差精確維持在 $\|V^T W V - I\|_\infty \le 10^{-14}$。

---

## 📜 授權協議 (License)
MIT License
