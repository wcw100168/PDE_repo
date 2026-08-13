# 多元素 DG 平流方程式實作與收斂驗證藍圖 (Multi-Element DG Advection)

**Status:** 🚧 **PLANNING PHASE**
**Target:** 從單一元素 (Single-Element) 升級為多元素 (Multi-Element) 架構，並進行空間 (h/p-refinement) 與時間的收斂性驗證。
**Future Proofing:** 架構設計必須相容於未來的「八面體網格 (Octahedron/Cubed-Sphere)」與「淺水波方程式 (SWE)」。

## 🎯 核心設計理念與未來擴展性 (Future Proofing)

為確保未來順利過渡到八面體上的 SWE，本階段的資料結構與實作必須遵守以下限制：

1. **全域狀態矩陣 $\mathbf{Q}$ 的維度設計**：
   - 必須採用 `(n_var, Np, K)` 的 3D 張量結構。
   - `n_var`: 物理變數數量（目前平流方程式 $n_{var}=1$，未來 SWE 將是 $h, hu, hv$ 等 $n_{var}=3$）。
   - `Np`: 單一元素體積節點數。
   - `K`: 總元素數量。
2. **無迴圈廣播 (Vectorization & Broadcasting)**：
   - 絕對禁止使用 `for k in range(K):` 來掃描元素計算 RHS。所有的微分矩陣相乘與幾何度量縮放，必須透過 NumPy 的 `np.newaxis` 與廣播機制一次性完成。
3. **邊界交換的純量化 (Vector-Reversal Strategy)**：
   - 相鄰元素的介面資料交換，依賴全域索引表 `vmapM` (Interior) 與 `vmapP` (Exterior)。
   - 根據使用者定義，由於介面節點的提取順序是沿著元素邊界繞行，相鄰元素的節點對應關係**必定是精確顛倒的 (Exactly Reversed)**。因此，建立 `vmapP` 時，只需將對應邊界的索引陣列進行 `[::-1]` 反轉即可，不需進行複雜的座標比對。

---

## 🏗️ 第一階段：網格拓樸與資料結構初始化 (Mesh Topology & Initialization)

本階段目標是建立多元素的幾何度量與連接關係表。

1. **網格讀取與連接性建立 (Mesh & Connectivity)**：
   - 讀取或生成包含 $K$ 個三角形的網格（例如 `total_elements = K`）。
   - 利用 `connectivity_demo.ipynb` 的頂點配對哈希法，建立 $EToE$ (Element-to-Element) 與 $EToF$ (Element-to-Face) 矩陣。
2. **全域索引映射表 (Index Mapping)**：
   - 生成 `vmapM`：大小為 `(3 * nfp, K)`，記錄所有介面節點在全域 $\mathbf{Q}$ 矩陣中的位置。
   - 生成 `vmapP`：利用 $EToE$ 和 $EToF$ 找出鄰居介面，並**嚴格套用 `[::-1]` 陣列反轉邏輯**來對齊節點，完成外部索引表的建構。
3. **幾何度量擴充 (Geometric Factors)**：
   - 計算每個元素的 Jacobian 行列式 $J$ 以及轉換係數 $rx, ry, sx, sy$。這些變數必須儲存為形狀為 `(K,)` 的 1D 陣列。
   - 計算每個元素每條邊的法向量 $\boldsymbol{n} = (n_x, n_y)$ 以及表面雅可比 $J_{face}$，儲存為 `(3, K)` 或展開為 `(3 * nfp, K)` 的陣列。
4. **狀態矩陣初始化**：
   - 初始化 $\mathbf{Q}$ 為 `(1, Np, K)`，並代入 $t=0$ 的精確解（例如動態波 $q = \sin(x)$ 或高斯脈衝）。

---

## ⚙️ 第二階段：全域 RHS 計算函數 (Global RHS Evaluation)

本階段將 `compute_rhs(Q, t)` 升級為支援 `(n_var, Np, K)` 的多元素版本。

1. **物理場動態求值**：
   - `u_arr, v_arr = velocity_field(X, Y, t)`，其中 `X, Y` 是形狀為 `(Np, K)` 的物理座標矩陣。
2. **體積項 (Volume Term - Split Form)**：
   - 利用參考微分矩陣 $D_r, D_s$ 進行廣播。
   - 範例邏輯：`dq_dr = D_r @ Q[0]`，然後 `D_x_q = rx[np.newaxis, :] * dq_dr + sx[np.newaxis, :] * dq_ds`。
   - 嚴格遵守 Split-Form 公式計算出 `volume_term`。
3. **介面通量與物理邊界 (Surface Flux & Boundaries)**：
   - **擷取狀態**：
     - `q_M = Q[0].flat[vmapM]` (內部狀態)
     - `q_P = Q[0].flat[vmapP]` (預設外部狀態為鄰居狀態)
   - **覆寫物理邊界**：
     - 找出 Domain Boundary 的索引（即 $EToE[k, f] == k$ 的位置）。
     - 將這些位置的 `q_P` 強制覆寫為解析解：`q_P[boundary_idx] = exact_solution(X_boundary, Y_boundary, t)`。
   - **計算迎風懲罰項 (Branchless Upwind)**：
     - 計算介面法向速度 $v_n = n_x u + n_y v$。
     - 懲罰項：`flux_penalty = 0.5 * (v_n - np.abs(v_n)) * (q_M - q_P)`。
4. **表面積分映射回體積 (Surface Assembly)**：
   - 將 `flux_penalty` 乘上 1D 積分權重 `weights_1d` 與表面雅可比 $J_{face}$。
   - 透過提取矩陣的轉置 $E^T$ 將介面數值加回所屬的元素體積節點，最後乘上反質量矩陣 $M_{inv}$。
5. **回傳總和**：`return volume_term + surface_term`，形狀必須維持 `(1, Np, K)`。

---

## ⏱️ 第三階段：全域動態步長與精確時間迭代 (Global Time Stepping)

本階段執行 LSRK54 時間推進，必須加入全局 CFL 搜尋與**嚴格的精確停止機制**。

1. **全域 CFL 動態步長限制**：
   - 步長必須由全域最差條件決定：`dt_global = CFL * np.min(h_min_array) / (V_max_global * N**2)`。
2. **精確停止機制的 LSRK54 迴圈**：
   - 必須嚴格依照以下邏輯撰寫，確保最後一步不會超越 $t_{final}$，且不會產生無窮迴圈：

```python
# 初始設定
t = 0.0
t_final = 1.0
tol = 1e-12    # 用於浮點數比較的安全容差
du = np.zeros_like(Q)

while t < t_final - tol:
    # 核心機制 1：動態步長檢查，確保最後一步精確踩在 t_final
    current_dt = min(dt_global, t_final - t)

    # 每個 Time step 開始前，清空殘差暫存
    du.fill(0.0)

    # LSRK54 的 5 個 Stage
    for stage in range(5):
        # 取得全域 RHS
        R_Q = compute_rhs(Q, t)

        # 低記憶體消耗更新殘差與解
        du = A_RK[stage] * du + current_dt * R_Q
        Q = Q + B_RK[stage] * du

    # 核心機制 2：推進時間使用 current_dt
    t += current_dt
```

---

## 📈 第四階段：空間與時間收斂驗證 (Convergence Validation)

本階段用於驗證多元素引擎的正確性。

1. **空間收斂驗證 (h-refinement)**：
   - 固定多項式階數 $k$（如 $k=3$）。
   - 依序使用不同解析度的網格（例如將三角形均勻切分為 4 個、16 個、64 個）。
   - 計算 $t = t_{final}$ 時的 $L_2$ 與 $L_\infty$ 誤差。
   - 預期結果：誤差衰減率 (Convergence Rate) 應接近 $\mathcal{O}(h^{k+1})$。
2. **誤差計算方式 (Global L2 Error)**：
   - 全域誤差必須是各個元素誤差的質量加權總和的平方根：
     `error_global = np.sqrt( np.sum( (Q_final - Q_exact)**2 * M_diag[:, np.newaxis] ) )`
     *(注意：`M_diag` 包含了每個元素的 $|J|$ 縮放)*
3. **驗證報告輸出**：
   - 繪製 $h$ vs. $L_2$ 誤差的雙對數圖 (Log-Log plot)。
   - 印出最終的 Convergence Rate 表格，確認求解器達到理論精度。

---

### 📊 附錄：收斂驗證表格輸出規範 (Convergence Table Formatting Rules)

為了確保實驗數據的易讀性與論文級別的排版標準，在輸出任何收斂測試（h-refinement 或 p-refinement）的結果時，**絕對禁止使用標準的 `print()` 輸出純文字表格**。

必須嚴格遵守以下 Pandas DataFrame 格式化與輸出限制：

1. **欄位命名與順序**：
   表格必須包含以下五個欄位，名稱需完全一致：
   * `N` (或 `k`，即多項式階數或網格數量)
   * `L2 Error`
   * `L2 C.R.` (L2 Convergence Rate)
   * `Linf Error`
   * `Linf C.R.` (Linf Convergence Rate)

2. **數值格式化要求**：
   * **Error (誤差)**：必須格式化為帶有兩位小數的科學記號（例如：`1.77e-02`）。
   * **C.R. (收斂率)**：必須格式化為帶有兩位小數的浮點數（例如：`3.19`）。
   * **第一列的 C.R.**：因為第一列無法計算收斂率，必須填入字串 `"-"`。

3. **強制使用 `display()` 函數**：
   必須引入 `from IPython.display import display`，並利用 Pandas 的 `Styler` 隱藏 index 後輸出。

**實作範本 (AI Agent 必須參考此結構撰寫輸出邏輯)：**

```python
import pandas as pd
from IPython.display import display

# 假設已計算出以下陣列 (長度皆為 len(N_list))
# N_list, l2_errors, l2_rates, linf_errors, linf_rates
# 注意：rates 陣列的第一個元素應設為 None 或 NaN，以便替換為 "-"

# 1. 處理 C.R. 第一列沒有數值的狀況，格式化為字串
formatted_l2_rates = ["-" if i == 0 else f"{r:.2f}" for i, r in enumerate(l2_rates)]
formatted_linf_rates = ["-" if i == 0 else f"{r:.2f}" for i, r in enumerate(linf_rates)]

# 2. 建立 DataFrame
df_convergence = pd.DataFrame({
    "N": N_list,
    "L2 Error": l2_errors,
    "L2 C.R.": formatted_l2_rates,
    "Linf Error": linf_errors,
    "Linf C.R.": formatted_linf_rates
})

# 3. 使用 Pandas Styler 進行 Error 的科學記號格式化，並隱藏 Index
styled_df = df_convergence.style.format({
    "L2 Error": "{:.2e}",
    "Linf Error": "{:.2e}"
}).hide(axis="index")

# 4. 嚴格規定：必須使用 display 輸出
display(styled_df)
```
