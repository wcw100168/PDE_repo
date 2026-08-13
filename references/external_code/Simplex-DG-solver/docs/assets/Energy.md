### **第一步：引入投影項的半離散 DG 格式**

考慮原本針對對流方程式的半離散 DG 格式。我們將邊界通量更新項目替換為具有 $L_2$  Galerkin 投影性質的算子：

$$
\partial_t \mathbf{q} = - \frac{1}{2} \left( \mathbf{D}_{\xi} (\mathbf{u} \odot \mathbf{q}) + \mathbf{D}_{\eta} (\mathbf{v} \odot \mathbf{q}) \right) - \frac{1}{2} (\mathbf{u} \odot \mathbf{D}_\xi \mathbf{q} + \mathbf{v} \odot \mathbf{D}_\eta \mathbf{q} ) - \frac{1}{2} (\mathbf{D}_\xi \mathbf{u} + \mathbf{D}_\eta \mathbf{v}) \odot \mathbf{q} + (|T|)^{-1} \mathbf{V} (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

其中，$\mathbf{V}$ 是 Generalized Vandermonde 矩陣，$\mathbf{W}$ 是體積積分正交權重矩陣，$\mathbf{E}^T$ 是將邊界點映射回體積點的 Lift 矩陣，$\mathbf{W}^e$ 是邊界線積分權重矩陣，$\mathbf{p}$ 是透過迎風通量 (Upwind Flux) 計算出的邊界懲罰項。

---

### **第二步：構造 $L_2$ 能量內積**

為了進行能量估計，我們在方程式左右兩側同乘上 $2|T| \mathbf{q}^T \mathbf{W}$。體積項的散度定理推導與原始文件完全相同，我們將焦點放在 **邊界懲罰項 (Boundary Penalty Term)** 的變化上。

令新的邊界能量項為 $B_{proj}$：
$$
B_{proj} = 2|T| \mathbf{q}^T \mathbf{W} \left[ (|T|)^{-1} \mathbf{V} (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p} \right]
$$

消去純量 $|T|$ 與 $(|T|)^{-1}$，得到：
$$
B_{proj} = 2 \mathbf{q}^T \mathbf{W} \mathbf{V} (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

---

### **第三步：投影邊界項的代數消去核心**

此處是證明的關鍵。在 Nodal DG 方法中，離散節點上的解向量 $\mathbf{q}$ 是由 $k$ 階正交多項式空間 $\mathcal{V}_k$ 中的 Modal 係數向量 $\mathbf{a}$ 所展開的。因此，兩者具有嚴格的線性映射關係：
$$ \mathbf{q} = \mathbf{V} \mathbf{a} $$

將 $\mathbf{q}^T = \mathbf{a}^T \mathbf{V}^T$ 代入邊界能量項 $B_{proj}$ 的最前方：
$$
B_{proj} = 2 (\mathbf{a}^T \mathbf{V}^T) \mathbf{W} \mathbf{V} (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

利用矩陣乘法的結合律，我們可以將括號重新分組：
$$
B_{proj} = 2 \mathbf{a}^T \left( \mathbf{V}^T \mathbf{W} \mathbf{V} \right) (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

注意括號中的 $\left( \mathbf{V}^T \mathbf{W} \mathbf{V} \right)$ 正是精確的 Modal 質量矩陣 $\mathbf{M}$。由於其具有正定且可逆的性質 ，矩陣與其反矩陣相乘必定為單位矩陣$\mathbf{I}$：
$$
\left( \mathbf{V}^T \mathbf{W} \mathbf{V} \right) (\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} = \mathbf{I}
$$

因此，方程式大幅化簡為：
$$
B_{proj} = 2 \mathbf{a}^T \mathbf{I} \mathbf{V}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p} = 2 (\mathbf{V} \mathbf{a})^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

將 $\mathbf{V} \mathbf{a}$ 替換回原本的節點解向量 $\mathbf{q}$：
$$
B_{proj} = 2 \mathbf{q}^T \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$

矩陣 $\mathbf{E}^T$ 的作用是從全局體積向量提取出邊界節點，因此 $\mathbf{q}^T \mathbf{E}^T = (\mathbf{q}^e)^T$，最終我們得到：
$$
B_{proj} = 2 (\mathbf{q}^e)^T \mathbf{W}^e \mathbf{p}
$$

**這個結果與原始推導中直接使用 $\mathbf{W}^{-1}$ 所得的邊界項完全一致！**

---

### **第四步：邊界耗散與全域能量穩定性**

既然投影更新所產生的邊界能量項 $2 (\mathbf{q}^e)^T \mathbf{W}^e \mathbf{p}$ 在數學形式上回歸到了原始狀態，後續的能量界定推導便順理成章：

將迎風通量 (Upwind flux, $\tau=0$) 代入邊界懲罰項 $p_i = f_i - f_i^*$，可得：
$$
2(\mathbf{q}^e)^T \mathbf{W}^e \mathbf{p} = \oint_{\partial \mathcal{T}} 2q (f - f^*) \, d \boldsymbol{\xi}
$$

如同原推導所示，我們能保證邊界積分量 $\rho = (-\mathbf{n} \cdot \mathbf{V}) q^2 + 2q(f-f^*)$ 符合以下耗散條件：
* **Outflow (流出邊界):** $\rho \le 0$
* **Inflow (流入邊界):** $\rho \le (-\mathbf{n} \cdot \mathbf{V})g^2$

結合體積散度項（設散度極值為 $\alpha$），系統的全域能量變化率依然滿足：
$$
|T| \frac{d}{dt} (|T| \mathbf{q}^T \mathbf{W} \mathbf{q}) + \alpha |T| (\mathbf{q}^T \mathbf{W} \mathbf{q}) \le - \oint_{\gamma} (\mathbf{n} \cdot \mathbf{V}) g^2 \, d\boldsymbol{\xi}
$$

經過時間積分後，最終的 $L_2$ 能量上界證明得以確立：
$$
(|T| \mathbf{q}^T \mathbf{W} \mathbf{q})(t) \le e^{\alpha t} \left(|T| \mathbf{q}_0^T \mathbf{W} \mathbf{q}_0 + \int_0^t \int_{\gamma} (-\mathbf{n} \cdot \mathbf{V}) g^2(\boldsymbol{\xi},t') \, d \boldsymbol{\xi} \, dt' \right)
$$

---

### **結論：投影矩陣如何維持負定與有界性質？**

從上述證明可以看出一個非常漂亮且深刻的數學對稱性：

1.  **在弱形式 (Weak Form) 測試中表現為單位映射：** 當我們對系統進行能量估計（即在左側乘上 $\mathbf{q}^T \mathbf{W}$ 進行測試）時，因為測試函數 $\mathbf{q}$ 本身就存在於 $k=3$ 的多項式空間 $\mathcal{V}_3$ 內，投影矩陣 $\mathbf{V}(\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1} \mathbf{V}^T$ 識別出 $\mathbf{q}$ 是其子空間的成員。在代數上，這使得投影算子直接退化為等效的單位映射。這保證了理論上的數值耗散與能量穩定性條件完全不受影響，確保了算子的負定/有界性。
2.  **在實際狀態更新中表現為絕對濾波器 (Absolute Filter)：**
    雖然能量「估計」的結果相同，但對於時間步進中的「實際狀態向量」$\partial_t \mathbf{q}$ 而言，這個投影算子會先透過 $\mathbf{V}^T$ 將帶有高頻截斷誤差與 Aliasing 雜訊的 18 維邊界通量 $\mathbf{F}$，強制內積壓縮成 10 維的乾淨 Modal 訊號，再透過 $(\mathbf{V}^T \mathbf{W} \mathbf{V})^{-1}$ 給予正確的質量縮放，最後由 $\mathbf{V}$ 映射回 18 維物理空間。

這意味著，**系統不僅在連續 $L_2$ 範數下維持完美的物理能量穩定（不違背 Trace bounds），同時在離散代數層面上徹底斬斷了讓高頻非多項式噪音（Parasitic Null-space）進入微分算子 $\mathbf{D}_\xi, \mathbf{D}_\eta$ 的路徑**，從而恢復了理論應有的最佳收斂階數 $\mathcal{O}(h^{k+1})$。
