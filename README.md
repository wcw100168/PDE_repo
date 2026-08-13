# AI-Assisted Numerical Mathematics Research Workflow

> **數學證明 × Numerical PDE × Research Software × AI Agent × Reproducibility**

本專案實踐 **Research-as-Code** 理念：將數學研究視為一個可追蹤、可驗證、可版本控制的工程專案，而非散落在 LaTeX、Python、ChatGPT 對話中的碎片。

```
Research Question → Mathematical Model → Proof → Implementation → Experiment → Evidence → Paper
```

所有階段都具備：**可追蹤性（provenance）**、**可版本控制（version control）**、**可驗證性（verification）**、**可重現性（reproducibility）**、**可由 AI 存取的結構化 context**。

---

## 目錄

- [核心理念](#核心理念)
- [快速開始](#快速開始)
- [目錄結構](#目錄結構)
- [三層資料模型](#三層資料模型)
- [數學管理流程](#數學管理流程)
- [Proof Policy — 證明撰寫十條規則](#proof-policy--證明撰寫十條規則)
- [驗證框架](#驗證框架)
- [AI 協作流程](#ai-協作流程)
- [Context 分層](#context-分層)
- [實驗管理](#實驗管理)
- [Git 協作](#git-協作)
- [進度報告與論文生成](#進度報告與論文生成)
- [外部材料處理](#外部材料處理)
- [分階段實施路線](#分階段實施路線)
- [十條核心規則](#十條核心規則)
- [參考文獻](#參考文獻)

---

## 核心理念

本專案**不**把研究視為「寫程式 + 寫數學 + 寫論文 + 偶爾問 AI」。

而是視為一套**統一的研究基礎設施**：

| 數學概念 | 軟體工程對應 |
|---------|------------|
| Definition | Interface |
| Assumption | Precondition |
| Lemma | Dependency / Library |
| Theorem | API Contract |
| Proof | Implementation |
| Counterexample | Failing Test |
| Numerical Experiment | Integration Test |
| Formal Proof (Lean) | Machine-Checked Test |
| Paper | Documentation |
| Git | Version Control |

**核心原則：**

> **Project 是 source of truth；Chat 是工作空間；Agent 是 project interface；Tests / verification 是裁判。**

---

## 快速開始

### 第一次使用

1. **填寫 `context/` 目錄下的核心文件：**
   - `context/project.md` — 用 500–1500 字描述你的研究
   - `context/notation.md` — 列出所有數學符號及其唯一正確解釋
   - `context/assumptions.md` — 列出所有全域假設

2. **把你正在研究的 theorem 拆成 dependency graph：**
   ```
   Theorem T001
   ├── Lemma L001
   │   ├── Definition D001
   │   └── Assumption A001
   ├── Lemma L002
   └── Assumption A002
   ```

3. **為每個 Definition / Lemma / Theorem 建立獨立文件：**
   - 使用 `math/` 目錄下的 `_TEMPLATE.md` 模板
   - 每個文件都包含 YAML frontmatter（ID、dependencies、status）

4. **更新 `context/current_state.md`：** 記錄目前進度

5. **初始化 Git：**
   ```bash
   cd 數學專案測試版
   git init
   git add .
   git commit -m "初始化研究專案骨架"
   ```

### 每日工作流

```
1. 閱讀 context/current_state.md
2. 在 Web Chat 中進行數學推導
3. 將新結果回寫到 math/ 目錄
4. 更新 context/current_state.md
5. 撰寫研究日誌 logs/research_log/
6. git commit
```

---

## 目錄結構

```
數學專案測試版/
│
├── math/                    ← 數學核心：定義、假設、引理、定理、證明
│   ├── definitions/         ← 每個 Definition 一個 .md 文件
│   ├── assumptions/         ← 每個 Assumption 一個 .md 文件
│   ├── lemmas/              ← 每個 Lemma 一個 .md 文件
│   ├── propositions/        ← 每個 Proposition 一個 .md 文件
│   ├── theorems/            ← 每個 Theorem 一個 .md 文件
│   └── proofs/              ← 每個 Proof 一個 .md 文件
│
├── src/                     ← 程式碼實作
│   ├── geometry/            ← 網格生成、座標轉換、度量項
│   ├── operators/           ← SBP 運算子、微分矩陣、插值
│   ├── discretization/      ← 空間離散化、求積
│   ├── solver/              ← 時間積分、RHS 計算
│   ├── boundary/            ← 邊界條件、數值通量
│   └── utils/               ← 輔助函式、I/O、視覺化
│
├── tests/                   ← 測試
│   ├── unit/                ← 單元測試
│   ├── mathematical/        ← 數學不變量測試
│   ├── invariants/          ← 物理不變量（守恆、自由流）
│   ├── convergence/         ← 收斂率測試
│   ├── stability/           ← 穩定性測試
│   └── regression/          ← 回歸測試
│
├── experiments/             ← 數值實驗
│   ├── configs/             ← 實驗配置 YAML
│   ├── runs/                ← 實驗執行結果
│   ├── analysis/            ← 結果分析
│   ├── figures/             ← 圖表
│   └── golden_cases/        ← 黃金標準測試案例
│
├── verification/            ← 驗證證據鏈
│   ├── theorem_checks/      ← 定理驗證記錄
│   ├── numerical_checks/    ← 數值驗證
│   ├── symbolic_checks/     ← 符號驗證（SymPy）
│   ├── counterexamples/     ← 反例資料庫
│   └── reports/             ← 驗證報告
│
├── paper/                   ← 論文
│   ├── manuscript.tex       ← 主文稿
│   ├── sections/            ← 分節文件
│   ├── figures/             ← 論文圖表
│   ├── tables/              ← 論文表格
│   └── bibliography/        ← 參考文獻
│
├── reports/                 ← 報告
│   ├── progress/            ← 給教授的進度報告
│   └── presentations/       ← 簡報
│
├── context/                 ← AI Context 層（最重要的橋樑）
│   ├── project.md           ← 專案全貌（AI 的 README）
│   ├── notation.md          ← 符號契約
│   ├── assumptions.md       ← 全域假設清單
│   ├── theorem_index.md     ← 定理索引（類似 API docs）
│   ├── experiment_index.md  ← 實驗索引
│   ├── current_state.md     ← 當前研究狀態（最常更新）
│   ├── open_questions.md    ← 待解問題
│   ├── proof_policy.md      ← 證明撰寫策略
│   └── verification_policy.md ← 驗證標準
│
├── decisions/               ← 研究決策記錄（ADR）
├── logs/                    ← 研究日誌
│   ├── research_log/        ← 每日研究記錄
│   └── meetings/            ← 會議記錄
│
├── references/              ← 外部參考材料
│   ├── papers/              ← 參考論文
│   ├── collaborators/       ← 同學/合作者的材料
│   └── external_code/       ← 外部程式碼
│
├── raw/                     ← 未整理的原始資料
│   ├── old_code/
│   ├── old_notes/
│   ├── old_papers/
│   └── miscellaneous/
│
└── archive/                 ← 已知舊版本存檔
```

---

## 三層資料模型

所有資料按可信度分為三層：

### 🟥 `raw/` — 原始資料

> 「不知道是否正確，但不想丟掉。」

- Agent 可以讀取，但**不能視為真實來源**
- 舊程式碼、舊筆記、未整理的實驗

### 🟨 `references/` — 外部參考

> 「外部材料，可以參考，但不是本專案的 canonical source。」

- 同學的程式碼、參考論文、外部 implementation
- 需要經過 **notation mapping** 才能引入

### 🟩 `math/` + `src/` + `tests/` + `experiments/` — Canonical

> 「目前我們認定的研究版本。」

- 所有外部材料進入 canonical 的路徑：

```
External → Analysis → Mapping → Verification → Canonical
```

---

## 數學管理流程

### 每個數學構件都有 ID

```
D001, D002, ...    ← Definitions
A001, A002, ...    ← Assumptions
L001, L002, ...    ← Lemmas
P001, P002, ...    ← Propositions
T001, T002, ...    ← Theorems
```

### 建立 Dependency Graph

```
T001
├── L001
│   ├── D001
│   └── A001
├── L002
│   ├── D003
│   └── A003
└── A004
```

### 每個 Lemma / Theorem 都有「Mathematical Contract」

```
Lemma L001: Stability

INPUT:
    u ∈ V_h
    CFL ≤ C
    SBP property

OUTPUT:
    ||u^{n+1}|| ≤ ||u^n||

GUARANTEES:
    Energy stability
```

### 每個 Proof 都拆成帶 ID 的步驟

```
Proof of T001:
  P001.1: Given ... (by A001)
  P001.2: Since ... (by L001)
  P001.3: Applying ... (by D003)
  P001.4: Therefore ... (conclusion)
```

### Proof 翻譯協議

中文 → 英文翻譯時，**逐步翻譯，不重寫**：

```
P001.1 → English
P001.2 → English
P001.3 → English
P001.4 → English
```

---

## Proof Policy — 證明撰寫十條規則

> 詳見 `context/proof_policy.md`

1. **永遠不要靜默地刪除中間步驟。**
2. **永遠不要引入未聲明的假設。**
3. **永遠不要靜默地弱化定理陳述。**
4. **每個非平凡的等式都需要理由。**
5. **每個不等式都需要理由。**
6. **離散與連續運算子不可靜默互換。**
7. **定義不可被近似解釋替代。**
8. **如果存在證明缺口，標記為 `GAP`。**
9. **AI 不能僅憑自身判斷將定理標記為已驗證。**
10. **翻譯必須精確保持數學結構。**

---

## 驗證框架

### 驗證等級 (V0–V5)

| 等級 | 名稱 | 說明 | 可信度 |
|-----|------|------|-------|
| V0 | Unchecked | 尚未檢查 | 不可信 |
| V1 | LLM Audited | AI 審查通過 | 僅供參考 |
| V2 | Symbolically Verified | SymPy / Mathematica 驗證 | 部分數學被機器驗證 |
| V3 | Numerically Supported | 數值實驗支持 | 數值行為一致 |
| V4 | Independently Reviewed | 獨立人工或 AI 審查 | 獨立檢查 |
| V5 | Formally Verified | Lean / Coq / Isabelle | 形式化證明 |

> **注意：** 這些不是嚴格的單調等級，而是不同的 evidence dimension。
> 一個 theorem 可以是 V3 (numerically) 但 V0 (formally)。

### 驗證記錄格式

```yaml
claim: T001
verification:
  - verifier:
      type: human
      name: Alice
    method: manual_review
    date: 2026-08-13
    commit: a81f9c2
    evidence: verification/theorem_checks/T001_alice.md
    status: passed

  - verifier:
      type: llm
      model: Claude-Opus-4.6
    method: proof_audit
    date: 2026-08-13
    commit: a81f9c2
    status: passed
```

### Risk-Based Verification

| 聲明類型 | 最低要求等級 |
|---------|------------|
| 核心收斂/穩定性定理 | V4 或更高 |
| 支援引理（代數） | V2 |
| 實作細節 | V3 |
| 格式/文件 | V1 |

### 反例搜索 (Adversarial Verification)

不要只問「這個 theorem 對嗎？」，而要問「你能不能證明它錯？」

```
Try to disprove the theorem.

Search for:
1. Counterexamples
2. Missing assumptions
3. Boundary cases
4. Degenerate cases
5. Dimension inconsistencies
6. Operator convention mismatch
7. Numerical instability
8. Parameter regimes where conclusion fails
```

---

## AI 協作流程

### 三種工具的分工

| 工具 | 職責 |
|-----|------|
| **Web Chat** | 思考、數學討論、推導、proof review |
| **Agent** | Repository 分析、重構、跨檔案調查、實驗、重現 |
| **Python / Lean** | 實際執行、測試、驗證 |

```
Web  = THINK（思考）
Agent = INVESTIGATE / EXECUTE（調查 / 執行）
Tests = JUDGE（裁判）
Git  = REMEMBER（記憶）
```

### AI 的四種角色

> **不要在同一個 prompt 中混合角色。**

| 角色 | 任務 | 範例 prompt |
|------|------|------------|
| **Architect** | 拆分 theorem dependency | 「把 T001 拆成最小 dependency graph」 |
| **Proof Writer** | 撰寫 proof 細節 | 「不要改變 assumptions，完成 P001.3」 |
| **Proof Auditor** | 只找 logical gap | 「不要修改 proof，只列出 logical gaps」 |
| **Adversarial Reviewer** | 嘗試找 counterexample | 「嘗試推翻 T001」 |

### AI 的權限模型

```
L0 — READ        只能讀
L1 — ANALYZE     可以產生報告
L2 — MODIFY      可以修改 branch
L3 — MERGE       可以進 main（預設不開放）
```

### AI 的不可越權區

```
AI MAY:
  ✓ analyze
  ✓ propose
  ✓ test
  ✓ refactor
  ✓ generate proof draft

AI MAY NOT:
  ✗ change theorem assumptions silently
  ✗ change numerical formulation
  ✗ delete failed experiments
  ✗ overwrite canonical proof
  ✗ modify reference results
  ✗ mark theorem verified without evidence
```

---

## Context 分層

### 為什麼需要 Context 層？

每次開新的 AI 對話，不需要重新教 AI 你的研究。
只需要讓 AI 讀取結構化的 context 文件。

### 四層 Context

| 層級 | 內容 | 何時使用 |
|-----|------|---------|
| **L0 — Global** | `project.md` + `notation.md` + `proof_policy.md` | 永遠提供 |
| **L1 — State** | `current_state.md` + `open_questions.md` | 每次 session |
| **L2 — Local** | 目前 theorem + dependencies | 討論特定問題時 |
| **L3 — Raw** | 完整論文、程式碼、實驗 | 需要時才載入 |

### Web Chat Session 模板

每次開新對話時，使用此模板：

```
PROJECT CONTEXT
[貼上 context/project.md]

NOTATION
[貼上 context/notation.md]

ASSUMPTIONS
[貼上 context/assumptions.md]

CURRENT STATE
[貼上 context/current_state.md]

CURRENT THEOREM
[貼上目標 theorem 文件]

DEPENDENCIES
[貼上相關 lemma 文件]

TASK
We are investigating the proof of T001.

Rules:
- Do not change notation.
- Do not introduce unstated assumptions.
- Explicitly mark logical gaps.
- Distinguish theorem, assumption, observation, and conjecture.
```

---

## 實驗管理

### 每個實驗都有唯一 ID 和配置文件

```yaml
# experiments/configs/EXP042.yaml
experiment:
  id: EXP042
  objective: "Verify fourth-order convergence"
  git_commit: "a81f9c2"

parameters:
  N: [16, 32, 64, 128]
  CFL: 0.4
  final_time: 1.0

result:
  convergence_order: 3.98

status: completed
```

### Golden Case — 黃金標準測試

與同學協作時，建立固定參數的參考結果：

```
experiments/golden_cases/
├── case_01/
│   ├── config.yaml
│   ├── reference_solution.npy
│   └── expected_metrics.json
└── case_02/
    └── ...
```

### 數學不變量 → 可執行測試

```python
# M 必須是對稱正定矩陣
assert np.allclose(M, M.T)

# Jacobian 必須大於零
assert np.all(J > 0)

# Free-stream preservation
assert free_stream_error < 1e-12
```

### 實驗結果不同時：Differential Diagnosis

不要直接叫 AI「找 bug」。沿資料流逐層比較：

```
Input → Grid → Geometry → Metric → Derivative
→ RHS → Boundary → RK Stage 1 → ... → Final Solution
```

找到**第一個產生不同結果的位置**。

---

## Git 協作

### 分支策略

```
main                          ← 保持可用
├── feature/alice-proof-T004  ← 證明工作
├── feature/bob-geometry      ← 程式碼工作
└── experiment/EXP042         ← 實驗工作
```

### Commit 規範

```
git commit -m "T002: proved under periodic BC"
git commit -m "L004: found missing regularity assumption"
git commit -m "EXP042: convergence test N=16..128"
git commit -m "A003: strengthened H² → H³"
```

### 數學也用 `git diff`

```diff
- Assume u ∈ H²(Ω)
+ Assume u ∈ H³(Ω)
```

---

## 進度報告與論文生成

### 報告從 Research State 自動生成

```
Canonical Mathematics + Experiments + Results
              ↓
         Research API
              ↓
   Progress Report / Paper / Slides
```

### 進度報告的資料來源

```
Agent 讀取:
  recent_theorems       → 最近完成的定理
  recent_experiments    → 最近的實驗
  verified_results      → 已驗證的結果
  failures              → 失敗記錄
  open_questions        → 待解問題
  next_steps            → 下一步計畫
```

### 論文不應是唯一真實來源

```
Canonical Math → Canonical Experiments → Verified Results → LaTeX Renderer → Paper
```

避免：`Code changed → Paper manually edited → Paper and code diverge`

---

## 外部材料處理

### 同學的數學符號不同時

1. **不要直接合併。** 先建立 Notation Mapping：

```
references/collaborators/alice/
├── original/              ← 保留原文
└── notation_mapping.md    ← 符號對應表
```

```yaml
Q_i → D_i    equivalence: confirmed
G   → J      equivalence: confirmed
W   → M      equivalence: probable   ← 需要進一步確認
```

2. **按照資料來源、shape、operation、定義判斷語義，不要單靠變數名稱。**

### 別人的論文

不要問「這篇 paper 說了什麼？」，而要問：

```
Extract the mathematical contract:
- theorem statement
- assumptions
- domain
- regularity requirements
- boundary conditions
- conclusion

Compare with our project:
  A1 → satisfied
  A2 → satisfied
  A3 → UNKNOWN   ← 不要讓 AI 猜測
```

### 同學的程式碼

第一階段：**Read-only Reconnaissance**

```
不要修改任何檔案。

分析 repository：
1. Module dependency graph
2. Entry points
3. Data flow
4. Mathematical operations
5. Likely numerical-sensitive locations
6. Unclear or suspicious areas
```

---

## 分階段實施路線

### 🥇 第一階段：現在就做（MVP）

```
✅ math/ + src/ + tests/ + experiments/ + paper/
✅ context/ 核心文件
✅ Git
✅ Research Log
✅ Decision Log
```

### 🥈 第二階段：驗證基礎

```
☐ verification/ 目錄啟用
☐ Golden cases
☐ Mathematical invariants → executable tests
☐ Reproduction workflow
```

### 🥉 第三階段：AI 整合

```
☐ generate_context.py（Context Compiler）
☐ Research API
☐ Automatic progress report
```

### 🏆 第四階段：Research CI

```
☐ Git push → 自動測試
☐ Symbolic checking
☐ Numerical regression
☐ Formal verification (Lean, optional)
```

### 🌟 第五階段：全自動化

```
☐ 自動論文渲染
☐ 自動圖表生成
☐ Claim-Evidence Graph
☐ Research Dashboard
```

### ⚠️ 不建議一開始做的事情

```
✗ 全部數學 formalize 成 Lean
✗ 建造完整 vector database
✗ 自製龐大 Agent framework
✗ 讓 AI 自動判定 theorem VERIFIED
✗ 花大量時間寫 infrastructure 而不做研究
```

> **原則：研究基礎設施的投入不能超過研究本身的收益。**

---

## 十條核心規則

1. **Git repository 是 source of truth。**
2. **Chat history 不是 source of truth。**
3. **外部材料先放 `references/` 或 `raw/`，不直接進 canonical。**
4. **每個 theorem 都有 dependency。**
5. **每個重要結果都要有 provenance。**
6. **Verification 必須記錄 verifier、method、commit、evidence。**
7. **AI 可以提出結論，但不能自己成為它唯一的證據。**
8. **數值 claim 優先使用 executable tests 驗證。**
9. **Web Chat 使用 context layer，不直接吞整個 repository。**
10. **論文、報告、簡報都應盡量從 canonical research state 生成。**

---

## 參考文獻

本 workflow 整合了以下成熟方法論：

### Research Software Engineering

- **A Research Software Engineering Workflow for CSE**
  [arXiv:2208.07460](https://arxiv.org/abs/2208.07460)
  — 針對 computational science 的研究程式工程化流程

### Reproducible Research

- **The Turing Way — Guide for Reproducible Research**
  [book.the-turing-way.org](https://book.the-turing-way.org/reproducible-research/reproducible-research/)
  — 可重現研究的完整指南，涵蓋 Git、測試、文件、協作

- **The Turing Way — Reproducible Project Template**
  [GitHub](https://github.com/the-turing-way/reproducible-project-template)
  — 可直接使用的研究專案模板

- **Ten Simple Rules for Reproducible Computational Research**
  [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285)
  — 可重現計算研究的十條規則

### Scientific Software Quality

- **Ten Simple Rules for Making Research Software More Robust**
  [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412)
  — 讓研究程式更穩健的十條規則

- **Ten Simple Rules for Making a Software Tool Workflow-Ready**
  [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009823)
  — 讓研究軟體適合放入 workflow 的十條規則

### Formal Mathematics / Proof Engineering

- **Lean / Mathlib**
  [leanprover-community.github.io](https://leanprover-community.github.io/contribute/how-to-contribute.html)
  — 大型協作式形式化數學專案，展示「數學 theorem 也可以是 software artifact」

### AI + Scientific Reproduction

- **Coding-agents can replicate scientific machine learning papers**
  [arXiv:2607.02134](https://arxiv.org/abs/2607.02134)
  — 2026 年 Agent 開始把「claim → evidence → reproduction」串起來

- **AutoMat**
  [arXiv:2605.00803](https://arxiv.org/abs/2605.00803)
  — Agent 重現 computational materials science 的成功率與限制

---

## 授權

此專案架構可自由使用與修改。

---

> *本專案骨架根據 [數學證明管理建議.md](數學證明管理建議.md) 中的完整討論設計而成。*
