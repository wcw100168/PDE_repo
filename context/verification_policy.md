# Verification Policy

<!-- 
  驗證標準與等級定義。
  定義多人協作時的驗證流程、記錄格式、以及各等級的含義。
-->

## Verification Levels

| Level | Name | Description | 可信程度 |
|-------|------|-------------|----------|
| V0 | Unchecked | 尚未經過任何檢查 | 不可信 |
| V1 | LLM Audited | AI 審查通過 | 僅供參考 |
| V2 | Symbolically Verified | SymPy / Mathematica 驗證 | 部分數學被機器驗證 |
| V3 | Numerically Supported | 數值實驗支持 | 數值行為一致 |
| V4 | Independently Reviewed | 獨立人工或獨立 AI 審查 | 獨立檢查 |
| V5 | Formally Verified | Lean / Coq / Isabelle | 形式化證明 |

> **注意：** 這些不是嚴格的單調等級，而是不同的 evidence dimension。
> 一個 theorem 可以是 V3 (numerically) 但 V0 (formally)。

## Verification Record Format

每個 verification 記錄必須回答：

```yaml
claim: T001                    # WHAT — 驗證了什麼
verification:
  - verifier:
      type: human              # WHO — 誰驗證
      name: Alice
    method: manual_review      # HOW — 怎麼驗證
    date: 2026-08-13           # WHEN — 何時
    commit: a81f9c2            # WHICH VERSION — 哪個版本
    evidence: verification/... # EVIDENCE — 證據在哪
    status: passed             # RESULT — 結果
```

## Rules for Mathematical Claims

A theorem **cannot** be marked VERIFIED unless:

1. At least one human review OR formal verification exists.
2. The verified commit is recorded.
3. Required assumptions are explicitly listed.

## Rules for Numerical Claims

A numerical result requires:

1. Reproducible experiment with recorded parameters.
2. Git commit hash.
3. Raw data or generated artifact preserved.

## Rules for AI Review

- LLM review may identify errors but **cannot by itself upgrade a claim to VERIFIED**.
- Different AI models reviewing independently contribute more confidence.
- AI model name and version must be recorded.

## Risk-Based Verification

| Claim Type | Minimum Required Level |
|-----------|----------------------|
| Core convergence/stability theorem | V4 or higher |
| Supporting lemma (algebraic) | V2 |
| Implementation detail | V3 |
| Formatting / documentation | V1 |
