# Proof Policy

<!-- 
  證明撰寫策略。
  所有 AI（Web Chat / Agent）在處理本專案的數學證明時，都必須遵守以下規則。
  此文件應在每次 AI session 開始時作為 L0 context 提供。
-->

## Core Rules

1. **Never remove an intermediate mathematical step silently.**
   每一個推導步驟都必須保留，除非使用者明確要求簡化。

2. **Never introduce an unstated assumption.**
   所有使用的假設必須在 `assumptions.md` 中有明確列出。

3. **Never weaken a theorem statement without explicitly reporting it.**
   如果需要弱化定理陳述，必須明確說明並記錄。

4. **Every nontrivial equality requires justification.**
   引用具體的 lemma、definition 或已知結果。

5. **Every inequality requires justification.**
   說明使用了什麼不等式（Cauchy-Schwarz、Young's、etc.）。

6. **Continuous and discrete operators must not be interchanged silently.**
   離散運算子和連續運算子在語義上不同，不可混用。

7. **Definitions must not be replaced by approximate interpretations.**
   使用精確定義，不可用近似解釋替代。

8. **If a proof gap exists, explicitly mark it as `GAP`.**
   格式：`GAP(P001.3): description of the gap`

9. **AI must not mark a theorem as mathematically verified solely by its own judgement.**
   AI 審查只能標記為 V1（LLM Audited），不能等同於 V4/V5。

10. **Translation must preserve mathematical structure and assumptions exactly.**
    翻譯時不得改變數學結構、假設或推導步驟。

## Proof Structure

每個 proof 應拆成帶 ID 的步驟：

```
Proof of T001:
  P001.1: Given ... (by A001)
  P001.2: Since ... (by L001)
  P001.3: Applying ... (by D003)
  P001.4: Therefore ... (conclusion)
```

## Translation Protocol

中文 → 英文翻譯時：

1. 每一步分別翻譯（P1 → English, P2 → English, ...）
2. 不要「重寫」或「美化」proof
3. 保持步驟 ID 對應
4. 保持所有數學符號不變

## AI Role Separation

不要在同一個 prompt 中混合以下角色：

- **Architect**: 拆分 theorem dependency
- **Proof Writer**: 撰寫 proof 細節
- **Proof Auditor**: 只找 logical gap，不修改 proof
- **Adversarial Reviewer**: 嘗試推翻 theorem
- **Translator**: 只翻譯，不改變數學結構
