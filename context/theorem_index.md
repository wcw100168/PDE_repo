# Theorem Index

<!-- 
  定理/引理索引。
  類似 API documentation，記錄每個數學結果的 statement、dependencies、status。
  AI 在討論某個 theorem 時，只需讀取此索引及其 dependencies。
-->

## Definitions

| ID | Name | File | Dependencies |
|----|------|------|--------------|
| D001 | | `math/definitions/D001.md` | — |

## Assumptions

| ID | Statement (short) | File |
|----|-------------------|------|
| A001 | Domain regularity | `context/assumptions.md` |
| A002 | Solution regularity | `context/assumptions.md` |
| A003 | Mapping smoothness | `context/assumptions.md` |
| A004 | SBP property | `context/assumptions.md` |

## Lemmas

| ID | Name | Dependencies | Status | Verification |
|----|------|-------------|--------|--------------|
| L001 | | D001, A001 | draft | V0 |

## Propositions

| ID | Name | Dependencies | Status | Verification |
|----|------|-------------|--------|--------------|

## Theorems

| ID | Name | Dependencies | Status | Verification |
|----|------|-------------|--------|--------------|
| T001 | | L001, A001–A004 | incomplete | V0 |

---

## Dependency Graph

```
T001
├── L001
│   ├── D001
│   └── A001
├── A002
├── A003
└── A004
```

## Status Legend

- `draft` — 初稿，尚未審查
- `proved` — 證明完成
- `incomplete` — 證明進行中
- `gap` — 存在已知邏輯缺口
- `verified` — 通過驗證
