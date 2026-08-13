# Theorem Index

## Definitions

| ID | Name | File | Dependencies |
|----|------|------|--------------|
| D001 | Radial Subdivided Octahedral Sphere Mapping | `math/definitions/D001_mapping.md` | A001, A002 |
| D002 | Preconditioned Cholesky Modal Orthogonalization | `math/definitions/D002_cholesky.md` | A004 |
| D003 | Closed-Form SBP Operator | `math/definitions/D003_sbp_operator.md` | A003 |

## Assumptions

| ID | Statement (short) | File |
|----|-------------------|------|
| A001 | Domain & Mesh Topology | `context/assumptions.md` |
| A002 | Radial Geometry Mapping | `context/assumptions.md` |
| A003 | Multidimensional SBP Operator Property | `context/assumptions.md` |
| A004 | Modal Cholesky Orthogonalization | `context/assumptions.md` |
| A005 | Peirce Subspace Decomposition | `context/assumptions.md` |

## Theorems

| ID | Name | Dependencies | Status | Verification |
|----|------|-------------|--------|--------------|
| T2.1 | Space Congruence Isomorphism Theorem | D002, D003, A004 | proved | V3 (Numerical $10^{-16}$) |
| T3.2 | Peirce Subspace Decomposition Theorem | D003, A005 | proved | V2 (Symbolic & Human) |
| T3.3 | Boundary Constraint System Uniqueness Theorem | T3.2, D003, A005 | proved | V2 (Symbolic & Human) |
| T4.1 | Split2 Two-Term Energy Stability Theorem | D001, D003, A003 | proved | V3 (Numerical $10^{-16}$) |
| T4.2 | Split3 Three-Term Mass Conservation Theorem | D001, D003, A003 | proved | V3 (Numerical $10^{-16}$) |

---

## Dependency Graph

```
T2.1 (Space Congruence Isomorphism)
├── D002 (Preconditioned Cholesky)
│   └── A004
└── D003 (Closed-Form SBP)
    └── A003

T3.3 (Boundary Constraint Uniqueness)
└── T3.2 (Peirce Subspace Decomposition)
    ├── D003 (Closed-Form SBP)
    └── A005

T4.1 (Split2 Energy Stability) & T4.2 (Split3 Mass Conservation)
├── D001 (Sphere Radial Mapping)
├── D003 (Closed-Form SBP)
└── A003 (SBP Property)
```

## Status Legend

- `draft` — 初稿，尚未審查
- `proved` — 證明完成，經推導無誤
- `incomplete` — 證明進行中
- `gap` — 存在已知邏輯缺口
- `verified` — 通過多重驗證 (V1 LLM, V2 Symbolic, V3 Numerical)
