# Notation Contract

<!-- 
  符號契約：防止 AI 偷換符號。
  此文件定義本專案中所有數學符號的「唯一正確解釋」。
  任何 AI 在處理本專案時都必須遵守此契約。
-->

## Domain & Coordinates

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\Omega$ | Physical domain | |
| $\hat{\Omega}$ | Reference domain | |
| $\xi$ | Reference coordinate | |
| $x$ | Physical coordinate | |

## Operators

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $D_i$ | SBP differentiation matrix in $i$-direction | Discrete operator |
| $M$ | Mass matrix | Symmetric positive definite |

## Solutions

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $u$ | Exact solution | |
| $u_h$ | Numerical solution | |

## Geometry

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $J$ | Jacobian determinant | $J > 0$ always |

---

## ⚠️ IMPORTANT: Forbidden Substitutions

以下替換在本專案中**嚴格禁止**：

```
Do NOT replace:
  J     with det(J)        — J is exclusively the Jacobian determinant
  D     with ∂              — D refers to discrete derivative operator only
  u_h   with u              — u_h and u are semantically different
  ξ     with x              — reference and physical coordinates must not be interchanged
```

## Conventions

- Einstein summation convention: **not** used unless explicitly stated
- Index ranges: stated explicitly for each expression
- Discrete vs continuous: always distinguished; never substitute one for the other
