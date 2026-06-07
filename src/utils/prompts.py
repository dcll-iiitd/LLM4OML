# ============================================================================
# File: src/utils/prompts.py
# ============================================================================
"""Prompt templates for proof generation, evaluation, correction, verification."""

from typing import Optional


class PromptManager:
    """Manages all prompt templates used in the pipeline."""

    # ------------------------------------------------------------------
    # GENERATION PROMPT
    # ------------------------------------------------------------------
    GENERATE_PROMPT = r"""
## System Context
You are an expert in optimization algorithms and mathematical proofs. Please reason step-by-step and show your work carefully.

## Reference Examples
Here are examples of algorithm descriptions and their convergence theories and proofs to use as templates:

### Example 1: Stochastic Gradient Descent (SGD)
*Algorithm Description:*
•  Update rule: $w_{{t+1}} = w_t - \eta_t g_t$
•  $g_t = \nabla f(w_t; z_t)$ is a stochastic gradient computed on a random sample $z_t$ at time $t$
•  Learning rate schedule: fixed ($\eta_t = \eta$) or decreasing (e.g., $\eta_t = \frac{{c}}{{\sqrt{{t}}}}$)

*Convergence Proof:*
Assumptions:
1. Objective function $f(w)$ is convex and differentiable
2. Gradients bounded: $\|\nabla f(w)\| \leq G$
3. Stochastic gradients unbiased: $\mathbb{{E}}[g_t | w_t] = \nabla f(w_t)$
4. Variance bound: $\mathbb{{E}}[\|g_t - \nabla f(w_t)\|^2] \leq \sigma^2$

Derivation:
From convexity:
$$f(w_t) - f(w^*) \leq \langle \nabla f(w_t), w_t - w^* \rangle$$

Using update rule and expanding squared norm:
$$\|w_{{t+1}} - w^*\|^2 = \|w_t - \eta_t g_t - w^*\|^2 = \|w_t - w^*\|^2 - 2\eta_t \langle g_t, w_t - w^* \rangle + \eta_t^2 \|g_t\|^2$$

Final convergence rate: $\mathbb{{E}}[f(\bar{{w}}_T) - f(w^*)] = O\left(\frac{{1}}{{\sqrt{{T}}}}\right)$

### Example 2: Adam
*Algorithm Description:*
•  First moment: $m_t = \beta_1 m_{{t-1}} + (1-\beta_1) g_t$
•  Second moment: $v_t = \beta_2 v_{{t-1}} + (1-\beta_2) g_t^2$
•  Bias correction: $\hat{{m}}_t = m_t/(1-\beta_1^t)$, $\hat{{v}}_t = v_t/(1-\beta_2^t)$
•  Update: $w_{{t+1}} = w_t - \eta \hat{{m}}_t / (\sqrt{{\hat{{v}}_t}} + \epsilon)$

Convergence rate: $\frac{{1}}{{T}} \sum_{{t=1}}^T \mathbb{{E}}[\|\nabla f(w_t)\|^2] \leq O\left(\frac{{\log T}}{{T}}\right)$

---

## Main Analysis Task

*ALGORITHM TO ANALYZE:*
{algorithm}

*ASSUMPTIONS:*
{assumptions}

## Required Analysis Components

### 1. Algorithm Details Expansion
1. *Formal Algorithm Description* — precise mathematical formulation, update rules, hyperparameters
2. *Pseudocode* — with initialization, iteration, and termination conditions
3. *Dry Run Example* — apply to $f(w)=(w-3)^2$ with $w_0=0$, $\eta=0.1$, 3 iterations showing gradient computation, update, and objective value

LaTeX structure:
\section*{{Algorithm Name}}
\section*{{Mathematical Formulation}}
\section*{{Pseudocode}}
\section*{{Dry Run Example}}

### 2. Convergence Theory Development
1. *Formal Assumptions* — precise mathematical definitions and parameter constraints
2. *Convergence Theorems* — rate, expected reduction, special cases
3. *Key Theoretical Properties* — stability, hyperparameter sensitivity, comparison to baselines

LaTeX structure:
\section*{{Assumptions}}
\section*{{Main Convergence Theorem}}
\section*{{Theoretical Properties}}
\section*{{Discussion}}

### 3. Complete Convergence Proof
*Required Elements:*
- Full expansion of all terms
- Explicit derivation of every inequality
- Detailed expectation calculations
- Complete algebraic manipulations
- Justification for every inequality/approximation

**CRITICAL: Annotate each non-trivial proof step with a line number in the format [Step N] so that errors can be pinpointed precisely in subsequent evaluation.**

LaTeX structure:
\section*{{Preliminaries (Definitions and Lemmas)}}
\section*{{Main Proof (Step-by-Step Derivation)}}
\section*{{Rate Derivation (With Constants)}}
\section*{{Special Cases}}

*IMPORTANT Requirements:*
- Derive everything from first principles
- Never skip algebraic steps
- Maintain strict adherence to the initial assumptions
- Use the reference examples as templates for mathematical rigor

Provide COMPLETE compilable LaTeX code only. No prose outside LaTeX.
"""

    # ------------------------------------------------------------------
    # EVALUATION PROMPT
    # ------------------------------------------------------------------
    EVALUATE_PROMPT = r"""
You are an expert mathematical reviewer specialising in optimization algorithm convergence proofs.

Evaluate the following LaTeX convergence proof carefully across five dimensions.
For EVERY issue found, record the specific [Step N] number from the proof where it occurs.

PROOF TO EVALUATE:
{proof}

---

## Correctness vs Completeness
First, score the mathematical correctness of the derivations.
Do NOT flag missing algebra, omitted constants, or informal exposition in the correctness verdict.
Only flag the proof as INCORRECT or INCOMPLETE if:
1. It contains false inequalities or invalid deductions
2. Claims a convergence rate different from what is derived
3. Misuses an assumption in a way that invalidates the theorem
4. Uses circular reasoning
5. Has a wrong theorem statement

---

## Evaluation Dimensions

### A. Convergence Rate Extraction
Extract the stated convergence rate (e.g., O(1/√T), O(log T / T), linear ρ^t).

### B. Assumption Usage Analysis
For each assumption listed (A1, A2, ...):
- Is it used in the proof?
- Is it used correctly?
- Flag with [Step N] if misapplied.

### C. Logical Soundness
Identify any of the following — cite exact [Step N]:
- Non-obvious claims without justification
- Missing connections between assumptions and derivations
- Gaps in inequality chains or expectation calculations
- Unjustified use of smoothness, convexity, or stochastic properties

### D. Mathematical Rigor
Flag with [Step N] any:
- Specific parameter choices used to conclude general behaviour
- "Small enough" / "large enough" hand-waving without formal conditions
- Decimal approximations substituted for exact expressions
- Informal probability/expectation arguments

### E. Completeness Assessment
Check for:
- Clear problem setup
- All assumptions introduced before use
- Complete derivation from update rule to convergence bound
- Final convergence guarantee clearly stated

---

## Output Format

Return ONLY valid JSON matching the schema below. No prose outside the JSON block.

```json
{{
  "hallucination_error": true/false,
  "missing_step": true/false,
  "operator_error": true/false,
  "completeness_score": 0-5,
  "assumption_use_score": 0-5,
  "correctness_verdict": "CORRECT|PROBABLY_CORRECT|INCOMPLETE|INCORRECT",
  "critical_errors": ["list of strings describing fatal mathematical errors"],
  "overall_verdict": "PASS|PASS_MINOR|CONDITIONAL|FAIL|REJECT",
  "detailed_feedback": "Full analysis text with [Step N] citations...",
  "hallucination_steps": [list of integer step numbers],
  "missing_step_indices": [list of integer step numbers],
  "operator_error_steps": [list of integer step numbers],
  "assumption_violation_steps": [list of integer step numbers],
  "flagged_steps": [list of integer step numbers to prioritise in correction]
}}
```

Scoring guide:
- completeness_score: 5=no gaps, 4=minor omissions, 3=some incomplete, 2=major gaps, 1=mostly incomplete, 0=empty/broken
- assumption_use_score: 5=all correct, 4=minor issues, 3=some incorrect, 2=multiple errors, 1=fundamental misuse, 0=ignored
- overall_verdict: PASS=publication-ready, PASS_MINOR=solid with trivial issues, CONDITIONAL=fixable gaps,
                   FAIL=significant errors requiring revision, REJECT=fundamental flaws
"""

    # ------------------------------------------------------------------
    # CORRECTION PROMPT
    # ------------------------------------------------------------------
    CORRECT_PROMPT = r"""
You are a mathematical rigor specialist. Your task is to CORRECT and IMPROVE a convergence proof based on detailed evaluation feedback.

---

## Input Materials

### 1. Original Proof Document
{previous_proof}

### 2. Evaluation Feedback
{feedback}

---

## Objectives

### Primary Goal
Generate a CORRECTED LaTeX document that:
1. Fixes ALL issues flagged in the feedback (especially MS, OP, HA items at specific [Step N] citations)
2. Maintains all CORRECT portions of the original proof unchanged
3. Preserves document structure (sections, theorem numbering, formatting)
4. Adds any missing lemmas or intermediate inequalities
5. Ensures mathematical rigor throughout

### Specific Requirements

#### A. Address Each Flagged Step
For each [Step N] cited in the evaluation:
- **MS (Missing Step):** Add the missing derivation, lemma, or intermediate inequality
- **OP (Operator Error):** Correct the mathematical statement
- **HA (Hallucination):** Remove or replace with correct, justified content

#### B. Convergence Rate Consistency
- If stated rate ≠ proved rate: either fix the proof OR adjust the theorem statement
- Add bridging inequalities where needed

#### C. Assumption Usage
- Verify all assumptions (A1, A2, ...) are used correctly
- If cited but not properly applied, add the missing steps

#### D. Preserve Step Annotations
- Keep all [Step N] annotations in the corrected proof so future evaluation rounds can cite them
- Renumber steps if the structure changes significantly

---

## Output Format

Provide the corrected proof as a COMPLETE compilable LaTeX document.

Begin with a comment header:
```
% CORRECTED VERSION
% Changes:
% - [Brief list of 2-3 key corrections]
```

Mark substantial changes inline:
```
% CORRECTED: <reason>
% ADDED: <reason>
```

End with:
```
% ===== CORRECTION SUMMARY =====
% 1. [Step N] Issue: [How fixed]
% 2. [Step N] Issue: [How fixed]
```

Provide COMPLETE compilable LaTeX code only.
"""

    # ------------------------------------------------------------------
    # VERIFICATION PROMPT
    # ------------------------------------------------------------------
    VERIFY_PROMPT = r"""
You are a strict mathematical reviewer. Verify whether the CORRECTED proof (V2) properly addresses the issues identified in the original evaluation (V1).

---

## Input Materials

### 1. Original Evaluation Report (V1)
{old_feedback}

### 2. Corrected Proof (V2)
{current_proof}

---

## Verification Tasks

### Per-Issue Check
For EACH issue flagged in V1, assess:
- Was it fixed? (FIXED / PARTIALLY FIXED / NOT FIXED / NEW ISSUE INTRODUCED)
- If partially fixed, what remains?

### New Error Detection
Scan V2 for newly introduced problems:
- Logical gaps
- Incorrect inequalities
- Inconsistent notation
- Broken references
- Computational errors

Cite all remaining or new errors with [Step N] references.

---

## Output Format

Return ONLY valid JSON matching the schema below. No prose outside the JSON block.

```json
{{
  "hallucination_error": true/false,
  "missing_step": true/false,
  "operator_error": true/false,
  "completeness_score": 0-5,
  "assumption_use_score": 0-5,
  "overall_verdict": "PASS|PASS_MINOR|CONDITIONAL|FAIL|REJECT",
  "detailed_feedback": "Issue-by-issue resolution status plus new error detection with [Step N] citations...",
  "hallucination_steps": [remaining or new hallucination step numbers],
  "missing_step_indices": [remaining or new missing step numbers],
  "operator_error_steps": [remaining or new operator error step numbers],
  "assumption_violation_steps": [remaining or new assumption violation step numbers],
  "flagged_steps": [steps that still need correction or are newly problematic]
}}
```

Be strict: a superficial fix that hides rather than resolves the problem should be rated NOT FIXED.
"""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_generation_prompt(self, algorithm: str, assumptions: str) -> str:
        return self.GENERATE_PROMPT.format(algorithm=algorithm, assumptions=assumptions)

    def get_initial_evaluation_prompt(self, proof: str) -> str:
        return self.EVALUATE_PROMPT.format(proof=proof)

    def get_correction_prompt(self, previous_proof: str, feedback: str) -> str:
        return self.CORRECT_PROMPT.format(previous_proof=previous_proof, feedback=feedback)

    def get_verification_prompt(self, old_feedback: str, current_proof: str) -> str:
        return self.VERIFY_PROMPT.format(old_feedback=old_feedback, current_proof=current_proof)