from openai import OpenAI
import os
import json
import tiktoken
from typing import Dict, Optional
import time
from dotenv import load_dotenv
load_dotenv

class AlgorithmAnalyzer:
    # Add class constants for prompts
    DEFAULT_SYSTEM_PROMPT = "You are an expert in optimization algorithms and mathematical proofs. Please reason step-by-step and show your work carefully."
    FEW_SHOT_EXAMPLES = r"""
        \subsection*{Example 1 (Stochastic Gradient Descent - SGD)}
        \textbf{Algorithm Description:}
        - Update rule: $w_{t+1} = w_t - \eta_t g_t$
        - $g_t = \nabla f(w_t; z_t)$ is a stochastic gradient computed on a random sample $z_t$ at time $t$
        - Learning rate schedule: fixed ($\eta_t = \eta$) or decreasing (e.g., $\eta_t = \frac{c}{\sqrt{t}}$)

        \textbf{Convergence Proof:}
        \textit{Assumptions:}
        \begin{enumerate}
        \item Objective function $f(w)$ is convex and differentiable
        \item Gradients bounded: $\|\nabla f(w)\| \leq G$
        \item Stochastic gradients unbiased: $\mathbb{E}[g_t | w_t] = \nabla f(w_t)$
        \item Variance bound: $\mathbb{E}[\|g_t - \nabla f(w_t)\|^2] \leq \sigma^2$
        \end{enumerate}

        \textit{Derivation:}
        From convexity:
        $$
        f(w_t) - f(w^*) \leq \langle \nabla f(w_t), w_t - w^* \rangle
        $$
        Using update rule and expanding squared norm:
        $$
        \|w_{t+1} - w^*\|^2 = \|w_t - \eta_t g_t - w^*\|^2 = \|w_t - w^*\|^2 - 2\eta_t \langle g_t, w_t - w^* \rangle + \eta_t^2 \|g_t\|^2
        $$
        Taking expectation:
        $$
        \mathbb{E}[\|w_{t+1} - w^*\|^2] \leq \mathbb{E}[\|w_t - w^*\|^2] - 2\eta_t \mathbb{E}[f(w_t) - f(w^*)] + \eta_t^2 (\mathbb{E}[\|\nabla f(w_t)\|^2] + \sigma^2)
        $$
        Summing over $t=1$ to $T$, telescoping and rearranging:
        $$
        \sum_{t=1}^T \eta_t \mathbb{E}[f(w_t) - f(w^*)] \leq \frac{\|w_0 - w^*\|^2}{2} + \sum_{t=1}^T \frac{\eta_t^2}{2}(G^2 + \sigma^2)
        $$
        Dividing both sides by $\sum_{t=1}^T \eta_t$, we get:
        $$
        \mathbb{E}[f(\bar{w}_T) - f(w^*)] \leq \frac{\|w_0 - w^*\|^2}{2\sum_{t=1}^T \eta_t} + \frac{(G^2 + \sigma^2)}{2} \cdot \frac{\sum_{t=1}^T \eta_t^2}{\sum_{t=1}^T \eta_t}
        $$
        For $\eta_t = \frac{c}{\sqrt{t}}$, this yields:
        $$
        \mathbb{E}[f(\bar{w}_T) - f(w^*)] = O\left(\frac{1}{\sqrt{T}}\right)
        $$

        \subsection*{Example 2 (Adaptive Moment Estimation - Adam)}
        \textbf{Algorithm Description:}
        - First moment estimate: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
        - Second moment estimate: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
        - Bias correction:
            $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$,
            $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
        - Update rule: $w_{t+1} = w_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

        \textbf{Convergence Proof:}
        \textit{Assumptions:}
        \begin{enumerate}
        \item The objective $f(w)$ is smooth: $\|\nabla f(w) - \nabla f(w')\| \leq L \|w - w'\|$ (L-smooth)
        \item Gradients bounded: $\|g_t\|_\infty \leq G$
        \item Coordinate-wise variance bounded: $\mathbb{E}[g_{t,i}^2] \leq \sigma_i^2$
        \end{enumerate}

        \textit{Derivation:}
        Define $r_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$, then:
        $$
        f(w_{t+1}) \leq f(w_t) - \eta \langle \nabla f(w_t), r_t \rangle + \frac{L\eta^2}{2} \|r_t\|^2
        $$
        Sum over all $t$:
        $$
        \sum_{t=1}^T \langle \nabla f(w_t), r_t \rangle \leq \frac{f(w_1) - f^*}{\eta} + \frac{L\eta}{2} \sum_{t=1}^T \|r_t\|^2
        $$
        Let $d_t = \frac{1}{\sqrt{\hat{v}_t} + \epsilon}$, then:
        $$
        \|r_t\|^2 = \|\hat{m}_t d_t\|^2 = d_t^\top (\hat{m}_t \circ \hat{m}_t)
        $$
        Under ADAM's parameter bounds, it can be shown that:
        $$
        \sum_{t=1}^T \|\nabla f(w_t)\|^2 \leq O\left(\log T\right)
        \Rightarrow \frac{1}{T} \sum_{t=1}^T \mathbb{E}[\|\nabla f(w_t)\|^2] \leq O\left(\frac{\log T}{T}\right)
        $$
        This implies sublinear convergence to stationary points for non-convex functions.

        \subsection*{Example 3 (RMSprop)}
        \textbf{Algorithm Description:}
        - Squared gradient averaging: $v_t = \gamma v_{t-1} + (1 - \gamma) g_t^2$
        - Adaptive learning rate: $w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$
        - Can be seen as diagonal preconditioning using historical second moments

        \textbf{Convergence Proof:}
        \textit{Assumptions:}
        \begin{enumerate}
        \item Convex function $f(w)$
        \item Bounded domain: $\|w - w^*\| \leq D$
        \item Subgradient bounded: $\|g_t\| \leq G$
        \end{enumerate}

        \textit{Regret Analysis:}
        Define regret:
        $$
        R(T) = \sum_{t=1}^T f(w_t) - f(w^*)
        $$
        Using adaptive step size $\eta_t = \frac{\eta}{\sqrt{v_t + \epsilon}}$, and bounding the inner product:
        $$
        \langle g_t, w_t - w^* \rangle \geq f(w_t) - f(w^*)
        $$
        Then, applying online mirror descent analysis:
        $$
        R(T) \leq \frac{D^2}{2\eta} \sum_{i=1}^d \sqrt{v_{T,i}} + \frac{\eta}{2(1 - \gamma)} \sum_{t=1}^T \|g_t\|_\infty^2
        $$
        Hence, for bounded gradients:
        $$
        R(T) = O(\sqrt{T})
        \Rightarrow \frac{1}{T} R(T) = O\left(\frac{1}{\sqrt{T}}\right)
        $$
        """

    def __init__(self, api_key, model, 
                 base_url: str = "https://integrate.api.nvidia.com/v1",  
                 max_context_length: int = 32768,  
                 max_generation_length: int = 24576):  
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_context_length = max_context_length
        self.max_generation_length = min(max_generation_length, max_context_length - 1000)
        self.results = {
            "model": model,
            "algorithm_description": "",
            "assumptions": "",
            "algorithm_details": "",
            "convergence_theory": "",
            "convergence_proof": ""
        }
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = None
            print("Warning: Using fallback token counter")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return max(1, len(text) // 3)

    def safe_call_llm(self, user_prompt: str, 
                    system_prompt: str = "You are an expert in optimization algorithms...", 
                    temperature: float = 0.7, 
                    max_tokens: int = None,
                    retries: int = 3) -> str:
        input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)
        available_tokens = self.max_context_length - input_tokens
        if max_tokens is None:
            max_tokens = min(self.max_generation_length, available_tokens - 100)
        else:
            max_tokens = min(max_tokens, available_tokens - 100)
        if max_tokens <= 100:
            raise ValueError(f"Input too long ({input_tokens} tokens). Max context: {self.max_context_length}")
        
        cot_system_prompt = """
        You are an expert in optimization algorithms and mathematical proofs. 
        Please reason step-by-step and show your work carefully."""
        
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": cot_system_prompt + system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def generate_algorithm_details(self, algorithm_description: str) -> str:
        prompt = f"""
        ## Algorithm Development Task
        Expand this algorithm description into a complete formal specification:
        "{algorithm_description}"

        Your output must include:
        1. **Formal Algorithm Description**:
           - Precise mathematical formulation
           - Update rules and equations
           - Hyperparameters and their roles
           
        2. **Pseudocode**:
           - Implementable pseudocode with clear steps
           - Include initialization, iteration, and termination conditions
           
        3. **Dry Run Example**:
           - Apply the algorithm to a simple quadratic function: f(w) = (w - 3)^2
           - Show 3 iterations with:
               - Initialization: w0 = 0
               - Step size: η = 0.1
               - Minibatch size: 1 (since it's deterministic in this case)
           - Show calculations for each step:
               - Gradient computation
               - Parameter update
               - Objective value

        Present the complete algorithm specification in this structure:
        \\section*{{Algorithm Name}}
        \\section*{{Mathematical Formulation}}
        \\section*{{Pseudocode}}
        \\section*{{Dry Run Example}}
        """
        return self.safe_call_llm(prompt, max_tokens=self.max_generation_length)

    def generate_convergence_theory(self, algorithm_description: str, assumptions: str) -> str:
        prompt = f"""
        ## Convergence Theory Development
        Given this algorithm:
        "{algorithm_description}"

        And these assumptions:
        "{assumptions}"

        Develop a rigorous convergence theory including:

        1. **Formal Statement of Assumptions**:
           - Precise mathematical definitions
           - Parameter constraints
           - Function properties

        2. **Convergence Theorems**:
           - Rate of convergence (e.g., O(1/T), linear, etc.)
           - Expected reduction properties
           - Special case behaviors

        3. **Key Theoretical Properties**:
           - Stability analysis
           - Sensitivity to hyperparameters
           - Comparison to baseline methods

        Present as a complete LaTeX document with:
        \\section*{{Assumptions}}
        \\section*{{Main Convergence Theorem}}
        \\section*{{Theoretical Properties}}
        \\section*{{Discussion}}
        """
        return self.safe_call_llm(prompt, max_tokens=self.max_generation_length)

    def generate_convergence_proof(self, algorithm_description: str, 
                                 convergence_theory: str, 
                                 assumptions: str) -> str:
        system_prompt = (
            self.DEFAULT_SYSTEM_PROMPT + 
            "\n\nHere are examples of algorithm descriptions and their convergence theories and proofs. " 
            "Use these examples as templates for your response:\n\n" + 
            self.FEW_SHOT_EXAMPLES
        )
        
        prompt = f"""
        ## Convergence Theory Development
        Given this algorithm:
        "{algorithm_description}"

        And these assumptions:
        "{assumptions}"

        Develop a rigorous convergence theory including:

        1. **Formal Statement of Assumptions**:
           - Precise mathematical definitions
           - Parameter constraints
           - Function properties

        2. **Convergence Theorems**:
           - Rate of convergence (e.g., O(1/T), linear, etc.)
           - Expected reduction properties
           - Special case behaviors

        3. **Key Theoretical Properties**:
           - Stability analysis
           - Sensitivity to hyperparameters
           - Comparison to baseline methods

        Present as a complete LaTeX document with:
        \\section*{{Assumptions}}
        \\section*{{Main Convergence Theorem}}
        \\section*{{Theoretical Properties}}
        \\section*{{Discussion}}
        """
        return self.safe_call_llm(
            prompt, 
            system_prompt=system_prompt,
            max_tokens=self.max_generation_length
        )

    def generate_convergence_proof(self, algorithm_description: str, 
                                 convergence_theory: str, 
                                 assumptions: str) -> str:
        # Build enhanced system prompt with examples
        system_prompt = (
            self.DEFAULT_SYSTEM_PROMPT + 
            "\n\nHere are examples of algorithm descriptions and their convergence proofs. " 
            "Use these examples as templates for your response:\n\n" + 
            self.FEW_SHOT_EXAMPLES
        )
        
        prompt = f"""
        ## Convergence Proof Development
        Given:
        1. Algorithm: "{algorithm_description}"
        2. Assumptions: "{assumptions}"
        3. Convergence Theory: "{convergence_theory[:4000]}"

        Provide a COMPLETE, STEP-BY-STEP convergence proof with:

        - Full mathematical expansions of all terms
        - Explicit derivation of inequalities
        - Detailed expectation calculations
        - Complete algebraic manipulations
        - Justification for every inequality/approximation

        Structure:
        \\section*{{Preliminaries (Definitions and Lemmas)}}
        \\section*{{Main Proof (Step-by-Step Derivation)}}
        \\section*{{Rate Derivation (With Constants)}}
        \\section*{{Special Cases}}

        IMPORTANT: 
        - Derive everything from first principles
        - Never skip algebraic steps
        - Maintain strict adherence to the initial assumptions
        """
        return self.safe_call_llm(
            prompt, 
            system_prompt=system_prompt,
            max_tokens=self.max_generation_length,
            temperature=0.3  # Lower temp for precision
        )
    
    def analyze_algorithm(self, algorithm_description: str, assumptions: str) -> Dict[str, str]:
        self.results["algorithm_description"] = algorithm_description
        self.results["assumptions"] = assumptions
        
        print("Generating algorithm details...")
        self.results["algorithm_details"] = self.generate_algorithm_details(algorithm_description)
        
        print("Generating convergence theory...")
        self.results["convergence_theory"] = self.generate_convergence_theory(
            algorithm_description, assumptions
        )
        
        print("Generating convergence proof...")
        self.results["convergence_proof"] = self.generate_convergence_proof(
            algorithm_description,
            self.results["convergence_theory"],
            assumptions
        )
        
        return self.results

    def save_results(self, filepath: str = "algorithm_analysis_7f.json") -> None:
    # def save_results(self, filepath: str = "algorithm_analysis_qwen.json") -> None:
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {filepath}")
        return filepath

def main():
    # Configuration - use your actual API key
    api_key = os.getenv("nim_api_key")
    base_url = "https://integrate.api.nvidia.com/v1"
    
    analyzer = AlgorithmAnalyzer(
        api_key=api_key,
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        # model="qwen/qwen3-235b-a22b",
        # qwen/qwen3-235b-a22b
        # nvidia/llama-3.3-nemotron-super-49b-v1
        base_url=base_url
    )
    
    # User-provided inputs
    algorithm_description = (
        "The algorithm combines momentum-based updates with adaptive learning rates, maintaining exponential moving averages of both gradients (first moment) and squared gradients (second moment). At each iteration, it computes bias-corrected moment estimates and uses these to update parameters with momentum-driven steps. It is designed for smooth nonconvex optimization problems where the stochastic gradients have bounded variance."
    )
    
    assumptions = (
        "smooth and nonconvex functions with bounded second-order moment of stochastic gradient."
    )

    # Run the analysis pipeline
    results = analyzer.analyze_algorithm(algorithm_description, assumptions)
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Algorithm Description Length: {len(algorithm_description)} chars")
    print(f"Algorithm Details Length: {len(results['algorithm_details'])} chars")
    print(f"Convergence Theory Length: {len(results['convergence_theory'])} chars")
    print(f"Convergence Proof Length: {len(results['convergence_proof'])} chars")

if __name__ == "__main__":
    main()