"""
Specialized Agents
==================

Pre-built agents for common tasks.
"""

from typing import Optional

from .base import BaseAgent, AgentConfig
from ..core.types import TaskRequest, TaskResult, AgentCapability


class InferenceAgent(BaseAgent):
    """
    Agent specialized for text generation/inference.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                agent_name="InferenceAgent",
                agent_type="inference",
                capabilities=[
                    AgentCapability.INFERENCE,
                    AgentCapability.SYNTHESIS,
                ],
                system_prompt="You are a helpful AI assistant.",
            )
        super().__init__(config)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Process inference request."""
        prompt = request.payload.get("prompt", "")
        max_tokens = request.payload.get("max_tokens", self.config.max_output_tokens)
        
        try:
            text = self.generate(prompt, max_tokens)
            return self._create_success_result(request, text, text)
        except Exception as e:
            return self._create_error_result(request, str(e))


class SummarizationAgent(BaseAgent):
    """
    Agent specialized for text summarization.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                agent_name="SummarizationAgent",
                agent_type="summarization",
                capabilities=[
                    AgentCapability.SUMMARIZATION,
                    AgentCapability.ANALYSIS,
                ],
                system_prompt=(
                    "You are an expert at summarizing text. "
                    "Provide clear, concise summaries that capture key points."
                ),
                temperature=0.3,  # More deterministic
            )
        super().__init__(config)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Process summarization request."""
        text = request.payload.get("text", "")
        max_length = request.payload.get("max_length", 200)
        style = request.payload.get("style", "concise")
        
        prompt = self._build_summarization_prompt(text, max_length, style)
        
        try:
            summary = self.generate(prompt, max_tokens=max_length * 2)
            return self._create_success_result(request, summary, summary)
        except Exception as e:
            return self._create_error_result(request, str(e))
    
    def _build_summarization_prompt(
        self,
        text: str,
        max_length: int,
        style: str,
    ) -> str:
        """Build summarization prompt."""
        return f"""Summarize the following text in a {style} style.
Keep the summary under {max_length} words.

Text to summarize:
{text}

Summary:"""


class CodeAgent(BaseAgent):
    """
    Agent specialized for code generation and analysis.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                agent_name="CodeAgent",
                agent_type="code",
                capabilities=[
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.ANALYSIS,
                ],
                system_prompt=(
                    "You are an expert programmer. "
                    "Write clean, efficient, well-documented code. "
                    "Explain your code when asked."
                ),
                temperature=0.2,  # Very deterministic for code
            )
        super().__init__(config)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Process code request."""
        task = request.payload.get("task", "generate")
        
        if task == "generate":
            return self._handle_generate(request)
        elif task == "explain":
            return self._handle_explain(request)
        elif task == "review":
            return self._handle_review(request)
        else:
            return self._create_error_result(request, f"Unknown code task: {task}")
    
    def _handle_generate(self, request: TaskRequest) -> TaskResult:
        """Handle code generation."""
        description = request.payload.get("description", "")
        language = request.payload.get("language", "python")
        
        prompt = f"""Write {language} code for the following:

{description}

Provide clean, well-documented code:"""
        
        try:
            code = self.generate(prompt)
            return self._create_success_result(request, code, code)
        except Exception as e:
            return self._create_error_result(request, str(e))
    
    def _handle_explain(self, request: TaskRequest) -> TaskResult:
        """Handle code explanation."""
        code = request.payload.get("code", "")
        
        prompt = f"""Explain the following code in detail:

```
{code}
```

Explanation:"""
        
        try:
            explanation = self.generate(prompt)
            return self._create_success_result(request, explanation, explanation)
        except Exception as e:
            return self._create_error_result(request, str(e))
    
    def _handle_review(self, request: TaskRequest) -> TaskResult:
        """Handle code review."""
        code = request.payload.get("code", "")
        
        prompt = f"""Review the following code for issues, improvements, and best practices:

```
{code}
```

Code Review:"""
        
        try:
            review = self.generate(prompt)
            return self._create_success_result(request, review, review)
        except Exception as e:
            return self._create_error_result(request, str(e))


class ReasoningAgent(BaseAgent):
    """
    Agent specialized for complex reasoning and planning.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                agent_name="ReasoningAgent",
                agent_type="reasoning",
                capabilities=[
                    AgentCapability.REASONING,
                    AgentCapability.PLANNING,
                    AgentCapability.ANALYSIS,
                ],
                system_prompt=(
                    "You are an expert at logical reasoning and problem-solving. "
                    "Think step by step. Consider multiple perspectives. "
                    "Explain your reasoning clearly."
                ),
                temperature=0.5,
            )
        super().__init__(config)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Process reasoning request."""
        problem = request.payload.get("problem", "")
        approach = request.payload.get("approach", "step_by_step")
        
        prompt = self._build_reasoning_prompt(problem, approach)
        
        try:
            reasoning = self.generate(prompt)
            return self._create_success_result(request, reasoning, reasoning)
        except Exception as e:
            return self._create_error_result(request, str(e))
    
    def _build_reasoning_prompt(self, problem: str, approach: str) -> str:
        """Build reasoning prompt."""
        if approach == "step_by_step":
            return f"""Problem: {problem}

Think through this step by step:
1."""
        elif approach == "pros_cons":
            return f"""Analyze the following, listing pros and cons:

{problem}

Analysis:"""
        else:
            return f"""Analyze and reason about the following:

{problem}

Reasoning:"""
