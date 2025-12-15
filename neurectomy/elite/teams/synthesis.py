"""
Synthesis Team
==============

8 agents specialized for content synthesis.
"""

from typing import List

from .base import EliteAgent, TeamCommander, TeamConfig, TeamRole
from ...agents.base import AgentConfig
from ...core.types import TaskRequest, TaskResult, AgentCapability


SYNTHESIS_TEAM_CONFIG = TeamConfig(
    team_id="synthesis_team",
    team_name="Synthesis Team",
    description="Specialized agents for content synthesis",
    primary_capabilities=[AgentCapability.SYNTHESIS],
)


class SynthesisCommander(TeamCommander):
    """Synthesis Team Commander."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="synthesis_commander",
            agent_name="Synthesis Commander",
            agent_type="commander",
            capabilities=[AgentCapability.SYNTHESIS, AgentCapability.PLANNING],
            system_prompt="You coordinate content synthesis operations.",
        )
        super().__init__(config, "synthesis_team", SYNTHESIS_TEAM_CONFIG)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Route synthesis tasks."""
        return self.route_task(request)


class ContentCreator(EliteAgent):
    """General content creation specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="content_creator",
            agent_name="Content Creator",
            agent_type="specialist",
            capabilities=[AgentCapability.SYNTHESIS],
            system_prompt="You create high-quality, engaging content.",
            temperature=0.7,
        )
        super().__init__(config, "synthesis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Create content."""
        topic = request.payload.get("topic", "")
        style = request.payload.get("style", "informative")
        prompt = f"Write {style} content about: {topic}"
        content = self.generate(prompt)
        return self._create_success_result(request, content, content)


class CodeCrafter(EliteAgent):
    """Code generation specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="code_crafter",
            agent_name="Code Crafter",
            agent_type="specialist",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.SYNTHESIS],
            system_prompt="You write clean, efficient, well-documented code.",
            temperature=0.2,
        )
        super().__init__(config, "synthesis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Generate code."""
        description = request.payload.get("description", "")
        language = request.payload.get("language", "python")
        prompt = f"Write {language} code for: {description}\n\nCode:"
        code = self.generate(prompt)
        return self._create_success_result(request, code, code)


class TranslationExpert(EliteAgent):
    """Language translation specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="translation_expert",
            agent_name="Translation Expert",
            agent_type="specialist",
            capabilities=[AgentCapability.TRANSLATION, AgentCapability.SYNTHESIS],
            system_prompt="You provide accurate, natural translations.",
            temperature=0.3,
        )
        super().__init__(config, "synthesis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Translate text."""
        text = request.payload.get("text", "")
        source = request.payload.get("source_language", "auto")
        target = request.payload.get("target_language", "English")
        prompt = f"Translate from {source} to {target}:\n\n{text}\n\nTranslation:"
        translated = self.generate(prompt)
        return self._create_success_result(request, translated, translated)


class StyleAdapter(EliteAgent):
    """Style transfer specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="style_adapter",
            agent_name="Style Adapter",
            agent_type="specialist",
            capabilities=[AgentCapability.SYNTHESIS],
            system_prompt="You adapt text to different styles and tones.",
        )
        super().__init__(config, "synthesis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Adapt style."""
        text = request.payload.get("text", "")
        target_style = request.payload.get("style", "formal")
        prompt = f"Rewrite in a {target_style} style:\n\n{text}\n\nAdapted:"
        adapted = self.generate(prompt)
        return self._create_success_result(request, {"style": target_style, "text": adapted}, adapted)


class FormatConverter(EliteAgent):
    """Format conversion specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="format_converter",
            agent_name="Format Converter",
            agent_type="specialist",
            capabilities=[AgentCapability.SYNTHESIS],
            system_prompt="You convert content between formats.",
        )
        super().__init__(config, "synthesis_team", TeamRole.SUPPORT)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Convert format."""
        content = request.payload.get("content", "")
        source_format = request.payload.get("source", "text")
        target_format = request.payload.get("target", "markdown")
        prompt = f"Convert from {source_format} to {target_format}:\n\n{content}\n\nConverted:"
        converted = self.generate(prompt)
        return self._create_success_result(request, {"format": target_format, "content": converted}, converted)


class QualityAssurer(EliteAgent):
    """Content quality assurance specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="quality_assurer",
            agent_name="Quality Assurer",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.SYNTHESIS],
            system_prompt="You evaluate and improve content quality.",
        )
        super().__init__(config, "synthesis_team", TeamRole.SUPPORT)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Assess quality."""
        content = request.payload.get("content", "")
        
        # Simple quality heuristics
        word_count = len(content.split())
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
        
        quality_score = min(1.0, word_count / 100) * 0.3  # Length component
        quality_score += 0.7 if 10 < avg_sentence_length < 25 else 0.3  # Readability
        
        return self._create_success_result(request, {
            "quality_score": round(quality_score, 2),
            "word_count": word_count,
            "readability": "good" if 10 < avg_sentence_length < 25 else "review",
        })


class OutputPolisher(EliteAgent):
    """Final output refinement specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="output_polisher",
            agent_name="Output Polisher",
            agent_type="specialist",
            capabilities=[AgentCapability.SYNTHESIS],
            system_prompt="You polish and refine output for final delivery.",
            temperature=0.3,
        )
        super().__init__(config, "synthesis_team", TeamRole.SUPPORT)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Polish output."""
        text = request.payload.get("text", "")
        prompt = f"Polish and improve this text for clarity and flow:\n\n{text}\n\nPolished:"
        polished = self.generate(prompt)
        return self._create_success_result(request, polished, polished)


def create_synthesis_team() -> List[EliteAgent]:
    """Create all Synthesis Team agents."""
    commander = SynthesisCommander()
    
    members = [
        ContentCreator(),
        CodeCrafter(),
        TranslationExpert(),
        StyleAdapter(),
        FormatConverter(),
        QualityAssurer(),
        OutputPolisher(),
    ]
    
    for member in members:
        commander.add_team_member(member)
    
    return [commander] + members
