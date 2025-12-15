"""
Analysis Team
=============

8 agents specialized for analysis tasks.
"""

from typing import List

from .base import EliteAgent, TeamCommander, TeamConfig, TeamRole
from ...agents.base import AgentConfig
from ...core.types import TaskRequest, TaskResult, AgentCapability


ANALYSIS_TEAM_CONFIG = TeamConfig(
    team_id="analysis_team",
    team_name="Analysis Team",
    description="Specialized agents for analysis tasks",
    primary_capabilities=[AgentCapability.ANALYSIS],
)


class AnalysisCommander(TeamCommander):
    """Analysis Team Commander."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="analysis_commander",
            agent_name="Analysis Commander",
            agent_type="commander",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.PLANNING],
            system_prompt="You coordinate analysis operations across the team.",
        )
        super().__init__(config, "analysis_team", ANALYSIS_TEAM_CONFIG)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Route analysis tasks."""
        return self.route_task(request)


class SentimentAnalyst(EliteAgent):
    """Sentiment analysis specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="sentiment_analyst",
            agent_name="Sentiment Analyst",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You analyze sentiment in text with high accuracy.",
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Analyze sentiment."""
        text = request.payload.get("text", "")
        prompt = f"Analyze the sentiment of: {text}\n\nSentiment (positive/negative/neutral):"
        result = self.generate(prompt, max_tokens=50)
        return self._create_success_result(request, {"sentiment": result}, result)


class EntityExtractor(EliteAgent):
    """Named entity recognition specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="entity_extractor",
            agent_name="Entity Extractor",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You extract named entities from text.",
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Extract entities."""
        text = request.payload.get("text", "")
        prompt = f"Extract named entities (people, places, organizations) from: {text}\n\nEntities:"
        result = self.generate(prompt)
        return self._create_success_result(request, {"entities": result.split(", ")}, result)


class TopicModeler(EliteAgent):
    """Topic modeling specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="topic_modeler",
            agent_name="Topic Modeler",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You identify topics and themes in text.",
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Model topics."""
        text = request.payload.get("text", "")
        prompt = f"Identify the main topics in: {text}\n\nTopics:"
        result = self.generate(prompt)
        return self._create_success_result(request, {"topics": result.split(", ")}, result)


class SummaryExpert(EliteAgent):
    """Summarization specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="summary_expert",
            agent_name="Summary Expert",
            agent_type="specialist",
            capabilities=[AgentCapability.SUMMARIZATION, AgentCapability.ANALYSIS],
            system_prompt="You create concise, accurate summaries.",
            temperature=0.3,
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Create summary."""
        text = request.payload.get("text", "")
        max_length = request.payload.get("max_length", 100)
        prompt = f"Summarize in {max_length} words or less:\n\n{text}\n\nSummary:"
        summary = self.generate(prompt, max_tokens=max_length * 2)
        return self._create_success_result(request, summary, summary)


class ClassificationAgent(EliteAgent):
    """Text classification specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="classification_agent",
            agent_name="Classification Agent",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You classify text into categories.",
            temperature=0.2,
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Classify text."""
        text = request.payload.get("text", "")
        categories = request.payload.get("categories", ["general"])
        prompt = f"Classify this text into one of: {categories}\n\nText: {text}\n\nClassification:"
        result = self.generate(prompt, max_tokens=20)
        return self._create_success_result(request, {"classification": result.strip()}, result)


class SimilarityMatcher(EliteAgent):
    """Semantic similarity specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="similarity_matcher",
            agent_name="Similarity Matcher",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You compute semantic similarity between texts.",
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Compute similarity."""
        text_a = request.payload.get("text_a", "")
        text_b = request.payload.get("text_b", "")
        
        # Simple Jaccard similarity
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        similarity = intersection / union if union > 0 else 0.0
        
        return self._create_success_result(request, {
            "similarity": round(similarity, 3),
            "method": "jaccard",
        })


class TrendDetector(EliteAgent):
    """Trend analysis specialist."""
    
    def __init__(self):
        config = AgentConfig(
            agent_id="trend_detector",
            agent_name="Trend Detector",
            agent_type="specialist",
            capabilities=[AgentCapability.ANALYSIS],
            system_prompt="You detect trends and patterns in data.",
        )
        super().__init__(config, "analysis_team", TeamRole.SPECIALIST)
    
    def process(self, request: TaskRequest) -> TaskResult:
        """Detect trends."""
        data = request.payload.get("data", [])
        
        if len(data) < 2:
            trend = "insufficient_data"
        elif data[-1] > data[0]:
            trend = "increasing"
        elif data[-1] < data[0]:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return self._create_success_result(request, {
            "trend": trend,
            "data_points": len(data),
        })


def create_analysis_team() -> List[EliteAgent]:
    """Create all Analysis Team agents."""
    commander = AnalysisCommander()
    
    members = [
        SentimentAnalyst(),
        EntityExtractor(),
        TopicModeler(),
        SummaryExpert(),
        ClassificationAgent(),
        SimilarityMatcher(),
        TrendDetector(),
    ]
    
    for member in members:
        commander.add_team_member(member)
    
    return [commander] + members
