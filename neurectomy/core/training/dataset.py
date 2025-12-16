"""Dataset handling for fine-tuning."""

from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class TrainingSample:
    """Single training sample."""
    prompt: str
    completion: str
    metadata: Dict = field(default_factory=dict)


class Dataset:
    """Training dataset."""
    
    def __init__(self, samples: List[TrainingSample] = None):
        self._samples = samples or []
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self._samples)
    
    def add(self, sample: TrainingSample) -> None:
        """Add a sample to the dataset."""
        self._samples.append(sample)
    
    @classmethod
    def from_jsonl(cls, path: str) -> 'Dataset':
        """Load dataset from JSONL file."""
        samples = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                samples.append(TrainingSample(
                    prompt=data.get("prompt", ""),
                    completion=data.get("completion", ""),
                    metadata=data.get("metadata", {}),
                ))
        return cls(samples)
    
    def to_jsonl(self, path: str) -> None:
        """Save dataset to JSONL file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in self._samples:
                json.dump({
                    "prompt": sample.prompt,
                    "completion": sample.completion,
                    "metadata": sample.metadata,
                }, f)
                f.write("\n")
    
    def split(self, train_ratio: float = 0.9) -> tuple:
        """Split into train/val sets."""
        split_idx = int(len(self._samples) * train_ratio)
        return (
            Dataset(self._samples[:split_idx]),
            Dataset(self._samples[split_idx:]),
        )
    
    def statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self._samples:
            return {
                "total_samples": 0,
                "avg_prompt_length": 0,
                "avg_completion_length": 0,
            }
        
        prompt_lengths = [len(s.prompt.split()) for s in self._samples]
        completion_lengths = [len(s.completion.split()) for s in self._samples]
        
        return {
            "total_samples": len(self._samples),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_completion_length": sum(completion_lengths) / len(completion_lengths),
            "total_tokens": sum(prompt_lengths) + sum(completion_lengths),
        }
