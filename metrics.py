"""
Metrics collection and monitoring for the agent.
Tracks response times, error rates, token usage, and verification stats.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Single metric entry"""
    timestamp: float = field(default_factory=time.time)
    metric_type: str = ""
    value: Any = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for the agent.
    Can export to various formats or send to monitoring services.
    """
    
    def __init__(self, enabled: bool = True, export_interval: int = 60):
        self.enabled = enabled
        self.export_interval = export_interval
        self.metrics: List[MetricEntry] = []
        self._last_export = time.time()
        self._aggregates = defaultdict(lambda: defaultdict(float))
    
    def record_llm_call(self, 
                        provider: str, 
                        model: str, 
                        duration: float, 
                        tokens_used: int,
                        success: bool = True,
                        error: Optional[str] = None):
        """Record an LLM API call"""
        if not self.enabled:
            return
        
        entry = MetricEntry(
            metric_type="llm_call",
            value=duration,
            tags={
                "provider": provider,
                "model": model,
                "success": str(success)
            },
            metadata={
                "tokens": tokens_used,
                "error": error
            }
        )
        self.metrics.append(entry)
        
        # Update aggregates
        self._aggregates["llm_calls"]["total"] += 1
        self._aggregates["llm_calls"]["success" if success else "failures"] += 1
        self._aggregates["llm_calls"]["tokens"] += tokens_used
        self._aggregates["llm_calls"]["duration"] += duration
    
    def record_verification(self, 
                           question: str, 
                           rounds: int, 
                           corrected: bool,
                           final_verdict: str):
        """Record a verification session"""
        if not self.enabled:
            return
        
        entry = MetricEntry(
            metric_type="verification",
            value=rounds,
            tags={
                "corrected": str(corrected),
                "final_verdict": final_verdict
            },
            metadata={
                "question": question[:100]  # Truncate for storage
            }
        )
        self.metrics.append(entry)
        
        self._aggregates["verifications"]["total"] += 1
        self._aggregates["verifications"]["corrected" if corrected else "uncorrected"] += 1
        self._aggregates["verifications"]["rounds"] += rounds
    
    def record_error(self, error_type: str, error_msg: str, context: Dict = None):
        """Record an error occurrence"""
        if not self.enabled:
            return
        
        entry = MetricEntry(
            metric_type="error",
            value=1,
            tags={"error_type": error_type},
            metadata={
                "message": error_msg,
                "context": context or {}
            }
        )
        self.metrics.append(entry)
        self._aggregates["errors"][error_type] += 1
    
    def get_aggregates(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        return {
            "llm_calls": dict(self._aggregates["llm_calls"]),
            "verifications": dict(self._aggregates["verifications"]),
            "errors": dict(self._aggregates["errors"]),
            "total_metrics": len(self.metrics)
        }
    
    def export_to_json(self, file_path: str):
        """Export metrics to JSON file"""
        data = {
            "export_time": time.time(),
            "aggregates": self.get_aggregates(),
            "metrics": [asdict(m) for m in self.metrics]
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Metrics exported to {file_path}")
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self._aggregates.clear()
        self._last_export = time.time()


# Global metrics collector
metrics = MetricsCollector()