"""AI model configuration for SPORE.

AIModelConfig holds credentials and endpoints for the optional external AI
model used only for search-query enrichment during web crawls.  It is NOT
used to generate SPORE's own responses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class AIModelConfig:
    name:     str = "Default Search Model"
    model_id: str = ""
    base_url: str = ""
    api_key:  str = ""

    @classmethod
    def from_env(cls) -> "AIModelConfig":
        """Load config from SPORE_AI_* environment variables."""
        return cls(
            name=os.getenv("SPORE_AI_MODEL_NAME", "Default Search Model"),
            model_id=os.getenv("SPORE_AI_MODEL_ID", ""),
            base_url=os.getenv("SPORE_AI_BASE_URL", ""),
            api_key=os.getenv("SPORE_AI_API_KEY", ""),
        )

    def as_display_dict(self) -> Dict[str, str]:
        """Return a safe display representation (API key masked)."""
        masked = "••••••••" if self.api_key else "(not set)"
        return {
            "name":     self.name     or "(not set)",
            "model_id": self.model_id or "(not set)",
            "base_url": self.base_url or "(not set)",
            "api_key":  masked,
        }
