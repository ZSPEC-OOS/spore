"""Interactive CLI for SPORE.

Run directly:
    python -m spore.cli
  or via the installed entry point:
    spore-cli
"""

from __future__ import annotations

import asyncio
from typing import List

from .engine import LanguageLearningEngine


class LearningInterface:

    def __init__(self) -> None:
        self.engine: LanguageLearningEngine = LanguageLearningEngine()
        self.command_history: List[str] = []

    async def run(self) -> None:
        print(
            """
🧠 SPORE — System for Progressive Online Research & Evolution
=============================================================
Commands:
  learn general    - Start general language acquisition
  learn <topic>    - Specialise in a specific topic
  stop             - Pause learning
  ask <question>   - Test knowledge
  visualize        - Show geometric dashboard launch/status
  status           - Check learning progress
  ai show          - Show AI model config (search/crawl only)
  ai config        - Configure AI model (name, model ID, base URL, API key)
  ai test          - Test AI model connection + web search provider
  exit             - Shutdown
"""
        )

        while True:
            cmd = input("\n> ").strip()
            lower = cmd.lower()
            self.command_history.append(cmd)

            if lower == "exit":
                print("👋 Shutting down SPORE.")
                break

            if lower == "learn general":
                print("🚀 Starting general language learning...")
                await self.engine.start_general_language_learning()
                continue

            if lower.startswith("learn "):
                topic = cmd[6:].strip()
                if topic:
                    print(f"🎯 Specialising in: {topic}")
                    await self.engine.specialize_topic(topic)
                continue

            if lower == "stop":
                self.engine.stop_learning()
                continue

            if lower.startswith("ask "):
                question = cmd[4:].strip()
                print(f"\n🤖 {self.engine.answer_question(question)}")
                continue

            if lower == "visualize":
                ready = self.engine.visualizer.readiness()
                print(
                    "\n".join(
                        [
                            "📊 Geometric Activation Visualizer is the only supported visualizer.",
                            f"Entry point: streamlit run {ready['entrypoint']}",
                            f"Artifacts root: {ready['artifacts_root']}",
                            f"Mind empty but ready: {ready['empty_but_ready']}",
                        ]
                    )
                )
                continue

            if lower == "status":
                print(
                    f"""
Phase:            {self.engine.phase.value}
Memories:         {len(self.engine.memory)}
Tracked concepts: {len(self.engine.concept_frequency)}
Topic:            {self.engine.topic or 'None'}
"""
                )
                continue

            if lower == "ai show":
                config = self.engine.ai_config.as_display_dict()
                print(
                    f"""
AI Model Configuration (Search/Crawl Only):
  Name:      {config['name']}
  Model ID:  {config['model_id']}
  Base URL:  {config['base_url']}
  API Key:   {config['api_key']}
"""
                )
                continue

            if lower == "ai config":
                current = self.engine.ai_config
                print("Leave any field empty to keep the current value.")
                name = input(f"Model Name [{current.name}]: ").strip() or current.name
                model_id = input(f"Model ID [{current.model_id or 'unset'}]: ").strip() or current.model_id
                base_url = input(f"Base URL [{current.base_url or 'unset'}]: ").strip() or current.base_url
                api_key = input("API Key [hidden, press Enter to keep current]: ").strip() or current.api_key
                self.engine.configure_ai_model(name, model_id, base_url, api_key)
                print("✅ AI model configuration updated.")
                continue

            if lower == "ai test":
                print("🧪 Running integration tests...")
                results = await self.engine.test_integrations()
                for label, (ok, message) in results.items():
                    icon = "✅" if ok else "❌"
                    print(f"{icon} {label}: {message}")
                continue

            print(
                "Unknown command. "
                "Try: learn general | learn <topic> | ask <question> | "
                "visualize | status | ai show | ai config | ai test | exit"
            )


def main() -> None:
    asyncio.run(LearningInterface().run())


if __name__ == "__main__":
    main()
