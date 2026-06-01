"""
Skill Loader — Loads SKILL.md instructions for agents at runtime.
Each agent loads its skill file to get system prompt, rules, and behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

_SKILLS_DIR = Path(__file__).parent


def load_skill(agent_name: str) -> Optional[str]:
    """
    Load SKILL.md content for a given agent.

    Args:
        agent_name: Agent name (e.g., "rag_agent", "web_search_agent")

    Returns:
        SKILL.md content as string, or None if not found.
    """
    skill_file = _SKILLS_DIR / f"{agent_name}.SKILL.md"
    if skill_file.exists():
        content = skill_file.read_text()
        logger.debug(f"Loaded skill: {agent_name} ({len(content)} chars)")
        return content
    logger.warning(f"Skill file not found: {skill_file}")
    return None


def load_model_assignment() -> Optional[str]:
    """Load MODEL_ASSIGNMENT.md for provider/model routing reference."""
    model_file = _SKILLS_DIR / "MODEL_ASSIGNMENT.md"
    if model_file.exists():
        return model_file.read_text()
    return None


def get_system_prompt(agent_name: str, fallback: str = "") -> str:
    """
    Get the system prompt for an agent from its SKILL.md.
    Falls back to provided default if skill file not found.
    """
    skill = load_skill(agent_name)
    if skill:
        return skill
    return fallback
