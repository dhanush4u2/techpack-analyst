"""Optional LLM verification hooks for candidate diffs."""
from __future__ import annotations

from typing import Dict


def verify_change(candidate_diff: Dict, config: Dict) -> Dict:
    """Return the candidate diff untouched unless verification is enabled.

    Integrators can plug Gemini/GPT calls here. The hook is kept synchronous to
    simplify orchestration; wrap async calls externally when necessary.
    """

    if not config.get("llm_verification_enabled", False):
        return candidate_diff
    # Placeholder: In production, fire an API call and annotate the diff object.
    candidate_diff.setdefault("verification", {})
    candidate_diff["verification"]["status"] = "skipped"
    return candidate_diff
