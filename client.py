"""
RF Spectrum Allocation - Client
================================
WebSocket-based EnvClient for interacting with the Spectrum environment.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SpectrumAction, SpectrumObservation, SpectrumState


class SpectrumEnv(EnvClient[SpectrumAction, SpectrumObservation, SpectrumState]):
    """Client for the RF Spectrum Allocation environment."""

    def _step_payload(self, action: SpectrumAction) -> dict:
        return {
            "assigned_band_index": action.assigned_band_index,
            "assigned_power_dbm": action.assigned_power_dbm,
            "justification": action.justification,
        }

    def _parse_result(self, payload: dict) -> StepResult[SpectrumObservation]:
        obs_data = payload.get("observation", {})
        obs = SpectrumObservation(
            spectrum_grid=obs_data.get("spectrum_grid", []),
            active_allocations=obs_data.get("active_allocations", []),
            current_request=obs_data.get("current_request", {}),
            regulatory_rules=obs_data.get("regulatory_rules", []),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 0),
            spectral_efficiency=obs_data.get("spectral_efficiency", 0.0),
            episode_reward_so_far=obs_data.get("episode_reward_so_far", 0.0),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SpectrumState:
        return SpectrumState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            accumulated_reward=payload.get("accumulated_reward", 0.0),
            requests_processed=payload.get("requests_processed", 0),
            requests_total=payload.get("requests_total", 0),
            successful_allocations=payload.get("successful_allocations", 0),
            rejected_requests=payload.get("rejected_requests", 0),
            interference_events=payload.get("interference_events", 0),
            preemptions=payload.get("preemptions", 0),
        )
