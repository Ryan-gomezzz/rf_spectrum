"""
RF Spectrum Allocation Environment - Models
=============================================
Typed Pydantic models for actions, observations, and state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State  # noqa: F401


# ── Spectrum domain types ────────────────────────────────────────────


@dataclass
class FrequencyBand:
    """A contiguous block of radio spectrum."""

    start_mhz: float
    end_mhz: float
    label: str  # e.g. "Band-A (700 MHz LTE)"

    @property
    def bandwidth_mhz(self) -> float:
        return self.end_mhz - self.start_mhz


@dataclass
class Allocation:
    """An existing spectrum allocation occupying a band."""

    band_index: int  # index into the spectrum grid
    user_type: str  # "primary" | "secondary" | "emergency"
    user_id: str
    power_dbm: float
    remaining_steps: int  # how many steps until this allocation expires


@dataclass
class AllocationRequest:
    """An incoming request for spectrum."""

    request_id: str
    requester_type: str  # "emergency", "commercial", "iot", "amateur", "military"
    requester_id: str
    bandwidth_needed_mhz: float
    preferred_band_index: Optional[int]
    priority: int  # 1 (highest) .. 5 (lowest)
    duration_steps: int
    power_dbm: float
    description: str  # natural language description of the request


# ── OpenEnv models ───────────────────────────────────────────────────


class SpectrumAction(Action):
    """Agent's allocation decision."""

    assigned_band_index: int = Field(
        ..., description="Index of the frequency band to assign (0-based), or -1 to reject"
    )
    assigned_power_dbm: float = Field(
        ..., description="Transmit power in dBm to allocate"
    )
    justification: str = Field(
        default="", description="Brief justification for the decision"
    )


class SpectrumObservation(Observation):
    """What the agent sees each step."""

    # Current spectrum state
    spectrum_grid: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of bands with occupancy info",
    )
    active_allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active allocations",
    )

    # Incoming request
    current_request: Dict[str, Any] = Field(
        default_factory=dict,
        description="The allocation request to process",
    )

    # Regulatory constraints
    regulatory_rules: List[str] = Field(
        default_factory=list,
        description="Active regulatory constraints",
    )

    # Episode info
    task_difficulty: str = Field(
        default="easy", description="easy | medium | hard"
    )
    step_number: int = Field(default=0)
    total_steps: int = Field(default=0)
    spectral_efficiency: float = Field(
        default=0.0,
        description="Current spectrum utilization ratio 0.0-1.0",
    )
    episode_reward_so_far: float = Field(default=0.0)
    last_action_error: Optional[str] = Field(default=None)

    # Metadata inherited from Observation base
    # done: bool, reward: float, metadata: dict


class SpectrumState(State):
    """Internal episode state."""

    task_name: str = ""
    accumulated_reward: float = 0.0
    requests_processed: int = 0
    requests_total: int = 0
    successful_allocations: int = 0
    rejected_requests: int = 0
    interference_events: int = 0
    preemptions: int = 0
