"""
RF Spectrum Allocation Environment - Core Logic
=================================================
Implements the OpenEnv Environment interface with step(), reset(), state().
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Ensure project root is on path when running from server/ directory or Docker
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import SpectrumAction, SpectrumObservation, SpectrumState
from scenarios import (
    SPECTRUM_GRID,
    TASK_REGISTRY,
    ScenarioRequest,
    get_rules,
    get_scenarios,
    get_spectrum_grid,
)


class SpectrumEnvironment(Environment):
    """
    RF Spectrum Allocation OpenEnv Environment.

    The agent manages a simulated radio spectrum, processing allocation requests
    and making assignment decisions while respecting regulatory constraints.
    """

    def __init__(self, task_name: str = "easy", episode_index: int = 0, seed: int | None = None):
        super().__init__()
        self._task_name = task_name
        self._episode_index = episode_index
        self._seed = seed
        self._state = SpectrumState(episode_id=str(uuid.uuid4()), step_count=0)

        # Will be populated on reset()
        self._scenarios: List[List[ScenarioRequest]] = []
        self._current_episode: List[ScenarioRequest] = []
        self._active_allocations: List[Dict[str, Any]] = []
        self._step_rewards: List[float] = []
        self._rules: List[str] = []

    # ── OpenEnv interface ────────────────────────────────────────────

    def reset(self, seed: int | None = None, task_name: str | None = None, 
              episode_index: int | None = None, **kwargs) -> SpectrumObservation:
        """Initialize a new episode."""
        if task_name is not None:
            self._task_name = task_name
        if episode_index is not None:
            self._episode_index = episode_index
        effective_seed = seed if seed is not None else self._seed

        self._scenarios = get_scenarios(self._task_name, seed=effective_seed)
        self._rules = get_rules(self._task_name)

        # Clamp episode index
        ep_idx = self._episode_index % len(self._scenarios)
        self._current_episode = self._scenarios[ep_idx]

        # Reset state
        self._active_allocations = []
        self._step_rewards = []
        self._state = SpectrumState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
            requests_total=len(self._current_episode),
        )

        return self._build_observation(done=False, reward=None)

    def step(self, action: SpectrumAction) -> SpectrumObservation:
        """Process one allocation decision."""
        step_idx = self._state.step_count

        if step_idx >= len(self._current_episode):
            return self._build_observation(done=True, reward=0.0, error="Episode already complete.")

        request = self._current_episode[step_idx]
        reward, error = self._evaluate_action(action, request)

        # Update allocations
        if action.assigned_band_index >= 0 and error is None:
            self._active_allocations.append({
                "band_index": action.assigned_band_index,
                "user_type": request.requester_type,
                "user_id": request.requester_id,
                "power_dbm": action.assigned_power_dbm,
                "remaining_steps": request.duration_steps,
                "request_id": request.request_id,
            })
            self._state.successful_allocations += 1

            # Handle preemption: remove lower-priority allocations in same band
            if request.priority <= 2:
                self._active_allocations = [
                    a for a in self._active_allocations
                    if a["band_index"] != action.assigned_band_index
                    or a["request_id"] == request.request_id
                ]
                # Count preemptions
                preempted = sum(
                    1 for a in self._active_allocations
                    if a["band_index"] == action.assigned_band_index
                    and a["request_id"] != request.request_id
                )
                self._state.preemptions += preempted
        elif action.assigned_band_index == -1:
            self._state.rejected_requests += 1

        # Decay remaining steps on all allocations
        for alloc in self._active_allocations:
            alloc["remaining_steps"] = max(0, alloc["remaining_steps"] - 1)
        self._active_allocations = [a for a in self._active_allocations if a["remaining_steps"] > 0]

        self._step_rewards.append(reward)
        self._state.step_count += 1
        self._state.requests_processed += 1
        self._state.accumulated_reward += reward

        done = self._state.step_count >= len(self._current_episode)

        return self._build_observation(done=done, reward=reward, error=error)

    @property
    def state(self) -> SpectrumState:
        return self._state

    # ── Reward computation ───────────────────────────────────────────

    def _evaluate_action(self, action: SpectrumAction, request: ScenarioRequest) -> tuple[float, str | None]:
        """
        Compute reward for a single action against ground truth.

        Returns (reward: 0.0-1.0, error_message or None).

        Reward breakdown:
        - Band selection:     0.35  (correct band assignment)
        - Power compliance:   0.25  (within regulatory limits)
        - Priority handling:  0.25  (correct preemption / rejection logic)
        - Justification:      0.15  (mentions relevant concepts)
        """
        total = 0.0
        error = None
        num_bands = len(SPECTRUM_GRID)

        # ── Band selection (0.35) ────────────────────────────────────
        if action.assigned_band_index == -1:
            # Agent chose to reject
            if request.gt_best_band_index == -1:
                total += 0.35  # correct rejection
            elif request.gt_reject_reason is not None:
                total += 0.15  # partially correct, rejection was acceptable
            else:
                total += 0.0  # incorrect rejection
                error = "Rejected a valid request that should have been allocated."
        elif 0 <= action.assigned_band_index < num_bands:
            if action.assigned_band_index == request.gt_best_band_index:
                total += 0.35  # optimal band
            elif action.assigned_band_index in request.gt_acceptable_bands:
                total += 0.25  # acceptable but not optimal
            else:
                # Check if band type is at least compatible
                chosen_type = SPECTRUM_GRID[action.assigned_band_index].band_type
                gt_type = SPECTRUM_GRID[request.gt_best_band_index].band_type if request.gt_best_band_index >= 0 else ""
                if chosen_type == gt_type:
                    total += 0.10  # right type of band, wrong specific band
                else:
                    total += 0.0
                    error = f"Assigned to incompatible band type '{chosen_type}'."

                # Penalty: assigning commercial traffic to protected band
                if chosen_type == "protected" and request.requester_type not in ("emergency", "military"):
                    total -= 0.1
                    error = "Regulatory violation: non-emergency user in protected band."
        else:
            error = f"Invalid band index: {action.assigned_band_index}."

        # ── Power compliance (0.25) ──────────────────────────────────
        if 0 <= action.assigned_band_index < num_bands:
            band = SPECTRUM_GRID[action.assigned_band_index]
            gt_max = request.gt_max_power_dbm

            if action.assigned_power_dbm <= gt_max:
                # Within limits
                if action.assigned_power_dbm <= band.max_power_dbm:
                    total += 0.25
                else:
                    total += 0.10  # exceeded band max but within gt
                    error = error or "Power exceeds band maximum."
            else:
                # Over the ground-truth maximum
                overshoot = action.assigned_power_dbm - gt_max
                if overshoot <= 3.0:
                    total += 0.15  # slightly over
                elif overshoot <= 10.0:
                    total += 0.05  # moderately over
                else:
                    total += 0.0  # way over
                error = error or f"Power {action.assigned_power_dbm} dBm exceeds limit {gt_max} dBm."

        # ── Priority handling (0.25) ──────────────────────────────────
        if request.priority == 1:
            # Emergency/military: should have been allocated immediately
            if action.assigned_band_index >= 0:
                total += 0.20
                if request.gt_should_preempt:
                    total += 0.05  # full marks for correct preemption awareness
            else:
                total += 0.0
                error = error or "Rejected emergency/military request."
        elif request.gt_should_preempt:
            if action.assigned_band_index >= 0:
                total += 0.25  # correctly allocated with preemption
            else:
                total += 0.05  # rejected but should have preempted
        else:
            # Standard priority handling
            if action.assigned_band_index >= 0 or request.gt_best_band_index == -1:
                total += 0.25
            else:
                total += 0.10

        # ── Justification quality (0.15) ─────────────────────────────
        justification = action.justification.lower()
        keywords_present = 0
        relevant_keywords = []

        if request.requester_type == "emergency":
            relevant_keywords = ["emergency", "priority", "public safety", "first responder"]
        elif request.requester_type == "military":
            relevant_keywords = ["military", "commandeer", "priority", "exclusive"]
        elif request.requester_type == "iot":
            relevant_keywords = ["iot", "sensor", "power", "unlicensed", "ism"]
        elif "cognitive" in request.description.lower() or "cbrs" in request.description.lower():
            relevant_keywords = ["cognitive", "secondary", "primary", "cbrs", "sensing", "gaa", "pal"]
        else:
            relevant_keywords = ["band", "power", "frequency", "allocat", "assign"]

        for kw in relevant_keywords:
            if kw in justification:
                keywords_present += 1

        if keywords_present >= 3:
            total += 0.15
        elif keywords_present >= 2:
            total += 0.10
        elif keywords_present >= 1:
            total += 0.05

        # Clamp to [0.0, 1.0]
        total = max(0.0, min(1.0, total))
        return round(total, 4), error

    # ── Observation builder ──────────────────────────────────────────

    def _build_observation(self, done: bool, reward: float | None,
                           error: str | None = None) -> SpectrumObservation:
        """Construct the observation for the current step."""
        step_idx = self._state.step_count

        # Current request (or empty if done)
        if step_idx < len(self._current_episode) and not done:
            req = self._current_episode[step_idx]
            current_request = {
                "request_id": req.request_id,
                "requester_type": req.requester_type,
                "requester_id": req.requester_id,
                "bandwidth_needed_mhz": req.bandwidth_needed_mhz,
                "preferred_band_index": req.preferred_band_index,
                "priority": req.priority,
                "duration_steps": req.duration_steps,
                "power_dbm": req.power_dbm,
                "description": req.description,
            }
        else:
            current_request = {}

        # Spectrum grid with occupancy
        grid = get_spectrum_grid()
        for band_info in grid:
            occupants = [
                a for a in self._active_allocations
                if a["band_index"] == band_info["index"]
            ]
            band_info["occupied"] = len(occupants) > 0
            band_info["occupants"] = [
                {
                    "user_type": a["user_type"],
                    "user_id": a["user_id"],
                    "power_dbm": a["power_dbm"],
                    "remaining_steps": a["remaining_steps"],
                }
                for a in occupants
            ]

        # Spectral efficiency
        occupied_bw = sum(
            SPECTRUM_GRID[a["band_index"]].end_mhz - SPECTRUM_GRID[a["band_index"]].start_mhz
            for a in self._active_allocations
        )
        total_bw = sum(b.end_mhz - b.start_mhz for b in SPECTRUM_GRID)
        efficiency = occupied_bw / total_bw if total_bw > 0 else 0.0

        return SpectrumObservation(
            spectrum_grid=grid,
            active_allocations=self._active_allocations.copy(),
            current_request=current_request,
            regulatory_rules=self._rules,
            task_difficulty=self._task_name,
            step_number=step_idx,
            total_steps=len(self._current_episode),
            spectral_efficiency=round(efficiency, 4),
            episode_reward_so_far=round(self._state.accumulated_reward, 4),
            last_action_error=error,
            done=done,
            reward=reward,
        )


# ── Grader ───────────────────────────────────────────────────────────

def grade_episode(rewards: List[float]) -> float:
    """
    Compute the final episode score from per-step rewards.
    Returns a float in [0.0, 1.0].
    """
    if not rewards:
        return 0.0
    return round(sum(rewards) / len(rewards), 4)
