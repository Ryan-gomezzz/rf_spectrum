"""
RF Spectrum Allocation - Scenario Generator
=============================================
Generates deterministic scenario pools for easy / medium / hard tasks.
Each scenario is a sequence of allocation requests with ground-truth
optimal actions for grading.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BandSpec:
    """Specification for a frequency band in the grid."""
    start_mhz: float
    end_mhz: float
    label: str
    band_type: str  # "licensed", "unlicensed", "protected", "shared"
    max_power_dbm: float
    guard_band_mhz: float  # required guard on each side


@dataclass
class ScenarioRequest:
    """A single allocation request within a scenario."""
    request_id: str
    requester_type: str
    requester_id: str
    bandwidth_needed_mhz: float
    preferred_band_index: Optional[int]
    priority: int
    duration_steps: int
    power_dbm: float
    description: str

    # Ground truth for grading
    gt_best_band_index: int  # -1 means reject is correct
    gt_acceptable_bands: List[int]  # any of these is acceptable
    gt_max_power_dbm: float
    gt_should_preempt: bool  # should this preempt a lower-priority user?
    gt_reject_reason: Optional[str]  # if rejection is correct, why


# ── Band grid (shared across all tasks) ──────────────────────────────

SPECTRUM_GRID: List[BandSpec] = [
    BandSpec(700.0, 710.0, "Band 0: 700 MHz Public Safety", "protected", 30.0, 1.0),
    BandSpec(710.0, 730.0, "Band 1: 700 MHz LTE", "licensed", 43.0, 0.5),
    BandSpec(730.0, 746.0, "Band 2: 700 MHz LTE-B", "licensed", 43.0, 0.5),
    BandSpec(850.0, 870.0, "Band 3: 850 MHz Cellular", "licensed", 40.0, 0.5),
    BandSpec(870.0, 890.0, "Band 4: 850 MHz Cellular-B", "licensed", 40.0, 0.5),
    BandSpec(1700.0, 1720.0, "Band 5: AWS-1 Uplink", "licensed", 38.0, 1.0),
    BandSpec(1720.0, 1755.0, "Band 6: AWS-1 Extended", "licensed", 38.0, 1.0),
    BandSpec(2400.0, 2450.0, "Band 7: 2.4 GHz ISM-A", "unlicensed", 20.0, 0.0),
    BandSpec(2450.0, 2483.5, "Band 8: 2.4 GHz ISM-B", "unlicensed", 20.0, 0.0),
    BandSpec(3550.0, 3600.0, "Band 9: CBRS PAL", "shared", 30.0, 0.5),
    BandSpec(3600.0, 3650.0, "Band 10: CBRS GAA", "shared", 23.0, 0.5),
    BandSpec(5150.0, 5250.0, "Band 11: 5 GHz UNII-1", "unlicensed", 23.0, 0.0),
]

REGULATORY_RULES_BASE = [
    "Emergency services (priority 1) must be allocated within protected or licensed bands.",
    "Protected bands (Band 0) are reserved for public safety; commercial use is prohibited.",
    "Guard bands must be maintained: no allocation may overlap another active allocation's guard band.",
    "Power must not exceed the band's maximum rated power.",
    "Unlicensed bands (ISM/UNII) are open-access but have strict power limits.",
    "CBRS shared bands: PAL users have priority over GAA users.",
    "Higher-priority requests may preempt lower-priority allocations in shared/licensed bands.",
]

REGULATORY_RULES_HARD = REGULATORY_RULES_BASE + [
    "Primary users in CBRS PAL bands cannot be preempted by secondary users.",
    "Military requests (priority 1) may commandeer any non-emergency band with 0-step notice.",
    "Concurrent allocations in adjacent bands must account for aggregate interference.",
    "IoT devices in unlicensed bands must use power ≤ 14 dBm to avoid interference.",
]


# ── Scenario builders ────────────────────────────────────────────────

def _build_easy_scenarios(seed: int = 42) -> List[List[ScenarioRequest]]:
    """Easy: mostly empty spectrum, obvious assignments, no conflicts."""
    rng = random.Random(seed)
    scenarios = []

    for ep in range(10):
        episode: List[ScenarioRequest] = []
        for step in range(5):
            req_type = rng.choice(["commercial", "iot", "amateur"])
            if req_type == "commercial":
                bw = rng.choice([10.0, 16.0, 20.0])
                best = rng.choice([1, 2, 3, 4])
                pwr = rng.uniform(30.0, 40.0)
                desc = f"Commercial LTE base station needs {bw} MHz in cellular band."
            elif req_type == "iot":
                bw = rng.choice([5.0, 10.0])
                best = rng.choice([7, 8, 11])
                pwr = rng.uniform(10.0, 14.0)
                desc = f"IoT sensor network requesting {bw} MHz in ISM band."
            else:
                bw = 5.0
                best = rng.choice([7, 8])
                pwr = rng.uniform(5.0, 10.0)
                desc = "Amateur radio operator requesting spectrum for local use."

            episode.append(ScenarioRequest(
                request_id=f"easy-{ep}-{step}",
                requester_type=req_type,
                requester_id=f"{req_type}-{rng.randint(100,999)}",
                bandwidth_needed_mhz=bw,
                preferred_band_index=best,
                priority=rng.choice([3, 4, 5]),
                duration_steps=rng.randint(2, 5),
                power_dbm=round(pwr, 1),
                description=desc,
                gt_best_band_index=best,
                gt_acceptable_bands=[best],
                gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                gt_should_preempt=False,
                gt_reject_reason=None,
            ))
        scenarios.append(episode)
    return scenarios


def _build_medium_scenarios(seed: int = 123) -> List[List[ScenarioRequest]]:
    """Medium: moderate occupancy, priority conflicts, guard band awareness."""
    rng = random.Random(seed)
    scenarios = []

    for ep in range(10):
        episode: List[ScenarioRequest] = []
        for step in range(8):
            roll = rng.random()

            if roll < 0.15:
                # Emergency request that should go to protected band
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="emergency",
                    requester_id=f"ems-{rng.randint(10,99)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=None,
                    priority=1,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=28.0,
                    description="Emergency dispatch requires immediate spectrum for first responder comms.",
                    gt_best_band_index=0,
                    gt_acceptable_bands=[0, 1],
                    gt_max_power_dbm=30.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.35:
                # Commercial wanting protected band (should be rejected or redirected)
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"telco-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=0,
                    priority=3,
                    duration_steps=rng.randint(3, 6),
                    power_dbm=35.0,
                    description="Mobile carrier requesting Band 0 for capacity expansion.",
                    gt_best_band_index=1,  # should redirect to Band 1
                    gt_acceptable_bands=[1, 2, 3, 4],
                    gt_max_power_dbm=43.0,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.55:
                # CBRS request with PAL/GAA priority awareness
                is_pal = rng.random() < 0.5
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="commercial" if is_pal else "iot",
                    requester_id=f"cbrs-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
                    preferred_band_index=9 if is_pal else 10,
                    priority=2 if is_pal else 4,
                    duration_steps=rng.randint(2, 6),
                    power_dbm=28.0 if is_pal else 20.0,
                    description=f"CBRS {'PAL licensee' if is_pal else 'GAA user'} requesting shared spectrum.",
                    gt_best_band_index=9 if is_pal else 10,
                    gt_acceptable_bands=[9, 10],
                    gt_max_power_dbm=30.0 if is_pal else 23.0,
                    gt_should_preempt=is_pal,
                    gt_reject_reason=None,
                ))
            else:
                # Standard commercial or IoT
                req_type = rng.choice(["commercial", "iot"])
                if req_type == "commercial":
                    best = rng.choice([1, 2, 3, 4, 5, 6])
                    bw = rng.choice([10.0, 16.0, 20.0])
                    pwr = rng.uniform(30.0, 40.0)
                    desc = f"Carrier needs {bw} MHz for urban macro cell."
                else:
                    best = rng.choice([7, 8, 11])
                    bw = rng.choice([5.0, 10.0])
                    pwr = rng.uniform(10.0, 14.0)
                    desc = f"IoT deployment needs {bw} MHz in unlicensed band."

                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type=req_type,
                    requester_id=f"{req_type}-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=bw,
                    preferred_band_index=best,
                    priority=rng.choice([2, 3, 4]),
                    duration_steps=rng.randint(2, 6),
                    power_dbm=round(pwr, 1),
                    description=desc,
                    gt_best_band_index=best,
                    gt_acceptable_bands=[best],
                    gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
        scenarios.append(episode)
    return scenarios


def _build_hard_scenarios(seed: int = 999) -> List[List[ScenarioRequest]]:
    """Hard: dense occupancy, cascading preemptions, military, cognitive radio dynamics."""
    rng = random.Random(seed)
    scenarios = []

    for ep in range(10):
        episode: List[ScenarioRequest] = []
        for step in range(12):
            roll = rng.random()

            if roll < 0.12:
                # Military commandeer
                target = rng.choice([1, 2, 3, 5, 6, 9])
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="military",
                    requester_id=f"mil-{rng.randint(10,99)}",
                    bandwidth_needed_mhz=20.0,
                    preferred_band_index=target,
                    priority=1,
                    duration_steps=rng.randint(5, 12),
                    power_dbm=43.0,
                    description="Military operation requires immediate exclusive spectrum access.",
                    gt_best_band_index=target,
                    gt_acceptable_bands=[target],
                    gt_max_power_dbm=43.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.22:
                # Emergency competing with existing emergency
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="emergency",
                    requester_id=f"ems-{rng.randint(10,99)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=0,
                    priority=1,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=28.0,
                    description="Multiple-alarm fire; dispatch needs additional spectrum for mutual aid.",
                    gt_best_band_index=0,
                    gt_acceptable_bands=[0, 1],
                    gt_max_power_dbm=30.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.35:
                # Overpowered IoT (should be rejected or power-capped)
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="iot",
                    requester_id=f"iot-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=rng.choice([7, 8]),
                    priority=5,
                    duration_steps=rng.randint(2, 4),
                    power_dbm=25.0,  # exceeds IoT limit of 14 dBm in hard rules
                    description="Smart city sensor grid requesting ISM band at elevated power.",
                    gt_best_band_index=rng.choice([7, 8]),
                    gt_acceptable_bands=[7, 8],
                    gt_max_power_dbm=14.0,  # must cap power
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.50:
                # Cognitive radio secondary user that must sense and avoid primary
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"cr-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
                    preferred_band_index=9,  # CBRS PAL - may have primary
                    priority=3,
                    duration_steps=rng.randint(2, 5),
                    power_dbm=23.0,
                    description="Cognitive radio secondary user sensing CBRS band for opportunistic access.",
                    gt_best_band_index=10,  # should be redirected to GAA
                    gt_acceptable_bands=[10, 7, 8],
                    gt_max_power_dbm=23.0,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.65:
                # Adjacent band interference scenario
                adj = rng.choice([(1, 2), (3, 4), (7, 8), (5, 6)])
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"telco-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=20.0,
                    preferred_band_index=adj[0],
                    priority=2,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=43.0,  # max power next to active adjacent band
                    description=f"High-power deployment in Band {adj[0]}, adjacent Band {adj[1]} is active.",
                    gt_best_band_index=adj[0],
                    gt_acceptable_bands=[adj[0]],
                    gt_max_power_dbm=38.0,  # should reduce power due to adjacency
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            else:
                # Dense general traffic
                req_type = rng.choice(["commercial", "commercial", "iot", "amateur"])
                best = rng.choice(list(range(len(SPECTRUM_GRID))))
                bw = rng.choice([5.0, 10.0, 16.0, 20.0])
                pwr = rng.uniform(15.0, 40.0)
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type=req_type,
                    requester_id=f"{req_type}-{rng.randint(100,999)}",
                    bandwidth_needed_mhz=bw,
                    preferred_band_index=best,
                    priority=rng.choice([2, 3, 4, 5]),
                    duration_steps=rng.randint(1, 4),
                    power_dbm=round(pwr, 1),
                    description=f"{req_type.title()} user needs {bw} MHz; spectrum congested.",
                    gt_best_band_index=best,
                    gt_acceptable_bands=[best] + [b for b in range(len(SPECTRUM_GRID))
                                                    if SPECTRUM_GRID[b].band_type == SPECTRUM_GRID[best].band_type
                                                    and b != best],
                    gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
        scenarios.append(episode)
    return scenarios


# ── Public API ───────────────────────────────────────────────────────

TASK_REGISTRY = {
    "easy": {
        "builder": _build_easy_scenarios,
        "rules": REGULATORY_RULES_BASE,
        "description": "Low-occupancy spectrum with straightforward allocation requests.",
        "steps_per_episode": 5,
    },
    "medium": {
        "builder": _build_medium_scenarios,
        "rules": REGULATORY_RULES_BASE,
        "description": "Moderate occupancy with priority conflicts and guard band awareness.",
        "steps_per_episode": 8,
    },
    "hard": {
        "builder": _build_hard_scenarios,
        "rules": REGULATORY_RULES_HARD,
        "description": "Dense urban spectrum: cascading preemptions, cognitive radio dynamics, military operations.",
        "steps_per_episode": 12,
    },
}


def get_scenarios(task_name: str, seed: int | None = None) -> List[List[ScenarioRequest]]:
    """Return the list of episodes for a given task."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_REGISTRY.keys())}")
    builder = TASK_REGISTRY[task_name]["builder"]
    return builder(seed=seed) if seed is not None else builder()


def get_rules(task_name: str) -> List[str]:
    """Return regulatory rules for a given task."""
    return TASK_REGISTRY[task_name]["rules"]


def get_spectrum_grid() -> List[Dict[str, Any]]:
    """Return the spectrum grid as serializable dicts."""
    return [
        {
            "index": i,
            "start_mhz": b.start_mhz,
            "end_mhz": b.end_mhz,
            "bandwidth_mhz": b.end_mhz - b.start_mhz,
            "label": b.label,
            "band_type": b.band_type,
            "max_power_dbm": b.max_power_dbm,
            "guard_band_mhz": b.guard_band_mhz,
        }
        for i, b in enumerate(SPECTRUM_GRID)
    ]
