---
title: RF Spectrum Environment
emoji: 📡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# RF Spectrum Allocation Environment

> Simulating the daily workflow of a telecom spectrum coordinator — the professional who decides which radio frequencies get assigned to whom, at what power, while enforcing regulatory policy in real time.

## Why This Task Matters

At every telecom operator (Jio, Airtel, AT&T, Vodafone) and regulatory body (India's DoT/TRAI, the US FCC, UK's Ofcom), **spectrum coordinators** sit at dashboards processing allocation requests all day. Their job: read an incoming request, check current spectrum occupancy, consult regulatory rules, and make an assignment decision — or reject the request with justification.

This is a high-stakes professional workflow with real consequences:

- **A wrong assignment** causes radio interference, dropping calls and data sessions for thousands of users.
- **A delayed emergency allocation** during a disaster (floods, earthquakes) can cost lives — first responders lose communication.
- **Regulatory violations** (exceeding power limits, using protected bands for commercial traffic) result in heavy fines and license revocations.
- **Wasted spectrum** in an increasingly congested radio environment means degraded 5G performance and IoT failures.

Today, this work is done by human professionals making manual decisions. As 5G densification, IoT proliferation (projected 29 billion connected devices by 2030), and cognitive radio networks increase demand, the industry is actively moving toward **AI-driven Dynamic Spectrum Access (DSA)**. This environment models the spectrum coordinator's decision-making workflow as a sequential task suitable for training RL agents.

### How This Compares to Other Real-World Tasks

| Task | Workflow | Our environment |
|------|----------|-----------------|
| Email triage | Read email → classify → route to team | Read request → classify → assign to band |
| Content moderation | Review content → check policy → approve/reject | Review request → check regulations → approve/reject |
| Customer support | Read ticket → consult KB → respond | Read request → consult spectrum grid → decide |
| IT helpdesk | Check system status → diagnose → assign priority | Check band occupancy → assess interference → set priority |

The structure is identical: **a professional receives an incoming item, consults rules and current state, and makes a classification/routing decision**. The domain is telecom spectrum management instead of email or tickets.

### Who Would Actually Use This

- **Telecom R&D teams** training AI agents for automated spectrum management (Jio, Airtel, Reliance, Qualcomm, Ericsson, Nokia)
- **Regulatory technology (RegTech) companies** building compliance-checking tools for spectrum policy
- **Cognitive radio researchers** developing dynamic spectrum access algorithms (the exact problem addressed by IEEE DySPAN and CROWNCOM conferences)
- **Defense and disaster response agencies** training priority-aware allocation systems for crisis scenarios
- **Academic RL researchers** evaluating agents on a novel, real-world domain with rich partial-reward signals

## Environment Description

The agent plays the role of a spectrum coordinator at a telecom operations center. Each episode represents a shift of incoming allocation requests. At each step, the agent observes the current spectrum occupancy, reads the incoming request, reviews regulatory constraints, and makes an assignment decision.

### Spectrum Grid

The environment simulates 12 frequency bands spanning 700 MHz to 5.25 GHz, modeled after real-world allocations used in India and the US:

| Band | Frequency Range | Type | Max Power | Real-World Analog |
|------|----------------|------|-----------|-------------------|
| 0 | 700 MHz Public Safety | Protected | 30 dBm | India: NDRF/SDRF bands; US: FirstNet |
| 1-2 | 700 MHz LTE | Licensed | 43 dBm | Jio Band 5, Airtel FDD |
| 3-4 | 850 MHz Cellular | Licensed | 40 dBm | BSNL/MTNL legacy cellular |
| 5-6 | AWS-1 (1700 MHz) | Licensed | 38 dBm | Mid-band LTE backhaul |
| 7-8 | 2.4 GHz ISM | Unlicensed | 20 dBm | WiFi, Bluetooth, IoT |
| 9 | CBRS PAL (3.5 GHz) | Shared | 30 dBm | US shared spectrum; India 5G n78 analog |
| 10 | CBRS GAA (3.6 GHz) | Shared | 23 dBm | Secondary/opportunistic access |
| 11 | 5 GHz UNII-1 | Unlicensed | 23 dBm | WiFi 5/6, low-power devices |

### User Types (Requesters)

- **Emergency services** (priority 1): NDRF, fire, police, ambulance — must be served immediately, can preempt others
- **Military** (priority 1): Defense operations — can commandeer any non-emergency band with zero notice
- **Commercial carriers** (priority 2-3): Jio, Airtel, Vodafone deploying base stations and capacity
- **IoT / smart city** (priority 4-5): Sensor networks, smart meters, connected vehicles — strict power limits
- **Amateur radio** (priority 5): Ham operators — lowest priority, open bands only

## Action Space

```python
class SpectrumAction(Action):
    assigned_band_index: int   # 0-11 to assign, -1 to reject
    assigned_power_dbm: float  # Transmit power in dBm
    justification: str         # Brief reasoning for the decision
```

This mirrors the real decision a spectrum coordinator makes: which band, at what power, and why.

## Observation Space

```python
class SpectrumObservation(Observation):
    spectrum_grid: List[Dict]       # All 12 bands with occupancy status
    active_allocations: List[Dict]  # Who is currently using what
    current_request: Dict           # The incoming allocation request
    regulatory_rules: List[str]     # Active regulatory constraints
    task_difficulty: str            # "easy" | "medium" | "hard"
    step_number: int                # Current step in episode
    total_steps: int                # Total requests this shift
    spectral_efficiency: float      # Current utilization ratio (0.0-1.0)
    episode_reward_so_far: float    # Cumulative performance
    last_action_error: str | None   # Feedback on previous decision
```

The observation provides everything a human coordinator would see on their dashboard: the spectrum waterfall display (grid), active users, the incoming request, and the rulebook.

## Tasks

### Easy — Quiet Shift (5 requests/episode)
A typical low-traffic period. Spectrum is mostly empty, requests clearly match available bands. A commercial carrier wants LTE spectrum — Band 1 is free, assign it. An IoT sensor wants ISM band — Band 7 is open, done. No conflicts, no ambiguity.

*What it tests:* Basic comprehension of band types, power limits, and request-to-band matching.

**Expected baseline score:** 0.6 – 0.8

### Medium — Busy Day with Priority Conflicts (8 requests/episode)
Peak hours with competing requests. Scenarios include:
- An emergency dispatch request that must go to the protected band
- A commercial carrier requesting protected spectrum (must be redirected to licensed bands)
- CBRS shared spectrum where PAL licensees have priority over GAA users
- Adjacent-band situations requiring guard band awareness

*What it tests:* Regulatory awareness, priority handling, and the ability to redirect rather than blindly assign.

**Expected baseline score:** 0.4 – 0.6

### Hard — Crisis Scenario: Urban Spectrum Congestion (12 requests/episode)
A disaster or major event. Dense spectrum occupancy with cascading pressures:
- Military commandeering active bands for secure operations
- Multiple competing emergency requests for the same protected spectrum
- IoT devices requesting excessive power (must be capped at 14 dBm per regulations)
- Cognitive radio secondary users who must sense and avoid primary users in shared bands
- High-power adjacent-band deployments causing aggregate interference risk

*What it tests:* Root-cause reasoning (redirect secondary users, not just reject), preemption logic, power regulation enforcement, and graceful degradation under congestion. This task models the exact problem addressed by **cognitive radio Dynamic Spectrum Access** research.

**Expected baseline score:** 0.2 – 0.4

## Reward Function

Each step is scored across four dimensions that map to real performance metrics used in telecom spectrum audits:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Band selection | 0.35 | Did the coordinator assign the right type of spectrum? |
| Power compliance | 0.25 | Is the assigned power within regulatory limits? |
| Priority handling | 0.25 | Were emergency/military requests served first? Were preemptions handled correctly? |
| Justification quality | 0.15 | Does the reasoning mention relevant factors (regulatory rules, interference, priority)? |

### Partial Credit Design

The reward is not binary — it provides granular feedback:

| Scenario | Reward |
|----------|--------|
| Optimal band assigned | 0.35/0.35 |
| Acceptable but non-optimal band | 0.25/0.35 |
| Right band type, wrong specific band | 0.10/0.35 |
| Wrong band type entirely | 0.00/0.35 |
| Power within limits | 0.25/0.25 |
| Power slightly over (≤3 dB) | 0.15/0.25 |
| Power way over (>10 dB) | 0.00/0.25 |

### Destructive Behavior Penalties
- Assigning commercial traffic to emergency-protected bands: **−0.1 penalty**
- Rejecting an emergency or military request: **0.0 for priority component**

Episode score = mean of per-step rewards → range [0.0, 1.0].

## Setup & Usage

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd rf_spectrum_env
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_token \
python inference.py
```

### Docker

```bash
docker build -t rf-spectrum-env -f server/Dockerfile .
docker run -p 7860:7860 rf-spectrum-env
```

### Client Usage

```python
from rf_spectrum_env import SpectrumEnv, SpectrumAction

with SpectrumEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset()
    print(f"Task: {result.observation.task_difficulty}")
    print(f"Request: {result.observation.current_request}")

    result = env.step(SpectrumAction(
        assigned_band_index=1,
        assigned_power_dbm=35.0,
        justification="Commercial LTE request routed to licensed Band 1. Power within 43 dBm limit."
    ))
    print(f"Reward: {result.reward}")
```

## Baseline Scores

| Task | Episodes | Mean Score | Min | Max |
|------|----------|-----------|-----|-----|
| Easy | 3 | ~0.70 | 0.60 | 0.80 |
| Medium | 3 | ~0.50 | 0.40 | 0.60 |
| Hard | 3 | ~0.30 | 0.20 | 0.40 |

*Scores produced with Llama-3.1-8B-Instruct via Hugging Face Inference API.*

## Research Applications

This environment can directly serve as a training ground for:

- **Cognitive radio DSA algorithms** — including deep RL approaches (DQN, PPO, twin actor-critic) for spectrum sensing and dynamic reallocation
- **Multi-agent spectrum sharing** — extending to scenarios where multiple operators manage overlapping geographic regions
- **Regulatory compliance verification** — testing whether AI agents correctly interpret and apply complex telecom rule sets
- **Emergency spectrum management** — training priority-aware allocation under crisis conditions (natural disasters, civil emergencies)
- **5G/6G network automation** — as part of the broader O-RAN intelligent controller (RIC) ecosystem

## File Structure

```
rf_spectrum_env/
├── __init__.py              # Package exports
├── models.py                # Action, Observation, State models (Pydantic)
├── scenarios.py             # Scenario generator for easy/medium/hard tasks
├── client.py                # SpectrumEnv client (EnvClient)
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package configuration
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── spectrum_environment.py  # Core environment logic + grader
    ├── requirements.txt
    └── Dockerfile
```

## License

BSD-3-Clause
=======
