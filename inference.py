"""
Baseline Inference Script for RF Spectrum Allocation Environment
=================================================================
Uses OpenAI-compatible API client to run an LLM agent against all three
task difficulty levels and report scores.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM (default allowed).
    MODEL_NAME     The model identifier to use for inference (default allowed).
    HF_TOKEN       Your Hugging Face / API key (required; no default).

STDOUT FORMAT (mandatory, do not alter):
    [START] task=<task_name> env=rf_spectrum_env model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Import environment components ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SpectrumAction, SpectrumObservation
from server.spectrum_environment import SpectrumEnvironment, grade_episode
from scenarios import TASK_REGISTRY

# ── Configuration ────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_RETRIES = 1
TEMPERATURE = 0.0
MAX_TOKENS = 300
API_TIMEOUT = 15.0  # seconds per request attempt

SYSTEM_PROMPT = textwrap.dedent("""
You are an RF Spectrum Manager agent. You process allocation requests and assign
frequency bands while respecting regulatory constraints.

For each request, you must output a JSON object with exactly these fields:
{
    "assigned_band_index": <int>,
    "assigned_power_dbm": <float>,
    "justification": "<string>"
}

Rules:
- assigned_band_index: The index (0-11) of the frequency band to assign, or -1 to reject.
- assigned_power_dbm: The transmit power in dBm. Must not exceed the band's maximum.
- justification: A brief explanation of your decision mentioning relevant factors.

Key regulatory principles:
- Protected bands (Band 0) are reserved for emergency/public safety only.
- Emergency and military requests (priority 1) take precedence over all others.
- Guard bands must be respected between adjacent allocations.
- Unlicensed bands (ISM/UNII) have strict power limits.
- CBRS: PAL users have priority over GAA users.
- IoT devices in unlicensed bands should use power ≤ 14 dBm.

Output ONLY the JSON object. No markdown, no explanation outside the JSON.
""").strip()


def build_user_prompt(observation: SpectrumObservation) -> str:
    """Build the user prompt from the current observation."""
    grid_summary = []
    for band in observation.spectrum_grid:
        status = "OCCUPIED" if band.get("occupied") else "FREE"
        occupants_str = ""
        if band.get("occupants"):
            occ_list = [f"{o['user_type']}({o['user_id']}) at {o['power_dbm']}dBm, {o['remaining_steps']} steps left"
                        for o in band["occupants"]]
            occupants_str = " | Occupants: " + "; ".join(occ_list)
        grid_summary.append(
            f"  Band {band['index']}: {band['label']} [{band['band_type']}] "
            f"{band['start_mhz']}-{band['end_mhz']} MHz "
            f"(max {band['max_power_dbm']} dBm, guard {band['guard_band_mhz']} MHz) "
            f"[{status}]{occupants_str}"
        )

    req = observation.current_request
    if not req:
        return "No more requests. Episode complete."

    rules_str = "\n".join(f"  - {r}" for r in observation.regulatory_rules)

    prompt = textwrap.dedent(f"""
SPECTRUM STATUS (Step {observation.step_number + 1}/{observation.total_steps}):
Difficulty: {observation.task_difficulty}
Spectral efficiency: {observation.spectral_efficiency:.1%}
Episode reward so far: {observation.episode_reward_so_far:.3f}

FREQUENCY BANDS:
{chr(10).join(grid_summary)}

INCOMING REQUEST:
  ID: {req.get('request_id', 'N/A')}
  Requester: {req.get('requester_type', 'N/A')} ({req.get('requester_id', 'N/A')})
  Priority: {req.get('priority', 'N/A')} (1=highest, 5=lowest)
  Bandwidth needed: {req.get('bandwidth_needed_mhz', 'N/A')} MHz
  Preferred band: {req.get('preferred_band_index', 'None')}
  Requested power: {req.get('power_dbm', 'N/A')} dBm
  Duration: {req.get('duration_steps', 'N/A')} steps
  Description: {req.get('description', 'N/A')}

REGULATORY RULES:
{rules_str}
""").strip()

    # Append look-ahead block for spectrum_auction task when upcoming requests are present
    if observation.upcoming_requests:
        upcoming_lines = []
        for i, upcoming in enumerate(observation.upcoming_requests):
            upcoming_lines.append(
                f"  Next+{i + 1}: {upcoming.get('requester_type', '?')} "
                f"(P{upcoming.get('priority', '?')}) needs "
                f"{upcoming.get('bandwidth_needed_mhz', '?')} MHz, "
                f"prefers Band {upcoming.get('preferred_band_index', '?')}"
            )
            upcoming_lines.append(f"    {upcoming.get('description', '')}")
        prompt += "\n\nUPCOMING REQUESTS (plan ahead — assign current request with future demand in mind):\n"
        prompt += "\n".join(upcoming_lines)

    prompt += (
        '\n\nRespond with a JSON object: {"assigned_band_index": <int>, '
        '"assigned_power_dbm": <float>, "justification": "<string>"}'
    )
    return prompt


def parse_action(response_text: str) -> SpectrumAction:
    """Extract a SpectrumAction from model response."""
    text = response_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return SpectrumAction(
            assigned_band_index=int(data.get("assigned_band_index", -1)),
            assigned_power_dbm=float(data.get("assigned_power_dbm", 20.0)),
            justification=str(data.get("justification", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return SpectrumAction(
                assigned_band_index=int(data.get("assigned_band_index", -1)),
                assigned_power_dbm=float(data.get("assigned_power_dbm", 20.0)),
                justification=str(data.get("justification", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: reject with no justification
    print(f"  [WARN] Could not parse action from response: {text[:100]}...", file=sys.stderr)
    return SpectrumAction(
        assigned_band_index=-1,
        assigned_power_dbm=0.0,
        justification="Could not determine appropriate allocation.",
    )


def _rule_based_action(obs: SpectrumObservation) -> SpectrumAction:
    """
    Fast, network-free fallback policy used when no LLM token is available.
    Picks the first free band that matches the request type and stays within
    power limits, falling back to a safe default if nothing matches.
    """
    req = obs.current_request
    if not req:
        return SpectrumAction(assigned_band_index=-1, assigned_power_dbm=0.0,
                              justification="No request to process.")

    requester_type = req.get("requester_type", "commercial")
    requested_power = float(req.get("power_dbm", 20.0))
    priority = req.get("priority", 3)

    occupied = {b["index"] for b in obs.spectrum_grid if b.get("occupied")}

    # Choose candidate bands by requester type and priority
    if requester_type in ("emergency", "military") or priority == 1:
        candidates = [0, 1, 2, 3, 4]  # protected + licensed
    elif requester_type == "iot":
        candidates = [7, 8, 10, 11]   # unlicensed / shared
    else:
        candidates = [1, 2, 3, 4, 5, 6, 9, 10]  # licensed / CBRS

    for band_info in obs.spectrum_grid:
        idx = band_info["index"]
        if idx not in occupied and idx in candidates:
            power = min(requested_power, float(band_info.get("max_power_dbm", 20.0)))
            return SpectrumAction(
                assigned_band_index=idx,
                assigned_power_dbm=power,
                justification=(
                    f"Assigned band {idx} ({band_info.get('label', '')}) for "
                    f"{requester_type} request; power {power:.1f} dBm within regulatory limit."
                ),
            )

    # All candidates occupied — use first free band or reject
    for band_info in obs.spectrum_grid:
        if band_info["index"] not in occupied:
            power = min(requested_power, float(band_info.get("max_power_dbm", 20.0)))
            return SpectrumAction(
                assigned_band_index=band_info["index"],
                assigned_power_dbm=power,
                justification=f"Fallback allocation to band {band_info['index']}; best available.",
            )

    return SpectrumAction(
        assigned_band_index=-1,
        assigned_power_dbm=0.0,
        justification="All bands occupied; rejecting request.",
    )


def run_episode(
    client: OpenAI,
    env: SpectrumEnvironment,
    task_name: str,
    episode_idx: int,
    use_llm: bool = True,
) -> float:
    """Run a single episode and return the score."""
    rewards: List[float] = []
    success = False
    step = 0
    score = 0.0

    # Mandatory [START] line
    print(f"[START] task={task_name} env=rf_spectrum_env model={MODEL_NAME}", flush=True)

    # Initialize conversation history (used only when use_llm=True)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        # env.reset() is inside the try block so the finally clause
        # (which prints [END]) always executes even if reset() raises.
        obs = env.reset(task_name=task_name, episode_index=episode_idx, seed=42)

        while not obs.done:
            if use_llm:
                user_prompt = build_user_prompt(obs)
                messages.append({"role": "user", "content": user_prompt})

                response_text = ""
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            stream=False,
                        )
                        response_text = completion.choices[0].message.content or ""
                        break
                    except Exception as e:
                        print(f"    [RETRY {attempt+1}] API error: {e}", file=sys.stderr)
                        if attempt == MAX_RETRIES:
                            response_text = (
                                '{"assigned_band_index": -1, "assigned_power_dbm": 0,'
                                ' "justification": "API failure"}'
                            )

                messages.append({"role": "assistant", "content": response_text})
                # Keep context window bounded: system prompt + last 6 exchanges
                if len(messages) > 13:
                    messages = [messages[0]] + messages[-12:]

                action = parse_action(response_text)
            else:
                action = _rule_based_action(obs)

            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            step += 1

            # Mandatory [STEP] line
            done_str = "true" if obs.done else "false"
            error_str = obs.last_action_error if obs.last_action_error else "null"
            action_str = f"assign(band={action.assigned_band_index},power={action.assigned_power_dbm:.1f})"
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} "
                f"done={done_str} error={error_str}",
                flush=True,
            )

        success = True

    except Exception as exc:
        # Ensure at least one [STEP] appears so the validator sees a complete triplet
        error_msg = str(exc).replace("\n", " ")
        print(
            f"[STEP] step={step + 1} action=null reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    finally:
        # Mandatory [END] line — guaranteed to run regardless of how we exit the try block
        score = grade_episode(rewards) if rewards else 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success_str = "true" if success else "false"
        print(
            f"[END] task={task_name} success={success_str} steps={step} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )

    print(f"  Episode {episode_idx} score: {score:.4f}", file=sys.stderr)
    return score


def main():
    """Run baseline inference across all tasks."""
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
    use_llm = bool(api_key)

    if not use_llm:
        print(
            "WARNING: HF_TOKEN not set — using fast rule-based agent so structured output is "
            "always produced without network calls.",
            file=sys.stderr,
        )

    print("=" * 60, file=sys.stderr)
    print("RF Spectrum Allocation — Baseline Inference", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"API:   {API_BASE_URL}", file=sys.stderr)
    print(f"LLM:   {'enabled' if use_llm else 'disabled (no token)'}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key or "missing-token", timeout=API_TIMEOUT)
    env = SpectrumEnvironment()

    results: Dict[str, List[float]] = {}
    num_episodes_per_task = 3

    for task_name in ["easy", "medium", "disaster_response", "hard", "spectrum_auction"]:
        print(f"\n{'-' * 40}", file=sys.stderr)
        print(f"TASK: {task_name.upper()}", file=sys.stderr)
        print(f"Description: {TASK_REGISTRY[task_name]['description']}", file=sys.stderr)
        print(f"{'-' * 40}", file=sys.stderr)

        task_scores = []
        for ep_idx in range(num_episodes_per_task):
            score = run_episode(client, env, task_name, ep_idx, use_llm=use_llm)
            task_scores.append(score)

        avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
        results[task_name] = task_scores
        print(f"\n  {task_name.upper()} average: {avg:.4f}", file=sys.stderr)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("RESULTS SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    overall_scores = []
    for task_name, scores in results.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        overall_scores.append(avg)
        print(f"  {task_name:8s}: {avg:.4f}  (episodes: {[f'{s:.3f}' for s in scores]})", file=sys.stderr)

    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"\n  OVERALL:  {overall_avg:.4f}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    sys.stdout.flush()
    return overall_avg


if __name__ == "__main__":
    try:
        main()
    except Exception as _fatal:
        # Emergency failsafe: if main() itself crashes, still emit one valid
        # [START]/[STEP]/[END] triplet so the validator can parse something.
        print(f"FATAL: {_fatal}", file=sys.stderr)
        _task = "easy"
        print(f"[START] task={_task} env=rf_spectrum_env model={MODEL_NAME}", flush=True)
        print(
            f"[STEP] step=1 action=assign(band=-1,power=0.0) reward=0.00 "
            f"done=true error=startup_failure",
            flush=True,
        )
        print(
            f"[END] task={_task} success=false steps=1 score=0.00 rewards=0.00",
            flush=True,
        )
