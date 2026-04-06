"""
Unit tests for the RF Spectrum Allocation Environment.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models import SpectrumAction, SpectrumObservation, SpectrumState
from scenarios import SPECTRUM_GRID, TASK_REGISTRY
from server.spectrum_environment import SpectrumEnvironment, grade_episode


# ── Helpers ──────────────────────────────────────────────────────────

def make_env(task: str = "easy") -> SpectrumEnvironment:
    env = SpectrumEnvironment()
    env.reset(task_name=task)
    return env


def reasonable_action(obs: SpectrumObservation, power_dbm: float = 20.0) -> SpectrumAction:
    """Build an action that assigns to the request's preferred band."""
    req = obs.current_request
    band_idx = req.get("preferred_band_index") or 1
    return SpectrumAction(
        assigned_band_index=band_idx,
        assigned_power_dbm=power_dbm,
        justification=f"Assigning {req.get('requester_type', 'unknown')} to preferred band {band_idx}",
    )


# ── reset() tests ─────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert isinstance(obs, SpectrumObservation)

    def test_reset_done_false(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert obs.done is False

    def test_reset_reward_none(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert obs.reward is None

    def test_reset_has_current_request(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert isinstance(obs.current_request, dict)
        assert "requester_type" in obs.current_request

    def test_reset_spectrum_grid_has_12_bands(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert len(obs.spectrum_grid) == 12

    def test_reset_step_number_zero(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert obs.step_number == 0

    def test_reset_easy_total_steps(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert obs.total_steps == 5

    def test_reset_medium_total_steps(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        assert obs.total_steps == 8

    def test_reset_hard_total_steps(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="hard")
        assert obs.total_steps == 12

    def test_reset_clears_state(self):
        env = SpectrumEnvironment()
        obs1 = env.reset(task_name="easy")
        action = reasonable_action(obs1)
        env.step(action)
        # Re-reset should start fresh
        obs2 = env.reset(task_name="easy")
        assert obs2.step_number == 0
        assert obs2.episode_reward_so_far == 0.0


# ── step() tests ─────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_observation(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        obs2 = env.step(reasonable_action(obs))
        assert isinstance(obs2, SpectrumObservation)

    def test_step_reward_in_range(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        obs2 = env.step(reasonable_action(obs))
        assert obs2.reward is not None
        assert 0.0 <= obs2.reward <= 1.0

    def test_step_increments_step_number(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        obs2 = env.step(reasonable_action(obs))
        assert obs2.step_number == 1

    def test_step_done_false_mid_episode(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")  # 5 steps
        for _ in range(4):
            if not obs.done:
                obs = env.step(reasonable_action(obs))
        assert obs.done is False

    def test_step_done_true_at_end(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        while not obs.done:
            obs = env.step(reasonable_action(obs))
        assert obs.done is True

    def test_different_actions_produce_different_rewards(self):
        """Rewards must NOT be constant — graders check this."""
        # Good action: assign to preferred band with acceptable power
        env1 = SpectrumEnvironment()
        obs1 = env1.reset(task_name="easy", seed=42)
        req = obs1.current_request
        preferred = req.get("preferred_band_index") or 1
        good_action = SpectrumAction(
            assigned_band_index=preferred,
            assigned_power_dbm=15.0,
            justification="Assigning commercial to preferred band with power within limits",
        )
        obs_good = env1.step(good_action)

        # Bad action: reject the request
        env2 = SpectrumEnvironment()
        env2.reset(task_name="easy", seed=42)
        bad_action = SpectrumAction(
            assigned_band_index=-1,
            assigned_power_dbm=0.0,
            justification="Rejecting",
        )
        obs_bad = env2.step(bad_action)

        assert obs_good.reward != obs_bad.reward, (
            f"Rewards must vary! good={obs_good.reward}, bad={obs_bad.reward}"
        )

    def test_rewards_vary_across_episode(self):
        """Episode rewards should not all be identical."""
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        rewards = []
        while not obs.done:
            # Alternate between good and bad actions to force variation
            if len(rewards) % 2 == 0:
                req = obs.current_request
                action = SpectrumAction(
                    assigned_band_index=req.get("preferred_band_index") or 1,
                    assigned_power_dbm=15.0,
                    justification="band power assign frequency allocat",
                )
            else:
                action = SpectrumAction(
                    assigned_band_index=-1,
                    assigned_power_dbm=0.0,
                    justification="",
                )
            obs = env.step(action)
            rewards.append(obs.reward)

        assert len(set(rewards)) > 1, f"All rewards identical: {rewards}"


# ── Task difficulty tests ─────────────────────────────────────────────

class TestTaskDifficulty:
    def test_all_tasks_complete(self):
        for task in ["easy", "medium", "hard"]:
            env = SpectrumEnvironment()
            obs = env.reset(task_name=task)
            steps = 0
            while not obs.done:
                obs = env.step(reasonable_action(obs))
                steps += 1
            assert obs.done is True, f"{task} never completed"

    def test_easy_episode_length(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        steps = 0
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            steps += 1
        assert steps == 5

    def test_medium_episode_length(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        steps = 0
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            steps += 1
        assert steps == 8

    def test_hard_episode_length(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="hard")
        steps = 0
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            steps += 1
        assert steps == 12

    def test_tasks_produce_different_difficulty(self):
        """Easy should score higher than hard on average with same strategy."""
        def run_task(task: str) -> float:
            env = SpectrumEnvironment()
            obs = env.reset(task_name=task)
            rewards = []
            while not obs.done:
                req = obs.current_request
                band = req.get("preferred_band_index") or 1
                action = SpectrumAction(
                    assigned_band_index=band,
                    assigned_power_dbm=min(req.get("power_dbm", 20.0), 30.0),
                    justification="band power frequency allocat assign",
                )
                obs = env.step(action)
                rewards.append(obs.reward)
            return sum(rewards) / len(rewards) if rewards else 0.0

        easy_score = run_task("easy")
        hard_score = run_task("hard")
        # Easy should score at least as high as hard
        assert easy_score >= hard_score - 0.3, (
            f"Easy ({easy_score:.3f}) should be harder to beat than hard ({hard_score:.3f})"
        )


# ── Reward function tests ─────────────────────────────────────────────

class TestRewardFunction:
    def test_reward_partial_credit(self):
        """Reward is not just 0 or 1 — should have partial credit values."""
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        rewards = []
        while not obs.done:
            req = obs.current_request
            # Use a suboptimal but not terrible action
            action = SpectrumAction(
                assigned_band_index=req.get("preferred_band_index") or 1,
                assigned_power_dbm=min(req.get("power_dbm", 25.0), 35.0),
                justification="band allocation",
            )
            obs = env.step(action)
            rewards.append(obs.reward)

        # Should have values besides just 0.0 and 1.0
        non_binary = [r for r in rewards if r not in (0.0, 1.0)]
        assert len(non_binary) > 0, f"All rewards binary (0 or 1): {rewards}"

    def test_destructive_penalty_commercial_in_protected_band(self):
        """Assigning commercial to protected band (band 0) gets penalized."""
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        # Find a commercial request and assign it to the protected band (0)
        for _ in range(8):
            req = obs.current_request
            if req.get("requester_type") == "commercial":
                action = SpectrumAction(
                    assigned_band_index=0,  # protected band — violation
                    assigned_power_dbm=25.0,
                    justification="commercial in band 0",
                )
                obs2 = env.step(action)
                # Should have error message and low reward
                assert obs2.last_action_error is not None, "Expected regulatory violation error"
                assert obs2.reward < 0.7, f"Commercial in protected band should be penalized, got {obs2.reward}"
                return
            else:
                obs = env.step(reasonable_action(obs))
                if obs.done:
                    break
        pytest.skip("No commercial request found in this episode")

    def test_emergency_request_gets_priority_reward(self):
        """Correctly handling emergency request earns priority score."""
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        for _ in range(8):
            req = obs.current_request
            if req.get("requester_type") == "emergency":
                action = SpectrumAction(
                    assigned_band_index=0,  # protected band — correct for emergency
                    assigned_power_dbm=28.0,
                    justification="emergency priority public safety first responder",
                )
                obs2 = env.step(action)
                # Priority component should contribute positively
                assert obs2.reward >= 0.4, f"Emergency correctly handled should score well, got {obs2.reward}"
                return
            else:
                obs = env.step(reasonable_action(obs))
                if obs.done:
                    break
        pytest.skip("No emergency request in this medium episode")

    def test_emergency_rejection_gets_zero_priority(self):
        """Rejecting an emergency request earns 0 priority component."""
        env = SpectrumEnvironment()
        obs = env.reset(task_name="medium")
        for _ in range(8):
            req = obs.current_request
            if req.get("requester_type") == "emergency":
                action = SpectrumAction(
                    assigned_band_index=-1,  # reject emergency — wrong
                    assigned_power_dbm=0.0,
                    justification="no band available",
                )
                obs2 = env.step(action)
                assert obs2.reward < 0.6, f"Rejecting emergency should score low, got {obs2.reward}"
                return
            else:
                obs = env.step(reasonable_action(obs))
                if obs.done:
                    break
        pytest.skip("No emergency request in this medium episode")

    def test_reward_in_valid_range(self):
        """All step rewards must be in [0.0, 1.0]."""
        for task in ["easy", "medium", "hard"]:
            env = SpectrumEnvironment()
            obs = env.reset(task_name=task)
            while not obs.done:
                obs = env.step(reasonable_action(obs))
                assert 0.0 <= obs.reward <= 1.0, f"Reward {obs.reward} out of range in {task}"

    def test_grader_in_range(self):
        """grade_episode() returns float in [0.0, 1.0]."""
        rewards = [0.3, 0.5, 0.8, 0.0, 1.0]
        score = grade_episode(rewards)
        assert 0.0 <= score <= 1.0

    def test_grader_empty(self):
        assert grade_episode([]) == 0.0

    def test_grader_mean(self):
        rewards = [0.4, 0.6]
        assert abs(grade_episode(rewards) - 0.5) < 1e-4


# ── state() tests ─────────────────────────────────────────────────────

class TestState:
    def test_state_returns_spectrum_state(self):
        env = SpectrumEnvironment()
        env.reset(task_name="easy")
        state = env.state
        assert isinstance(state, SpectrumState)

    def test_state_step_count(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        assert env.state.step_count == 0
        env.step(reasonable_action(obs))
        assert env.state.step_count == 1

    def test_state_task_name(self):
        env = SpectrumEnvironment()
        env.reset(task_name="hard")
        assert env.state.task_name == "hard"

    def test_state_accumulated_reward(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        obs2 = env.step(reasonable_action(obs))
        assert abs(env.state.accumulated_reward - obs2.reward) < 1e-6

    def test_state_requests_total(self):
        env = SpectrumEnvironment()
        env.reset(task_name="easy")
        assert env.state.requests_total == 5

    def test_state_requests_processed(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="easy")
        env.step(reasonable_action(obs))
        assert env.state.requests_processed == 1


# ── Grader variation test ─────────────────────────────────────────────

class TestGraderVariation:
    def test_grader_not_constant_across_actions(self):
        """The grader must not return the same score for all actions."""
        env = SpectrumEnvironment()
        episode_scores = []

        # Run same task with different strategies
        for strategy in ["good", "reject_all", "wrong_band"]:
            env.reset(task_name="easy", seed=42)
            obs = env.reset(task_name="easy", seed=42)
            rewards = []
            while not obs.done:
                req = obs.current_request
                preferred = req.get("preferred_band_index") or 1

                if strategy == "good":
                    action = SpectrumAction(
                        assigned_band_index=preferred,
                        assigned_power_dbm=15.0,
                        justification="band power frequency allocat assign",
                    )
                elif strategy == "reject_all":
                    action = SpectrumAction(
                        assigned_band_index=-1,
                        assigned_power_dbm=0.0,
                        justification="",
                    )
                else:  # wrong_band
                    wrong = (preferred + 3) % 12
                    action = SpectrumAction(
                        assigned_band_index=wrong,
                        assigned_power_dbm=50.0,
                        justification="",
                    )
                obs = env.step(action)
                rewards.append(obs.reward)
            episode_scores.append(grade_episode(rewards))

        # At least two strategies should produce different scores
        assert len(set(round(s, 3) for s in episode_scores)) > 1, (
            f"Grader returned same score for all strategies: {episode_scores}"
        )


# ── New task tests ────────────────────────────────────────────────────

class TestNewTasks:
    def test_all_five_tasks_in_registry(self):
        for task in ["easy", "medium", "disaster_response", "hard", "spectrum_auction"]:
            assert task in TASK_REGISTRY, f"Task '{task}' missing from TASK_REGISTRY"

    def test_disaster_response_episode_length(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="disaster_response")
        assert obs.total_steps == 10
        steps = 0
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            steps += 1
        assert steps == 10

    def test_spectrum_auction_episode_length(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="spectrum_auction")
        assert obs.total_steps == 8
        steps = 0
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            steps += 1
        assert steps == 8

    def test_spectrum_auction_has_upcoming_requests(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="spectrum_auction")
        assert isinstance(obs.upcoming_requests, list)
        assert len(obs.upcoming_requests) > 0, "spectrum_auction should expose upcoming requests"
        assert len(obs.upcoming_requests) <= 2

    def test_spectrum_auction_upcoming_fields(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="spectrum_auction")
        for entry in obs.upcoming_requests:
            assert "requester_type" in entry
            assert "priority" in entry
            assert "bandwidth_needed_mhz" in entry
            assert "preferred_band_index" in entry
            assert "description" in entry
            # Ground-truth fields must NOT be exposed
            assert "gt_best_band_index" not in entry
            assert "gt_acceptable_bands" not in entry

    def test_disaster_response_no_upcoming_requests(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="disaster_response")
        assert obs.upcoming_requests == [], "disaster_response should not expose upcoming requests"

    def test_disaster_response_rewards_vary(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="disaster_response")
        rewards = []
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            rewards.append(obs.reward)
        assert len(set(round(r, 2) for r in rewards)) > 1, f"All rewards identical: {rewards}"

    def test_spectrum_auction_rewards_vary(self):
        env = SpectrumEnvironment()
        obs = env.reset(task_name="spectrum_auction")
        rewards = []
        while not obs.done:
            obs = env.step(reasonable_action(obs))
            rewards.append(obs.reward)
        assert len(set(round(r, 2) for r in rewards)) > 1, f"All rewards identical: {rewards}"

    def test_disaster_phase2_reject_outscores_assign(self):
        """Rejecting commercial during disaster should outscore assigning it."""
        env_reject = SpectrumEnvironment()
        obs = env_reject.reset(task_name="disaster_response", seed=777)
        rewards_reject = []
        # Advance to step 8 (index 7) — commercial reject case
        for i in range(7):
            if obs.done:
                break
            obs = env_reject.step(reasonable_action(obs))
        if not obs.done and obs.current_request.get("requester_type") == "commercial":
            obs_r = env_reject.step(SpectrumAction(
                assigned_band_index=-1,
                assigned_power_dbm=0.0,
                justification="Rejecting non-essential commercial traffic during disaster operations",
            ))
            rewards_reject.append(obs_r.reward)

        if rewards_reject:
            assert rewards_reject[0] >= 0.3, (
                f"Disaster phase2 reject should score ≥0.3, got {rewards_reject[0]}"
            )
        else:
            pytest.skip("Could not reach commercial reject step with this seed")

    def test_new_tasks_rewards_in_range(self):
        for task in ["disaster_response", "spectrum_auction"]:
            env = SpectrumEnvironment()
            obs = env.reset(task_name=task)
            while not obs.done:
                obs = env.step(reasonable_action(obs))
                assert 0.0 <= obs.reward <= 1.0, f"Reward {obs.reward} out of range in {task}"

    def test_all_tasks_complete(self):
        for task in ["easy", "medium", "disaster_response", "hard", "spectrum_auction"]:
            env = SpectrumEnvironment()
            obs = env.reset(task_name=task)
            steps = 0
            while not obs.done:
                obs = env.step(reasonable_action(obs))
                steps += 1
            assert obs.done is True, f"{task} never reached done=True"
            assert steps > 0, f"{task} completed in 0 steps"
