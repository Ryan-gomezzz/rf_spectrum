import sys
sys.path.insert(0, '.')
from server.spectrum_environment import SpectrumEnvironment, grade_episode
from models import SpectrumAction

env = SpectrumEnvironment()

for task in ["easy", "medium", "hard"]:
    obs = env.reset(task_name=task)
    print(f"\nTask: {task}, Steps: {obs.total_steps}, Done: {obs.done}")

    rewards = []
    while not obs.done:
        req = obs.current_request
        action = SpectrumAction(
            assigned_band_index=req.get("preferred_band_index", 1) or 1,
            assigned_power_dbm=min(req.get("power_dbm", 20.0), 30.0),
            justification=f"Assigning {req.get('requester_type', 'unknown')} to preferred band with power compliance"
        )
        obs = env.step(action)
        rewards.append(obs.reward)
        print(f"  Step {obs.step_number}: reward={obs.reward:.3f}, error={obs.last_action_error}")

    score = grade_episode(rewards)
    print(f"  Episode score: {score:.4f}")
    print(f"  Reward variance: {max(rewards) - min(rewards):.3f} (MUST be > 0)")

# Also test with deliberately bad actions to prove reward varies
print("\n--- Testing reward variance with bad actions ---")
obs = env.reset(task_name="easy")
bad_action = SpectrumAction(assigned_band_index=0, assigned_power_dbm=50.0, justification="yolo")
obs_bad = env.step(bad_action)
print(f"Bad action reward: {obs_bad.reward:.3f}")

obs = env.reset(task_name="easy")
good_req = obs.current_request
good_action = SpectrumAction(
    assigned_band_index=good_req.get("preferred_band_index", 1) or 1,
    assigned_power_dbm=20.0,
    justification="Assigning commercial request to licensed band with appropriate power level"
)
obs_good = env.step(good_action)
print(f"Good action reward: {obs_good.reward:.3f}")
print(f"Rewards differ: {obs_bad.reward != obs_good.reward} (MUST be True)")
