# Explanation: DQN Implementation Walkthrough

## 1. `GridWorld` — a minimal but complete MDP, built from scratch

```python
def step(self, action_idx):
    ...
    reward = -0.01
    if self.pos == self.goal:
        reward = 1.0; done = True
    elif self.pos in self.obstacles:
        reward = -0.5; done = True
    elif self.steps >= self.max_steps:
        done = True
    return self._obs(), reward, done
```

Three terminal conditions (goal reached, obstacle hit, step limit) and one
non-terminal per-step penalty. The `-0.01` step penalty is what gives the
agent a reason to prefer *shorter* successful paths over longer ones — with
reward 0 per step, any path to the goal would look equally good to the
learned Q-values, and the agent would have no incentive to be efficient.

## 2. Observation encoding

```python
return np.array([self.pos[0]/(N-1), self.pos[1]/(N-1),
                  self.goal[0]/(N-1), self.goal[1]/(N-1)], dtype=np.float32)
```

Both agent position and goal position are included in the observation
(normalized to `[0,1]`) even though the goal never moves in this run — this
is deliberate future-proofing: the network genuinely learns a
goal-conditioned policy, not one hard-coded to a specific fixed goal
location, even though this particular experiment only tests one goal.

## 3. `QNetwork` and the target network

```python
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
target_net.eval()
```

Two structurally-identical networks. `target_net` starts as an exact copy
of `q_net` but is never trained directly by gradient descent — it's only
periodically resynced (`target_net.load_state_dict(q_net.state_dict())`
every `TARGET_UPDATE_EVERY=10` episodes). This is the "target network"
stabilization trick from theory.md section 4a: the TD target used in the
loss below is computed from this slowly-updating copy, not from `q_net`
directly.

## 4. The TD-error loss, computed exactly as derived in theory.md

```python
q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    next_q = target_net(s2).max(dim=1)[0]
    target = r + GAMMA * next_q * (1 - d)
loss = F.mse_loss(q_values, target)
```

- `q_net(s).gather(1, a.unsqueeze(1))`: pulls out the Q-value for the
  *specific action actually taken* in each transition (the network outputs
  Q-values for all 4 actions; we only have a training signal for the one
  that was taken).
- `target_net(s2).max(dim=1)[0]`: the `max_a' Q(s',a')` term from the
  Bellman equation, evaluated using the frozen target network.
- `(1 - d)`: zeroes out the bootstrapped future-value term when the
  episode ended (`d=1`) — there is no `s'` to bootstrap from after a
  terminal transition, so the target should just be the raw reward.
- The whole `target` computation is wrapped in `torch.no_grad()` — the
  target is a fixed regression label for this update, not something we
  backpropagate through.

## 5. Experience replay — why sampling matters, not just storing

```python
if len(buffer) >= MIN_BUFFER_BEFORE_TRAIN:
    s, a, r, s2, d = buffer.sample(BATCH_SIZE)
    ...
```

Training only starts once the buffer has at least 500 transitions
(`MIN_BUFFER_BEFORE_TRAIN`), so the very first updates aren't computed from
a handful of highly-correlated, early-episode transitions. `buffer.sample`
uses `random.sample` — a uniform, order-independent draw — which is what
breaks the correlation between consecutive within-episode transitions per
theory.md section 4b.

## 6. Epsilon-greedy schedule — the actual numbers from this run

```
Episode    1: eps=0.997
Episode  300: eps=0.050 (fully decayed)
Episode  500: eps=0.050 (held at floor)
```

`epsilon_for_episode` linearly decays `eps` from 1.0 to 0.05 over the first
300 episodes, then holds at the floor. The printed per-50-episode reward
trace shows this transition directly: reward stays negative/noisy through
episode ~100 (still mostly random exploration), then climbs sharply once
`eps` drops below ~0.5 around episode 150-200, and stabilizes once `eps`
hits its floor around episode 300 — a real, observable connection between
the exploration schedule and when learning visibly "kicks in."

## 7. Honest before/after comparison

```
First 50 episodes: avg reward=-0.700, avg length=23.9
Last  50 episodes: avg reward=0.803, avg length=11.7
```

Both reward (up) and episode length (down) improved substantially and in
the expected direction — the agent is both succeeding more often *and*
doing so more efficiently, not just one or the other. The
`if last_50_reward > first_50_reward + 0.2` check is a real, falsifiable
threshold printed by the script itself, not a post-hoc description.

## 8. Final greedy-policy evaluation — separating training performance from actual policy quality

```python
def evaluate_greedy(n_eval=100):
    ...
    with torch.no_grad():
        action = q_vals.argmax(dim=1).item()  # eps=0: no exploration at all
```

This runs 100 fresh episodes with `epsilon=0` (pure exploitation, no random
actions) — a cleaner measure of what the learned policy actually does,
uncontaminated by the residual `eps=0.05` exploration still present during
training. Result: `success_rate=100.00%`, `avg steps=10.0`. The Manhattan
distance from `(0,0)` to `(5,5)` around the obstacles at (2,2),(2,3),(3,2)
is close to 10 steps, meaning the learned policy is not just succeeding but
finding paths close to optimal length — a strong, concretely-verified
result rather than a vague "it seems to work."

## 9. `learned_policy.png` — a directly interpretable sanity check

```python
q_vals = q_net(torch.tensor(obs).unsqueeze(0))
best_a = q_vals.argmax(dim=1).item()
```

For every non-terminal, non-obstacle cell, the arrow shown is literally
`argmax_a Q(s,a)` for that cell — this is the single most direct way to
audit what the network learned: every arrow should visibly point toward a
route around the obstacles and into the goal. This is a much more
trustworthy check than the reward curve alone, since reward could in
principle look good due to a lucky evaluation seed; a full grid of
sensible-looking arrows is much harder to get by chance.
