# Person Behavior

Persons in the simulation make mode choice decisions, accumulate travel experience, and update their beliefs across days within a season.

---

## Mode Choice

Each `TrafficPersonAgent` chooses between car and bus at the moment of creation by comparing the **expected generalized cost** of each mode.

The generalized cost formula (from `compute_expected_generalized_cost()` in `traffic/agents/traffic_person_agent.py`):

```
effective_tt = expected_tt - uncertainty_multiplier * uncertainty
cost = value_of_time * experience_weight * effective_tt + toll
```

The person chooses whichever mode has the lower cost. Key inputs:

| Input | Car source | Bus source |
|-------|-----------|------------|
| `expected_tt` | `expected_tt_car` | `expected_tt_bus` |
| `uncertainty` | `tt_unc_car` | `tt_unc_bus` |
| `experience_weight` | `experience_weight_car` | `experience_weight_bus` |
| `toll` | `current_toll_car` (road toll) | `bus_user_fee` |

The `uncertainty_multiplier` acts as a risk-aversion parameter. Higher values penalize uncertain modes more, so a person with limited bus experience will perceive bus travel as riskier even if the expected time is similar to car.

---

## Belief System

Expected travel times are computed by `compute_experience_beliefs()` in `season/persons.py`. The formula blends a prior with exponentially time-decayed observations:

```
expected_tt = (prior_weight * prior + W * mu_data) / (prior_weight + W)
```

Where:

- `W` = sum of time-decayed observation weights: `sum(exp(-time_decay_rate * age_in_days))`
- `mu_data` = weighted average of observed travel times
- `prior` = the mode-specific prior (`prior_car` or `prior_bus`)

**Uncertainty** is inversely proportional to effective sample size, inflated by staleness:

```
base_unc = 1 / max(total_weight, epsilon)
uncertainty = base_unc * (1 + staleness_scale * staleness)
```

Where `staleness` is the number of days since the last trip on that mode.

---

## Prior Updating

After each trip, the mode-specific prior is nudged toward the realized travel time via `slow_prior_update()`:

```
prior_new = prior + eta * (realized_tt - prior)
```

With `eta = 0.05`, this is a very slow update -- the prior moves only 5% of the way toward each new observation. This reflects the assumption that baseline expectations change gradually.

---

## Experience Accumulation

The `SeasonPerson.record_experience()` method appends each completed trip to the person's `history` list. Records include:

- `day_index`, `mode`, `toll_paid`
- `realized_tt`, `wait_time`, `onboard_time`
- `cumtime_lost_min`, `realized_cost`

Between days, `update_beliefs_from_history()` recomputes `expected_tt` and `uncertainty` for both modes using the **complete** history up to that point. Older observations are weighted less via the exponential decay.

---

## Population Heterogeneity

Person traits are drawn from distributions defined in `PopulationParams` (`season/configs.py`):

| Parameter | Default distribution |
|-----------|---------------------|
| `value_of_time` | `lognorm(s=0.64, scale=40/60)` |
| `experience_weight_car` | `1.0` (scalar) |
| `experience_weight_bus` | `skewnorm(6, loc=1.15, scale=0.3)` |
| `prior_car` | `22.0` minutes |
| `prior_bus` | `60.0` minutes |
| `uncertainty_multiplier` | `1.0` (scalar) |
| `time_decay_rate` | `0.1` (scalar) |

Each field can be a scalar (same for everyone) or a frozen scipy distribution (heterogeneous population). The bus experience weight skews higher than car, reflecting that people generally perceive bus travel time as more costly than equivalent car travel time.
