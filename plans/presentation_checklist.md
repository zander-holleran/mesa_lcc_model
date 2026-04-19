# Presentation Wrap-Up Checklist

## 1. Model Validation (Speed-Density Benchmark)
- [ ] Run the full validation sweep (sweep_run.ipynb with `car_preference=0.94`, multiple traffic percentiles and seeds)
- [ ] Process Tier 2 data through Edie generalized cells in the benchmark notebook
- [ ] Fit Greenshields, Greenberg, Underwood to ABM scatter
- [ ] Overlay fitted curves on NCHRP CI bands — confirm parameters land within ranges
- [ ] Record fitted parameter table (v_f, k_j, k_0, v_0) with % deviation from literature means
- [ ] Export key figures: `kv_scatter_fit`, `kv_scatter_ci` (NCHRP overlay), sensitivity plots
- [ ] Write 2-3 sentence findings summary for the presentation

## 2. Policy Sweep (Toll + Bus Intervention Experiment)
- [ ] Design the sweep space:
  - **Toll regimes:** no toll, static flat tolls ($5, $10, $15, $20), dynamic PI toll (target ~30 vehicles, kp/ki combos)
  - **Bus regimes:** no bus (interval=0 or very high), 30-min, 15-min, 10-min, 5-min headways
  - **Traffic demand:** traffic_percentile at [20, 40, 60, 80, 90, 95, 99] (low to extreme)
  - **Seeds:** 3-5 per combo for variance estimation
- [ ] Enable `BusCostConfig.default()` for bus cost accounting
- [ ] Set population_size appropriately (4000+ for statistical power)
- [ ] Multi-day seasons (10-20 days) so belief formation / mode shift has time to stabilize
- [ ] Run the sweep (ParallelSweepRunner, estimate wall time first)
- [ ] Verify no failures; re-run any failed configs

## 3. Analysis of Sweep Results
- [ ] Load the sweep results parquet
- [ ] **Descriptive analysis:**
  - Summary table of key metrics by toll regime x bus regime
  - How does avg_tt, share_bus, avg_cum_time_lost, total_rev change across regimes?
  - Distribution of realized costs by mode
- [ ] **Best overall regime:** which toll+bus combo minimizes avg_cost_vot_standardized?
- [ ] **Robustness across demand levels:**
  - Which regime handles low traffic_percentile (20-40) best? (don't over-toll light days)
  - Which regime handles high traffic_percentile (90-99) best? (congestion control)
  - Which regime has lowest variance in outcomes across demand levels?
- [ ] **Bus system economics:** bus_cost_annual vs toll revenue collected — is it self-funding?
- [ ] **Mode shift dynamics:** how does share_bus evolve over the 10-20 day season?
- [ ] Export 4-6 analysis figures for slides

## 4. Presentation Slides — Content Assembly
- [ ] LCC background and problem context (see outline below)
- [ ] Model architecture slides (micro → macro)
- [ ] Validation results (benchmark figures)
- [ ] Policy experiment design
- [ ] Key results and findings
- [ ] Recommendations / takeaways
- [ ] Technical appendix (optional)

## 5. Technical Prep
- [ ] Ensure all figures export to `notebooks/figures/` with consistent sizing/style
- [ ] Git commit all analysis code and results before presentation
- [ ] Test that the key notebooks run end-to-end without errors

---

# Presentation Outline

## Slide 1: Title
**Agent-Based Modeling of Traffic Congestion and Transit Policy in Little Cottonwood Canyon**

## Slide 2-3: The LCC Problem
- SR-210 is a two-lane mountain highway serving Alta and Snowbird ski resorts
- Single access corridor — no alternative routes
- Severe congestion on powder days (traffic percentile 90+): 2+ hour travel times for a 14-mile road
- UDOT has been evaluating interventions: tolling, enhanced bus service, gondola
- The impending question: what combination of pricing + transit delivers the best outcomes?

## Slide 4: Research Question
- Can an agent-based model capture the emergent congestion dynamics of a constrained mountain corridor?
- What toll + bus service combination minimizes traveler cost across varying demand levels?
- How do travelers adapt their mode choice over a season of repeated trips?

## Slide 5-6: Model Architecture — Micro Level
- **Road geometry:** GeoDataFrame of ~50m waypoints along SR-210, with speed limits, grade, curvature
- **Vehicle physics:** empirical acceleration curves (from stop-sign data), car-following with braking distances
- **Step separation:** all agents compute speed, then all move — prevents order-dependent artifacts
- **Crash and closure events:** stochastic incidents that block road segments
- Diagram: vehicle → road segment → speed/brake decision → move

## Slide 7: Model Architecture — Agent Decision Making
- **TrafficPersonAgent:** each person decides car vs bus before entering the simulation
- **Generalized cost:** `VOT * experience_weight * expected_tt + toll`
- Uncertainty-adjusted: persons with less experience or stale beliefs face higher effective uncertainty
- Mode choice is endogenous — it emerges from beliefs, not assigned exogenously

## Slide 8: Model Architecture — Belief Formation (Season Level)
- **SeasonPerson** persists across days, accumulating trip history
- Exponential time-decay weighting on past experiences
- Bayesian-style prior + data fusion: `expected_tt = (prior_wt * prior + W * mu_data) / (prior_wt + W)`
- Priors update slowly via `prior += eta * (realized - prior)`
- Heterogeneous population: VOT drawn from lognorm, experience weights from skewnorm

## Slide 9: Tolling System
- Composable architecture: **Signal** (reads model state) → **Transform** (maps to toll) → **TollConfig** (wrappers)
- Static flat tolls, piecewise-linear, step function, PI controller (feedback-driven)
- PI transform: drives toward a target vehicle count, integral resets when congestion clears

## Slide 10: Bus System
- Buses dispatched at configurable headways, fixed capacity
- Bus cost estimation (post-hoc, zero sim overhead): fleet sizing, labor, capital, ops
- Based on TCRP, FTA, BLS, NTD parameters

## Slide 11: Data Collection & Orchestration
- **HybridDataCollector** — 4 tiers: scalars/histograms, spatial snapshots, events, full state
- **SeasonOrchestrator:** runs N days, resets model between days, persists SeasonPersons
- **ParallelSweepRunner:** cross-product sweep space, ProcessPoolExecutor, aggregated results

## Slide 12: Technical Backend
- **Mesa** (ABM framework) for agent scheduling and model structure
- **NumPy structured arrays** (`VehicleState`) for vectorized vehicle physics — not per-agent Python objects
- **SciPy** for statistical distributions (lognorm VOT, skewnorm experience weights) and curve fitting
- **GeoPandas** for road geometry
- **ProcessPoolExecutor** for parallel sweeps across CPU cores
- **Parquet** for output storage (trip logs, day summaries, season summaries)
- Deterministic seeding throughout — reproducible results given identical config

## Slide 13: Validation — Speed-Density Curves
- Tier 2 spatial data → Edie generalized k/v cells
- Fitted Greenshields, Greenberg, Underwood models to ABM output
- Comparison against NCHRP 17-65 literature CI bands
- [Insert benchmark figures here]
- Key finding: ABM parameters fall within / near established ranges for a steep two-lane highway

## Slide 14: Policy Experiment Design
- Sweep space table: toll regimes x bus headways x traffic demand levels
- N total configurations, M seeds each
- Multi-day seasons (belief formation needs time to converge)

## Slide 15-17: Results — Descriptive
- Summary heatmap: avg_tt by toll x bus x demand
- Mode share evolution over season days
- Travel time distributions by regime

## Slide 18: Results — Best Overall Regime
- Which toll+bus combo minimizes VOT-standardized cost?
- Revenue vs bus system cost — is it self-funding?

## Slide 19: Results — Demand Robustness
- Low demand days: does the toll unnecessarily penalize light traffic?
- High demand days: which regime most effectively controls congestion?
- Variance across demand levels — which regime is most stable?

## Slide 20: Conclusions & Recommendations
- Key takeaway on optimal intervention mix
- Limitations: single corridor, no demand elasticity beyond mode choice, no network effects
- Future work: gondola as third mode, dynamic headway adjustment, multi-season adaptation

## Appendix (if needed)
- Detailed parameter tables
- Full sensitivity analysis figures
- Determinism/reproducibility protocol
