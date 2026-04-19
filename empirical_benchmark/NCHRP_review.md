# NCHRP 17-65 Parameter Review for SR-210 Benchmark Models

**Source Document:** Washburn, S.S., Watson, D., Bian, Z., Luttinen, T., Al-Kaisy, A., Jafari, A., Dowling, R., & Elias, A. (2018). *Improved Analysis of Two-Lane Highway Capacity and Operational Performance.* NCHRP Web-Only Document 255. National Academies Press. DOI: 10.17226/25179.

**PDF:** [`empirical_benchmark/25179.pdf`](25179.pdf)

**Purpose:** Extract empirical parameter estimates from this 679-page report to validate and narrow the SR-210 engineering estimate ranges used in [`classical_traffic_models_benchmark.ipynb`](classical_traffic_models_benchmark.ipynb).

---

## 1. Free-Flow Speed (v_f)

| Value / Range | Source | Road Type | Measured/Modeled |
|---|---|---|---|
| 45–65 mph (HCM FFS curves) | [Figure 1-1, p.2](25179.pdf#page=11) (HCM 2010, Exhibit 15-2) | Two-lane highways, general | Model-derived |
| BFFS ≈ posted speed + 10 mph | [Section 1.2.5, p.7](25179.pdf#page=16) (HCM 2010) | General guidance | Rule of thumb |
| FFS = BFFS − a × HV% | [Eq 3-1, p.65](25179.pdf#page=74) (Washburn et al. 2018) | All two-lane segment types | Regression from simulation |
| BFFS_HC = Min(BFFS_T, 44.32 + 0.3728 × BFFS_T − 6.868 × HorizClass) | [Eq 3-3, p.66](25179.pdf#page=75) | Horizontal curves | Model (R² = 0.996) |
| ~59–62 mph (95–100 km/h, 0% HV) | [Figure 2-12, p.46](25179.pdf#page=55) (German HBS 2015) | Two-lane, bendiness class 1, grade class 1 | Empirical curves |
| Low-speed highway threshold: < 50 mph | [Table 3-23, p.87](25179.pdf#page=96) | Distinct LOS criteria for low-speed facilities | Policy classification |

### SR-210 Interpretation

SR-210 has posted speed limits of 25–50 mph, so BFFS ≈ 35–60 mph by the HCM rule ([p.7](25179.pdf#page=16)). However, with HorizClass 3–5 (tight curves, radius <650 ft per [Table 2-13, p.56](25179.pdf#page=65)) and Vertical Class 5 (grades >9%, length >0.3 mi per [Table 2-14, p.57](25179.pdf#page=66)), the FFS drops substantially. The document classifies SR-210 as a **low-speed highway (< 50 mph)** requiring different LOS thresholds ([Table 3-23, p.87](25179.pdf#page=96)).

**Conclusion:** The current v_f range of **34–40 mph is well-supported** as a realistic passenger-car FFS on the steepest sections.

---

## 2. Capacity (veh/hr/direction)

| Value / Range | Source | Road Type | Measured/Modeled |
|---|---|---|---|
| 1700 pc/h directional (base) | [Section 1.2.7, p.8](25179.pdf#page=17) (HCM 2010) | Two-lane highway, ideal conditions | HCM standard |
| 3200 pc/h two-way (max) | [Section 1.2.7, p.8](25179.pdf#page=17) (HCM 2010) | Two-lane highway, ideal conditions | HCM standard |
| ~1200–1400 veh/h (0% HV) | [Figure 2-12, p.46](25179.pdf#page=55) (German HBS 2015) | Two-lane, grade class 1 | Empirical |
| ~800–1000 veh/h (10–20% HV) | [Figure 2-12, p.46](25179.pdf#page=55) (German HBS 2015) | Two-lane, grade class 1 | Empirical |
| ~1400–1600 veh/h (onset of congestion) | [Figure 1-5, p.9](25179.pdf#page=18) (Luttinen et al. 2005) | Finnish arterial highway 4, grade-separated | Field data |
| Capacity drop: 300–400 pc/h below capacity under congestion | [p.8](25179.pdf#page=17) (Finnish studies cited by Washburn) | Congested two-lane | Field-measured |

### SR-210 Interpretation

The base 1700 pc/h assumes ideal conditions (level terrain, no HV, standard lane widths). SR-210's extreme grades (up to 11%), tight curves, and no-passing constraints degrade capacity severely. From [Figure 2-11, p.43](25179.pdf#page=52), a 5-mile 6% upgrade with 30% trucks drops mixed-flow speed to ~30 mph at capacity, with capacity itself reduced to ~1200–1400 veh/hr/lane. For SR-210's steeper grades, capacity is even lower.

**Conclusion:** The current range of 600–2000 veh/hr/dir should **narrow to approximately 700–1400 veh/hr/dir**.

---

## 3. Optimum Density (k_0) — Density at Capacity

| Value / Range | Source | Road Type | Measured/Modeled |
|---|---|---|---|
| 20–25 veh/km (32–40 veh/mi) | [p.8](25179.pdf#page=17) (Finnish studies) | Two-lane highway | Field-measured |
| LOS E boundary: 40 veh/km (64 veh/mi) | [Table 2-3, p.14](25179.pdf#page=23) (German HBS 2001) | Two-lane, density as service measure | Policy threshold |
| Follower density at LOS D/E boundary: 12.0 followers/mi/ln (high-speed), 15.0 (low-speed) | [Table 3-23, p.87](25179.pdf#page=96) (Washburn et al. 2018) | Two-lane highways | Simulation-derived |
| Implied k_0 from capacity/FFS: 700–1400 / 35–55 ≈ 13–40 veh/mi | Derivable from [Eq 3-5, p.67](25179.pdf#page=76) and Section 1.2.7 | SR-210 estimate | Derived |
| German HBS speed-flow curves show capacity at ~800–1400 veh/h depending on HV% | [Figure 2-12, p.46](25179.pdf#page=55) (HBS 2015) | Two-lane, grade class 1 | Empirical curves |

### SR-210 Interpretation

The Finnish field data places optimum density at 20–25 veh/km = 32–40 veh/mi for generic two-lane highways ([Section 1.2.7, p.8](25179.pdf#page=17): "As traffic density exceeds optimum density (approximately 20–25 veh/km) the maximum flow rate is approximately 300–400 pc/h lower than capacity"). This is the density at which flow peaks — exactly the physical meaning of k_0 in the Underwood model.

For SR-210, the optimum density should be **lower** than the generic two-lane value because:
- Steep grades (up to 11%) cause platoons to form at lower densities, reaching capacity sooner.
- The German HBS LOS E boundary of 40 veh/km = 64 veh/mi ([Table 2-3, p.14](25179.pdf#page=23)) represents near-capacity operation, suggesting k_0 should be well below 64 veh/mi.
- From the NCHRP speed model ([Eq 3-5, p.67](25179.pdf#page=76)), ATS = FFS at flows below 100 veh/h, meaning the free-flow regime is narrow. With SR-210 capacity estimated at 700–1400 veh/hr and capacity speeds of 25–40 mph, k_0 = capacity/speed ≈ 18–56 veh/mi, with the lower end more plausible for steeper segments.

**Conclusion:** **Recommended range: 15–50 veh/mi.** The Finnish field optimum of 32–40 veh/mi anchors the mid-range for generic two-lane highways, but SR-210's extreme grades justify the lower bound extending to ~15 veh/mi.

---

## 4. Speed Constant (v_0) — Greenberg Model

The Greenberg model defines v = v_0 × ln(k_j / k), where v_0 is the speed at which flow is maximized (i.e., the speed at capacity). NCHRP 17-65 does not use the Greenberg model, but several passages constrain what v_0 should be for a two-lane highway like SR-210.

| Value / Implied Range | Source | Derivation |
|---|---|---|
| ~30 mph (mixed-flow speed at capacity, 5-mi 6% upgrade, 30% trucks) | [Figure 2-11, p.43](25179.pdf#page=52) (Dowling et al. 2014) | Speed-flow curve shows capacity speed dropping to ~25–35 mph on steep upgrades with heavy vehicles |
| S_calib,cap (speed at capacity) in mixed-flow model | [Eq 2-14, p.47](25179.pdf#page=56) | Defines the speed at which flow reaches capacity — directly analogous to v_0 |
| ATS at capacity ≈ FFS − m × (v_d − 0.1)^p | [Eq 3-5, p.67](25179.pdf#page=76) | At capacity flow (~700–1400 veh/hr for SR-210), ATS drops well below FFS |
| Speed at LOS E: ~40–50 km/h (25–31 mph) for grade class 3–4, 10–20% HV | [Figure 2-12, p.46](25179.pdf#page=55) (German HBS 2015) | Speed-flow curves for severe vertical alignment show capacity speeds of 25–35 mph |
| Truck crawl speeds on 8–10% grades: 22–30 mph | [Figures 3-3 to 3-5, pp.79–80](25179.pdf#page=88) | On SR-210's steepest grades, heavy vehicles set a speed floor that constrains the entire traffic stream |
| Finnish highway speed at ~1400 veh/h: ~60–70 km/h (37–43 mph) | [Figure 1-5, p.9](25179.pdf#page=18) (Luttinen et al. 2005) | Speed at near-capacity flow on a flat two-lane highway — upper bound for v_0 |
| Implied from q_max and k_j: v_0 = q_max × e / k_j | Derivable from Greenberg critical point | With q_max = 700–1400 veh/hr and k_j = 60–150 veh/mi → v_0 = 13–63 mph; constrained by grade to lower half |

### SR-210 Interpretation

The Greenberg v_0 parameter represents the speed at maximum flow (capacity). For SR-210, several NCHRP evidence lines converge:

**Lower bound (~10–15 mph):** On the steepest SR-210 segments (9–11% grade), truck crawl speeds of 22–25 mph ([Figures 3-3 to 3-5, pp.79–80](25179.pdf#page=88)) set a ceiling for the traffic stream in no-passing zones. With passenger cars unable to pass, the mixed-flow speed at capacity approaches truck crawl speed. Figure 2-11 ([p.43](25179.pdf#page=52)) shows this effect explicitly: on a 5-mile 6% upgrade with 30% trucks, the mixed-flow speed-flow pattern shows capacity speed of ~25–30 mph. SR-210 has steeper grades and heavier no-passing constraints, pushing v_0 lower.

**Upper bound (~35–40 mph):** The Finnish field data ([Figure 1-5, p.9](25179.pdf#page=18)) shows speed at near-capacity flow (~1400 veh/h) of ~60–70 km/h (37–43 mph) on a flat two-lane highway. This represents the maximum plausible v_0 for any two-lane highway. SR-210's grades ensure v_0 is well below this.

**Mid-range (~20–30 mph):** The German HBS speed-flow curves ([Figure 2-12, p.46](25179.pdf#page=55)) show that for grade class 3–4 with 10–20% heavy vehicles, speed at capacity falls to approximately 40–50 km/h (25–31 mph). SR-210's Vertical Class 5 geometry (grades > 9%, [Table 2-14, p.57](25179.pdf#page=66)) is more severe than the HBS grade classes shown, further reducing v_0.

**Cross-check from Greenberg critical point:** q_max = v_0 × k_j / e. With SR-210 capacity of 700–1400 veh/hr (Section 2) and k_j of 60–150 veh/mi (Section 5): v_0 = q_max × e / k_j = (700–1400) × 2.718 / (60–150) ≈ 13–63 mph. The physically plausible subset (given grade constraints) is the lower half of this range.

**Conclusion:** **Recommended range: 10–35 mph.** The lower bound reflects SR-210's steepest segments where truck crawl speeds dominate; the upper bound reflects capacity speed on gentler lower sections. The literature Greenberg calibrations (Drake: 32.8 mph on a flat expressway; Lu & Meng Beijing: 19.1 mph; Lu & Meng JJT: 29.2 mph) bracket this range.

---

## 5. Jam Density (k_j)

NCHRP 17-65 does **not** use classical speed-density models (Greenshields/Greenberg/Underwood) and does not report jam density directly. Indirect evidence:

| Value / Implied Range | Source | Derivation |
|---|---|---|
| LOS F: > 40 veh/km (> 64 veh/mi) | [Table 2-3, p.14](25179.pdf#page=23) (German HBS) | Density above this = breakdown |
| Max follower density observed: ~22–32 followers/mi | [Tables 3-21 and 3-22, p.87](25179.pdf#page=96) (Washburn et al. 2018) | Experimental design, LOS D conditions |
| Implied k_c from capacity/speed | Derivable from [Eq 3-5, p.67](25179.pdf#page=76) | At capacity: k_c = capacity_flow / ATS_at_capacity |

### SR-210 Interpretation

For a low-speed two-lane highway, if capacity speed is ~25–35 mph and capacity is ~700–1400 veh/hr, then k_c = capacity/speed = 20–56 veh/mi. Since k_j = 2×k_c for Greenshields, this implies k_j ≈ 40–112 veh/mi for SR-210. Combined with the German LOS F threshold of 64+ veh/mi ([Table 2-3, p.14](25179.pdf#page=23)), the upper bound of 250 veh/mi is almost certainly too high for a mountain two-lane road.

**Conclusion:** The current range of 50–250 veh/mi should **narrow to approximately 60–150 veh/mi**.

---

## 6. Truck Upgrade Speed Curves

From [Figures 3-3 to 3-5, pp.79–80](25179.pdf#page=88) and [Equation 3-20, p.81](25179.pdf#page=90):

| Grade | SUT Crawl Speed | Intermediate ST Crawl Speed | Interstate ST Crawl Speed |
|---|---|---|---|
| 6% | ~45 mph | ~35 mph | ~35 mph |
| 8% | ~30 mph | ~30 mph | ~25 mph |
| 10% | ~25 mph | ~25 mph | ~22 mph |

**Upgrade speed model:** V = 75 + a×L + b×L² + c×L³ ([Eq 3-20, p.81](25179.pdf#page=90))

Coefficients for each grade slope (1–10%) and truck type are provided in:
- Single-unit truck: [Table 3-17, p.81](25179.pdf#page=90)
- Intermediate semi-trailer: [Table 3-18, p.81](25179.pdf#page=90)
- Interstate semi-trailer: [Table 3-19, p.82](25179.pdf#page=91)

These are directly usable for calibrating heavy vehicle and bus behavior in the SR-210 ABM.

---

## 7. Vertical Alignment Classification for SR-210

From [Table 2-14, p.57](25179.pdf#page=66): SR-210 segments with grades >9% and segment lengths >0.3 mi classify as **Vertical Class 5** (most severe). This is the highest grade classification in the NCHRP methodology, meaning the steepest FFS-HV% slope coefficients from [Table 3-1, p.66](25179.pdf#page=75) apply:

| Coefficient | Vertical Class 5 Value |
|---|---|
| a0 | −0.38360 |
| a1 | 0.01074 |
| a2 | 0.01945 |
| a3 | −0.69848 |
| a4 | 0.01069 |
| a5 | 0.12700 |

The speed-flow slope coefficients for Vertical Class 5 from [Table 3-2, p.69](25179.pdf#page=78) (passing zone / passing constrained segments):

| Coefficient | Vertical Class 5 Value |
|---|---|
| b0 | 23.9144 |
| b1 | −0.6925 |
| b2 | 1.9473 |
| b3 | Varies |
| b4 | Varies |
| b5 | 3.5115 |

---

## 8. Speed-Flow Relationship Shape

The report confirms that the speed-flow relationship for two-lane highways is **concave-up**, not linear as assumed in HCM 2010 ([Section 1.2.1, p.2](25179.pdf#page=11); [Section 4.1.2, p.91](25179.pdf#page=100)). This is consistent with the Greenberg and Underwood model shapes used in the benchmark notebook.

The proposed average speed model is:

**ATS = FFS − m × (v_d − 0.1)^p** ([Eq 3-5, p.67](25179.pdf#page=76))

where:
- v_d = flow rate in the analysis direction (1000s of veh/h)
- m = speed-flow slope coefficient (varies by vertical class, segment type, length, HV%)
- p = power coefficient (varies by segment type)

For flow rates < 100 veh/h, ATS = FFS (speed is unaffected by traffic). This provides an empirical anchor: the free-flow regime extends to approximately 100 veh/h directional flow.

---

## Summary: Recommended Updated SR-210 Engineering Estimates

| Parameter | Current Range | NCHRP-Informed Range | Change | Key Citation |
|---|---|---|---|---|
| v_f (mph) | 34–40 | **34–40** | Confirmed | HCM BFFS rule + grade/curve corrections ([p.7](25179.pdf#page=16), [p.66](25179.pdf#page=75)) |
| k_j (veh/mi) | 50–250 | **60–150** | Narrowed | German HBS LOS F > 64 veh/mi ([Table 2-3, p.14](25179.pdf#page=23)); Greenshields k_j=2k_c derivation |
| k_0 (veh/mi) | 15–40 | **15–50** | Widened high end | Finnish field optimum 32–40 veh/mi ([p.8](25179.pdf#page=17)); SR-210 grade constraints lower the low end |
| v_0 (mph) | — | **10–35** | New | Mixed-flow speed at capacity on steep upgrades ([Fig 2-11, p.43](25179.pdf#page=52)); truck crawl speeds ([Figs 3-3–3-5, pp.79–80](25179.pdf#page=88)); German HBS grade-class curves ([Fig 2-12, p.46](25179.pdf#page=55)) |
| Capacity (veh/hr/dir) | 600–2,000 | **700–1,400** | Narrowed | HCM base 1700 pc/h ([p.8](25179.pdf#page=17)) degraded by grade/HV; German HBS ([Fig 2-12, p.46](25179.pdf#page=55)) |

---

## Additional Notes

- **Follower density** is the recommended service measure for two-lane highways in the new NCHRP methodology, replacing PTSF and ATS ([Section 2.3.7, p.37](25179.pdf#page=46)). This is calculated as FD = (PF/100) × (v_d/S) in units of followers/mi/ln ([Eq 3-24, p.87](25179.pdf#page=96)).
- **Critical headway** for identifying following vehicles is **2.5 seconds** ([p.62](25179.pdf#page=71)), revised down from the HCM's 3.0 seconds.
- **Capacity on opposing flow:** Directional capacity is reduced when opposing flow exceeds ~400–450 veh/h ([p.8](25179.pdf#page=17), citing Luttinen 2001a). For SR-210 where opposing (downhill) traffic shares the same roadway, this interaction matters.
- The document does **not** reference Greenshields, Greenberg, or Underwood models. The classical speed-density models are not part of the HCM two-lane highway methodology. The NCHRP methodology uses empirical regression models calibrated from microsimulation (SwashSim) rather than theoretical fundamental diagrams.
