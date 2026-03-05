# Development Workflow

## Planning New Features

Before implementing any feature:

1. **Analyze existing implementation** (if any)
   - What does the current feature do?
   - How does it work technically?
   - What triggers it?

2. **Create plan file** in `plan/feature-name.md` with:
   - Current state analysis
   - Proposed solution
   - Comparison: new vs existing approach
   - Trade-offs and rationale

## Git Workflow

Before making any code changes:

1. **Create a feature branch** from main:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/short-description
   ```

2. **Make commits** with clear messages on the feature branch

3. **Push and create PR** when ready for review:
   ```bash
   git push -u origin feature/short-description
   ```

Branch naming conventions:
- `feature/description` - new features
- `fix/description` - bug fixes
- `refactor/description` - code refactoring
- `perf/description` - performance optimizations

## Implementation

- Reference the plan file: `plan/feature-name.md`
- Follow the approved approach
- Work on your feature branch (not main)

## Testing Protocol

### Standard Tests
After functional changes:
1. Run tests related to the modified code
2. Show test output
3. If tests fail, analyze and fix
4. Don't mark task as complete until tests pass


### Optimization Changes Only

**IMPORTANT:** For any performance optimization or refactoring that should not change model behavior:

#### Step 0: Create a feature branch
```bash
git checkout -b perf/short-description
```

#### Step 1: Save baselines (before making changes)
```bash
python tests/optimization_check.py --save-baseline
python tests/performance_free_flow.py --save-baseline --runs 1
python tests/performance_congestion.py --save-baseline --runs 1
```

#### Step 2: Make optimization changes

#### Step 3: Verify determinism (iterate until passing)
```bash
python tests/optimization_check.py --verify
```
- This compares: steps, crashes, finished agents, model time series, trip logs
- **ALL outputs must match** - if they differ, the optimization changed behavior
- If verification fails, fix the issue and re-run until it passes

#### Step 4: Verify performance (after optimization_check passes)
```bash
python tests/performance_free_flow.py --verify --runs 1
python tests/performance_congestion.py --verify --runs 1
```
- Reports: time change %, steps/sec change %
- Confirms outputs still match baseline

#### Step 5: Report results
Show verification results including:
- PASS/FAIL status for all three tests
- Performance comparison (speedup/slowdown %)

#### Step 6: Clean up baselines
```bash
python tests/optimization_check.py --clean
python tests/performance_free_flow.py --clean
python tests/performance_congestion.py --clean
```

**Baseline files location:** `tests/baselines/`

**Test configurations:**
| Test | Persons | Days | Traffic | Purpose |
|------|---------|------|---------|---------|
| optimization_check | 1000 | 1 | 50th %ile | Quick determinism check |
| performance_free_flow | 3000 | 3 | 50th %ile | Performance under normal traffic |
| performance_congestion | 3000 | 3 | 90th %ile | Performance under heavy traffic |

---

## Change Type Classification

**Optimization changes** include:
- Performance improvements (algorithm efficiency, caching, etc.)
- Code refactoring that shouldn't change behavior
- Data structure changes for speed
- Vectorization or parallelization

**Functional changes** include:
- New features
- Bug fixes that change output
- Changes to model logic or parameters