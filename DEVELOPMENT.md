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

3. **Wait for approval** before coding

## Implementation

- Reference the plan file: `plan/feature-name.md`
- Follow the approved approach

## Testing Protocol

### Standard Tests
After functional changes:
1. Run tests related to the modified code
2. Show test output
3. If tests fail, analyze and fix
4. Don't mark task as complete until tests pass


### Optimization Changes Only

**IMPORTANT:** For any performance optimization or refactoring that should not change model behavior:

1. **Before making changes:**
```bash
   python tests/optimization_check.py --save-baseline
```

2. **Make optimization changes**

3. **After changes:**
```bash
   python tests/optimization_check.py --verify
```
   - This compares: steps, crashes, finished agents, model time series, trip logs
   - Shows performance impact (speedup/slowdown %)
   - **ALL outputs must match** - if they differ, the optimization changed behavior

4. **Show verification results** before completing task

5. **Clean up when done:**
```bash
   python tests/optimization_check.py --clean
```

**Baseline files location:** `tests/baselines/`

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