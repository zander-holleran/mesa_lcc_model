# LCC Traffic Model Documentation

Welcome to the documentation for the Little Cottonwood Canyon (LCC) Traffic Model -- a Mesa-based agent-based simulation of vehicle traffic in LCC, Utah.

For the project README and source code, see the [GitHub repository](https://github.com/zander-holleran/mesa_lcc_model).

---

## What's in this wiki

**[Getting Started](getting-started/installation.md)** -- Install dependencies, run your first simulation, and learn the project's terminology.

- [Installation & Setup](getting-started/installation.md) -- prerequisites, dependencies, data preparation
- [First Run](getting-started/first-run.md) -- guided walkthrough of `notebooks/season_run.ipynb`
- [Glossary](getting-started/glossary.md) -- definitions for every domain-specific term in the project

**[Domain Model](domain/lcc-context.md)** -- How the simulation models real-world LCC traffic.

- [LCC Context](domain/lcc-context.md) -- the real-world setting and policy questions
- [Person Behavior](domain/person-behavior.md) -- mode choice, beliefs, and learning
- [Vehicle Physics](domain/vehicle-physics.md) -- acceleration, braking, driver decisions
- [Road Network](domain/road-network.md) -- geometry, segments, and waypoints
- [Bus Service](domain/bus-service.md) -- dispatch, boarding, and bus-person relationships
- [Generation & Lifecycle](domain/generation-and-lifecycle.md) -- spawning logic, termination, and cross-cutting model behavior

**[Architecture](architecture/system-overview.md)** -- Code structure, agent types, and data systems.

- [System Overview](architecture/system-overview.md) -- season/day/step hierarchy
- [TrafficModel Lifecycle](architecture/traffic-model-lifecycle.md) -- step-by-step execution order
- [Agent Taxonomy](architecture/agent-taxonomy.md) -- all agent types and their responsibilities
- [Data Collection](architecture/data-collection.md) -- the 4-tier HybridDataCollector
- [Tolling System](architecture/tolling-system.md) -- Signal, Transform, and TollConfig composition

**[Experiments & Results](experiments/index.md)** -- Documentation of simulation experiments (coming soon).

---

## Quick links

- [Glossary](getting-started/glossary.md) -- look up any term
- [season_run.ipynb](https://github.com/zander-holleran/mesa_lcc_model/blob/main/notebooks/season_run.ipynb) -- the primary notebook
- [DEVELOPMENT.md](https://github.com/zander-holleran/mesa_lcc_model/blob/main/DEVELOPMENT.md) -- development workflow and testing protocol
