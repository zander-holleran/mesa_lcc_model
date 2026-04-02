# Little Cottonwood Canyon Context

## The Setting

Little Cottonwood Canyon (LCC) is a narrow canyon in the Wasatch Range east of Salt Lake City, Utah. It provides access to two major ski resorts -- Snowbird and Alta -- via a single road: **SR-210 (Highway 210)**.

## The Problem

There is essentially **one road in and one road out**. No viable alternate routes exist. This creates severe traffic congestion during peak periods:

- **Weekend mornings** during ski season see heavy inbound traffic
- **Powder days** (fresh snowfall) generate surge demand that can exceed road capacity
- **Avalanche control** operations periodically close the road for unpredictable durations, trapping vehicles already in the canyon and blocking new entries

The combination of high demand, limited capacity, and stochastic closures makes LCC a compelling case study for transportation policy analysis.

## Policy Context

The Utah Department of Transportation (UDOT) has explored several interventions:

- **Tolling** -- congestion pricing to manage demand
- **Enhanced bus service** -- increased frequency, dedicated lanes, or priority boarding
- **Vehicle restrictions** -- requiring chains or traction devices during winter storms

This simulation enables testing these interventions in a controlled environment where person-level behavior, vehicle dynamics, and policy levers interact over both single days and full seasons.

## What the Model Captures

The simulation models the **inbound (uphill) direction** of SR-210 from the canyon mouth to the ski resorts. Key elements:

- Road geometry derived from real GIS data (curvature, speed limits, segment spacing)
- Vehicle arrival patterns calibrated to empirical vehicle count data
- Person-level mode choice between car and bus, informed by travel time beliefs
- Avalanche closures modeled as timed `BlockerAgent` instances
- Dynamic and static tolling via the composable `TollConfig` system
