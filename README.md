# Hyperion

**Urban compound flood intelligence. The next six hours, not the next thirty years.**

Hyperion produces a calibrated 0 to 6 hour nowcast and an operational score for compound urban flooding, where rain, tide, river and terrain stack on top of each other.

This repository documents the method and architecture. The implementation lives in a private repository.

---

## The gap this exists for

Flood risk products tell a city what the next thirty years look like, scored per parcel, for insurance and planning. That work matters and is well served.

An operator on a Tuesday afternoon has a different question. It is raining, the tide is coming in, and they need to decide in the next hour whether to close an underpass, move equipment, or pre-position a crew. A thirty year risk map does not answer that.

**The gap is operational, not actuarial.** Hyperion is built for the window where a decision still changes the outcome.

This is also why the project does not try to be a per-address risk score. That product exists. This one does not compete with it.

---

## Compound flooding

Most flood tools model one driver well. Real urban flooding is usually several at once:

- **Rain** falling faster than drainage can clear
- **Tide** raising the level the drainage has to discharge into
- **River** stage backing up through the same system
- **Terrain** deciding where the water that cannot leave will collect

Individually, each may stay under its own threshold. Together they produce a street that floods anyway. Hyperion models the combination rather than the components in isolation.

---

## Method

```
Federal observation and forecast data
-> Event Lab, per region pipeline built on real event labels
-> Model, gradient boosting and XGBoost, guarded by a leakage inspector
-> Composite nowcast, 0 to 6 hour, calibrated
-> Ops score, alert, report
-> Authenticated API, operator surface
```

**Real events, not synthetic ones.** The models are trained and validated against labeled data from Hurricanes Helene and Idalia. Big Bend serves as the anchor lab because it carries the cleanest event labels.

**The leakage inspector is not optional.** In flood modeling it is very easy to build a model that appears excellent and is quietly reading a feature that already contains the answer. Every model run passes leakage inspection before any result is reported. A number that has not passed it does not exist.

**Calibration over accuracy.** An operational score has to mean the same thing every time it is issued. A probability that is well calibrated and modest is more useful to someone making a call than a confident number that is right on average.

---

## Phases

| Phase | Region | Why |
| --- | --- | --- |
| 1 | Florida and Louisiana | US federal data is free, clean and well labeled. The right place to prove the core. |
| 2 | Recife, Brazil | Real impact, fragmented data. Only worth attempting once the core is stable. |

The ordering is deliberate. Recife is where this matters most and where the data is hardest. Starting there would have meant fighting the data and the method at the same time. Phase 1 exists to make the method the only unknown when phase 2 begins.

Active labs in phase 1: Tampa Bay, Big Bend (the Helene and Idalia anchor), and New Orleans compound.

---

## Stack

| Layer | Technology |
| --- | --- |
| Pipeline | Python, multi-lab structure with a canonical lab template |
| Models | Gradient boosting, XGBoost, leakage inspection |
| Service | FastAPI, authenticated, deployed on Hetzner |
| Operator surface | Web frontend consuming the API |
| Data | Federal observation and forecast sources, NASA imagery |

The frontend consumes the existing API rather than reimplementing any flood logic in the browser. Compute is deliberately modest: data and API on commodity hardware, GPU only if training demands it.

---

## What this is not, yet

- Not a commercial product
- Recife is not in production
- No per-address insurance scoring, and that is a design decision rather than a missing feature

---

## Why this is public

The implementation is private. The method is not, because a forecasting method that cannot be examined should not be trusted by anyone making an operational decision with it.

Part of [Mycellium Lab](https://mycelliumlab.com).

---

**Cesar Augusto** · AI Systems Engineer, Mycellium Lab
[GitHub](https://github.com/cesaremcasa) · [LinkedIn](https://www.linkedin.com/in/cesar-augusto-22943a351/) · [korvo.dev](https://korvo.dev)
