# Legal characterization fixture

This fixture is a four-cell, three-hour **synthetic** compound-flood payload
used by the PR-1 characterization tests. It contains no observed records,
household locations, person identifiers, credentials, or other private data.
The field names mirror the public NOAA/NWS/USGS/NASA contracts used by
Hyperion; the values are deterministic test values and are not copied from a
source response.

The fixture is released under [CC0 1.0](LICENSE). It is intentionally small
enough for API, lab, score, frontend, and historical-backtest tests to run
without a network call or a large checked-in dataset. `backtest.csv` is a
six-hour frozen event window with repeated locations across time, allowing the
temporal split gate to prove that timestamps—not random row order—separate
calibration from evaluation.
