# Test fixture provenance

`legacy_full_match_characterization.jsonl` is a lossless metrics projection of the real
`planner_reflection_memory_reflex-20260714T025618Z` journal. It retains every Decision,
ExecutionReport, Planning module output, and terminal EpisodeResult while removing observation
bulk and unrelated module telemetry. The readable one-event-per-line fixture freezes the v1.0
baseline used to validate report/replay compatibility; it is not synthetic gameplay data.

The source journal contained 4,232 events. The projection contains 1,948 events and must retain
these characterization totals: 946 decisions, 754 fallback decisions, 903 execution reports,
782 tracked control NoOps, 121 meaningful commands, 30 meaningful successes, 71 terminal
cancellations, and 39 unique validation-rejected command IDs.
