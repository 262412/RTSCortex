# LLM-PySC2 worker bridge

This package stays on Python 3.9 and communicates with the Python 3.11 RTSCortex runtime
through the versioned JSON API. It contains no planner or model client.

The v0.1 initial delivery implements transport and validated action rendering. The live
`RTSCortexMainAgent` hook, SC2 installation, map setup, and execution feedback integration
belong to the real-environment milestone because this compute node does not currently have
StarCraft II installed.

The upstream source is pinned at `third_party/LLM-PySC2`; do not edit the submodule directly.
Any required environment hooks must be maintained as a small reviewed patch in `patches/`.
