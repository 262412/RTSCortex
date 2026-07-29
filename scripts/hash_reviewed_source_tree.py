"""Hash the complete reviewed Worker source tree for runtime attestation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_reviewed_llm_pysc2_runtime import reviewed_source_tree_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--field",
        choices=(
            "reviewed_tree_sha256",
            "reviewed_tree_file_count",
            "reviewed_untracked_file_count",
        ),
    )
    arguments = parser.parse_args()
    manifest = reviewed_source_tree_manifest(arguments.repository)
    if arguments.field is not None:
        print(manifest[arguments.field])
        return
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
