#!/bin/bash
set -e

DATABASE_PATH="${1:-data/database.jsonl}"

if [ ! -f "$DATABASE_PATH" ]; then
    echo "No duplicates found (database file does not exist)"
    exit 0
fi

PYTHON_SCRIPT="
import json
import sys
from collections import Counter

database_path = sys.argv[1]

try:
    with open(database_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print('No duplicates found (database file does not exist)')
    sys.exit(0)

if not lines:
    print('No duplicates found (database is empty)')
    sys.exit(0)

sample_keys = []
for line in lines:
    sample = json.loads(line)
    sample_keys.append((sample['model_id'], sample['prompt_text'], sample['attempt_number']))

counts = Counter(sample_keys)
duplicates = [(key, count) for key, count in counts.items() if count > 1]

if duplicates:
    print(f'Found {len(duplicates)} duplicate key(s):')
    for key, count in duplicates:
        model_id, prompt_text, attempt_number = key
        truncated_prompt = prompt_text[:40] + '...' if len(prompt_text) > 40 else prompt_text
        print(f'  - model_id={model_id}, prompt_text=\"{truncated_prompt}\", attempt_number={attempt_number} (appears {count} times)')
    sys.exit(1)
else:
    print('No duplicates found')
    sys.exit(0)
"

python3 -c "$PYTHON_SCRIPT" "$DATABASE_PATH"
