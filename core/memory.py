
import json
from datetime import datetime

def log(entry, path="aria_log.jsonl"):
    entry["timestamp"] = datetime.utcnow().isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
