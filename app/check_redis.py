import redis
import json

r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
job_id = "d2d787a7-90d5-4bdc-847c-e16ffb3476bb"
val = r.get(f"telecom_job:{job_id}")
if val:
    print(json.dumps(json.loads(val), indent=2))
else:
    print(f"Job {job_id} not found in Redis.")
