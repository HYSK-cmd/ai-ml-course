# HLD: <system name>

> Week 7 deliverable -- copy to `docs/w7_url_shortener.md`. Target: 800-1500 words + one diagram
> (ASCII or Mermaid). Interview-style: every number is derived, every choice names its trade-off.

## 1. Requirements
**Functional** (what it does -- 3 to 5 bullets)
**Non-functional** (availability target, latency p99, durability, consistency needs, scale)
**Out of scope**

## 2. Back-of-envelope estimates
Writes/day -> writes/s (avg and peak), read:write ratio -> reads/s, storage per record x records x
retention, bandwidth, cache size for the hot set (80/20). Show the arithmetic.

## 3. API
Endpoints with request/response shapes, status codes, idempotency, auth, rate limits.

## 4. Data model
Entities, keys, indexes. Which store and WHY (SQL vs KV vs wide-column). Partition key choice.

## 5. High-level design
Diagram: clients -> LB -> services -> cache -> DB / queues. Walk through one read and one write.

## 6. Deep dives (pick 2-3)
e.g. ID generation (base62 counter vs hash vs Snowflake), cache strategy and invalidation,
sharding + consistent hashing, replication and failover, hot keys, abuse / rate limiting.

## 7. Trade-offs and bottlenecks
What breaks first at 10x? What did you choose NOT to do and why? Consistency vs availability choices.

## 8. AWS mapping
Map each box to an AWS service you'd use (and one you considered and rejected).
