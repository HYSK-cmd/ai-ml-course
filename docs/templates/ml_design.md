# ML System Design: <product>

> Week 8 deliverable -- copy to `docs/w8_ml_design.md` and write it about the recommender you built,
> as if proposing it for a real streaming product with 50M users. 1000-2000 words.

## 1. Business problem & success metric
Who is the user, what decision does the model make, what business metric moves (and guardrails).

## 2. ML framing
Prediction target, label definition (implicit vs explicit feedback, what counts as positive),
offline metric(s) that proxy the business metric -- and where they disagree.

## 3. Data
Sources, volume, freshness, label delay, biases (popularity, position, selection), splits that avoid
leakage (why temporal), privacy.

## 4. Features
User, item, context, cross features. Which are computed offline (batch) vs online (streaming)?
Feature store? How do you guarantee training/serving consistency?

## 5. Model
Stage 1 retrieval (two-tower, in-batch negatives, logQ) and stage 2 ranking (GBM or DNN); why two
stages; candidate count vs latency; cold start (users and items).

## 6. Offline evaluation
Your actual numbers: recall@50, NDCG@10, coverage vs the popularity baseline. Ablations (no ranker,
no genre features, n_probe). Error analysis: which users do you fail?

## 7. Online evaluation
A/B design: unit of randomization, primary metric, sample size (use your calculator), duration,
novelty effects, guardrails, ramp plan.

## 8. Serving
Architecture, latency budget per stage, ANN index refresh, caching, batching, rate limiting,
fallbacks, cost per 1k requests.

## 9. Monitoring
Data drift (PSI on features), prediction drift, feedback loops, retraining cadence, alerting.

## 10. Risks & next steps
