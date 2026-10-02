COURSE.weeks.push({
  n: 6, id: "w6", title: "MLOps",
  tagline: "Reproducible training, experiment tracking, data versioning, a batched streaming inference server for your GPT, containers, and the monitoring that tells you when a model is going bad.",
  hours: { learn: 6, project: 7.5 },
  goals: [
    "Make a training run reproducible and traceable: seeds, pinned environment, tracked params/metrics/artifacts, versioned data",
    "Derive the throughput/latency trade-off of dynamic batching and implement a batcher with asyncio futures",
    "Serve your GPT over HTTP with validation, streaming (SSE) and Prometheus metrics",
    "Containerize the server with a multi-stage Dockerfile and a health check",
    "Detect data drift with PSI and KS and explain what to do when it fires",
  ],
  schedule: [
    ["Mon", "L1 The ML lifecycle, reproducibility, MLflow and DVC"],
    ["Tue", "L2 Model serving and dynamic batching (play with the simulator)"],
    ["Wed", "L3 Containers and deployment strategies"],
    ["Thu", "L4 Monitoring: percentiles, Prometheus, drift"],
    ["Fri", "Flashcards; derive PSI as a symmetrized KL"],
    ["Sat", "Milestones 1–4: batcher, metrics, drift, FastAPI app + GPTEngine"],
    ["Sun", "Milestones 5–6: MLflow tracking, Docker, DVC, load test with locust"],
  ],
  lessons: [
    {
      id: "w6l1", title: "The ML lifecycle and reproducibility", minutes: 65,
      summary: "Why ML systems rot, what you must record to reproduce a model, and how MLflow and DVC record it.",
      keypoints: [
        "A model is code + data + config + environment + randomness; reproducibility requires versioning all five.",
        "Seed Python, numpy and torch RNGs; deterministic GPU kernels cost speed.",
        "Experiment tracking: params once, metrics with a step, artifacts (checkpoints, plots) per run.",
        "DVC stores large data content-addressed outside git; git tracks small pointer files.",
        "Most ML system complexity is outside the model: data, config, serving, monitoring.",
      ],
      body: R`
<h2>Why ML needs its own ops discipline</h2>
<p>Sculley et al.'s "Hidden Technical Debt in Machine Learning Systems" (2015) has one famous figure: the ML code is a small box in the middle of a large system of data collection, feature extraction, configuration, serving infrastructure and monitoring. Software rots when code changes; ML systems also rot when <em>data</em> changes, silently, with no exception thrown. MLOps is the set of practices that make the whole loop (data → train → evaluate → deploy → monitor → retrain) repeatable and observable.</p>
<h2>What makes a run reproducible</h2>
<p>A trained model is a function of: <strong>code</strong> (git commit), <strong>data</strong> (exact snapshot), <strong>configuration</strong> (every hyperparameter), <strong>environment</strong> (package versions, CUDA, hardware) and <strong>randomness</strong> (seeds). Change any one and you get a different model.</p>
<ul>
<li><strong>Seeds</strong>: <code>random.seed</code>, <code>np.random.default_rng(seed)</code> passed explicitly (as your forge does), <code>torch.manual_seed</code>, and generators for data sampling (<code>get_batch(..., generator=g)</code>). Pass RNGs down instead of relying on global state: that's why every forge API takes a seed or a generator.</li>
<li><strong>GPU nondeterminism</strong>: some CUDA kernels (atomics in scatter-add, some convolution algorithms) are nondeterministic. <code>torch.use_deterministic_algorithms(True)</code> forces deterministic ones at a speed cost. In practice teams accept tiny run-to-run variation and compare models over several seeds.</li>
<li><strong>Environment</strong>: a lock file, and a container image built from it (L3).</li>
</ul>
<h2>Experiment tracking with MLflow</h2>
<p>An <em>experiment</em> groups <em>runs</em>. Each run records:</p>
<ul>
<li><strong>params</strong>: logged once (<code>mlflow.log_params(asdict(cfg))</code>), the full config.</li>
<li><strong>metrics</strong>: time series keyed by <code>step</code> (<code>mlflow.log_metrics({"val_loss": v}, step=s)</code>), so you can overlay learning curves across runs.</li>
<li><strong>artifacts</strong>: files: checkpoints, plots, the tokenizer, an evaluation report.</li>
<li>tags: git commit, dataset version, who ran it.</li>
</ul>
<p>The tracking backend stores runs; MLflow 3 deprecates the plain-file store in favor of a database, so use <code>sqlite:///mlflow.db</code> and browse with <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>. The model registry adds named model versions with stages (staging, production), the hand-off point to deployment. The Week 6 test trains a tiny model with <code>cfg.mlflow=True</code> and checks params, a val-loss history with ≥2 points, and a <code>.pt</code> artifact.</p>
<h2>Data versioning with DVC</h2>
<p>Git can't hold gigabytes of data. DVC hashes data files, stores them in a content-addressed cache (local or S3/GCS remote), and commits a tiny pointer file (<code>data/shakespeare.txt.dvc</code> with the hash) to git. Checking out an old commit + <code>dvc checkout</code> restores the exact data that commit trained on. <code>dvc.yaml</code> pipelines go further: stages with declared inputs/outputs, re-run only when inputs change (Make for ML). The repo's <code>.gitignore</code> already ignores <code>data/*</code> but allows <code>data/*.dvc</code> for exactly this.</p>
<div class="callout pitfall"><b>"It worked on my machine" for ML</b>The most common irreproducibility in practice isn't seeds, it's data: a feature table that was regenerated, a filter that changed upstream, a label that got backfilled. Version datasets and log the dataset version with every run.</div>
<div class="callout prod"><b>In production</b>On AWS this maps to: S3 (with versioning) for data and artifacts, SageMaker Experiments or a hosted MLflow for tracking, a model registry gating deployment, and pipelines in Step Functions, SageMaker Pipelines or Airflow. The concepts are tool-independent; MLflow + DVC are the open-source reference.</div>
`,
      quiz: [
        { q: "Two runs of the same commit and config produce different val losses. Which is NOT a likely cause?", options: ["Unseeded data sampling", "Nondeterministic GPU kernels", "A dataset regenerated between runs", "Using a git branch"], answer: 3, why: "A branch name doesn't change computation; the other three do." },
        { q: "What does git store when you DVC-track a 2 GB dataset?", options: ["The dataset", "A small .dvc file with the content hash and metadata", "Nothing", "A compressed copy"], answer: 1, why: "Data lives in the DVC cache/remote; git versions the pointer." },
        { q: "Why log metrics with a step instead of only the final value?", options: ["MLflow requires it", "To compare learning curves across runs and diagnose over/underfitting", "To save space", "Steps are needed for artifacts"], answer: 1, why: "Curves reveal divergence, overfitting and schedule effects that a final number hides." },
      ],
      explain: "List everything you'd need to record to retrain your Shakespeare GPT bit-for-bit (or nearly) six months from now, and which tool records each piece.",
      resources: [
        { title: "Sculley et al. — Hidden Technical Debt in Machine Learning Systems", url: "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", note: "" },
        { title: "MLflow Tracking docs", url: "https://mlflow.org/docs/latest/ml/tracking/", note: "" },
        { title: "DVC — Get Started: data versioning", url: "https://dvc.org/doc/start", note: "" },
        { title: "Chip Huyen — Designing Machine Learning Systems, ch. 1, 6", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/", note: "" },
        { title: "PyTorch — Reproducibility notes", url: "https://pytorch.org/docs/stable/notes/randomness.html", note: "" },
      ],
      cards: [
        { f: "Five things a trained model depends on", b: "Code, data, config, environment, randomness." },
        { f: "MLflow: params vs metrics vs artifacts", b: "Params: config, logged once. Metrics: time series with step. Artifacts: files (checkpoints, plots)." },
        { f: "How does DVC version data with git?", b: "Content-addressed cache/remote for the data; git tracks small .dvc pointer files with hashes." },
      ],
    },
    {
      id: "w6l2", title: "Model serving and dynamic batching", minutes: 80,
      summary: "Online inference, latency budgets, why batching multiplies GPU throughput, how to build a dynamic batcher on asyncio futures, and streaming tokens with SSE.",
      keypoints: [
        "GPU forward cost ≈ fixed overhead + small per-item cost, so batching raises throughput dramatically.",
        "Dynamic batching: flush when the batch is full or the oldest request has waited max_wait.",
        "Each request awaits a Future; the batch worker sets results in order.",
        "Run blocking model calls off the event loop (asyncio.to_thread).",
        "Streaming (SSE) improves time-to-first-token; validate inputs at the boundary (pydantic → 422).",
      ],
      body: R`
<h2>Online inference</h2>
<p>Batch inference (score a table nightly) is easy: big batches, throughput is all that matters. Online inference answers a request in milliseconds while many users wait. The design questions: what latency budget (p50, p99) do callers need, what throughput must one machine sustain, and what does it cost per 1,000 requests?</p>
<h2>Why batching wins on GPUs</h2>
<p>A forward pass for $b$ requests costs roughly $t(b) = \alpha + \beta b$: a fixed part $\alpha$ (kernel launches, reading all the weights from memory, Python overhead) and a small per-request part $\beta$. For decode-heavy LLM work $\alpha$ dominates: generating one token for one user reads every weight once, and generating one token for 16 users reads them... once. Throughput is</p>
$$\text{throughput}(b) = \frac{b}{\alpha + \beta b}\ \xrightarrow{b\to\infty}\ \frac1\beta,$$
<p>versus $1/(\alpha+\beta)$ unbatched. With $\alpha = 18$ ms and $\beta = 1.5$ ms, batch 1 caps at ~51 req/s while batch 16 reaches ~380 req/s. The cost: a request may wait for others to arrive.</p>
<h3>Dynamic batching</h3>
<p>Don't wait for a fixed batch size (latency would be unbounded at low traffic); don't skip batching (throughput collapses at high traffic). The rule: take the oldest waiting request, then collect more until <strong>the batch is full</strong> or <strong>max_wait has elapsed</strong>, then run. At low load you add at most max_wait of latency; at high load batches fill instantly and you get full throughput.</p>
<div class="widget" data-widget="batching"></div>
<p>Try: at 40 req/s batch 1 is fine. At 120 req/s batch 1's queue grows without bound (p99 explodes) while the batcher stays in tens of milliseconds. Raise max_wait at low traffic and watch p50 rise for no throughput gain: the knob is a latency tax you pay only when traffic is light.</p>
<h3>Building it on asyncio</h3>
<pre><code># the shape of it (you write the real thing)
async def submit(item):
    fut = loop.create_future()
    await queue.put((item, fut))
    return await fut                 # caller sleeps until the worker sets the result

async def worker():
    while running:
        batch = [await queue.get()]                       # block for the first
        deadline = now() + max_wait
        while len(batch) &lt; max_batch and (t := deadline - now()) &gt; 0:
            try: batch.append(await asyncio.wait_for(queue.get(), t))
            except asyncio.TimeoutError: break
        results = await asyncio.to_thread(process_batch, [i for i, _ in batch])
        for (_, fut), r in zip(batch, results): fut.set_result(r)</code></pre>
<p>Details the tests check: results must go to the right caller (zip preserves order); an exception in <code>process_batch</code> is set on every future of <em>that</em> batch and the worker keeps running; a sync <code>process_batch</code> runs in a thread so the loop keeps serving; <code>stop()</code> drains everything already submitted; a full batch doesn't wait for the deadline.</p>
<h3>Batching variable-length prompts</h3>
<p>Your <code>GPTEngine</code> core version groups requests with equal prompt lengths and settings into one <code>generate</code> call. Real servers (vLLM, TGI) pad and mask, and go further with <strong>continuous batching</strong>: sequences join and leave the running batch at every decode step instead of waiting for the slowest sequence to finish.</p>
<h2>Streaming with Server-Sent Events</h2>
<p>For text generation, users perceive <em>time to first token</em>. SSE is plain HTTP: response type <code>text/event-stream</code>, the body is a series of <code>data: ...\n\n</code> lines, one per event, ending with a sentinel (<code>data: [DONE]</code>, OpenAI's convention). FastAPI's <code>StreamingResponse</code> takes a generator; your <code>GPTEngine.stream</code> yields one decoded token at a time using the KV cache.</p>
<h2>The API boundary</h2>
<p>Validate everything at the edge: pydantic models with <code>Field(min_length=1, max_length=2000)</code>, <code>ge=1, le=512</code> etc. make FastAPI return 422 automatically. An unbounded <code>max_new_tokens</code> is a denial-of-service hole. Health endpoints (<code>/health</code>) let load balancers and orchestrators know when to route traffic.</p>
<div class="callout prod"><b>In production</b>Model optimization multiplies everything above: int8/int4 weight quantization (less memory traffic per token), <code>torch.compile</code> or TensorRT (fused kernels), speculative decoding (a small model drafts tokens the big one verifies). But batching is usually the first and biggest win, and getting it wrong (blocking the loop, no timeout, unbounded queue) is the classic serving outage.</div>
`,
      quiz: [
        { q: "Forward cost is 20 ms + 2 ms per request. Max throughput with batch size 1 vs 16?", options: ["45 vs 45 req/s", "45 vs ~308 req/s", "50 vs 800 req/s", "45 vs 720 req/s"], answer: 1, why: "1/0.022 ≈ 45; 16/(0.020+0.032) = 16/0.052 ≈ 308." },
        { q: "Traffic is very light (1 req/s), max_batch=32, max_wait=50 ms. Each request's added latency is about…", options: ["0 ms", "≈ 50 ms (it waits out the deadline alone)", "≈ 1 s", "≈ 32 × forward time"], answer: 1, why: "No one else arrives, so every request pays the full wait. That's why max_wait is kept small." },
        { q: "process_batch raises for one batch. Correct batcher behavior?", options: ["Crash the server", "Every caller in that batch gets the exception; the batcher continues", "Retry forever", "Drop the requests silently"], answer: 1, why: "Failures are isolated per batch; later requests must still be served." },
      ],
      explain: "Explain dynamic batching to a backend engineer: why it raises GPU throughput, what max_batch_size and max_wait_ms trade off, and how asyncio futures connect callers to the batch worker.",
      resources: [
        { title: "NVIDIA Triton — Dynamic batching", url: "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher", note: "" },
        { title: "Anyscale — How continuous batching enables 23x throughput in LLM inference", url: "https://www.anyscale.com/blog/continuous-batching-llm-inference", note: "" },
        { title: "FastAPI docs — Request body validation, StreamingResponse, lifespan events", url: "https://fastapi.tiangolo.com/advanced/events/", note: "" },
        { title: "MDN — Server-sent events", url: "https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events", note: "" },
      ],
      cards: [
        { f: "Throughput with batch size b and cost α + βb", b: "b/(α + βb), approaching 1/β for large b." },
        { f: "Dynamic batching flush condition", b: "Batch full OR oldest request waited max_wait." },
        { f: "SSE wire format", b: "Content-Type text/event-stream; events as 'data: ...\\n\\n'; end with a sentinel like 'data: [DONE]'." },
      ],
    },
    {
      id: "w6l3", title: "Containers and deployment", minutes: 55,
      summary: "Docker images and layers, multi-stage builds, GPUs in containers, config via environment, and safe rollout strategies.",
      keypoints: [
        "An image is a stack of cached layers; order Dockerfile steps from least to most frequently changing.",
        "Multi-stage builds keep build tools out of the runtime image.",
        "GPU containers need a CUDA-compatible base and the NVIDIA container toolkit on the host.",
        "12-factor: configuration comes from the environment, not baked into the image.",
        "Roll out with canary or shadow traffic and automatic rollback on SLO breach.",
      ],
      body: R`
<h2>Images and layers</h2>
<p>A Dockerfile is a recipe; each instruction creates a cached layer. If a layer's inputs haven't changed, Docker reuses it. So order matters: copy <code>pyproject.toml</code> and install dependencies <em>before</em> copying source code, and a code edit won't reinstall PyTorch.</p>
<pre><code># sketch — your Dockerfile is a Week 6 deliverable
FROM python:3.13-slim AS build
WORKDIR /app
COPY pyproject.toml .
RUN pip wheel --wheel-dir /wheels ".[w4,w6]"

FROM python:3.13-slim
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-index /wheels/*
COPY forge serve ./
ENV FORGE_CKPT=/models/ckpt.pt FORGE_DEVICE=cpu
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]</code></pre>
<p><strong>Multi-stage builds</strong> build wheels (compilers, headers) in one stage and copy only the results into a slim runtime stage: smaller images, faster pulls, smaller attack surface. <strong>GPU images</strong> start from an <code>nvidia/cuda</code> runtime base (or use the CUDA-bundled PyTorch wheels) and run with <code>--gpus all</code>; on Windows that means Docker Desktop with the WSL2 backend. The model checkpoint is usually mounted or downloaded at startup rather than baked in, so one image serves many model versions.</p>
<h2>Configuration and health</h2>
<p>The 12-factor rule: config that varies between environments (model path, device, batch size, URLs, secrets) comes from environment variables, which is why your server has <code>app_from_env()</code>. A <code>HEALTHCHECK</code> (or Kubernetes liveness/readiness probes) lets the orchestrator restart a hung container and hold traffic until the model has loaded.</p>
<h2>Where it runs</h2>
<ul>
<li><strong>Kubernetes</strong>: a Deployment keeps N replicas running, a Service load-balances them, a HorizontalPodAutoscaler adds replicas on CPU/GPU utilization or queue length.</li>
<li><strong>AWS</strong>: images in ECR; run on ECS/Fargate or EKS; or SageMaker real-time endpoints, which wrap the same container contract with autoscaling and A/B variants.</li>
</ul>
<h2>Rolling out a new model safely</h2>
<ul>
<li><strong>Shadow</strong>: the new model receives a copy of live traffic; its outputs are logged, never shown. Zero user risk; measures latency, errors and prediction drift against the old model.</li>
<li><strong>Canary</strong>: send 1–5% of traffic to the new version; watch error rate, latency and business metrics; promote gradually or roll back automatically.</li>
<li><strong>Blue/green</strong>: two full environments; flip the router; instant rollback.</li>
<li><strong>A/B test</strong>: a controlled experiment for the business metric (Week 8).</li>
</ul>
<div class="callout prod"><b>In production</b>Your AWS internship background is directly relevant here: ML deployment is regular service deployment plus model-specific checks (prediction distributions, feature freshness, offline/online metric agreement). Interviewers for ML engineer roles probe this boundary.</div>
`,
      quiz: [
        { q: "Why COPY pyproject.toml and install dependencies before copying source code?", options: ["Docker requires it", "So code changes don't invalidate the cached dependency layer", "To reduce image size", "Security"], answer: 1, why: "Layer caching: a changed layer invalidates everything after it." },
        { q: "Which rollout strategy has zero user-facing risk?", options: ["Canary", "Blue/green", "Shadow", "A/B"], answer: 2, why: "Shadow traffic's outputs are never returned to users." },
      ],
      explain: "Explain how you would deploy a new version of your GPT server to production on AWS with minimal risk, from image build to full rollout.",
      resources: [
        { title: "Docker docs — Multi-stage builds; build cache", url: "https://docs.docker.com/build/building/multi-stage/", note: "" },
        { title: "The Twelve-Factor App", url: "https://12factor.net/", note: "config, processes, disposability" },
        { title: "Google SRE Book — Release engineering, canarying", url: "https://sre.google/sre-book/release-engineering/", note: "" },
      ],
      cards: [
        { f: "Benefit of multi-stage Docker builds", b: "Build tools stay in the builder stage; runtime image is smaller and safer." },
        { f: "Shadow vs canary deployment", b: "Shadow: copy of traffic, outputs discarded. Canary: small % of real traffic actually served by the new version." },
      ],
    },
    {
      id: "w6l4", title: "Monitoring: percentiles, Prometheus and drift", minutes: 75,
      summary: "Why averages lie, how histograms give you percentiles, the Prometheus model, and detecting data drift with PSI and KS.",
      keypoints: [
        "Report latency percentiles (p50/p95/p99), never just the mean.",
        "Prometheus histograms export cumulative bucket counts; quantiles are estimated by interpolation inside a bucket.",
        "Counters only go up (rates come from deltas); gauges go up and down; histograms count observations into buckets.",
        "Data drift: P(x) changes. Concept drift: P(y|x) changes. Labels often arrive late.",
        "PSI = Σ(a−e)ln(a/e) = KL(a‖e) + KL(e‖a); KS = max gap between ECDFs.",
      ],
      body: R`
<h2>Averages lie</h2>
<p>If 99 requests take 10 ms and one takes 5 s, the mean is 60 ms, a number no user experienced. Tail latency matters because users and fan-out services hit the tail: a page that calls 100 backends sees each backend's p99 on almost every page load. Track p50 (typical), p95/p99 (tail) and set SLOs on them ("p99 &lt; 300 ms over 30 days").</p>
<div class="widget" data-widget="latency"></div>
<h2>Histograms and quantile estimation</h2>
<p>Storing every latency is expensive; Prometheus instead counts observations into fixed buckets with upper bounds $le$, exported <strong>cumulatively</strong>:</p>
<pre><code>request_latency_seconds_bucket{le="0.1"} 7      # observations ≤ 0.1 s
request_latency_seconds_bucket{le="0.5"} 9
request_latency_seconds_bucket{le="+Inf"} 10
request_latency_seconds_sum 2.31
request_latency_seconds_count 10</code></pre>
<p>Cumulative counts let a server aggregate histograms from many replicas by adding them. To estimate the $q$-quantile: the target rank is $q\cdot N$; find the bucket where the cumulative count first reaches it, and interpolate linearly between the bucket's lower and upper bounds (assuming observations are spread uniformly inside it). If the rank lands in +Inf, return the highest finite bound. That's PromQL's <code>histogram_quantile</code>, and your <code>Histogram.quantile</code>. Accuracy depends on bucket placement: put bounds around your SLO.</p>
<h2>The Prometheus model</h2>
<ul>
<li><strong>Pull</strong>: Prometheus scrapes <code>GET /metrics</code> on every target every ~15 s. Your app exposes text; it never pushes.</li>
<li><strong>Counter</strong>: monotonically increasing (requests, errors); dashboards use <code>rate()</code> over it. <strong>Gauge</strong>: a current value (queue length, GPU memory). <strong>Histogram</strong>: buckets + sum + count.</li>
<li>The <strong>four golden signals</strong> (Google SRE): latency, traffic, errors, saturation. For a model server add: batch size distribution (are you actually batching?), queue depth, tokens/s, and GPU utilization.</li>
<li>Thread safety: handlers run concurrently, so increments need a lock (the test hammers a counter from 8 threads).</li>
</ul>
<h2>Model monitoring: drift</h2>
<p>A model can be up, fast and wrong. Quality degrades when the world changes:</p>
<ul>
<li><strong>Data (covariate) drift</strong>: the input distribution $P(x)$ shifts (new user segment, a broken upstream feature, seasonality).</li>
<li><strong>Concept drift</strong>: $P(y\mid x)$ changes (fraudsters adapt, tastes change). Only visible with labels.</li>
<li><strong>Label delay</strong>: labels arrive days or weeks later (did the loan default?), so input and prediction drift are the early warning system.</li>
</ul>
<h3>Population Stability Index</h3>
<p>Bin a feature using quantiles of the reference (training) distribution, compute the fraction of reference ($e_i$) and live ($a_i$) data per bin, clip zeros to a small $\epsilon$, then</p>
$$\mathrm{PSI} = \sum_i (a_i - e_i)\ln\frac{a_i}{e_i} = \mathrm{KL}(a\|e) + \mathrm{KL}(e\|a).$$
<p>(Expand: $\sum a\ln\frac ae + \sum e\ln\frac ea$.) It's a symmetrized KL divergence on the binned distributions, from Week 1. Rules of thumb: &lt;0.1 stable, 0.1–0.25 moderate, &gt;0.25 significant shift. Quantile bins make each reference bin hold ~1/bins of the data; ties at edges go left (<code>searchsorted(..., side="left")</code>), the convention your test's hand-computed example uses.</p>
<h3>Kolmogorov–Smirnov</h3>
<p>The two-sample KS statistic is $D = \max_x|F_a(x) - F_b(x)|$, the largest vertical gap between the empirical CDFs. Compute both ECDFs at every observed value with <code>searchsorted(side="right")</code> after sorting: $O((n+m)\log(n+m))$. No binning, sensitive to any distribution change, but with large samples it flags tiny, harmless shifts as significant: use the statistic as an effect size, not just its p-value.</p>
<h3>What to do when drift fires</h3>
<p>Investigate before retraining: is it a pipeline bug (a feature suddenly all zeros, units changed) or a real-world shift? Pipeline bugs are fixed upstream; real shifts trigger retraining on recent data, ideally automated with the same validation gates as the original model. Alert on drift in the features the model relies on most.</p>
<div class="callout prod"><b>In production</b>A typical model dashboard: request rate, error rate, p50/p99 latency, batch size, queue depth (service health); prediction distribution and PSI on top features vs training (model health); and, when labels arrive, rolling accuracy/AUC (quality). Your server exposes the first group; the drift module is the second.</div>
`,
      quiz: [
        { q: "Buckets (cumulative): le=1 → 10, le=2 → 20 (20 observations total). Estimated median?", options: ["1.0", "1.5", "0.5", "2.0"], answer: 0, why: "Rank 0.5·20 = 10 is reached exactly at the end of the first bucket: interpolate to its upper bound 1.0. (The test's histogram case.)" },
        { q: "PSI between identical distributions is…", options: ["1", "≈ 0", "0.25", "undefined"], answer: 1, why: "All bins match: every term $(a-e)\\ln(a/e)$ is 0." },
        { q: "Your fraud model's inputs look stable (low PSI) but precision has dropped a lot over 3 months. Most likely…", options: ["Data drift", "Concept drift: the relationship between features and fraud changed", "A latency problem", "A bug in PSI"], answer: 1, why: "Stable $P(x)$ but changing $P(y|x)$ is concept drift; only labels reveal it." },
      ],
      explain: "Explain why p99 latency matters more than mean latency, and how a Prometheus histogram lets you estimate p99 without storing every request.",
      resources: [
        { title: "Prometheus docs — Histograms and summaries", url: "https://prometheus.io/docs/practices/histograms/", note: "" },
        { title: "Google SRE Book — Monitoring Distributed Systems (four golden signals)", url: "https://sre.google/sre-book/monitoring-distributed-systems/", note: "" },
        { title: "Dean & Barroso — The Tail at Scale", url: "https://research.google/pubs/the-tail-at-scale/", note: "" },
        { title: "Chip Huyen — Data Distribution Shifts and Monitoring", url: "https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html", note: "" },
      ],
      cards: [
        { f: "PSI formula and its relation to KL", b: "Σ(a−e)ln(a/e) = KL(a‖e) + KL(e‖a)" },
        { f: "Data drift vs concept drift", b: "Data: P(x) changes. Concept: P(y|x) changes." },
        { f: "Why Prometheus buckets are cumulative", b: "Histograms from many replicas can be summed, and quantiles estimated by interpolating within the bucket where rank q·N falls." },
        { f: "Four golden signals", b: "Latency, traffic, errors, saturation." },
      ],
    },
  ],
  project: {
    title: "serve — a batched, streaming, observable GPT server",
    dir: "serve/, Dockerfile, loadtest/, forge/gpt/train.py (MLflow)",
    pitch: "Build the asyncio dynamic batcher, Prometheus metrics, PSI/KS drift detection and a FastAPI app with SSE streaming around your GPT. Track training in MLflow, version data with DVC, containerize, and prove batching's throughput gain with a load test.",
    test: "pytest tests/w6 -q",
    overview: R`
<p>The app tests use a fake engine, so you can build the whole HTTP layer before touching the model. The engine tests build a tiny random GPT checkpoint with your Week 4 code. The ops tests check deliverables: an MLflow-tracked training run, a multi-stage <code>Dockerfile</code> with a <code>HEALTHCHECK</code>, DVC tracking of the dataset, and <code>loadtest/results.json</code> with your measured numbers.</p>
<p>Extra tools for this week (not preinstalled): <code>pip install -e ".[ops]"</code> gives you <code>dvc</code> and <code>locust</code>. Docker Desktop is needed to build the image.</p>
<p>Run the real server: <code>$env:FORGE_CKPT="checkpoints/gpt/ckpt.pt"; $env:FORGE_DEVICE="cuda"; uvicorn serve.app:app_from_env --factory --port 8000</code>, then open <code>http://localhost:8000/docs</code>: FastAPI's interactive page lets you send <code>/generate</code> requests from the browser.</p>
`,
    milestones: [
      { id: "w6m1", core: true, title: "DynamicBatcher", test: "pytest tests/w6/test_batcher.py -q",
        detail: R`<p>Routing, max batch size, max-wait flush, immediate flush of full batches, error isolation, off-loop execution of sync functions, draining on stop.</p>`,
        hints: [
          R`<p>State: an <code>asyncio.Queue</code> of (item, future) pairs and one background task created in <code>start()</code>. Create the queue in <code>start()</code> (inside the running loop), not in <code>__init__</code>.</p>`,
          R`<p>For stop: put a sentinel (None) on the queue; the worker finishes the current batch, sees the sentinel, then drains anything left. Set a flag so <code>submit</code> raises RuntimeError afterwards.</p>`,
          R`<p><code>inspect.iscoroutinefunction(fn)</code> → <code>await fn(items)</code>, else <code>await asyncio.to_thread(fn, items)</code>. Wrap in try/except and <code>set_exception</code> on each future of the batch.</p>`,
        ] },
      { id: "w6m2", core: true, title: "Counter, Histogram, Registry", test: "pytest tests/w6/test_metrics_drift.py -q -k \"counter or histogram or thread or registry\"",
        detail: R`<p>Exact exposition format, cumulative buckets with inclusive <code>le</code>, quantile interpolation, thread safety.</p>`,
        hints: [
          R`<p>Per-bucket counts with <code>bisect.bisect_left(bounds, v)</code> (inclusive upper bound); one extra slot for +Inf. Render running sums.</p>`,
          R`<p>Quantile: rank = q·N; walk buckets accumulating counts; in the bucket where cum + c ≥ rank, return lo + (hi − lo)·(rank − cum)/c.</p>`,
          R`<p>Guard every mutation with a <code>threading.Lock</code>.</p>`,
        ] },
      { id: "w6m3", core: true, title: "PSI, KS and DriftMonitor", test: "pytest tests/w6/test_metrics_drift.py -q -k \"psi or ks_ or monitor\"",
        detail: R`<p>Match the hand-computed PSI exactly and scipy's KS statistic including ties.</p>`,
        hints: [
          R`<p>Inner edges: <code>np.quantile(expected, np.arange(1, bins) / bins)</code>; bin index <code>np.searchsorted(edges, x, side="left")</code>; proportions via <code>np.bincount(..., minlength=bins) / n</code>; clip to eps.</p>`,
          R`<p>KS: sort both; evaluate both ECDFs at all values of the concatenation with <code>searchsorted(..., side="right") / n</code>; max absolute difference.</p>`,
          R`<p>DriftMonitor window: <code>collections.deque(maxlen=window)</code>.</p>`,
        ] },
      { id: "w6m4", core: true, title: "FastAPI app and GPTEngine", test: "pytest tests/w6/test_app.py -q",
        detail: R`<p>Validation (422s), batching across concurrent requests, SSE format, /metrics, batch-equals-individual for the real engine, streaming equals batch generation past block_size.</p>`,
        hints: [
          R`<p>Start/stop the batcher in a FastAPI <code>lifespan</code> async context manager. Use an HTTP middleware to time requests and increment counters (skip /metrics itself).</p>`,
          R`<p>Pydantic body: <code>prompt: str = Field(min_length=1, max_length=2000)</code>, <code>max_new_tokens: int = Field(64, ge=1, le=512)</code>, <code>temperature: float = Field(1.0, gt=0, le=5)</code>, <code>top_k: int | None = Field(None, ge=1)</code>.</p>`,
          R`<p>GPTEngine.stream: prefill the prompt with <code>step</code>, then sample one token at a time with <code>step</code> on the single new token; rebuild the cache when it reaches block_size, exactly like <code>generate</code>, so the two agree.</p>`,
        ] },
      { id: "w6m5", core: true, title: "MLflow tracking in train()", test: "pytest tests/w6/test_mlops.py -q -k mlflow",
        detail: R`<p>When <code>cfg.mlflow</code>: set tracking URI/experiment, start a run, log all config params, log train/val loss at every eval with <code>step</code>, log the checkpoint artifact, end the run.</p>`,
        hints: [
          R`<p>Import mlflow inside the <code>if cfg.mlflow:</code> branch so training works without it.</p>`,
          R`<p>The test needs ≥2 val_loss points: log at each eval interval AND at the end.</p>`,
          R`<p>Browse your real runs: <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>.</p>`,
        ] },
      { id: "w6m6", core: true, title: "Dockerfile, DVC, load test", test: "pytest tests/w6/test_mlops.py -q -k \"docker or dvc or load\"",
        detail: R`<p>Multi-stage Dockerfile with HEALTHCHECK and uvicorn; <code>dvc init</code> + <code>dvc add data/shakespeare.txt</code>; a locust load test against the real server comparing <code>max_batch_size=1</code> vs batched, recorded in <code>loadtest/results.json</code> (keys: rps_batch1, rps_batched, p50_ms_batched, p99_ms_batched, max_batch_size, max_wait_ms). Batched must be ≥3× batch-1 throughput.</p>`,
        hints: [
          R`<p>locustfile: one <code>HttpUser</code> with a task posting to <code>/generate</code> with a short prompt and e.g. 32 new tokens. Run headless: <code>locust -f loadtest/locustfile.py --headless -u 64 -r 16 -t 60s --host http://localhost:8000</code>.</p>`,
          R`<p>Run the server on CUDA for the test: the GPU is where batching shines. Use enough concurrent users (≥ 4× max batch) to keep batches full.</p>`,
          R`<p>Read p50/p99 from locust's summary or from your own /metrics histogram; they should roughly agree.</p>`,
        ] },
      { id: "w6m7", core: false, title: "Stretch: int8 weight quantization and a shadow endpoint",
        detail: R`<p>Quantize your GPT's Linear weights to int8 per output channel (scale = max|w|/127) and dequantize on the fly; measure memory, speed and val-loss change. Add a shadow route that sends a copy of each request to a second engine and logs disagreements.</p>`,
        hints: [R`<p>Per-channel scales keep quantization error small; compare logits of the fp32 and int8 models on a held-out batch before measuring speed.</p>`] },
    ],
  },
});
