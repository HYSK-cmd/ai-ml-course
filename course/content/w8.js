COURSE.weeks.push({
  n: 8, id: "w8", title: "ML System Design",
  tagline: "Design and build a production-shaped recommender: leakage-free data, two-tower retrieval with logQ correction, an ANN index on your k-means, a GBM re-ranker, a service, and the statistics to launch it.",
  hours: { learn: 6.5, project: 7.5 },
  goals: [
    "Run an ML system design discussion end to end with a repeatable framework",
    "Derive the in-batch softmax sampling bias and the logQ correction, and see it change real metrics",
    "Build an IVF index and reason about the recall/latency trade-off and HNSW/PQ alternatives",
    "Construct ranking labels and features without leakage, and explain training–serving skew",
    "Size an A/B test, avoid peeking, and use CUPED; write a full ML design doc for your recommender",
  ],
  schedule: [
    ["Mon", "L1 The ML system design framework + L2 Two-tower retrieval and logQ"],
    ["Tue", "L3 Approximate nearest-neighbor search"],
    ["Wed", "L4 Ranking, features and leakage"],
    ["Thu", "L5 Experimentation and launch decisions"],
    ["Fri", "Flashcards; outline your ML design doc"],
    ["Sat", "Milestones 1–4: data split, metrics/baseline, two-tower, IVF"],
    ["Sun", "Milestones 5–8: pipeline (pytest -m slow), A/B stats, service, design doc"],
  ],
  lessons: [
    {
      id: "w8l1", title: "The ML system design framework", minutes: 60,
      summary: "A repeatable structure for designing ML systems, applied to movie recommendations, including the two-stage architecture and its latency budget.",
      keypoints: [
        "Start from the business goal and turn it into a measurable ML objective plus guardrails.",
        "Define labels precisely, including what counts as negative and when labels become available.",
        "Always establish a simple baseline (popularity) before modeling.",
        "Large catalogs use two stages: cheap retrieval of hundreds of candidates, then an expensive ranker.",
        "Offline metrics are proxies; only online experiments measure the business metric.",
      ],
      body: R`
<h2>The framework</h2>
<ol>
<li><strong>Clarify the problem.</strong> Who are the users, what decision does the model make, at what scale and latency? "Recommend movies on the home page for 50M users, < 200 ms."</li>
<li><strong>Business metric → ML objective.</strong> The business wants long-term engagement (watch time, retention). The model can optimize a proxy: P(user watches and likes item). Add guardrails: diversity, freshness, no harmful content, latency.</li>
<li><strong>Data and labels.</strong> Implicit feedback (clicks, watches) is plentiful but noisy and biased by what was shown; explicit (ratings) is clean but sparse. Your choice: a rating ≥ 4 is a positive; everything else is unlabeled (not necessarily negative). Labels arrive with delay.</li>
<li><strong>Features.</strong> User (history, demographics), item (genre, popularity, age), context (time, device), cross features (user–genre affinity).</li>
<li><strong>Model.</strong> Baseline first (popularity), then retrieval + ranking.</li>
<li><strong>Offline evaluation.</strong> A split that mimics deployment (temporal), ranking metrics (recall@K, NDCG@K, coverage), slices (new vs heavy users).</li>
<li><strong>Serving.</strong> Latency budget, caching, fallbacks, index refresh.</li>
<li><strong>Online evaluation.</strong> A/B test on the business metric (L5).</li>
<li><strong>Monitoring and iteration.</strong> Drift, feedback loops, retraining cadence (Week 6).</li>
</ol>
<h2>Two-stage architecture</h2>
<p>Scoring every one of 10M items with a heavy model for every request is impossible within 200 ms. So:</p>
<ul>
<li><strong>Retrieval (candidate generation)</strong>: a cheap model whose scores are dot products of precomputed vectors, so an ANN index can find the top few hundred items among millions in milliseconds. Optimizes <em>recall</em>: don't miss good items.</li>
<li><strong>Ranking</strong>: a richer model (GBM or DNN with cross features) scores only those candidates. Optimizes <em>precision at the top</em>.</li>
<li><strong>Re-ranking / policy</strong>: business rules, diversity, deduplication, freshness boosts.</li>
</ul>
<pre class="mermaid">flowchart LR
  U[Request: user id] --> F[Feature lookup]
  F --> R[Two-tower user vector]
  R --> A[ANN index: top 200]
  A --> K[Ranker: GBM on rich features]
  K --> P[Policy: filter seen, diversity]
  P --> O[Top 10]
  U -. unknown user .-> POP[Popularity fallback]</pre>
<p>A plausible latency budget: feature lookup 10 ms, user-vector compute 5 ms, ANN 5 ms, ranking 200 candidates 30 ms, policy 5 ms, network overhead the rest. Every box is something you've now built: the ranker is your Week 2 GBM, the index uses your Week 2 k-means, the service uses your Week 6 metrics and Week 7 rate limiter.</p>
<h2>Feedback loops</h2>
<p>A recommender trains on data it influenced: items it showed get clicks, items it didn't never get a chance. Left alone this narrows toward popular items (popularity bias) and makes offline evaluation optimistic. Mitigations: exploration (show some random/novel items), logging propensities, and debiasing techniques like the logQ correction you'll implement for training-time sampling bias.</p>
<div class="callout prod"><b>In production</b>This exact architecture (two-tower retrieval → ANN → ranker → policy) runs at YouTube, Pinterest, Instagram, Spotify and in Amazon's recommendation stack. Interviewers expect you to draw it, give the latency budget, and discuss cold start, freshness and feedback loops.</div>
`,
      quiz: [
        { q: "Why not score every item with the best model for every request?", options: ["It would overfit", "Latency and cost: millions of items × a heavy model per request is far beyond a 200 ms budget", "Rankers can't score many items", "ANN is more accurate"], answer: 1, why: "Two stages trade a little recall for orders-of-magnitude less compute." },
        { q: "In implicit feedback, an item the user never interacted with is…", options: ["a confirmed negative", "unlabeled: maybe disliked, maybe never seen", "a positive", "noise to delete"], answer: 1, why: "Missing interactions are mostly 'not exposed', which is why negatives are sampled and biases appear." },
      ],
      explain: "Walk through designing a home-page movie recommender using the nine-step framework, and justify the two-stage architecture with a latency budget.",
      resources: [
        { title: "Chip Huyen — Designing Machine Learning Systems", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/", note: "the framework book" },
        { title: "Covington et al. — Deep Neural Networks for YouTube Recommendations", url: "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", note: "two-stage architecture" },
        { title: "Eugene Yan — System Design for Recommendations and Search", url: "https://eugeneyan.com/writing/system-design-for-discovery/", note: "" },
        { title: "Alex Xu & Ali Aminian — Machine Learning System Design Interview", url: "https://bytebytego.com/", note: "" },
      ],
      cards: [
        { f: "Two stages of a large-scale recommender", b: "Retrieval (cheap, high recall, ANN over embeddings) then ranking (expensive, precise, few hundred candidates)." },
        { f: "Why is offline evaluation of recommenders optimistic?", b: "Feedback loops and exposure bias: logged data only contains items the old system chose to show." },
      ],
    },
    {
      id: "w8l2", title: "Two-tower retrieval and the logQ correction", minutes: 90,
      summary: "Matrix factorization grown up: user and item towers trained with in-batch negatives, the popularity bias that sampling introduces, and the correction that fixes it on your own data.",
      keypoints: [
        "Two towers map users and items to normalized vectors; score = dot product / temperature.",
        "In-batch negatives: other positives in the batch serve as negatives; loss = softmax cross-entropy over the batch.",
        "Items appear as in-batch negatives in proportion to their popularity $q_j$, biasing the model against popular items.",
        "logQ correction: train with logits $s_{ij} - \\log q_j$; serve with the uncorrected $s_{ij}$.",
        "On MovieLens: recall@50 0.156 → 0.245 and NDCG@10 0.040 → 0.095 from the correction alone.",
      ],
      body: R`
<h2>From matrix factorization to two towers</h2>
<p>Collaborative filtering factorizes the user×item interaction matrix: $R\approx UV^\top$, with a $d$-dimensional vector per user and per item (a rank-$d$ approximation, Week 1 L2). A two-tower model generalizes this: each "tower" is a network mapping a user's features (id embedding, history, context) or an item's features (id embedding, genres, text) to a vector. Because the score is a dot product of separately computed vectors, item vectors can be precomputed and indexed, which is what makes retrieval fast. The towers can't model fine-grained user–item interactions; that's the ranker's job.</p>
<p>Your towers: <code>Embedding(n_users, 64)</code> → MLP → L2 normalize; <code>Embedding(n_items, 64) + Linear(genres)</code> → MLP → L2 normalize. Genre features give items a cold-start path: a new item with genres but no id history still gets a sensible vector.</p>
<h2>Training with in-batch negatives</h2>
<p>We want $P(i\mid u)\propto\exp(s(u,i)/\tau)$ over the whole catalog, but a softmax over millions of items per example is too expensive. Sampled softmax approximates the denominator with a few negatives. The cheapest source of negatives is free: in a batch of $B$ positive pairs $(u_r, i_r)$, use every other item in the batch as a negative for row $r$:</p>
$$\text{logits} = \frac{U I^\top}{\tau}\in\mathbb{R}^{B\times B},\qquad \mathcal{L} = \mathrm{CE}(\text{logits}, \text{labels}=[0,1,\dots,B-1]).$$
<p>One matrix multiply gives $B^2$ scores; the diagonal holds positives. The temperature $\tau$ (0.05 here) sharpens the softmax over normalized vectors whose dot products lie in $[-1,1]$.</p>
<h3>The bias</h3>
<p>Batches are sampled from interactions, so item $j$ appears in a batch (and therefore as a negative for everyone else in it) with probability proportional to its popularity $q_j$. The sampled-softmax estimate of $\log P(i\mid u)$ is biased: popular items are over-represented in the denominator, so training pushes their scores <em>down</em> far more than a full softmax would. The model learns "popular ⇒ penalize". On MovieLens this showed up plainly while building your course: the uncorrected model scored recall@50 = 0.156 vs popularity's 0.155, and NDCG@10 of 0.040 vs popularity's 0.079, while recommending 96% of the catalog in top-10 lists. Lots of diversity, too little relevance.</p>
<h3>The logQ correction</h3>
<p>The importance-sampling fix (Bengio & Senécal; Yi et al. 2019 for two-towers): when an item is sampled with probability $q_j$, subtract $\log q_j$ from its logit before the softmax:</p>
$$s^c_{rj} = \frac{u_r^\top v_j}{\tau} - \log q_j.$$
<p>Intuition: a popular item's logit is reduced in training, so the model doesn't need to push its learned score down to compensate; at serving time (no correction) popular items get their deserved scores back. Apply it to every column (positives included, as in the paper's formulation). Estimate $q_j$ as the item's frequency in the training stream. With it, on the same data: recall@50 = 0.245 and NDCG@10 = 0.095, now clearly above popularity. Train once with it off and once with it on; seeing this yourself is the point of the milestone.</p>
<h3>False negatives</h3>
<p>If two rows in a batch share the same item, each row's off-diagonal copy is labeled a negative while actually being that row's positive. Mask those logits to $-\infty$. With popular items in batches of 1024, this happens constantly.</p>
<h2>Evaluating retrieval</h2>
<ul>
<li><strong>Temporal split per user</strong>: last 20% of each user's interactions are test. A random split trains on the future (and the test catches it).</li>
<li><strong>Exclude seen items</strong>: never recommend what the user already interacted with in train.</li>
<li><strong>Recall@K</strong> $= |\text{top-}K\cap\text{truth}|/|\text{truth}|$: did retrieval find the relevant items?</li>
<li><strong>NDCG@K</strong>: rewards relevant items near the top: $\mathrm{DCG} = \sum_{\text{hits at rank } r} 1/\log_2(r+1)$, normalized by the ideal DCG of $\min(K,|\text{truth}|)$ hits at the top ranks.</li>
<li><strong>Coverage</strong>: fraction of the catalog that appears in anyone's top-K. Popularity covers 2.5%; a good personalized model covers far more.</li>
</ul>
<div class="callout prod"><b>In production</b>Production two-towers add many features (watch history pooled with attention, context), use mixed negative sampling (in-batch + uniformly sampled items to fix the bias toward never-sampled tail items), train on billions of events in streaming fashion, and refresh item embeddings and the index continuously.</div>
`,
      quiz: [
        { q: "Item A appears in 10% of interactions, item B in 0.1%. Under in-batch negatives without correction, which is penalized more as a negative?", options: ["B", "A, about 100× more often", "Equal", "Neither appears as a negative"], answer: 1, why: "Negatives are other batch items, sampled proportional to popularity." },
        { q: "The logQ-corrected training logit is…", options: ["$s + \\log q_j$", "$s - \\log q_j$", "$s / q_j$", "$s \\cdot q_j$"], answer: 1, why: "Subtract the log sampling probability; at serving time use the raw score." },
        { q: "NDCG@3 for a user with truth {A} and recommendations [X, A, Y]?", options: ["1", "$1/\\log_2 3 \\approx 0.63$", "0.5", "0.33"], answer: 1, why: "DCG = 1/log₂(2+1); ideal DCG = 1/log₂(2) = 1." },
        { q: "Why mask same-item off-diagonal logits in a batch?", options: ["To save compute", "They're false negatives: the same item is actually a positive for that row", "To apply logQ", "To normalize vectors"], answer: 1, why: "Training would push a user away from an item they interacted with." },
      ],
      explain: "Explain why in-batch negative sampling makes a two-tower model under-recommend popular items, and how subtracting log(q) during training fixes it.",
      resources: [
        { title: "Yi et al. — Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", url: "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", note: "logQ correction for two-towers" },
        { title: "Koren, Bell, Volinsky — Matrix Factorization Techniques for Recommender Systems", url: "https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf", note: "" },
        { title: "Harper & Konstan — The MovieLens Datasets", url: "https://grouplens.org/datasets/movielens/", note: "" },
        { title: "Järvelin & Kekäläinen — Cumulated gain-based evaluation (NDCG)", url: "https://dl.acm.org/doi/10.1145/582415.582418", note: "" },
      ],
      cards: [
        { f: "In-batch softmax logits", b: "$UI^\\top/\\tau$, a B×B matrix with positives on the diagonal." },
        { f: "logQ correction", b: "Train with $s_{ij} - \\log q_j$ (q = item sampling frequency); serve with raw $s_{ij}$." },
        { f: "Recall@K", b: "|top-K ∩ relevant| / |relevant|" },
        { f: "IDCG in NDCG@K", b: "DCG of min(K, |truth|) hits placed at ranks 1, 2, …" },
      ],
    },
    {
      id: "w8l3", title: "Approximate nearest-neighbor search", minutes: 70,
      summary: "Why exact search doesn't scale, IVF from k-means, the n_probe trade-off, and the ideas behind product quantization and HNSW.",
      keypoints: [
        "Brute force costs $O(Nd)$ per query; fine for 10⁵ vectors, not for 10⁸ at thousands of QPS.",
        "IVF: k-means into $L$ lists; search only the $n_{probe}$ lists nearest the query.",
        "Recall rises and speed falls with $n_{probe}$; $L\\approx\\sqrt N$ is a common starting point.",
        "Store each list contiguously so scanning it is one matmul, not a gather.",
        "PQ compresses vectors into short codes; HNSW navigates a layered proximity graph in ~log N hops.",
      ],
      body: R`
<h2>The problem</h2>
<p>Given $N$ item vectors and a query $q$, find the top-$k$ by inner product (maximum inner product search, MIPS; with normalized vectors it's cosine similarity and equivalent to nearest neighbors in L2). Brute force is a matrix-vector product: $O(Nd)$. For $N=10^5$, $d=64$ that's 6.4M multiply-adds, under a millisecond with BLAS (your test measured ~1–3 ms for 100k–200k). For $N = 10^8$ it's 100× too slow for a latency budget, and memory-bandwidth bound.</p>
<h2>IVF: inverted file index</h2>
<ol>
<li><strong>Train</strong>: run k-means (yours, from Week 2) on a sample to get $L$ centroids, the coarse quantizer.</li>
<li><strong>Add</strong>: assign each vector to its nearest centroid. Sort vectors by list and store them contiguously with an offsets array, so list $c$ is <code>vecs[offsets[c]:offsets[c+1]]</code>.</li>
<li><strong>Search</strong>: score the query against the $L$ centroids, take the top $n_{probe}$ lists, score only their vectors, return the top-$k$ (with <code>argpartition</code> then a small sort).</li>
</ol>
<p>Work per query ≈ $Ld + \frac{n_{probe}}{L}Nd$. Minimizing over $L$ gives $L\approx\sqrt{n_{probe}N}$, hence the $\sqrt N$ rule of thumb. Recall drops when a true neighbor lives in a list whose centroid is not among the $n_{probe}$ nearest, which happens near cluster boundaries; raising $n_{probe}$ trades speed for recall. The test (200k clustered vectors, 512 lists, n_probe 16) wants recall@10 ≥ 0.9 and ≥5× speedup per query over brute force. The reference gets 0.993 recall at 14× with contiguous slices, and only 5.5× when it gathered rows with fancy indexing (a copy per query): memory layout again.</p>
<h2>Product quantization (compression)</h2>
<p>Billions of 128-d float vectors don't fit in RAM (512 GB). PQ splits each vector into $m$ sub-vectors, runs k-means with 256 centroids in each subspace, and stores each vector as $m$ one-byte codes: 128 floats (512 B) become e.g. 16 bytes. Distances are computed approximately from per-query lookup tables. IVF+PQ ("IVFPQ") is the classic billion-scale FAISS index.</p>
<h2>HNSW (graphs)</h2>
<p>Hierarchical Navigable Small World graphs connect each vector to its near neighbors in several layers (sparse at the top, dense at the bottom, like a skip list). Search greedily walks toward the query from an entry point at the top layer, descending layers: about $O(\log N)$ hops. Excellent recall/latency, at the cost of memory for edges and slower builds. It's the default in many vector databases.</p>
<table>
<tr><th>index</th><th>strength</th><th>weakness</th></tr>
<tr><td>Flat (brute force)</td><td>exact, no training</td><td>O(N) per query</td></tr>
<tr><td>IVF</td><td>simple, fast, tunable</td><td>recall loss at boundaries, needs retraining if data shifts</td></tr>
<tr><td>IVF-PQ</td><td>huge scale in RAM</td><td>approximate distances</td></tr>
<tr><td>HNSW</td><td>best recall/latency</td><td>memory, build time, deletes</td></tr>
</table>
<div class="callout prod"><b>In production</b>FAISS (Meta), ScaNN (Google) and vector databases (pgvector, OpenSearch k-NN, Pinecone) implement these. Operational questions: how often to rebuild the index as item embeddings change, how to filter (e.g. exclude seen items or region-restricted items) without destroying recall, and how to shard an index across machines (Week 7's consistent hashing).</div>
`,
      quiz: [
        { q: "N = 10⁶ vectors, IVF with L = 1000 lists and n_probe = 10. Roughly what fraction of vectors are scanned per query?", options: ["10%", "1%", "0.1%", "100%"], answer: 1, why: "n_probe/L = 10/1000 = 1% (plus 1000 centroid scores)." },
        { q: "Raising n_probe…", options: ["lowers recall and latency", "raises recall and latency", "changes neither", "requires retraining"], answer: 1, why: "More lists scanned: fewer missed neighbors, more work." },
        { q: "Why store each IVF list contiguously?", options: ["To save memory", "Scanning a list becomes one contiguous slice matmul instead of a fancy-index gather that copies rows", "k-means requires it", "For thread safety"], answer: 1, why: "Avoiding per-query copies was worth ~2.6× in the reference." },
      ],
      explain: "Explain how an IVF index answers a query, why n_probe trades recall for latency, and when you'd choose HNSW or PQ instead.",
      resources: [
        { title: "Pinecone — FAISS: The Missing Manual (IVF, PQ, HNSW)", url: "https://www.pinecone.io/learn/series/faiss/", note: "" },
        { title: "Jégou et al. — Product Quantization for Nearest Neighbor Search", url: "https://ieeexplore.ieee.org/document/5432202", note: "" },
        { title: "Malkov & Yashunin — HNSW", url: "https://arxiv.org/abs/1603.09320", note: "" },
        { title: "ann-benchmarks.com", url: "https://ann-benchmarks.com/", note: "recall vs QPS for real libraries" },
      ],
      cards: [
        { f: "IVF search in one sentence", b: "Score centroids, scan only the n_probe nearest lists, take the top-k." },
        { f: "Rule of thumb for IVF list count", b: "L ≈ √N (more precisely √(n_probe·N) minimizes work)." },
        { f: "What does product quantization do?", b: "Splits vectors into sub-vectors and replaces each with a 1-byte k-means code: huge memory compression, approximate distances." },
      ],
    },
    {
      id: "w8l4", title: "Ranking, features and leakage", minutes: 70,
      summary: "The second stage: building ranking training data without leakage, the features that matter, training–serving skew, and feature stores.",
      keypoints: [
        "Ranker labels: split TRAIN in time again; candidates from the past part, labels from the future part.",
        "Features for ranker training must be computed only from data before the label window, including the retrieval model itself.",
        "Training–serving skew: the same feature computed differently offline and online silently breaks models.",
        "Feature stores give point-in-time-correct offline joins and low-latency online reads from one definition.",
        "Popularity features in the ranker correct retrieval's biases; position bias distorts click labels.",
      ],
      body: R`
<h2>Building ranker training data</h2>
<p>The ranker learns $P(\text{relevant}\mid u, i, \text{features})$ over candidates that retrieval actually produces, so its training data must look like serving. Recipe (yours, stretch milestone):</p>
<ol>
<li>Split the <em>training</em> interactions in time once more: per user, the last 10% are the "future" (labels), the rest the "history".</li>
<li>Train a two-tower model <strong>on history only</strong>, build an index, retrieve 200 candidates per user (excluding history items).</li>
<li>Label a candidate 1 if it appears in that user's future, else 0. Downsample negatives (positives are ~1%).</li>
<li>Compute features from history only: retrieval score, item popularity, user activity, mean ratings, genre affinity.</li>
<li>Fit your GBM. At serving time recompute the same features from the full training data with the full-train two-tower.</li>
</ol>
<div class="callout pitfall"><b>The leak you'd almost certainly write</b>Using the full-train two-tower to produce candidates and the retrieval-score feature for ranker training. That model has already seen the "future" interactions, so the score of a future item is suspiciously high: the ranker learns "retrieval score is nearly perfect", which is false at test time. Training a second two-tower on history only is the fix. Same logic for every feature: compute it as of the moment of the prediction ("point-in-time correctness").</div>
<p>In the reference build, the ranker on top of uncorrected retrieval lifted NDCG@10 from 0.040 to 0.089 and recall@50 from 0.156 to 0.226. Much of that came from the popularity feature undoing retrieval's popularity bias: a ranker sees the bias in its features and corrects it, the way logQ corrects it inside retrieval. With logQ-corrected retrieval the ranker adds a smaller gain (0.095 → 0.103 NDCG@10).</p>
<h2>Features that matter</h2>
<ul>
<li><strong>User</strong>: activity level, recency, average rating given, genre profile.</li>
<li><strong>Item</strong>: popularity (log count, recent trend), average rating, age, content (genre, text embeddings).</li>
<li><strong>Cross</strong>: user–genre affinity, the retrieval dot product, similarity to recently watched items. GBMs and DNN rankers exist to exploit crosses the two-tower can't.</li>
<li><strong>Context</strong>: time of day, device, page position.</li>
</ul>
<h2>Training–serving skew and feature stores</h2>
<p>A feature computed by a SQL job for training and by a Java service at request time will eventually disagree (time zones, default values, a filter applied in one place only), and the model degrades with no error. Defenses: compute features once from a shared definition; log the features actually used at serving time and train on those logs; monitor feature distributions online vs offline (Week 6's PSI).</p>
<p>A <strong>feature store</strong> (Feast, Tecton, SageMaker Feature Store) makes this systematic: features defined once; an offline store that supports point-in-time joins ("give me each user's features as of each label's timestamp") for training; an online store (a low-latency KV store, often an LSM like your Week 7 project) for serving; and materialization jobs keeping them consistent.</p>
<h2>Biases in ranking labels</h2>
<p><strong>Position bias</strong>: items shown at the top get clicked more regardless of relevance. Training on raw clicks teaches the model to reproduce the old ranking. Mitigations: add position as a training feature and set it to a constant at serving, inverse propensity weighting, or randomized exploration slots to estimate propensities.</p>
<div class="callout prod"><b>In production</b>The same two-stage pattern is retrieval-augmented generation (RAG) for LLMs: embed and retrieve passages with an ANN index (high recall), re-rank them with a cross-encoder (precision), then generate. Leakage, freshness and skew problems carry over directly.</div>
`,
      quiz: [
        { q: "You compute 'item popularity' for ranker training over the entire training set including the label window. Problem?", options: ["None", "Leakage: popularity includes the very interactions being predicted", "It's too slow", "Popularity isn't a valid feature"], answer: 1, why: "Features must be computed from data before the label window (point-in-time)." },
        { q: "What does a feature store's offline 'point-in-time join' guarantee?", options: ["Faster training", "Each training example gets feature values as they were at that example's timestamp", "Features are normalized", "Online and offline use different code"], answer: 1, why: "Prevents future information leaking into training features." },
      ],
      explain: "Explain how to build training data for a second-stage ranker without leakage, including why the retrieval model used for ranker training must differ from the serving one.",
      resources: [
        { title: "Eugene Yan — Feature stores: a hierarchy of needs", url: "https://eugeneyan.com/writing/feature-stores/", note: "" },
        { title: "Feast documentation — point-in-time joins", url: "https://docs.feast.dev/getting-started/concepts/point-in-time-joins", note: "" },
        { title: "Joachims et al. — Unbiased Learning-to-Rank with Biased Feedback", url: "https://arxiv.org/abs/1608.04468", note: "position bias" },
      ],
      cards: [
        { f: "Training–serving skew", b: "A feature computed differently offline (training) and online (serving), silently degrading the model." },
        { f: "Point-in-time correctness", b: "Each training example's features use only data available at that example's timestamp." },
        { f: "Ranker label construction", b: "Split train in time; candidates from the history part (with a history-only retrieval model), labels from the future part." },
      ],
    },
    {
      id: "w8l5", title: "Experimentation and launch decisions", minutes: 75,
      summary: "A/B tests from the ground up: the z-test, power and sample size derived, peeking and multiple testing, CUPED, and how to decide whether to launch.",
      keypoints: [
        "Randomize users (not requests) into control and treatment; compare the primary metric with a pre-registered test.",
        "Sample size per arm: $n = (z_{1-\\alpha/2}\\sqrt{2\\bar p(1-\\bar p)} + z_{1-\\beta}\\sqrt{p_1(1-p_1)+p_2(1-p_2)})^2/\\delta^2$.",
        "Halving the minimum detectable effect quadruples the required sample.",
        "Peeking and stopping at the first p < 0.05 inflates false positives; fix the horizon or use sequential methods.",
        "CUPED reduces variance with a pre-experiment covariate: $Y' = Y - \\theta(X - \\bar X)$, $\\theta = \\mathrm{cov}(X,Y)/\\mathrm{var}(X)$.",
      ],
      body: R`
<h2>Why experiments</h2>
<p>Offline NDCG went up; will watch time go up? Not necessarily: offline metrics are proxies computed on logged data shaped by the old system. A randomized controlled experiment is the only reliable way to measure causal impact on the business metric. Randomize by <em>user</em> (so each user has a consistent experience and the unit matches the metric), typically by hashing the user id into buckets.</p>
<h2>The two-proportion z-test</h2>
<p>Control converts $x_A$ of $n_A$ users ($\hat p_A$), treatment $x_B$ of $n_B$ ($\hat p_B$). Under $H_0: p_A = p_B$, pool $\hat p = \frac{x_A+x_B}{n_A+n_B}$ and</p>
$$z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p(1-\hat p)\big(\frac1{n_A}+\frac1{n_B}\big)}},\qquad p\text{-value} = 2\big(1-\Phi(|z|)\big).$$
<p>Example (tested): 200/2000 vs 250/2000 gives $z\approx 2.50$, $p\approx 0.0124$.</p>
<h2>Power and sample size, derived</h2>
<p>Choose the significance level $\alpha$ (false-positive rate, usually 0.05), the power $1-\beta$ (chance of detecting a real effect, usually 0.8), and the minimum detectable effect $\delta = p_2 - p_1$ worth acting on. We reject when $|\hat\Delta| > z_{1-\alpha/2}\,\sigma_0$, where $\sigma_0 = \sqrt{2\bar p(1-\bar p)/n}$ is the standard error under $H_0$ (per-arm size $n$, $\bar p = (p_1+p_2)/2$). Under the alternative, $\hat\Delta\sim\mathcal{N}(\delta, \sigma_1^2)$ with $\sigma_1 = \sqrt{(p_1(1-p_1)+p_2(1-p_2))/n}$. Power $\approx P(\hat\Delta > z_{1-\alpha/2}\sigma_0) = \Phi\big(\frac{\delta - z_{1-\alpha/2}\sigma_0}{\sigma_1}\big)$. Setting this equal to $1-\beta$ means $\delta = z_{1-\alpha/2}\sigma_0 + z_{1-\beta}\sigma_1$; substitute and solve for $n$:</p>
$$n = \frac{\Big(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\Big)^2}{\delta^2}.$$
<p>For $p_1 = 10\%$, $\delta = 2$ points: 3,841 users per arm. Since $n\propto 1/\delta^2$, detecting 1 point needs ~4×. Your <code>simulate_power</code> verifies the formula by Monte Carlo: at $n = 3{,}841$ the simulated power is ≈ 0.80, and with no true effect the rejection rate is ≈ 0.05.</p>
<div class="widget" data-widget="abpower"></div>
<h2>Ways to fool yourself</h2>
<ul>
<li><strong>Peeking</strong>: checking daily and stopping the first time $p<0.05$ can push the real false-positive rate above 20%. Fix the sample size in advance, or use sequential tests (always-valid p-values, alpha spending).</li>
<li><strong>Multiple testing</strong>: 20 metrics at α = 0.05 give one "significant" result by chance. Pre-register one primary metric; correct the rest (Bonferroni, Benjamini–Hochberg).</li>
<li><strong>Novelty and primacy effects</strong>: users click new things because they're new. Run long enough (often two full weeks to cover weekly cycles) and look at the trend.</li>
<li><strong>Sample ratio mismatch</strong>: a 50/50 split that comes back 50.8/49.2 with millions of users indicates a bug in assignment or logging. Check it first.</li>
<li><strong>Interference</strong>: in marketplaces and social networks treatment users affect control users; randomize by cluster or region.</li>
</ul>
<h2>CUPED: free variance reduction</h2>
<p>Much of the variance in a user's metric is predictable from their pre-experiment behavior $X$ (last month's watch time). Use $Y' = Y - \theta(X - \bar X)$ with $\theta = \mathrm{cov}(X,Y)/\mathrm{var}(X)$: same expected treatment effect (since $X$ is unaffected by treatment), variance reduced by a factor $1-\rho^2$ where $\rho = \mathrm{corr}(X,Y)$. With $\rho = 0.7$, variance halves, equivalent to doubling the sample. Microsoft, Netflix and Booking use it routinely.</p>
<h2>Interleaving for rankers</h2>
<p>To compare two rankers quickly, interleave their results in one list per user (team-draft interleaving) and credit clicks to the ranker that contributed the clicked item. It needs orders of magnitude fewer users than an A/B test to detect a preference, then an A/B test confirms the business impact.</p>
<h2>Launching</h2>
<p>Launch when the primary metric improves significantly, guardrails (latency, errors, diversity, revenue) don't regress, the effect makes sense, and the cost is justified. Then ramp (1% → 10% → 50% → 100%) with monitoring, keeping a small long-term holdout to measure the cumulative effect of many launches.</p>
<div class="callout prod"><b>In production</b>Your design doc's online-evaluation section should name the unit of randomization, primary and guardrail metrics, the sample size from your own calculator, the duration, and the ramp plan. That paragraph is where many ML design interviews are won or lost.</div>
`,
      quiz: [
        { q: "Baseline 5%, you want to detect +0.5 points instead of +1 point. Required sample per arm changes by about…", options: ["×2", "×4", "×0.5", "unchanged"], answer: 1, why: "$n\\propto 1/\\delta^2$." },
        { q: "You check results daily and stop at the first p < 0.05. Your actual false-positive rate is…", options: ["exactly 5%", "below 5%", "well above 5%", "0%"], answer: 2, why: "Repeated looks give repeated chances to cross the threshold by chance." },
        { q: "CUPED with pre-period correlation ρ = 0.8 reduces variance by…", options: ["20%", "64%", "80%", "36%"], answer: 1, why: "Remaining variance is $1-\\rho^2 = 0.36$, a 64% reduction." },
        { q: "With 1M users split 50/50 you observe 503,000 vs 497,000. First action?", options: ["Launch", "Investigate sample ratio mismatch (assignment/logging bug)", "Run longer", "Ignore it"], answer: 1, why: "Under a fair 50/50 split the difference between arms has SD $2\\sqrt{N/4} = 1{,}000$, so a 6,000-user gap is 6 standard deviations: something is broken." },
      ],
      explain: "Derive the sample-size formula for a two-proportion A/B test from the null and alternative distributions, and explain why peeking inflates false positives.",
      resources: [
        { title: "Kohavi, Tang, Xu — Trustworthy Online Controlled Experiments", url: "https://experimentguide.com/", note: "the A/B testing book" },
        { title: "Deng et al. — CUPED (Improving the Sensitivity of Online Controlled Experiments)", url: "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf", note: "" },
        { title: "Evan Miller — How Not To Run an A/B Test (peeking)", url: "https://www.evanmiller.org/how-not-to-run-an-ab-test.html", note: "" },
        { title: "Chapelle et al. — Large-scale validation and analysis of interleaved search evaluation", url: "https://dl.acm.org/doi/10.1145/2094072.2094078", note: "" },
      ],
      cards: [
        { f: "Two-proportion z statistic (pooled)", b: "$(\\hat p_B-\\hat p_A)/\\sqrt{\\hat p(1-\\hat p)(1/n_A+1/n_B)}$" },
        { f: "Sample size scaling with MDE", b: "$n \\propto 1/\\delta^2$" },
        { f: "CUPED adjustment", b: "$Y' = Y - \\theta(X-\\bar X)$, $\\theta = \\mathrm{cov}(X,Y)/\\mathrm{var}(X)$; variance × $(1-\\rho^2)$." },
        { f: "Sample ratio mismatch", b: "Observed split differs significantly from the designed split: an assignment or logging bug; check before reading results." },
      ],
    },
  ],
  project: {
    title: "recsys — a two-stage MovieLens recommender, served",
    dir: "recsys/, docs/w8_ml_design.md",
    pitch: "Load MovieLens-1M with a leakage-free per-user temporal split, train a two-tower model with in-batch negatives and logQ correction on your GPU, index items with an IVF built on your k-means, serve recommendations with metrics, rate limiting and a cold-start fallback, and size the A/B test that would launch it.",
    test: "pytest tests/w8 -q",
    slow: "pytest tests/w8 -m \"slow and not stretch\" -s",
    overview: R`
<p>The fast tests (14) cover data loading and splitting, metrics, the popularity baseline, the IVF index (200k vectors: recall@10 ≥ 0.9 and ≥5× faster than brute force per query), the A/B statistics, the service, and the design doc. The slow tests build the full pipeline: on the reference, building takes ~40 s on the RTX 3060 and yields recall@50 0.245 vs popularity 0.155 and NDCG@10 0.095 vs 0.079. Your pipeline must beat popularity by 30% on recall@50 and 5% on NDCG@10, with 5× its catalog coverage.</p>
<p>Before training with logQ correction, train once without it (<code>TwoTowerConfig(logq_correction=False)</code>) and record both evaluations in your design doc. Watching the bias appear and disappear on real data is the lesson.</p>
`,
    rubric: R`
<p>Score your <code>docs/w8_ml_design.md</code> 0–2 per line (aim for ≥16/20):</p>
<ul>
<li>Business goal, ML objective and guardrails are distinct and measurable.</li>
<li>Label definition, negatives, and label delay are explicit.</li>
<li>The split is justified (temporal, per user) and leakage risks are named.</li>
<li>Features: user, item, cross, context; offline vs online computation; skew prevention.</li>
<li>Architecture diagram with a latency budget per stage.</li>
<li>Retrieval details: in-batch negatives, logQ (with your measured before/after numbers), false-negative masking.</li>
<li>Offline results table vs popularity, including coverage and one slice (e.g. light vs heavy users).</li>
<li>A/B plan: unit, primary + guardrail metrics, sample size from your calculator, duration, ramp.</li>
<li>Serving: index refresh, caching, cold start, rate limiting, fallbacks.</li>
<li>Monitoring and retraining: drift metrics, feedback loops, cadence.</li>
</ul>
`,
    milestones: [
      { id: "w8m1", core: true, title: "Data: loaders, per-user temporal split, encode_ids", test: "pytest tests/w8/test_recsys_core.py -q -k \"loaders or split or encode\"",
        detail: R`<p>Exact row counts and dtypes, no leakage, the toy split exactly, ids built from train only, no mutation of inputs.</p>`,
        hints: [
          R`<p><code>pd.read_csv(path, sep="::", engine="python", encoding="latin-1", names=[...])</code>.</p>`,
          R`<p>Sort by (user, timestamp, item); per-user size via <code>groupby(...).transform("size")</code> and rank via <code>groupby(...).cumcount()</code>; test rows are rank ≥ n − ceil(frac·n) for users with n ≥ min_train + 1.</p>`,
          R`<p>Build the index dicts from sorted unique train ids; filter test rows to known users and items before mapping.</p>`,
        ] },
      { id: "w8m2", core: true, title: "Metrics and the popularity baseline", test: "pytest tests/w8/test_recsys_core.py -q -k \"metrics or popularity\"",
        detail: R`<p>Hand-computed recall/hit/NDCG/coverage including users without recommendations and users with empty truth; popularity with id tie-break and exclusions.</p>`,
        hints: [
          R`<p>Average only over users whose truth set is non-empty; <code>recs.get(u, [])</code> for missing users.</p>`,
          R`<p>Rank r (0-based) contributes 1/log2(r + 2).</p>`,
        ] },
      { id: "w8m3", core: true, title: "Two-tower model with in-batch negatives and logQ correction",
        detail: R`<p>Implement <code>TwoTower</code>, <code>genre_matrix</code>, <code>train_two_tower</code> (GPU), <code>export_embeddings</code>. Checked by the slow pipeline tests. Compare logq on/off and keep both numbers for your doc.</p>`,
        hints: [
          R`<p>Logits <code>U @ I.T / temperature</code> (B×B); labels <code>arange(B)</code>; cross-entropy.</p>`,
          R`<p>q = <code>bincount(train items) / total</code>; corrected logits <code>logits - log(q[i])[None, :]</code> (columns). False negatives: <code>same = i[None, :] == i[:, None]</code>, mask <code>same &amp; ~eye</code> with -inf.</p>`,
          R`<p>Keep all training ids on the GPU and index batches from a permutation there; 20 epochs over ~460k pairs takes well under a minute.</p>`,
        ] },
      { id: "w8m4", core: true, title: "IVF index", test: "pytest tests/w8/test_recsys_core.py -q -k \"brute or ivf\"",
        detail: R`<p>Exact brute force; IVF with your KMeans, contiguous lists, custom ids, padding with -1/-inf.</p>`,
        hints: [
          R`<p>Top-k: <code>np.argpartition(-scores, k-1)[:k]</code> then sort those k.</p>`,
          R`<p>add(): assign by argmax of <code>vectors @ centroids.T</code>, <code>order = argsort(assign, kind="stable")</code>, store <code>vecs[order]</code>, <code>ids[order]</code> and <code>offsets = searchsorted(assign[order], arange(L+1))</code>.</p>`,
          R`<p>search(): for each probed list, score the slice <code>vecs[a:b] @ q</code> (no gather), concatenate, top-k.</p>`,
        ] },
      { id: "w8m5", core: true, title: "Pipeline: build, evaluate, Recommender with fallback", test: "pytest tests/w8 -m \"slow and not stretch\" -s",
        detail: R`<p>Beat popularity on recall@50 (×1.3), NDCG@10 (×1.05) and coverage (×5); never recommend seen items; cold-start fallback for unknown users.</p>`,
        hints: [
          R`<p>Ask the index for n + (a margin) candidates, then filter seen items and truncate: filtering after search can otherwise leave you short.</p>`,
          R`<p>Keep a raw-id lookup array for items (encoded index → raw MovieLens id) for the API.</p>`,
          R`<p>If you're below popularity, check logQ is on, embeddings are normalized, and the split/encoding didn't shift ids between train and test.</p>`,
        ] },
      { id: "w8m6", core: true, title: "A/B statistics", test: "pytest tests/w8/test_recsys_core.py -q -k \"sample_size or ztest or power\"",
        detail: R`<p>Sample size exactly 3841 for (10%, +2 pts); z-test; vectorized power simulation that recovers 0.80 and α.</p>`,
        hints: [
          R`<p><code>from statistics import NormalDist</code>; <code>NormalDist().inv_cdf(1 - alpha/2)</code>.</p>`,
          R`<p>Vectorize the simulation: <code>rng.binomial(n, p, size=n_sims)</code> for each arm, compute all z's at once.</p>`,
        ] },
      { id: "w8m7", core: true, title: "Recommendation service", test: "pytest tests/w8/test_recsys_core.py -q -k service",
        detail: R`<p>FastAPI app with /recommend/{user_id}?k=…, 422 for bad k, fallback flag, metrics, and your Week 7 rate limiter.</p>`,
        hints: [R`<p>Reuse <code>serve.metrics</code> and the same middleware pattern as <code>serve/app.py</code>; <code>Query(10, ge=1, le=100)</code> for k.</p>`] },
      { id: "w8m8", core: true, title: "ML design document", test: "pytest tests/w8/test_recsys_core.py -q -k design_doc",
        detail: R`<p>Write <code>docs/w8_ml_design.md</code> from the template (≥1000 words) using your measured numbers. Score it with the rubric.</p>`,
        hints: [R`<p>Lead with the results table (yours vs popularity, logQ off vs on); then explain how you got there. Ask the tutor for an interviewer-style critique.</p>`] },
      { id: "w8m9", core: false, title: "Stretch: GBM re-ranker without leakage", test: "pytest tests/w8 -m stretch -s",
        detail: R`<p>Follow L4: a history-only two-tower for candidates and features, labels from the future slice, your Week 2 GradientBoostingClassifier as the ranker. Target NDCG@10 &gt; 1.15× popularity.</p>`,
        hints: [
          R`<p>Downsample negatives (keep all positives, ~10% of negatives) or GBM training on 400k rows will be slow.</p>`,
          R`<p>Your GBM's <code>decision_function</code> is the ranking score; no need for calibrated probabilities.</p>`,
        ] },
    ],
  },
});
