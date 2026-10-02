COURSE.weeks.push({
  n: 7, id: "w7", title: "System Design & HLD",
  tagline: "Estimation, scaling, replication and partitioning, storage engines, caches, queues and rate limiting. You build the primitives and write a full high-level design.",
  hours: { learn: 6.5, project: 7 },
  goals: [
    "Do back-of-envelope estimates for QPS, storage and bandwidth in under five minutes",
    "Explain replication, partitioning, quorums and CAP/PACELC trade-offs with concrete systems",
    "Implement consistent hashing, a Bloom filter and an LSM-tree store with crash recovery",
    "Choose and implement the right rate-limiting algorithm and wire it into your Week 6 server",
    "Write an interview-grade HLD for a URL shortener",
  ],
  schedule: [
    ["Mon", "L1 Scaling fundamentals and estimation"],
    ["Tue", "L2 Replication, partitioning, consistency"],
    ["Wed", "L3 Storage engines: the LSM-tree (directly your project)"],
    ["Thu", "L4 Caching, queues and rate limiting + L5 the HLD framework"],
    ["Fri", "Flashcards; draft the URL-shortener HLD outline"],
    ["Sat", "Milestones 1–5: hash ring, Bloom filter, LSM store with WAL recovery"],
    ["Sun", "Milestones 6–8: compaction, rate limiters + server integration, finish the HLD doc"],
  ],
  lessons: [
    {
      id: "w7l1", title: "Scaling fundamentals and estimation", minutes: 70,
      summary: "Latency numbers, back-of-envelope math, availability, stateless services and load balancing: the vocabulary of every design discussion.",
      keypoints: [
        "Memory ~100 ns, SSD random read ~100 µs, same-datacenter round trip ~0.5 ms, cross-continent ~150 ms.",
        "1 day ≈ 86,400 s ≈ 10⁵ s: 1M requests/day ≈ 12 QPS average; plan peak at 2–5× average.",
        "Availability multiplies for serial dependencies: 99.9% × 99.9% = 99.8%.",
        "Scale out stateless services behind a load balancer; push state into dedicated stores.",
        "L4 balancers route connections; L7 balancers route requests by content.",
      ],
      body: R`
<h2>Numbers to carry in your head</h2>
<table>
<tr><th>operation</th><th>time</th></tr>
<tr><td>L1 cache reference</td><td>~1 ns</td></tr>
<tr><td>Main memory reference</td><td>~100 ns</td></tr>
<tr><td>Read 1 MB sequentially from memory</td><td>~10 µs</td></tr>
<tr><td>SSD random read (4 KB)</td><td>~100 µs</td></tr>
<tr><td>Round trip within a datacenter</td><td>~0.5 ms</td></tr>
<tr><td>Read 1 MB sequentially from SSD</td><td>~0.5–1 ms</td></tr>
<tr><td>Disk (HDD) seek</td><td>~10 ms</td></tr>
<tr><td>Round trip US ↔ Europe</td><td>~80–150 ms</td></tr>
</table>
<p>The orders of magnitude matter, not the digits: memory is ~1000× faster than SSD random reads, which are ~100× faster than disk seeks; a network hop inside a datacenter costs about as much as a few SSD reads. So: cache hot data in memory, read sequentially when you can (the reason LSM-trees exist), and minimize cross-region round trips.</p>
<h2>Back-of-envelope estimation</h2>
<p>One day is 86,400 s; round to $10^5$. So:</p>
<ul>
<li>100M requests/day ≈ $10^8/10^5 = 1{,}000$ QPS average; peak 2–5× ⇒ design for ~5k QPS.</li>
<li>Storage: records/day × bytes/record × retention. 100M URLs/day × 500 B × 5 years ≈ $10^8\times500\times1800\approx 9\times10^{13}$ B ≈ 90 TB.</li>
<li>Bandwidth: QPS × response size. 5k QPS × 1 KB = 5 MB/s, trivial; 5k QPS × 1 MB images = 5 GB/s, not trivial.</li>
<li>Cache: the 80/20 rule: caching the hottest 20% of a day's reads covers most traffic.</li>
<li>ML: an embedding table of 100M users × 64 dims × 4 B = 25.6 GB: fits on one big machine, not on one GPU.</li>
</ul>
<p>Write the arithmetic down in the interview; the reasoning is graded more than the exact number.</p>
<h2>Availability</h2>
<p>"Three nines" (99.9%) allows 8.8 hours of downtime a year; four nines, 53 minutes. Serial dependencies multiply: a request needing services at 99.9% and 99.95% succeeds 99.85% of the time. Redundancy helps: two independent replicas each at 99% give $1 - 0.01^2 = 99.99\%$ (if failures are truly independent, which they often aren't: same rack, same bad deploy).</p>
<h2>Scaling out</h2>
<ul>
<li><strong>Vertical</strong> (bigger machine): simple, a hard ceiling, a single point of failure.</li>
<li><strong>Horizontal</strong> (more machines): needs the service to be <strong>stateless</strong>: any replica can serve any request because state lives in databases, caches and queues. Your model server is stateless (the model is read-only); the KV cache is per-request state that lives only for the request.</li>
<li><strong>Load balancers</strong>: L4 (TCP) route connections by address/port, fast and content-blind. L7 (HTTP) inspect requests: route by path or header, terminate TLS, retry. Algorithms: round robin, least connections (good for uneven request costs like LLM generation), consistent hashing (sticky routing, e.g. to a replica holding a cache), power-of-two-choices.</li>
<li><strong>CDN</strong>: cache static and cacheable content at the edge, near users.</li>
</ul>
<div class="callout prod"><b>In production / AWS</b>Route 53 (DNS) → CloudFront (CDN) → ALB (L7) or NLB (L4) → auto-scaling groups / ECS / EKS services → ElastiCache (Redis) → RDS/Aurora or DynamoDB → S3. Knowing what each box costs and how it fails is the core of an HLD interview, and your AWS experience gives you real anchors for these.</div>
`,
      quiz: [
        { q: "50M requests per day, peak factor 3. Design peak QPS?", options: ["≈ 50", "≈ 600", "≈ 1,750", "≈ 50,000"], answer: 2, why: "50M / 86,400 ≈ 580 average; × 3 ≈ 1,750." },
        { q: "A request depends serially on three services, each 99.9% available. End-to-end availability?", options: ["99.9%", "≈ 99.7%", "99.99%", "≈ 97%"], answer: 1, why: "0.999³ ≈ 0.997." },
        { q: "LLM requests vary 100× in cost. Which load-balancing algorithm fits best?", options: ["Round robin", "Least outstanding requests / least connections", "Random", "DNS round robin"], answer: 1, why: "Round robin ignores cost; least-outstanding routes around busy replicas." },
      ],
      explain: "Estimate QPS, storage and cache size for a service that creates 10M short links per day and serves 100 reads per link created, explaining each step.",
      resources: [
        { title: "Jeff Dean — Latency numbers every programmer should know (interactive)", url: "https://colin-scott.github.io/personal_website/research/interactive_latency.html", note: "" },
        { title: "Alex Xu — System Design Interview vol. 1, ch. 1–3", url: "https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF", note: "" },
        { title: "The System Design Primer", url: "https://github.com/donnemartin/system-design-primer", note: "" },
      ],
      cards: [
        { f: "Seconds in a day (rounded)", b: "86,400 ≈ 10⁵" },
        { f: "Memory vs SSD random read latency", b: "~100 ns vs ~100 µs (≈1000×)" },
        { f: "Downtime per year at 99.9% availability", b: "≈ 8.8 hours" },
        { f: "L4 vs L7 load balancer", b: "L4 routes TCP connections; L7 routes HTTP requests by content." },
      ],
    },
    {
      id: "w7l2", title: "Replication, partitioning and consistency", minutes: 85,
      summary: "How data systems survive failures and outgrow one machine, and the consistency trade-offs that follow.",
      keypoints: [
        "Leader-follower replication: writes to the leader; async followers can serve stale reads.",
        "Quorums: with N replicas, R + W > N makes reads overlap the latest write.",
        "Hash partitioning spreads load; range partitioning keeps scans cheap but risks hot spots.",
        "Consistent hashing moves only ~1/N of keys when a node joins; virtual nodes even out load.",
        "CAP: during a partition choose consistency or availability. PACELC: otherwise choose latency or consistency.",
      ],
      body: R`
<h2>Replication</h2>
<p>Copies of data on several machines give fault tolerance and read scaling.</p>
<ul>
<li><strong>Leader–follower</strong> (Postgres, MySQL, most of RDS): all writes go to the leader, which streams a log to followers. <em>Synchronous</em> replication waits for a follower's ack (durable, slower, a dead follower blocks writes); <em>asynchronous</em> doesn't (fast, but a leader crash loses the unreplicated tail, and followers serve stale reads). Failover promotes a follower: risky (split brain, lost writes) and must be fenced.</li>
<li><strong>Multi-leader</strong>: writes accepted in several regions; conflicts must be resolved (last-writer-wins, CRDTs).</li>
<li><strong>Leaderless</strong> (Dynamo, Cassandra): clients write to $W$ of $N$ replicas and read from $R$. If $R + W > N$, every read set intersects every write set, so a read sees the latest acknowledged write (with versioning to pick it). $N=3, W=2, R=2$ is the classic setting; $W=1, R=1$ is fast but eventually consistent.</li>
</ul>
<h3>Consistency models (strongest to weakest)</h3>
<ul>
<li><strong>Linearizable</strong>: behaves like a single copy; once a write is acknowledged, every later read sees it.</li>
<li><strong>Read-your-writes</strong>: you see your own writes (route a user's reads to the leader for a while after they write).</li>
<li><strong>Monotonic reads</strong>: you never see time go backwards (pin a user to one replica).</li>
<li><strong>Eventual</strong>: replicas converge if writes stop.</li>
</ul>
<h3>CAP and PACELC</h3>
<p>CAP: when a network <em>partition</em> separates replicas, a system must either refuse some requests (stay <strong>C</strong>onsistent) or serve possibly-stale data (stay <strong>A</strong>vailable). It's a statement about behavior during failures. PACELC adds the everyday trade-off: <em>else</em> (no partition), choose <strong>L</strong>atency or <strong>C</strong>onsistency, since synchronous coordination costs round trips. DynamoDB and Cassandra lean PA/EL (tunable); Spanner leans PC/EC (TrueTime, consistent but pays latency).</p>
<h2>Partitioning (sharding)</h2>
<p>When data or write load exceeds one machine, split it.</p>
<ul>
<li><strong>Range partitioning</strong> (by key ranges): efficient range scans; sequential keys (timestamps) create a hot partition.</li>
<li><strong>Hash partitioning</strong>: even spread; range scans hit every partition.</li>
<li><strong>Hot keys</strong> (a celebrity's profile) defeat any scheme: add a random suffix to spread writes, cache reads aggressively.</li>
</ul>
<h3>Consistent hashing</h3>
<p>The naive scheme <code>node = hash(key) mod N</code> remaps almost every key when $N$ changes (only keys where both mods agree stay: about $1/(N+1)$ of them). Consistent hashing places nodes and keys on a ring (hash space $[0, 2^{64})$); a key belongs to the first node clockwise. Adding a node only takes keys from its clockwise successor's arc: expected movement $1/(N+1)$, and only <em>to</em> the new node. Removing a node hands its keys to its successor only.</p>
<p>With one point per node, arcs have very uneven lengths (the max arc is ~$\ln N/N$ of the ring). <strong>Virtual nodes</strong> give each node many points ($v$ ≈ 100–200), so each node owns many small arcs and its share concentrates around $1/N$ (coefficient of variation shrinks roughly like $1/\sqrt v$). They also let you weight nodes by capacity and spread a dead node's load across many survivors.</p>
<div class="widget" data-widget="ring"></div>
<p>Lookup is a binary search over sorted ring positions: <code>bisect.bisect(points, hash(key)) % len(points)</code>. Replica placement walks clockwise collecting the next $n$ <em>distinct</em> physical nodes (skip vnodes of nodes already chosen). That's Dynamo's preference list and your <code>get_nodes</code>.</p>
<div class="callout prod"><b>In production</b>Consistent hashing is in DynamoDB/Cassandra partitioning, memcached client libraries, load balancers with session affinity, and sharded embedding servers for recommenders. ML-specific: sharding a huge embedding table across parameter servers by hashed id is the same problem.</div>
`,
      quiz: [
        { q: "N = 5 replicas. Which (R, W) guarantees reads overlap the latest write?", options: ["R=2, W=2", "R=2, W=3", "R=3, W=3", "R=1, W=4"], answer: 2, why: "Need R + W > N = 5: only 3+3 = 6 qualifies (2+3 = 5 and 1+4 = 5 do not)." },
        { q: "Going from 9 to 10 nodes with consistent hashing, roughly what fraction of keys move?", options: ["90%", "50%", "10%", "1%"], answer: 2, why: "About 1/(N+1) = 1/10, all to the new node. mod-N hashing would move ~90%." },
        { q: "Why add virtual nodes?", options: ["To speed up lookups", "To balance load: many small arcs average out arc-length variance", "To add replication", "To support range scans"], answer: 1, why: "One point per node leaves very uneven arcs; many points per node concentrate each node's share near 1/N." },
        { q: "PACELC 'EL' means…", options: ["When there's no partition, the system prefers lower latency over strong consistency", "Eventually linearizable", "Election by leader", "Elastic load"], answer: 0, why: "Else (no partition): Latency vs Consistency." },
      ],
      explain: "Explain consistent hashing with virtual nodes: why mod-N hashing fails when nodes change, what moves when a node joins, and why virtual nodes balance load.",
      resources: [
        { title: "Kleppmann — Designing Data-Intensive Applications, ch. 5–6, 9", url: "https://dataintensive.net/", note: "replication, partitioning, consistency: the core reading" },
        { title: "DeCandia et al. — Dynamo: Amazon's Highly Available Key-value Store", url: "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf", note: "ring, vnodes, quorums" },
        { title: "Abadi — Consistency Tradeoffs in Modern Distributed Database System Design (PACELC)", url: "https://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf", note: "" },
      ],
      cards: [
        { f: "Quorum overlap condition", b: "R + W > N" },
        { f: "Keys moved when adding a node to a consistent-hash ring of N", b: "≈ 1/(N+1), all to the new node." },
        { f: "CAP in one sentence", b: "During a network partition, a replicated system must give up either consistency or availability." },
        { f: "Range vs hash partitioning", b: "Range: cheap scans, hot-spot risk. Hash: even load, scans hit all partitions." },
      ],
    },
    {
      id: "w7l3", title: "Storage engines: the LSM-tree", minutes: 90,
      summary: "B-trees vs log-structured merge-trees, then every component of an LSM store: WAL, memtable, SSTables, sparse indexes, Bloom filters, compaction and crash recovery.",
      keypoints: [
        "B-trees update in place (good reads, random writes); LSM-trees append and merge (sequential writes, more read work).",
        "Write path: WAL append (durable) → memtable → flush to an immutable sorted SSTable.",
        "Deletes are tombstones; compaction merges SSTables, keeps newest versions, drops tombstones.",
        "Bloom filter false-positive rate ≈ $(1-e^{-kn/m})^k$, minimized at $k = (m/n)\\ln 2$.",
        "A WAL record needs a length and a checksum so a torn tail after a crash can be detected and ignored.",
        "Atomic file replace (write temp, fsync, rename) and a manifest keep the set of live files consistent.",
      ],
      body: R`
<h2>Two families</h2>
<p><strong>B-trees</strong> (Postgres, MySQL InnoDB) keep sorted pages on disk and update them in place: a read is $O(\log n)$ page reads, a write rewrites a page (random I/O plus a write-ahead log). <strong>LSM-trees</strong> (LevelDB, RocksDB, Cassandra, ScyllaDB, the storage under DynamoDB-like systems) never update in place: writes are buffered in memory and flushed as sorted immutable files, merged later in the background. Writes become sequential I/O (fast), at the cost of reads that may consult several files and of background compaction work.</p>
<p>The trade-off is stated as amplification: <em>write amplification</em> (bytes written to disk per byte of user data, from compaction), <em>read amplification</em> (files touched per read), <em>space amplification</em> (stale versions not yet compacted). Tuning an LSM store is choosing among them.</p>
<h2>The write path</h2>
<ol>
<li><strong>WAL append.</strong> Append the operation to <code>wal.log</code> and flush it to the OS before acknowledging. If the process crashes, the WAL replays it. (fsync on every write survives power loss too, at a large latency cost; group commit batches fsyncs.)</li>
<li><strong>Memtable.</strong> Apply the operation to an in-memory map (a sorted structure in real engines; a dict sorted at flush time is fine for you). Deletes write a <strong>tombstone</strong>, not a removal: an older SSTable may still hold the key, and the tombstone must shadow it.</li>
<li><strong>Flush.</strong> When the memtable is big enough, write it in key order as a new <strong>SSTable</strong> (sorted string table), then truncate the WAL: its contents are now durable in the SSTable.</li>
</ol>
<h2>The read path</h2>
<p>Check the memtable, then SSTables from <em>newest to oldest</em>; the first hit wins (a tombstone means "deleted"). To keep this cheap:</p>
<ul>
<li><strong>Sparse index</strong>: keep every 16th key and its byte offset in memory. Binary-search the index, seek to that block, scan at most 16 records. Memory stays small no matter how big the file.</li>
<li><strong>Bloom filter</strong> per SSTable: answers "definitely not here" for most absent keys without touching disk. The test counts SSTable reads for 1,000 absent keys and demands &lt;5% of the naive count.</li>
</ul>
<h3>Bloom filters, derived</h3>
<p>A bit array of $m$ bits and $k$ hash functions. Insert: set the $k$ bits. Query: if any of the $k$ bits is 0, the key was never inserted (no false negatives). After $n$ inserts a given bit is still 0 with probability $(1-1/m)^{kn}\approx e^{-kn/m}$, so a false positive (all $k$ bits set by chance) has probability</p>
$$p \approx \big(1 - e^{-kn/m}\big)^k.$$
<p>Minimizing over $k$ gives $k^* = \frac mn\ln 2$, and then $p = 2^{-k^*}$, i.e. $m = -\frac{n\ln p}{(\ln 2)^2}$. For $p = 1\%$: $m/n\approx 9.6$ bits per key and $k = 7$, exactly what <code>test_bloom_sizing_formulas</code> checks. Generate the $k$ indices with <strong>double hashing</strong>, $h_i = h_1 + i\,h_2 \bmod m$, from one cryptographic digest (Kirsch–Mitzenmacher show it's as good as $k$ independent hashes). Store real bits: <code>bits[i >> 3] |= 1 << (i & 7)</code>.</p>
<h2>Compaction</h2>
<p>Flushes keep adding files, and reads slow down. Compaction merges SSTables: a k-way merge of sorted runs (<code>heapq.merge</code>) keeping only the newest version of each key. Strategies: <em>size-tiered</em> (merge files of similar size; low write amplification, more space) and <em>leveled</em> (LevelDB/RocksDB: each level 10× larger with non-overlapping key ranges; fewer files per read, more write amplification). Tombstones can be dropped only when the merge includes <em>every</em> older file that might hold the key; your full compaction (merge everything) qualifies.</p>
<h2>Crash safety</h2>
<ul>
<li><strong>Torn writes.</strong> A crash mid-append leaves a partial record at the end of the WAL. Frame each record as <code>[crc32][length][payload]</code>; on replay, stop at the first record whose length runs past EOF or whose checksum doesn't match. Everything before it is intact. The test appends garbage to a WAL written by a process killed with <code>os._exit</code>.</li>
<li><strong>Atomic file creation.</strong> Write an SSTable to <code>name.tmp</code>, flush + fsync, then <code>os.replace</code> to the final name: readers see either nothing or the complete file.</li>
<li><strong>Which files are live?</strong> If you crash after writing a compacted file but before deleting its inputs, the directory holds both. Without care, a key deleted (tombstone dropped in the merged file) can <em>resurrect</em> from an old input. Real engines record the live file set in a <strong>MANIFEST</strong> updated atomically; files not in it are garbage from a crash. (The reference solution does this; the tests don't force it, but you should.)</li>
</ul>
<div class="callout pitfall"><b>Windows note</b>You can't delete a file another handle still has open on Windows. Open SSTables for each read (or keep handles and close them before deleting) so compaction can remove old files.</div>
<div class="callout prod"><b>In production</b>RocksDB is the embedded engine inside Kafka Streams state stores, CockroachDB, TiKV and many feature stores; Cassandra and ScyllaDB are distributed LSM stores. Feature stores that serve ML features at low latency usually sit on one of these. Knowing why writes are cheap and point reads need Bloom filters lets you reason about their latency.</div>
`,
      quiz: [
        { q: "Why can't a delete simply remove the key from the memtable?", options: ["It can", "An older SSTable may still contain the key; without a tombstone, reads would find the old value", "Memtables are immutable", "For performance"], answer: 1, why: "Reads fall through to older files; the tombstone must shadow them until compaction drops it." },
        { q: "Bloom filter with m/n = 9.6 bits per key. Optimal k?", options: ["3", "5", "7", "10"], answer: 2, why: "$k = (m/n)\\ln 2 \\approx 9.6 \\times 0.693 \\approx 6.6 \\to 7$." },
        { q: "After a crash, the last WAL record is half-written. Correct recovery?", options: ["Fail to start", "Replay up to the last valid (length + CRC verified) record and ignore the rest", "Discard the whole WAL", "Replay everything and hope"], answer: 1, why: "The torn tail was never acknowledged; everything before it was." },
        { q: "Compared with B-trees, LSM-trees typically have…", options: ["faster random writes, more read amplification", "faster reads, slower writes", "no compaction cost", "in-place updates"], answer: 0, why: "Sequential, batched writes; reads may check several files." },
      ],
      explain: "Walk through what happens on put, get (including a key that doesn't exist), delete, flush, compaction and recovery after a crash in an LSM-tree store.",
      resources: [
        { title: "Kleppmann — DDIA ch. 3: Storage and Retrieval", url: "https://dataintensive.net/", note: "LSM vs B-tree; read this before coding" },
        { title: "O'Neil et al. — The Log-Structured Merge-Tree", url: "https://www.cs.umb.edu/~poneil/lsmtree.pdf", note: "" },
        { title: "LevelDB implementation notes", url: "https://github.com/google/leveldb/blob/main/doc/impl.md", note: "log format, manifest, compaction" },
        { title: "Kirsch & Mitzenmacher — Less Hashing, Same Performance", url: "https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf", note: "double hashing for Bloom filters" },
      ],
      cards: [
        { f: "Bloom filter optimal k and bits per key for false-positive rate p", b: "$m/n = -\\ln p/(\\ln 2)^2$, $k = (m/n)\\ln 2$ (≈ 9.6 bits, k=7 for 1%)." },
        { f: "LSM write path", b: "WAL append → memtable → flush to SSTable (then truncate WAL)." },
        { f: "When may compaction drop a tombstone?", b: "Only when every older file that could hold the key is part of the merge." },
        { f: "How to make WAL recovery tolerate torn writes", b: "Frame records with length + CRC; stop replay at the first invalid record." },
      ],
    },
    {
      id: "w7l4", title: "Caching, queues and rate limiting", minutes: 80,
      summary: "Cache patterns and their failure modes, asynchronous work with queues and logs, and the four rate-limiting algorithms compared.",
      keypoints: [
        "Cache-aside is the default; invalidation and stampedes are the hard parts.",
        "Single-flight collapses concurrent misses for the same key into one backend call.",
        "Queues decouple producers and consumers; at-least-once delivery requires idempotent consumers.",
        "Token bucket allows bursts up to capacity with a long-run rate; sliding-window log is exact but stores timestamps.",
        "The sliding-window counter approximates with two counters: prev·(1 − elapsed/window) + current.",
      ],
      body: R`
<h2>Caching</h2>
<ul>
<li><strong>Cache-aside</strong> (lazy): read cache; on miss read the DB and populate. Simple; stale until TTL or explicit invalidation.</li>
<li><strong>Write-through</strong>: write cache and DB together. Fresh reads, slower writes.</li>
<li><strong>Write-back</strong>: write cache, flush to DB later. Fast writes, data-loss risk.</li>
</ul>
<p>Eviction: LRU (recency), LFU (frequency), TTL (age). An LRU with TTL is an <code>OrderedDict</code> with <code>move_to_end</code> on access and expiry timestamps (your stretch <code>TTLCache</code>).</p>
<p><strong>Failure modes.</strong> <em>Stampede</em>: a hot key expires and 1,000 concurrent requests all miss and hit the database at once. Fixes: single-flight (one caller computes, the rest wait for its result), early probabilistic refresh, or serving stale while refreshing. <em>Invalidation races</em>: a slow reader can repopulate the cache with an old value after a writer invalidated it; versioned values or delete-after-write with short TTLs limit the damage. <em>Cold start</em>: an empty cache after a deploy can flatten the database: warm it.</p>
<h2>Queues and logs</h2>
<p>A queue decouples a producer from a slow or bursty consumer: sending an email, resizing an image, scoring a batch, retraining a model. Two families:</p>
<ul>
<li><strong>Message queues</strong> (SQS, RabbitMQ): a message is delivered to one consumer and deleted when acknowledged. Visibility timeouts redeliver unacknowledged messages.</li>
<li><strong>Logs</strong> (Kafka, Kinesis): an append-only, partitioned, retained log. Consumers track offsets; many consumer groups can read the same data independently and replay history. Ordering is per partition, so choose the partition key (e.g. user id) to keep related events ordered.</li>
</ul>
<p><strong>Delivery semantics.</strong> At-most-once loses messages; at-least-once may duplicate them; "exactly-once" is at-least-once plus idempotent processing. Make consumers idempotent: an idempotency key per message and a "processed" record (or upserts keyed by event id). The transactional outbox pattern writes the event in the same DB transaction as the state change, so they can't disagree.</p>
<h2>Rate limiting</h2>
<p>Protect services from abuse and overload, and enforce quotas.</p>
<table>
<tr><th>algorithm</th><th>state</th><th>behavior</th></tr>
<tr><td>Fixed window</td><td>count per window</td><td>simple; allows 2× limit across a window boundary</td></tr>
<tr><td>Sliding-window log</td><td>timestamps of recent requests</td><td>exact; memory ∝ limit per key</td></tr>
<tr><td>Sliding-window counter</td><td>two counters</td><td>O(1); approximates with a weighted previous window</td></tr>
<tr><td>Token bucket</td><td>tokens + last refill time</td><td>bursts up to capacity, long-run rate r; the most common in APIs</td></tr>
</table>
<h3>Token bucket</h3>
<p>Capacity $b$, refill rate $r$ tokens/s. On each request, lazily refill: <code>tokens = min(b, tokens + (now - last)*r)</code>; allow if <code>tokens >= cost</code> and subtract. No timers. <code>retry_after = (cost - tokens)/r</code> fills the HTTP <code>Retry-After</code> header. The check-and-update must hold a lock: two threads reading "1 token left" would both pass (the test hammers it from 8 threads and demands exactly 1,000 allowed).</p>
<h3>Sliding windows</h3>
<p>Fixed windows let a client send the full limit at 0.9 s and again at 1.1 s: 2× the limit in 0.2 s. The <strong>log</strong> keeps every allowed timestamp in a deque, evicts those older than the window, allows if fewer than the limit remain: exact. The <strong>counter</strong> (Cloudflare's) keeps only this window's and the previous window's counts and estimates</p>
$$\text{estimate} = \text{prev}\times\Big(1 - \frac{\text{elapsed in current window}}{\text{window}}\Big) + \text{current},$$
<p>assuming the previous window's requests were spread uniformly. At 1.1 s after 10 requests at 0.9 s: $10\times0.9 + 0 = 9 < 10$, so exactly one more passes (tested).</p>
<p><strong>Distributed limits</strong>: with many server replicas, keep counters in Redis (atomic <code>INCR</code> + expiry, or a Lua script for token buckets), or give each replica 1/N of the quota and accept some slack.</p>
<div class="callout prod"><b>In production</b>LLM APIs rate-limit on both requests and tokens per minute: a token bucket whose cost is the request's token count (your <code>allow(cost=...)</code> parameter). Recommenders put a cache in front of expensive per-user computation with TTLs tuned to how fast recommendations should change.</div>
`,
      quiz: [
        { q: "Token bucket, capacity 5, rate 2/s, starts full. 6 requests arrive at t=0. How many pass, and when can the next one pass?", options: ["5; at t=0.5 s", "6; immediately", "2; at t=1 s", "5; at t=1 s"], answer: 0, why: "Burst up to capacity 5; the bucket refills one token per 0.5 s." },
        { q: "Why must consumers of an at-least-once queue be idempotent?", options: ["For speed", "Messages can be delivered more than once (redelivery after timeouts/crashes)", "Queues reorder messages", "To save storage"], answer: 1, why: "Redelivery is how at-least-once avoids losing messages; duplicate processing must be harmless." },
        { q: "Fixed-window limiter, 10 per second. Worst-case requests allowed within any 0.2 s span?", options: ["10", "2", "20", "11"], answer: 2, why: "10 at the end of one window and 10 at the start of the next." },
      ],
      explain: "Compare token bucket, sliding-window log and sliding-window counter rate limiters: what state each keeps, what bursts each allows, and which you'd use for an LLM API priced per token.",
      resources: [
        { title: "Cloudflare — How we built rate limiting capable of scaling to millions of domains", url: "https://blog.cloudflare.com/counting-things-a-lot-of-different-things/", note: "the sliding-window counter" },
        { title: "Stripe — Scaling your API with rate limiters", url: "https://stripe.com/blog/rate-limiters", note: "" },
        { title: "Kleppmann — DDIA ch. 11: Stream processing", url: "https://dataintensive.net/", note: "logs vs queues, exactly-once" },
        { title: "AWS Builders' Library — Avoiding insurmountable queue backlogs; Caching challenges and strategies", url: "https://aws.amazon.com/builders-library/", note: "" },
      ],
      cards: [
        { f: "Token bucket refill rule", b: "tokens = min(capacity, tokens + elapsed × rate), lazily on each request." },
        { f: "Sliding-window counter estimate", b: "prev × (1 − elapsed/window) + current" },
        { f: "Cache stampede fix", b: "Single-flight (one caller recomputes, others wait), early refresh, or serve-stale." },
        { f: "Exactly-once processing in practice", b: "At-least-once delivery + idempotent consumers (idempotency keys)." },
      ],
    },
    {
      id: "w7l5", title: "The HLD framework: a URL shortener end to end", minutes: 70,
      summary: "A repeatable structure for system design interviews, applied in full to a URL shortener, with notes on news feeds and chat.",
      keypoints: [
        "Structure: requirements → estimates → API → data model → high-level design → deep dives → trade-offs.",
        "Clarify functional and non-functional requirements before drawing anything.",
        "ID generation: hashing, a counter with base62, pre-allocated ranges, or Snowflake-style ids.",
        "Read-heavy systems: cache-aside + CDN; write analytics asynchronously through a queue.",
        "Always name what breaks first at 10× and what you chose not to do.",
      ],
      body: R`
<h2>The framework</h2>
<ol>
<li><strong>Requirements</strong> (5 min): functional (what it does) and non-functional (scale, latency, availability, consistency, durability). Ask; don't assume.</li>
<li><strong>Estimates</strong>: QPS (avg and peak), storage, bandwidth, cache size.</li>
<li><strong>API</strong>: endpoints, request/response, errors, idempotency, auth, limits.</li>
<li><strong>Data model</strong>: entities, keys, access patterns, which store and why.</li>
<li><strong>High-level design</strong>: boxes and arrows; walk one read and one write through it.</li>
<li><strong>Deep dives</strong>: the 2–3 hardest parts.</li>
<li><strong>Trade-offs and bottlenecks</strong>: what fails at 10×, what you'd monitor, what you deliberately left out.</li>
</ol>
<h2>Worked example: URL shortener</h2>
<p><strong>Requirements.</strong> Create a short link for a long URL (optionally custom alias, expiry); redirect short → long; basic click analytics. Non-functional: 100M new links/day, 100:1 read:write, redirect p99 &lt; 50 ms, highly available reads, links durable for 5 years.</p>
<p><strong>Estimates.</strong> Writes $10^8/10^5\approx 1{,}200$/s (peak ~5k). Reads ~120k/s (peak ~500k). Storage: $10^8\times 365\times5\approx 1.8\times10^{11}$ links × ~500 B ≈ 90 TB. Key space: base62 with 7 chars gives $62^7\approx 3.5\times10^{12}$, enough.</p>
<p><strong>API.</strong> <code>POST /links {long_url, alias?, ttl?} → 201 {short}</code> (idempotency key header to avoid duplicate links on retries; 409 on alias conflict). <code>GET /{short} → 301/302 Location</code> (302 if you want every click to reach you for analytics; 301 lets browsers cache). Rate-limit creation per API key.</p>
<p><strong>Data model.</strong> <code>links(short PK, long_url, owner, created, expires)</code>: a pure key–value access pattern at huge scale → DynamoDB/Cassandra (partition key = short code). Clicks go to an append-only stream, not the links table.</p>
<p><strong>High-level design.</strong></p>
<pre class="mermaid">flowchart LR
  C[Client] --> CDN[CDN / edge cache]
  CDN --> LB[Load balancer]
  LB --> R[Redirect service]
  LB --> W[Create service]
  R --> RC[(Redis cache)]
  R --> DB[(KV store, sharded by short code)]
  W --> IDS[ID allocator]
  W --> DB
  R --> Q[[Click stream: Kafka/Kinesis]]
  Q --> A[Analytics consumers] --> OLAP[(Analytics store)]</pre>
<p><strong>Deep dive 1: ID generation.</strong> Options: (a) hash the long URL and take 7 base62 chars (collisions need checks; same URL from different users collides on purpose or not); (b) a global counter encoded in base62 (no collisions, but a single counter is a bottleneck and IDs are guessable); (c) <em>range allocation</em>: an ID service hands each create-server a block of 1M counter values; servers allocate locally with no coordination per request (best for this scale); (d) Snowflake-style 64-bit ids (timestamp + machine + sequence), longer codes. Pick (c), and shuffle/encrypt the counter if guessability matters.</p>
<p><strong>Deep dive 2: reads at 500k/s.</strong> Cache-aside in Redis keyed by short code (a small hot set: 20% of daily-created links ≈ 20M × 500 B = 10 GB, fits); CDN caching of redirects for the hottest links; KV store read replicas. Analytics must not slow redirects: emit click events asynchronously to a stream; aggregate downstream.</p>
<p><strong>Trade-offs.</strong> 301 vs 302 (cacheability vs analytics fidelity); eventual consistency for newly created links on read replicas (route a creator's first reads to the primary, or write-through the cache on create); hot links (celebrity tweets) handled by CDN; expired-link cleanup via TTL on the KV store.</p>
<h2>Two more patterns to know</h2>
<ul>
<li><strong>News feed</strong>: fan-out on write (push new posts into followers' precomputed feeds: fast reads, expensive for celebrities) vs fan-out on read (merge followees' posts at read time). Real systems are hybrid: push for normal users, pull for celebrities, then an ML ranker orders the feed (Week 8).</li>
<li><strong>Chat</strong>: persistent WebSocket connections to gateway servers, a presence service, messages partitioned by conversation id for ordering, delivery receipts with idempotent message ids.</li>
</ul>
<div class="callout prod"><b>In production</b>Your Week 7 deliverable is this document for the URL shortener in your own words, with your own estimates and an AWS mapping. Use the rubric on the project page. It's also the exact format of most ML engineer system design rounds; Week 8 adapts it for ML systems.</div>
`,
      quiz: [
        { q: "Why prefer range-allocated counters over one global counter for ID generation?", options: ["They're shorter", "Servers allocate locally from pre-assigned blocks: no per-request coordination, no single bottleneck", "They avoid base62", "They make IDs random"], answer: 1, why: "Coordination happens once per block (e.g. 1M ids), not per request." },
        { q: "You want click analytics on every redirect. 301 or 302?", options: ["301", "302", "Either", "200 with a meta refresh"], answer: 1, why: "301 is permanent and cached by browsers, so repeat clicks never reach you." },
      ],
      explain: "Present the URL shortener design in under five minutes as you would in an interview: requirements, estimates, API, data model, the diagram, and two deep dives.",
      resources: [
        { title: "Alex Xu — System Design Interview vol. 1, ch. 8 (URL shortener), 11 (news feed), 12 (chat)", url: "https://bytebytego.com/", note: "" },
        { title: "Twitter — Announcing Snowflake (ID generation)", url: "https://blog.x.com/engineering/en_us/a/2010/announcing-snowflake", note: "" },
        { title: "Hello Interview — System design delivery framework", url: "https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery", note: "" },
      ],
      cards: [
        { f: "HLD interview structure", b: "Requirements → estimates → API → data model → high-level design → deep dives → trade-offs." },
        { f: "base62 codes of length 7", b: "62⁷ ≈ 3.5 × 10¹² possible ids." },
        { f: "Fan-out on write vs read", b: "Write: precompute followers' feeds (fast reads, costly for celebrities). Read: merge at read time. Real systems combine both." },
      ],
    },
  ],
  project: {
    title: "sysdesign — distributed-systems primitives + an HLD",
    dir: "sysdesign/, serve/app.py (rate limiting), docs/w7_url_shortener.md",
    pitch: "Build a consistent-hash ring with virtual nodes, a bit-packed Bloom filter, an LSM-tree KV store with a checksummed WAL that survives real process crashes, and four rate limiters wired into your Week 6 server. Then write a full HLD.",
    test: "pytest tests/w7 -q",
    slow: "pytest tests/w7 -q -m stretch",
    overview: R`
<p>The LSM store is the centerpiece. One test runs a separate Python process that writes 300 keys and dies with <code>os._exit(0)</code> (no close, no flush): your store must recover everything from the WAL. Another appends garbage to the WAL and expects a clean recovery. A third runs 4,000 random operations against a dict oracle with small memtables, frequent compactions and periodic reopening: the definitive correctness check.</p>
<p>Design your on-disk formats before coding: WAL record framing, SSTable layout (data, index, Bloom filter, footer) and file naming that preserves age order. Sketch them in a comment at the top of <code>lsm.py</code>.</p>
`,
    rubric: R`
<p>Score your <code>docs/w7_url_shortener.md</code> 0–2 on each line (aim for ≥14/18):</p>
<ul>
<li>Requirements: functional and non-functional, with numbers, and explicit out-of-scope items.</li>
<li>Estimates: QPS (avg/peak), storage, cache size, all with visible arithmetic.</li>
<li>API: endpoints, status codes, idempotency, rate limits.</li>
<li>Data model: schema, partition key, store choice justified by access pattern.</li>
<li>Diagram: every box earns its place; one read path and one write path walked through.</li>
<li>Deep dive 1 (ID generation): ≥3 options compared, one chosen with reasons.</li>
<li>Deep dive 2 (scaling reads / caching / hot keys).</li>
<li>Trade-offs: consistency choices, 301 vs 302, what breaks first at 10×, monitoring.</li>
<li>AWS mapping: a concrete service per box and one alternative you rejected.</li>
</ul>
<p>Then paste the doc into the tutor box below and ask for an interviewer-style critique.</p>
`,
    milestones: [
      { id: "w7m1", core: true, title: "Consistent-hash ring with virtual nodes", test: "pytest tests/w7/test_hashring_bloom.py -q -k \"empty or deterministic or node or replica or lookup\"",
        detail: R`<p>Order-independent placement, ~1/N movement only to the new node, balance with 200 vnodes, replica lists, O(log n) lookups.</p>`,
        hints: [
          R`<p>Keep a sorted list of ring positions and a dict position → node. <code>bisect.insort</code> on add.</p>`,
          R`<p>Lookup: <code>i = bisect.bisect(points, h) % len(points)</code>.</p>`,
          R`<p>get_nodes: walk from i, skipping nodes already collected, until you have min(n, number of nodes).</p>`,
        ] },
      { id: "w7m2", core: true, title: "Bloom filter", test: "pytest tests/w7/test_hashring_bloom.py -q -k test_bloom",
        detail: R`<p>Exact m and k formulas, &lt;2% false positives at 1% target, bit-packed, serializable.</p>`,
        hints: [
          R`<p>m = ceil(-n ln p / (ln 2)²); k = max(1, round(m/n · ln 2)).</p>`,
          R`<p>Indices: two 64-bit ints from one sha256 digest; force h2 odd; <code>(h1 + i*h2) % m</code>.</p>`,
          R`<p>Serialize as a small header (m, k via <code>struct.pack</code>) followed by the bytearray.</p>`,
        ] },
      { id: "w7m3", core: true, title: "LSM store: WAL, memtable, CRUD, flush, SSTable reads, scan", test: "pytest tests/w7/test_lsm.py -q -k \"crud or flush or scan or persistence\"",
        detail: R`<p>Unicode keys/values with tabs and newlines, tombstones that shadow older files, range scans merging all levels.</p>`,
        hints: [
          R`<p>WAL record: <code>payload = json.dumps([op, key, value]).encode()</code>; write <code>struct.pack(">II", zlib.crc32(payload), len(payload)) + payload</code>; then <code>file.flush()</code>.</p>`,
          R`<p>SSTable: binary records <code>[key_len u32][val_len i32 (-1 = tombstone)][key][value]</code>, then a JSON sparse index (every 16th key + offset), then the Bloom filter bytes, then a footer with the index and filter offsets. Read the footer on open.</p>`,
          R`<p>Name SSTables with a zero-padded sequence number so sorting names sorts by age; search newest first.</p>`,
        ] },
      { id: "w7m4", core: true, title: "Crash recovery and torn tails", test: "pytest tests/w7/test_lsm.py -q -k \"crash or torn\"",
        detail: R`<p>Replay the WAL on open; stop at the first invalid record; keep working afterwards.</p>`,
        hints: [
          R`<p>Read the whole WAL into memory on open and parse with an offset; check <code>pos + 8 + n &lt;= len(data)</code> and the CRC before applying.</p>`,
          R`<p>Open the WAL for appending (<code>"ab"</code>) after replay. Truncate it only after a flush has made its contents durable in an SSTable.</p>`,
          R`<p>If later writes after a torn tail disappear on the next reopen, you appended after the garbage: rewrite the WAL (or truncate it at the last valid offset) during recovery.</p>`,
        ] },
      { id: "w7m5", core: true, title: "Compaction, Bloom skipping, and the oracle test", test: "pytest tests/w7/test_lsm.py -q -k \"compaction or bloom or oracle\"",
        detail: R`<p>Full compaction to one file dropping tombstones and shadowed values; automatic compaction at the threshold; &lt;5% SSTable reads for absent keys.</p>`,
        hints: [
          R`<p>Merge oldest to newest into a dict (or k-way merge with heapq), drop tombstones, write one new SSTable, then delete the inputs.</p>`,
          R`<p>Count a read in <code>stats["sstable_reads"]</code> only after the Bloom filter says "maybe".</p>`,
          R`<p>For crash safety add a MANIFEST (JSON list of live files) replaced atomically after each flush/compaction; ignore and delete files not listed on open.</p>`,
        ] },
      { id: "w7m6", core: true, title: "Rate limiters and server integration", test: "pytest tests/w7/test_ratelimit.py -q",
        detail: R`<p>Token bucket with retry_after, exact sliding log, sliding counter, LRU-bounded keyed limiter, and 429 + Retry-After in your FastAPI app (not on /health or /metrics).</p>`,
        hints: [
          R`<p>Every limiter: <code>with self.lock:</code> around read-modify-write; use the injected clock, never time.monotonic directly.</p>`,
          R`<p>Sliding counter: window index <code>floor(now / window)</code>; on change, prev = current if the new index is exactly old+1, else 0.</p>`,
          R`<p>In the app's HTTP middleware: compute the key from <code>X-API-Key</code> or <code>request.client.host</code>; return a JSONResponse 429 with <code>headers={"Retry-After": "1"}</code> before calling the route.</p>`,
        ] },
      { id: "w7m7", core: true, title: "HLD document", test: "pytest tests/w7/test_ratelimit.py -q -k hld",
        detail: R`<p>Write <code>docs/w7_url_shortener.md</code> from <code>docs/templates/hld.md</code>: ≥800 words with all sections. Score yourself with the rubric below.</p>`,
        hints: [R`<p>Timebox: 90 minutes. Draw the diagram first (Mermaid renders on GitHub), then write estimates; deep dives last.</p>`] },
      { id: "w7m8", core: false, title: "Stretch: TTLCache and SingleFlight; leveled compaction",
        detail: R`<p>Implement the caching primitives (<code>pytest tests/w7 -m stretch</code>) and put a TTLCache + SingleFlight in front of the server's generate path for identical greedy requests. Optionally implement two-level leveled compaction and measure write amplification.</p>`,
        hints: [R`<p>SingleFlight: a dict key → {event, result, error} under a lock; the first caller computes, others wait on the event; remove the entry when done.</p>`] },
    ],
  },
});
