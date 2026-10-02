COURSE.weeks.push({
  n: 4, id: "w4", title: "Deep Learning II",
  tagline: "From embeddings and tokenizers to attention, the transformer block, and efficient LLM training and inference. You train a GPT on your RTX 3060.",
  hours: { learn: 6.5, project: 7 },
  goals: [
    "Implement byte-level BPE and explain the vocabulary-size vs sequence-length trade-off",
    "Derive scaled dot-product attention, including the √d and the causal mask, and count a GPT's parameters by hand",
    "Explain bf16 vs fp16, why the KV cache turns generation from O(T²) to O(T) per token, and its memory cost",
    "Train a character-level GPT on Shakespeare to validation loss < 1.60 in about 6 minutes on your GPU",
  ],
  schedule: [
    ["Mon", "L1 PyTorch internals, GPUs and mixed precision"],
    ["Tue", "L2 Sequences, embeddings and tokenization"],
    ["Wed", "L3 Attention and the transformer block (the core lesson)"],
    ["Thu", "L4 Training and inference for LLMs: schedules, KV cache, sampling"],
    ["Fri", "Flashcards; derive the param-count formula and the KV-cache memory formula"],
    ["Sat", "Milestones 1–3: tokenizers, sampling, GPT model"],
    ["Sun", "Milestones 4–6: KV cache, training loop, the 6-minute Shakespeare run"],
  ],
  lessons: [
    {
      id: "w4l1", title: "PyTorch internals, GPUs and mixed precision", minutes: 70,
      summary: "What a torch.Tensor really is, how your forge maps onto PyTorch, how GPU execution and memory work, and why bf16 is the default for training.",
      keypoints: [
        "A tensor is a view: storage + shape + strides + offset. transpose/view are free; .contiguous() copies.",
        "CUDA kernels launch asynchronously; time with torch.cuda.synchronize() or events.",
        "Training memory = weights + grads + optimizer state (AdamW: 2 extra copies) + activations.",
        "bf16 keeps fp32's 8-bit exponent (range) with fewer mantissa bits; fp16 needs loss scaling.",
        "Training compute ≈ 6 × parameters × tokens FLOPs.",
      ],
      body: R`
<p>This week you switch from your numpy engine to PyTorch, which you now understand from the inside: <code>requires_grad</code>, the graph, <code>.backward()</code>, <code>.grad</code> accumulation, <code>no_grad</code>, <code>zero_grad</code>, <code>state_dict</code>, <code>Module.__setattr__</code> registration. All the same ideas, with a C++ engine and GPU kernels behind them.</p>
<h2>Tensors are views</h2>
<p>A <code>torch.Tensor</code> (like a numpy array) is a <strong>storage</strong> (flat memory) plus metadata: <strong>shape</strong>, <strong>strides</strong> (how many elements to skip per index step in each dim) and an offset. A $(3,4)$ row-major tensor has strides $(4,1)$. <code>.T</code> just swaps them to $(1,4)$: no data moves. <code>view</code>, slicing and <code>transpose</code> are free; operations that need a particular layout call <code>.contiguous()</code> and copy. In attention you'll write <code>y.transpose(1, 2).contiguous().view(B, T, C)</code>: <code>view</code> requires contiguous memory, the transpose made it non-contiguous, so you copy once explicitly.</p>
<div class="callout pitfall"><b>In-place ops and autograd</b>Modifying a tensor in place that autograd saved for the backward pass raises "a variable needed for gradient computation has been modified by an inplace operation". Your forge had the same hazard silently (optimizers update <code>p.data</code> in place, which is safe only because it happens after backward).</div>

<h2>How the GPU runs your code</h2>
<ul>
<li><strong>Asynchronous launches.</strong> Python enqueues kernels and returns immediately; the GPU runs them in order. A <code>.item()</code>, <code>.cpu()</code> or <code>print(tensor)</code> forces a sync. So timing with <code>time.time()</code> around GPU code measures launch time unless you <code>torch.cuda.synchronize()</code> first. Calling <code>loss.item()</code> every step also costs a sync; log every N steps.</li>
<li><strong>Bandwidth vs compute.</strong> Your RTX 3060 peaks around 13 TFLOP/s in fp32 and roughly 25 TFLOP/s for bf16 tensor-core matmuls (fp32 accumulation), but has only ~360 GB/s of memory bandwidth. Elementwise ops (GELU, adds, LayerNorm) are memory-bound: they move bytes, not FLOPs. That's why fused kernels and <code>torch.compile</code> help: fewer round-trips to memory.</li>
<li><strong>Memory budget</strong> for training a model with $P$ parameters in fp32 with AdamW: weights $4P$ + grads $4P$ + Adam's $m,v$ $8P$ = $16P$ bytes, plus <em>activations</em>, which scale with batch × sequence × layers × width and usually dominate. Your 10.8M-param GPT needs ~170 MB of state; activations at batch 64 × 256 tokens are several hundred MB more. 12 GB is plenty.</li>
</ul>
<h2>Mixed precision</h2>
<table>
<tr><th>format</th><th>exponent bits</th><th>mantissa bits</th><th>max</th><th>notes</th></tr>
<tr><td>fp32</td><td>8</td><td>23</td><td>3e38</td><td>baseline</td></tr>
<tr><td>fp16</td><td>5</td><td>10</td><td>65504</td><td>small gradients underflow → needs loss scaling</td></tr>
<tr><td>bf16</td><td>8</td><td>7</td><td>3e38</td><td>same range as fp32, less precision; no scaling needed (Ampere+)</td></tr>
</table>
<p><code>torch.autocast("cuda", dtype=torch.bfloat16)</code> runs matmuls in bf16 on tensor cores while keeping numerically sensitive ops (softmax, layer norm, losses) and the master weights in fp32. Your 3060 (Ampere) supports bf16, so you never need fp16's <code>GradScaler</code>. Expect 2–3× speedups on matmul-heavy models.</p>
<h2>Back-of-envelope compute</h2>
<p>A forward pass through a transformer costs about $2P$ FLOPs per token (each parameter does one multiply-add), backward about $4P$: <mark>training costs ≈ $6PD$ FLOPs for $D$ tokens</mark>. Your run: $P\approx 10.8$M, $D = 2500 \text{ steps}\times 64\times256\approx 41$M tokens ⇒ $6\times10.8\text{M}\times 41\text{M}\approx 2.7\times10^{15}$ FLOPs. At an effective ~8 TFLOP/s that's ~6 minutes, which is what the reference run took. (Attention's $O(T^2)$ term adds a bit on top.) Being able to do this estimate before launching a job is a senior-engineer skill.</p>
<div class="callout prod"><b>In production</b>The same accounting sizes real clusters: GPU-hours ≈ 6PD / (GPUs × achieved FLOP/s). "MFU" (model FLOP utilization, achieved/peak) of 40–50% is considered good for large training runs.</div>
`,
      quiz: [
        { q: "Why does <code>y.transpose(1,2).view(B,T,C)</code> fail but <code>.contiguous().view(...)</code> work?", options: ["transpose copies data incorrectly", "view needs memory laid out to match the new shape; the transposed tensor isn't contiguous", "view only works on CPU", "B, T, C are wrong"], answer: 1, why: "transpose only swaps strides; view can't reinterpret non-contiguous memory. contiguous() copies into the right layout (or use reshape, which copies if needed)." },
        { q: "Why is bf16 safer than fp16 for training without loss scaling?", options: ["More mantissa bits", "Same 8-bit exponent as fp32, so small gradients don't underflow", "It's faster", "It uses less memory than fp16"], answer: 1, why: "Range comes from exponent bits. fp16's max is 65504 and its smallest normal ≈ 6e-5, so tiny gradients flush to zero without scaling." },
        { q: "Approximate training FLOPs for a 100M-parameter model on 2B tokens?", options: ["$2\\times10^{17}$", "$1.2\\times10^{18}$", "$6\\times10^{15}$", "$1.2\\times10^{21}$"], answer: 1, why: "$6PD = 6\\times10^8\\times2\\times10^9 = 1.2\\times10^{18}$." },
      ],
      explain: "Explain what tensor strides are and why transpose is free but view sometimes fails; then estimate how long your Shakespeare GPT should take to train using 6PD.",
      resources: [
        { title: "Edward Yang — PyTorch internals", url: "http://blog.ezyang.com/2019/05/pytorch-internals/", note: "strides, storage, dispatch" },
        { title: "PyTorch — Automatic Mixed Precision recipe", url: "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html", note: "" },
        { title: "Horace He — Making Deep Learning Go Brrrr From First Principles", url: "https://horace.io/brrr_intro.html", note: "compute vs memory bound" },
        { title: "Kaplan et al. — Scaling Laws (Appendix: 6N FLOPs)", url: "https://arxiv.org/abs/2001.08361", note: "" },
      ],
      cards: [
        { f: "Training FLOPs rule of thumb", b: "≈ 6 × parameters × tokens" },
        { f: "bf16 vs fp16 bit layout", b: "bf16: 8 exponent, 7 mantissa. fp16: 5 exponent, 10 mantissa." },
        { f: "AdamW training memory for P params in fp32 (excluding activations)", b: "≈ 16P bytes (weights, grads, m, v)." },
        { f: "Why must you synchronize before timing CUDA code?", b: "Kernel launches are asynchronous; without a sync you time the launch, not the work." },
      ],
    },
    {
      id: "w4l2", title: "Sequences, embeddings and tokenization", minutes: 75,
      summary: "Language modeling as next-token prediction, embeddings as learned lookups, why RNNs struggled, and byte-level BPE in detail.",
      keypoints: [
        "Autoregressive factorization: $p(x_{1:T}) = \\prod_t p(x_t\\mid x_{\\lt t})$; train with next-token cross-entropy.",
        "An embedding lookup equals a one-hot vector times a weight matrix; its gradient is a scatter-add into rows.",
        "RNNs pass information through a T-step chain: vanishing gradients and no parallelism over time.",
        "BPE greedily merges the most frequent adjacent pair; encode applies merges in learned order.",
        "Byte-level BPE never produces unknown tokens; bigger vocab = shorter sequences but bigger embedding/softmax.",
      ],
      body: R`
<h2>Language modeling</h2>
<p>Any distribution over sequences factorizes exactly by the chain rule of probability:</p>
$$p(x_1,\dots,x_T) = \prod_{t=1}^T p(x_t\mid x_1,\dots,x_{t-1}).$$
<p>So a model that predicts the next token given the prefix <em>is</em> a full generative model. Training: maximize likelihood, i.e. minimize the average cross-entropy of each next token. One forward pass over a window of $T$ tokens produces $T$ predictions at once (position $t$ predicts token $t+1$), which is why <code>get_batch</code> returns <code>y</code> = <code>x</code> shifted left by one and why causal masking is essential: position $t$ must not see token $t+1$.</p>
<p>Baseline to beat: a bigram model (a $V\times V$ table of counts) gets about 2.5 nats on Shakespeare characters. A uniform guess over 65 characters is $\ln 65 = 4.17$. Your GPT will reach ~1.5.</p>

<h2>Embeddings</h2>
<p>A token id $i$ becomes a vector $E[i]\in\mathbb{R}^C$ by table lookup. Mathematically that's $\mathrm{onehot}(i)^\top E$, a linear layer on a one-hot input, so its gradient is $\mathrm{onehot}(i)\,\bar e^\top$: only the used rows get gradient, accumulated with scatter-add when a token repeats (exactly your Week 1 <code>getitem</code> backward with <code>np.add.at</code>). <strong>Weight tying</strong> reuses $E$ as the output projection: logits $= hE^\top$. The model scores a token by the dot product between its hidden state and the token's embedding. It saves $VC$ parameters and usually improves quality; your GPT does it.</p>

<h2>Recurrent networks, briefly</h2>
<p>An RNN keeps a state $h_t = \tanh(W_hh_{t-1} + W_xx_t)$. Backprop through time (BPTT) unrolls this into a $T$-layer deep network with shared weights, so Week 3's vanishing/exploding analysis applies along time: the gradient from step $t$ to step $t-k$ contains $\prod W_h^\top D$, so dependencies more than a few dozen steps back are hard to learn. LSTMs add a gated cell state $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$: an additive, residual-like path (Jacobian ≈ $f_t$, which can be ≈ 1) that carries information much further. Two structural limits remain: the computation is sequential in $t$ (no parallelism across the sequence on a GPU), and information between distant tokens must squeeze through every intermediate state (path length $O(T)$). Attention fixes both: path length 1 between any two positions, and all positions computed in parallel.</p>

<h2>Tokenization</h2>
<p>Characters give tiny vocabularies but long sequences (attention cost grows with $T^2$) and make the model learn spelling. Words give short sequences but huge vocabularies and unknown words. Subwords sit in between. <strong>Byte-pair encoding</strong> (Sennrich 2016; GPT-2's byte-level version):</p>
<ol>
<li>Start from bytes: vocabulary = 256 tokens, text = its UTF-8 bytes. Any string is representable, so there's never an "unknown token" (emoji and Korean included).</li>
<li>Count all adjacent pairs; merge the most frequent pair into a new token id (256, 257, …); repeat until the vocabulary size is reached.</li>
<li>To <strong>encode</strong> new text: start from bytes and apply merges in the order they were learned (lowest merge id first among the pairs present), until no learned pair remains.</li>
<li>To <strong>decode</strong>: concatenate each token's bytes and UTF-8-decode with <code>errors="replace"</code> (a token sequence can end mid-character).</li>
</ol>
<p>Worked example (it's in your tests): "aaabdaaabac". Pair counts: aa=4, ab=2, … → merge (a,a)→256: [256,a,b,d,256,a,b,a,c]. Now (256,a)=2 and (a,b)=2 tie. The spec breaks ties by <em>earliest first occurrence</em>: (256,a) appears first → 257 = "aaa". Then (257,b)→258 = "aaab". Encoding gives [258, d, 258, a, c]. (In Python, a dict preserves insertion order and <code>max</code> returns the first maximal key, which implements this tie-break for free.)</p>
<div class="callout pitfall"><b>Encoding order matters</b>Encoding by "merge the most frequent pair in this new text" gives different, inconsistent tokenizations. Merges must replay in learned order; the test <code>test_bpe_encode_uses_merge_order_not_frequency</code> catches it.</div>
<p>GPT-2 also pre-splits text with a regex (words, numbers, punctuation, whitespace) so merges never cross word boundaries: no token for "dog." vs "dog!" variants. That's your stretch goal. Production tokenizers (tiktoken, SentencePiece) are this algorithm with fast data structures: a naive implementation re-scans the whole sequence per merge ($O(\text{merges}\times n)$), fine for 100k characters, painful for gigabytes.</p>
<div class="callout"><b>The trade-off</b>Vocab size $V$ costs $2VC$ parameters if untied (embedding + head) and a $V$-way softmax per position. A larger $V$ makes sequences shorter (more text per context window, fewer attention FLOPs). GPT-2 uses 50,257; Llama 3 uses 128k. Your char model uses 65; a BPE-512 model on Shakespeare compresses text about 2× relative to bytes.</div>
<div class="callout prod"><b>In production</b>Tokenization bugs are real incidents: mismatched tokenizer versions between training and serving, whitespace normalization differences, numbers split oddly (why LLMs are bad at arithmetic), and multilingual text costing 2–4× more tokens (and money) per word. Always version the tokenizer with the model, which is why your checkpoint stores both.</div>
`,
      quiz: [
        { q: "BPE has learned merges (a,b)→256 then (256,256)→257. How is 'abab' encoded?", options: ["[97,98,97,98]", "[256,256]", "[257]", "[256,97,98]"], answer: 2, why: "Apply merges in learned order: first ab→256 gives [256,256], then (256,256)→257." },
        { q: "Why can byte-level BPE encode any string?", options: ["It has a huge vocabulary", "Its base vocabulary is all 256 bytes, and any UTF-8 text is a byte sequence", "It falls back to an UNK token", "It uses characters"], answer: 1, why: "Worst case, a string is encoded as raw bytes." },
        { q: "Why do RNNs train slower than transformers on GPUs for the same data?", options: ["More parameters", "Their computation is sequential across time steps, so positions can't be processed in parallel", "They need bigger batches", "They can't use cross-entropy"], answer: 1, why: "$h_t$ depends on $h_{t-1}$; a transformer computes all positions of a training sequence at once." },
        { q: "The gradient of an embedding table after one batch is nonzero…", options: ["everywhere", "only in rows of tokens that appeared in the batch", "only in the first row", "nowhere, embeddings are frozen"], answer: 1, why: "A lookup is a one-hot matmul: only selected rows receive gradient (scatter-added for repeats)." },
      ],
      explain: "Walk through training BPE on 'aaabdaaabac' to three merges and encoding it, including the tie-break, and explain why merges must be applied in learned order.",
      resources: [
        { title: "Karpathy — Let's build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE", note: "minbpe; same spec as your tests" },
        { title: "Sennrich et al. — Neural Machine Translation of Rare Words with Subword Units", url: "https://arxiv.org/abs/1508.07909", note: "the BPE paper" },
        { title: "Olah — Understanding LSTM Networks", url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", note: "" },
        { title: "Karpathy — makemore part 1 (bigram LM)", url: "https://www.youtube.com/watch?v=PaCmpygFfXo", note: "the baseline" },
      ],
      cards: [
        { f: "Autoregressive factorization", b: "$p(x_{1:T}) = \\prod_t p(x_t\\mid x_{\\lt t})$" },
        { f: "BPE tie-break used in this course", b: "Among equally frequent pairs, pick the one whose first occurrence is earliest." },
        { f: "BPE encode rule", b: "Repeatedly apply the present pair with the LOWEST merge id (learned order)." },
        { f: "Weight tying in a language model", b: "Output projection uses the token embedding matrix: logits = h Eᵀ." },
      ],
    },
    {
      id: "w4l3", title: "Attention and the transformer block", minutes: 100,
      summary: "Scaled dot-product attention from first principles, the causal mask, multi-head attention, positional information, and the full GPT block with its parameter count.",
      keypoints: [
        "Attention: $\\mathrm{softmax}(QK^\\top/\\sqrt{d})V$, a data-dependent weighted average of values.",
        "Divide by $\\sqrt d$ because $q\\cdot k$ has variance $d$; without it softmax saturates.",
        "Causal mask: set scores for $j > i$ to $-\\infty$ before softmax.",
        "Multi-head: split $C$ into $h$ heads of size $C/h$; same FLOPs, multiple attention patterns.",
        "Block (pre-LN): $x + \\mathrm{attn}(\\mathrm{LN}(x))$, then $x + \\mathrm{MLP}(\\mathrm{LN}(x))$; per-block params ≈ $12C^2$.",
        "Attention costs $O(T^2C)$; the MLP costs $O(TC^2)$.",
      ],
      body: R`
<h2>Attention from first principles</h2>
<p>Each position should gather information from other positions, but which ones depends on the content. So let every position emit three vectors via learned linear maps: a <strong>query</strong> $q_i$ (what am I looking for?), a <strong>key</strong> $k_j$ (what do I contain?), a <strong>value</strong> $v_j$ (what do I pass on?). Position $i$'s output is a weighted average of values, weighted by query–key match:</p>
$$\alpha_{ij} = \mathrm{softmax}_j\Big(\frac{q_i^\top k_j}{\sqrt d}\Big),\qquad o_i = \sum_j\alpha_{ij}v_j\qquad\Longleftrightarrow\qquad O = \mathrm{softmax}\Big(\frac{QK^\top}{\sqrt d}\Big)V.$$
<p>With $X\in\mathbb{R}^{T\times C}$: $Q = XW_Q$, $K = XW_K$, $V=XW_V$. Your <code>c_attn</code> is one Linear$(C, 3C)$ that computes all three at once, then splits.</p>
<h3>Why $\sqrt d$</h3>
<p>If components of $q$ and $k$ are roughly independent with unit variance, $q^\top k = \sum_{m=1}^d q_mk_m$ has variance $d$ (Week 3's variance argument). With $d=64$, scores have std 8: softmax of such spread-out numbers is nearly one-hot, and the softmax Jacobian $\mathrm{diag}(p)-pp^\top$ is then ≈ 0, so gradients vanish. Dividing by $\sqrt d$ restores unit variance. Toggle it in the widget at $d=256$ and watch the rows saturate.</p>
<div class="widget" data-widget="attention"></div>
<h3>The causal mask</h3>
<p>For language modeling, position $i$ may only attend to $j\le i$. Set the scores for $j>i$ to $-\infty$ before the softmax; $e^{-\infty}=0$, so those weights vanish exactly and no information (or gradient) flows from the future. The test <code>test_causal_mask_no_future_leak</code> changes tokens 20–31 and checks that logits at positions 0–19 are bit-identical. With a KV cache, queries are positions $S-T..S-1$ and keys are $0..S-1$: build the mask from absolute positions (key position > query position), not from a fixed lower-triangular $T\times T$ matrix.</p>
<h3>Multi-head attention</h3>
<p>One head computes one attention pattern. Split $C$ into $h$ heads of size $d = C/h$: reshape $Q,K,V$ from $(B,T,C)$ to $(B,h,T,d)$, attend per head (batched matmul $(B,h,T,d)@(B,h,d,T)$), concatenate back to $(B,T,C)$, and apply the output projection <code>c_proj</code>. Same parameters and FLOPs as one big head, but different heads can track different relations (previous token, matching bracket, subject of the verb).</p>
<h3>Positional information</h3>
<p>Attention is permutation-equivariant: shuffle the inputs and the outputs shuffle the same way. Order must be injected:</p>
<ul>
<li><strong>Learned absolute</strong> (GPT-2, yours): add a learned vector $P[t]$ to the token embedding. Simple; can't extrapolate past <code>block_size</code>.</li>
<li><strong>Sinusoidal</strong> (original transformer): fixed $\sin/\cos$ at geometric frequencies.</li>
<li><strong>RoPE</strong> (Llama, stretch goal): rotate each 2-D pair of $q$ and $k$ components by an angle proportional to position. Then $q_i^\top k_j$ depends only on $i-j$: relative position for free, and it plays well with KV caches.</li>
</ul>

<h2>The transformer block</h2>
<pre><code>x = x + attn(ln_1(x))      # communicate between positions
x = x + mlp(ln_2(x))       # compute per position: Linear(C,4C) -> GELU -> Linear(4C,C)</code></pre>
<p>Attention mixes information across positions; the MLP processes each position independently. Residual connections (Week 3) keep gradients flowing through dozens of blocks; pre-LN keeps the residual stream unnormalized and training stable. A final LayerNorm and the tied output projection give logits.</p>
<h3>Counting parameters</h3>
<p>Per block, with biases: two LayerNorms $2\cdot 2C$; <code>c_attn</code> $3C^2+3C$; attention <code>c_proj</code> $C^2 + C$; MLP $4C^2+4C$ and $4C^2+C$. Total $12C^2 + 13C$. Plus embeddings $VC + TC$ and the final LayerNorm $2C$; the tied head adds nothing:</p>
$$P = VC + T_{max}C + L(12C^2+13C) + 2C.$$
<p>For your Shakespeare model ($V=65, T=256, L=6, C=384$): 10,770,816. GPT-2 small ($V=50257, T=1024, L=12, C=768$): 124,439,808, the famous "124M". The test checks both, so derive it before you code.</p>
<h3>Where the compute goes</h3>
<p>Per layer: QKV/projection/MLP matmuls cost $\approx 24TC^2$ FLOPs (forward), attention scores and weighted sum $\approx 4T^2C$. For $T < 6C$ the MLP and projections dominate; for long contexts the $T^2$ term takes over, which motivates FlashAttention (same math, tiled to avoid materializing the $T\times T$ matrix in slow memory) and the whole long-context literature.</p>
<div class="callout pitfall"><b>Rules for your implementation</b>You must write the attention yourself: no <code>F.scaled_dot_product_attention</code>, no <code>nn.MultiheadAttention</code> (a test greps your source). Use <code>masked_fill(mask, float("-inf"))</code> and <code>F.softmax</code>. Initialize with std 0.02 and scale every <code>c_proj</code> by $1/\sqrt{2L}$ (each block adds two contributions to the residual stream, so their variances must shrink for the sum to stay ~unit variance: the Week 3 variance argument again).</div>
<div class="callout prod"><b>In production</b>Everything about LLM serving cost follows from this lesson: parameters set memory and per-token FLOPs, the $T^2$ term sets the cost of long prompts, and the KV cache (next lesson) sets how many concurrent users fit on a GPU.</div>
`,
      quiz: [
        { q: "Why scale attention scores by $1/\\sqrt d$?", options: ["To make attention sum to 1", "Dot products of d-dim random vectors have variance d; scaling keeps softmax out of saturation", "To reduce memory", "To make it causal"], answer: 1, why: "Unscaled scores grow like √d, pushing softmax toward one-hot with vanishing gradients." },
        { q: "With a KV cache holding 10 positions, you feed 3 new tokens. Query 0 of this call (absolute position 10) may attend to keys…", options: ["0..12", "0..10", "10..12", "only 10"], answer: 1, why: "Causality on absolute positions: key position ≤ query position, so keys 0–10." },
        { q: "Per-block parameters of a GPT block with width C (biases included)?", options: ["$4C^2$", "$8C^2 + 9C$", "$12C^2 + 13C$", "$16C^2$"], answer: 2, why: "Attention $4C^2+4C$, MLP $8C^2+5C$, two LNs $4C$." },
        { q: "Multi-head attention with h heads vs one head of size C: parameters and FLOPs are…", options: ["h times larger", "about the same", "h times smaller", "quadratic in h"], answer: 1, why: "Same projections; heads just split the C dimension." },
      ],
      explain: "Explain scaled dot-product attention to someone who knows MLPs: what Q, K and V are, why we divide by sqrt(d), how the causal mask works, and why multiple heads help.",
      resources: [
        { title: "Karpathy — Let's build GPT: from scratch, in code, spelled out", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "closest match to your project" },
        { title: "Vaswani et al. — Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", note: "" },
        { title: "Anthropic — A Mathematical Framework for Transformer Circuits", url: "https://transformer-circuits.pub/2021/framework/index.html", note: "residual stream view" },
        { title: "Su et al. — RoFormer (RoPE)", url: "https://arxiv.org/abs/2104.09864", note: "stretch goal" },
        { title: "nanoGPT model.py", url: "https://github.com/karpathy/nanoGPT/blob/master/model.py", note: "read AFTER you write yours" },
      ],
      cards: [
        { f: "Scaled dot-product attention", b: "$\\mathrm{softmax}(QK^\\top/\\sqrt d)\\,V$" },
        { f: "GPT parameter count", b: "$VC + T_{max}C + L(12C^2+13C) + 2C$ (tied head)" },
        { f: "How is causality enforced?", b: "Scores for key position > query position set to $-\\infty$ before softmax." },
        { f: "Why scale residual projections by $1/\\sqrt{2L}$ at init?", b: "The residual stream sums 2L block outputs; scaling keeps its variance ~constant with depth." },
        { f: "Attention vs MLP FLOPs per layer", b: "attention ~$4T^2C$; projections+MLP ~$24TC^2$." },
      ],
    },
    {
      id: "w4l4", title: "Training and inference for LLMs", minutes: 85,
      summary: "The training recipe (AdamW, warmup + cosine, clipping, bf16, eval), then inference: the KV cache derived, its memory cost, and sampling strategies.",
      keypoints: [
        "AdamW with weight decay only on matrices/embeddings; warmup then cosine to ~10% of peak; clip grad norm at 1.",
        "Small data overfits quickly: watch val loss, use dropout, keep the best checkpoint.",
        "KV cache: store past keys/values so each new token costs O(T) instead of recomputing O(T²).",
        "KV cache memory = 2 × layers × tokens × C × bytes per value (per sequence).",
        "Temperature rescales logits; top-k keeps the k best; top-p keeps the smallest set with cumulative prob ≥ p.",
      ],
      body: R`
<h2>The training recipe</h2>
<ul>
<li><strong>AdamW</strong>, $\beta=(0.9, 0.99)$ for small models, weight decay 0.1 applied only to parameters with ≥2 dimensions (matrices and embeddings). Decaying biases and LayerNorm gains toward zero has no regularization benefit and hurts. Use two parameter groups.</li>
<li><strong>Schedule</strong>: linear warmup ~100 steps, cosine decay to <code>min_lr</code> ≈ peak/10 (your <code>lr_at</code>, Week 1 L5).</li>
<li><strong>Gradient clipping</strong> at global norm 1.0: <code>torch.nn.utils.clip_grad_norm_</code>, after <code>backward()</code>, before <code>step()</code>.</li>
<li><strong>bf16 autocast</strong> around the forward pass and loss only; the backward runs in matching precision automatically.</li>
<li><strong>Evaluation</strong>: every N steps, switch to <code>model.eval()</code> under <code>torch.no_grad()</code>, average the loss over several random batches from train and val, switch back to <code>train()</code>. Single-batch estimates are too noisy to compare.</li>
</ul>
<p>On Tiny Shakespeare (1.1M characters) the reference run with dropout 0.2 reaches val ≈ 1.46 at step 2000 and then creeps <em>up</em> to 1.50 by 2500 while train loss keeps falling: a 10.8M-parameter model memorizes 1M characters quickly. Production practice: keep the checkpoint with the best val loss (stretch), or use more data. The data-to-parameter ratio is the whole story here: Chinchilla's compute-optimal rule is ~20 tokens per parameter; you have ~0.1.</p>

<h2>Inference: the KV cache</h2>
<p>Generating token $t+1$ needs the hidden state at position $t$, which attends over keys and values of positions $0..t$. Naively you re-run the model over the whole prefix every step: step $t$ costs $O(t)$ token-forwards, $O(T^2)$ total, with all earlier positions recomputed identically every time (causality means their hidden states never change).</p>
<p>So cache them. Per layer, keep $K$ and $V$ for all past positions. To process new tokens: compute their $q,k,v$ only, append $k,v$ to the cache, attend the new queries over all cached keys (with the absolute-position causal mask from L3), and add positional embeddings at the correct offset (the cache must know its length). Each new token now costs one token-forward plus an $O(t)$ attention read.</p>
<p>Your test checks the invariant that defines correctness: feeding the sequence as a 10-token chunk, then 15 single tokens, then a 5-token chunk must give the same logits as one full forward pass (to 1e-4). Get the mask or the position offset wrong and it fails.</p>
<h3>KV cache memory</h3>
$$\text{bytes} = 2\ (\text{K and V})\times L\times T\times C\times\text{bytes per value}\quad\text{per sequence}.$$
<p>For a 7B Llama-2 ($L=32$, $C=4096$, fp16) at $T=4096$: $2\cdot32\cdot4096\cdot4096\cdot2 \approx 2.1$ GB per sequence. That's why serving throughput is limited by KV memory, and why grouped-query attention (fewer K/V heads), paged KV caches (vLLM) and KV quantization exist. Your model: $2\cdot6\cdot256\cdot384\cdot4$ bytes ≈ 4.7 MB per sequence.</p>
<h3>Past the context window</h3>
<p>Learned positions stop at <code>block_size</code>. Without a cache, crop the context to the last <code>block_size</code> tokens each step. With a cache, when it fills up, rebuild it from the last <code>block_size - 1</code> tokens (or fall back to no-cache). Within the window, cached and uncached greedy generation must agree token for token (tested).</p>

<h2>Sampling</h2>
<p>Given logits $z$ for the next token:</p>
<ul>
<li><strong>Greedy</strong>: argmax. Deterministic, often repetitive. (<code>top_k=1</code> in your API.)</li>
<li><strong>Temperature</strong> $\tau$: sample from $\mathrm{softmax}(z/\tau)$. $\tau<1$ sharpens (more conservative), $\tau>1$ flattens (more random), $\tau\to0$ approaches greedy.</li>
<li><strong>Top-k</strong>: keep the $k$ largest logits, set the rest to $-\infty$, renormalize.</li>
<li><strong>Top-p (nucleus)</strong>: sort probabilities descending; keep the smallest prefix whose cumulative probability reaches $p$ (always at least the top-1). Adapts to the model's confidence: few candidates when it's sure, many when it isn't. Precisely: keep token $i$ (in sorted order) iff the cumulative probability <em>before</em> it is $< p$.</li>
</ul>
<p>Order in your <code>sample_next</code>: divide by temperature, apply top-k then top-p, softmax, <code>torch.multinomial</code> with the provided generator (so seeded generation is reproducible, which the tests and the Week 6 server rely on).</p>
<div class="callout prod"><b>In production</b>Serving systems split inference into <em>prefill</em> (process the prompt in one parallel pass, compute-bound) and <em>decode</em> (one token at a time, memory-bandwidth-bound, reading the whole KV cache and all weights per token). Batching many sequences' decode steps together amortizes the weight reads, which is exactly what your Week 6 dynamic batcher does at the request level.</div>
`,
      quiz: [
        { q: "KV cache size for one sequence: L=24, C=2048, T=2048, bf16?", options: ["≈ 100 MB", "≈ 400 MB", "≈ 800 MB", "≈ 4 GB"], answer: 1, why: "$2\\cdot24\\cdot2048\\cdot2048\\cdot2$ bytes ≈ 403 MB." },
        { q: "Probabilities sorted: [0.5, 0.3, 0.15, 0.05]. Which tokens does top-p = 0.8 keep?", options: ["only the first", "first two", "first three", "all four"], answer: 1, why: "Cumulative before each: 0, 0.5, 0.8, 0.95. Keep while the cumulative-before is < 0.8: tokens 1 and 2." },
        { q: "Why exclude LayerNorm weights and biases from weight decay?", options: ["They have no gradients", "Shrinking them toward zero doesn't regularize capacity and can hurt normalization; decay is meant for weight matrices", "PyTorch doesn't allow it", "It makes training slower"], answer: 1, why: "Decay limits the scale of linear maps; gains/biases are 1-D affine parameters where it just distorts the model." },
        { q: "Validation loss bottoms out at step 2000 then rises while train loss keeps falling. Best response?", options: ["Train longer", "Keep the best-val checkpoint; add data or regularization", "Raise the learning rate", "Remove dropout"], answer: 1, why: "Classic overfitting on a small corpus." },
      ],
      explain: "Explain how a KV cache works, why it makes generation O(T) per token instead of O(T^2), what must be true for its logits to match a full forward pass, and how much memory it uses.",
      resources: [
        { title: "Kipply — Transformer Inference Arithmetic", url: "https://kipp.ly/transformer-inference-arithmetic/", note: "KV cache and FLOP math" },
        { title: "Holtzman et al. — The Curious Case of Neural Text Degeneration (nucleus sampling)", url: "https://arxiv.org/abs/1904.09751", note: "" },
        { title: "Hoffmann et al. — Chinchilla scaling laws", url: "https://arxiv.org/abs/2203.15556", note: "" },
        { title: "Kwon et al. — PagedAttention / vLLM", url: "https://arxiv.org/abs/2309.06180", note: "how KV memory is managed in serving" },
      ],
      cards: [
        { f: "KV cache memory per sequence", b: "$2 \\times L \\times T \\times C \\times$ bytes per value" },
        { f: "Top-p keep rule (sorted descending)", b: "Keep token i iff cumulative probability before it < p (top-1 always kept)." },
        { f: "Which parameters get weight decay in GPT training?", b: "Only ≥2-D tensors (weight matrices, embeddings); not biases or LayerNorm." },
        { f: "Prefill vs decode", b: "Prefill: whole prompt in parallel, compute-bound. Decode: one token per step, memory-bandwidth-bound." },
      ],
    },
  ],
  project: {
    title: "forge.gpt — tokenizer, GPT with KV cache, and a real training run",
    dir: "forge/gpt/",
    pitch: "Write byte-level BPE, a GPT-2-architecture decoder with your own attention and a KV cache, sampling filters, and an AMP training loop with checkpoints. Then train on Tiny Shakespeare on your GPU to val loss < 1.60.",
    test: "pytest tests/w4 -q",
    slow: "pytest tests/w4 -m slow -s",
    overview: R`
<p>This week uses PyTorch (CUDA build already installed in your venv: <code>python -c "import torch; print(torch.cuda.is_available())"</code>). Allowed building blocks are listed in the <code>model.py</code> docstring; attention itself must be yours.</p>
<p>The architecture spec is fixed so <code>param_count</code> has one right answer: pre-LN blocks, Linear(C,3C) for QKV, GELU MLP with 4C hidden, final LN, tied head, learned positions, biases everywhere except the head. The reference model is 10,770,816 parameters for the default config.</p>
<p>The slow test trains the default config for 2,500 steps (batch 64 × 256 characters). Reference on this machine: 6 min 7 s, val loss 1.498. After it finishes, sample from it: <code>forge generate --ckpt checkpoints/gpt/ckpt.pt --prompt "ROMEO:"</code> works once Week 5's CLI exists; until then use <code>load_checkpoint</code> in a Python shell.</p>
`,
    milestones: [
      { id: "w4m1", core: true, title: "CharTokenizer and byte-level BPE", test: "pytest tests/w4/test_tokenizer.py -q",
        detail: R`<p>Wikipedia example with the tie-break, merge-order encoding, unicode round-trips, save/load, compression &gt; 1.8× on held-out Shakespeare, training on 100k chars in &lt; 2 min.</p>`,
        hints: [
          R`<p>Count pairs with a plain dict over <code>zip(ids, ids[1:])</code>; <code>max(counts, key=counts.get)</code> returns the first maximal key in insertion order, which is the earliest-occurrence tie-break.</p>`,
          R`<p>Keep <code>merges: dict[(a,b)] -> new_id</code> (insertion-ordered) and <code>vocab: dict[id] -> bytes</code> with <code>vocab[new] = vocab[a] + vocab[b]</code>. Stop early if the best pair occurs only once.</p>`,
          R`<p>Encode: while len ≥ 2, find the pair present in the sequence with the smallest merge id (<code>min(set(zip(...)), key=lambda p: merges.get(p, inf))</code>); stop if it isn't in merges; otherwise merge all its occurrences left to right.</p>`,
        ] },
      { id: "w4m2", core: true, title: "Sampling: filter_logits and sample_next", test: "pytest tests/w4/test_model.py -q -k \"filter or sample_next\"",
        detail: R`<p>Top-k, top-p (exact boundary behavior), temperature, seeded multinomial; never modify the input logits.</p>`,
        hints: [
          R`<p>top-k: <code>kth = torch.topk(logits, k).values[:, -1:]</code>; mask <code>logits &lt; kth</code>.</p>`,
          R`<p>top-p: sort descending, softmax, cumsum; <code>remove = (cum - probs) &gt;= p</code>; force <code>remove[:, 0] = False</code>; scatter the mask back to the original order with the sort indices.</p>`,
          R`<p>Clone first: <code>logits = logits.clone()</code>.</p>`,
        ] },
      { id: "w4m3", core: true, title: "GPT model: attention, blocks, tying, init, param_count", test: "pytest tests/w4/test_model.py -q -k \"forbidden or param_count or tying or forward or causal or backward\"",
        detail: R`<p>Exact parameter count, tied embedding, scaled c_proj init, initial loss ≈ ln V, no future leakage, gradients reach every parameter.</p>`,
        hints: [
          R`<p>Shapes: <code>q, k, v = self.c_attn(x).split(C, dim=2)</code>; each <code>.view(B, T, h, C // h).transpose(1, 2)</code> → (B, h, T, d). Scores <code>q @ k.transpose(-2, -1) / sqrt(d)</code> → (B, h, T, S).</p>`,
          R`<p>Tie with <code>self.lm_head.weight = self.wte.weight</code> AFTER creating both. Initialize by iterating <code>named_parameters()</code>: names ending in <code>c_proj.weight</code> get std 0.02/sqrt(2L), other ≥2-D params std 0.02, Linear biases zero (leave LayerNorm alone).</p>`,
          R`<p>Mask from absolute positions: <code>qpos = arange(S - T, S)[:, None]</code>, <code>kpos = arange(S)[None, :]</code>, <code>scores.masked_fill(kpos &gt; qpos, -inf)</code>. Works for full passes (S = T) and cached steps alike.</p>`,
        ] },
      { id: "w4m4", core: true, title: "KV cache: init_cache, step, generate", test: "pytest tests/w4/test_model.py -q -k \"kv_cache or generate\"",
        detail: R`<p>Chunked + single-token stepping must match the full forward to 1e-4; generation past block_size; reproducible seeded sampling.</p>`,
        hints: [
          R`<p>A cache object with one (k, v) slot per layer and a position counter. Attention appends with <code>torch.cat([cache.k, k], dim=2)</code>.</p>`,
          R`<p>Positional embedding for a step: <code>wpe(arange(cache.pos, cache.pos + t))</code>. Increment <code>cache.pos</code> after the forward.</p>`,
          R`<p>In <code>generate</code> with cache: first call <code>step</code> on the whole prompt (cropped), then one token at a time; when <code>cache.pos == block_size</code>, start a fresh cache from the last <code>block_size - 1</code> tokens.</p>`,
        ] },
      { id: "w4m5", core: true, title: "get_batch, lr_at, save/load checkpoints", test: "pytest tests/w4/test_train.py -q -k \"batch or lr or checkpoint\"",
        detail: R`<p>Seeded batches with shifted targets, the exact warmup + cosine formula, and checkpoints that carry config and tokenizer.</p>`,
        hints: [
          R`<p><code>ix = torch.randint(0, len(data) - block_size, (B,), generator=g)</code>; stack slices for x and the +1-shifted slices for y.</p>`,
          R`<p>Checkpoint dict: <code>{"config": asdict(model.cfg), "state_dict": ..., "tokenizer": {...}, "step": ...}</code>. Create the parent directory first.</p>`,
          R`<p><code>torch.load(path, map_location=device, weights_only=False)</code> because the file contains plain Python dicts beyond tensors.</p>`,
        ] },
      { id: "w4m6", core: true, title: "train() and the Shakespeare run", test: "pytest tests/w4/test_train.py -q -k tiny",
        detail: R`<p>The tiny CPU test must reach val &lt; 1.0 on a repeated sentence and greedy-generate it back. Then run the real thing (≈6 min).</p><div class="cmd"><code>pytest tests/w4 -m slow -s</code></div>`,
        hints: [
          R`<p>Order inside the loop: set lr for this step, get batch, forward under autocast, <code>zero_grad(set_to_none=True)</code>, backward, clip, step.</p>`,
          R`<p>Two AdamW param groups: <code>[p for p in params if p.dim() &gt;= 2]</code> with weight decay, the rest with 0. <code>fused=True</code> on CUDA is a free speedup.</p>`,
          R`<p>If val loss is stuck near 2.5 (bigram level), attention probably isn't working: check the mask direction and that heads are reshaped back correctly before <code>c_proj</code>.</p>`,
        ] },
      { id: "w4m7", core: false, title: "Stretch: RoPE, GPT-2 regex pre-split, BPE model, best-checkpoint saving",
        detail: R`<p>Add <code>pos="rope"</code> to the config (no wpe; rotate q,k by position, which must respect the cache offset). Add regex pre-splitting to BPE and train a BPE-512 model: compare val loss per <em>character</em> (convert from per-token). Save the checkpoint with the best val loss during training.</p>`,
        hints: [R`<p>Loss per character = loss per token × tokens/characters. A BPE model with 2× compression and loss 3.0/token is 1.5/char.</p>`] },
    ],
  },
});
