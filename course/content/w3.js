COURSE.weeks.push({
  n: 3, id: "w3", title: "Deep Learning I",
  tagline: "MLPs, initialization, normalization and CNNs, built as a PyTorch-style library on your Week 1 autograd and trained on MNIST with plain numpy.",
  hours: { learn: 6, project: 7.5 },
  goals: [
    "Write backprop for a whole MLP as matrix recurrences and say where vanishing gradients come from",
    "Derive Xavier and He initialization from a variance argument",
    "Derive the BatchNorm backward pass and explain train vs eval behavior",
    "Implement convolution with im2col and its backward with col2im",
    "Train a CNN to ≥97.8% on MNIST with a framework you wrote, in about a minute on CPU",
  ],
  schedule: [
    ["Mon", "L1 MLPs and backprop in matrix form"],
    ["Tue", "L2 Initialization and activations (play with the widget until the bars stay green)"],
    ["Wed", "L3 Normalization, residuals and regularization"],
    ["Thu", "L4 Convolutions, im2col and training diagnostics"],
    ["Fri", "Flashcards; derive BN backward on paper"],
    ["Sat", "Milestones 1–3: Module system, Linear/losses, Dropout/BN/LN"],
    ["Sun", "Milestones 4–6: Conv2d/MaxPool, DataLoader, MNIST runs (pytest -m slow)"],
  ],
  lessons: [
    {
      id: "w3l1", title: "MLPs and backprop in matrix form", minutes: 70,
      summary: "Stack linear maps and nonlinearities, then derive the layer-by-layer backward recurrence that your autograd computes automatically.",
      keypoints: [
        "Without nonlinearities, depth collapses to one linear map.",
        "Universal approximation: one wide hidden layer can approximate any continuous function; depth makes it efficient.",
        "Backward recurrence: $\\delta_\\ell = (\\delta_{\\ell+1}W_{\\ell+1}^\\top)\\odot\\phi'(Z_\\ell)$ and $\\nabla W_\\ell = H_{\\ell-1}^\\top\\delta_\\ell$.",
        "Gradients at early layers are products of many Jacobians: they vanish or explode geometrically.",
        "A backward pass costs about 2× a forward pass.",
      ],
      body: R`
<h2>The multilayer perceptron</h2>
<p>A layer computes $H_\ell = \phi(Z_\ell)$, $Z_\ell = H_{\ell-1}W_\ell + b_\ell$ (batch-as-rows; with PyTorch's weight layout it's $H_{\ell-1}W_\ell^\top$). Stack $L$ of them and finish with a linear layer producing logits. Why nonlinearity is essential: $W_3W_2W_1$ is a single matrix, so a deep linear network can't represent anything a one-layer one can't. The nonlinearity $\phi$ lets each layer bend space.</p>
<p><strong>Universal approximation.</strong> With one hidden layer and enough units, an MLP can approximate any continuous function on a compact set (Cybenko 1989, Hornik 1991). Intuition with ReLUs: each unit contributes a "hinge", and sums of hinges make any piecewise-linear function. The catch is "enough units": some functions need exponentially many units in one layer but only polynomially many with depth, because deep nets reuse intermediate features (compositionality). That's the real argument for depth.</p>

<h2>Backprop through a whole MLP</h2>
<p>Last week autograd handled this op by op. Write it out once as matrices so you can reason about it. Define $\delta_\ell = \partial L/\partial Z_\ell$ (shape $N\times d_\ell$). For softmax-CE at the top, $\delta_L = (P - Y)/N$. Then, by the matmul and elementwise rules from Week 1 L3:</p>
$$\nabla W_\ell = H_{\ell-1}^\top\delta_\ell,\qquad \nabla b_\ell = \mathbf 1^\top\delta_\ell,\qquad \delta_{\ell-1} = \big(\delta_\ell W_\ell^\top\big)\odot\phi'(Z_{\ell-1}).$$
<p>Every term is something you computed in the forward pass ($H$, $Z$) or a matmul with a weight you already have. That's why the backward pass costs about two matmuls per forward matmul: one for the weight gradient and one to pass $\delta$ down. Training FLOPs ≈ 3× forward FLOPs, a rule you'll use to estimate GPT training time in Week 4.</p>
<h3>Where vanishing and exploding gradients come from</h3>
<p>Unroll the recurrence: $\delta_1 = \delta_L\,(W_L^\top D_{L-1})(W_{L-1}^\top D_{L-2})\cdots$ with $D_\ell = \mathrm{diag}(\phi'(Z_\ell))$. A product of $L$ random matrices scales like $c^L$ for some per-layer gain $c$. If $c<1$, early layers get no signal (vanishing); if $c>1$, they get enormous updates (exploding). Sigmoid makes it worse: $\sigma'\le 0.25$, so even with well-scaled weights each layer multiplies the gradient by at most 1/4. The next two lessons are a toolbox for keeping $c\approx 1$: initialization, non-saturating activations, normalization, residual connections.</p>
<div class="callout"><b>Debug recipe: overfit one batch</b>Before training for real, take 32 examples and train until the loss is ~0. If you can't, the bug is in the model or the optimizer, not the data or hyperparameters. It catches wrong labels, a missing <code>zero_grad</code>, a detached graph, a sign error. Do this every time you build a new model.</div>

<h2>How your library is shaped</h2>
<p>PyTorch's design, which you're copying, separates three things:</p>
<ul>
<li><strong>Tensors with autograd</strong> (your Week 1 work): the math.</li>
<li><strong>Modules</strong>: objects that own parameters and define <code>forward</code>. Assigning a <code>Parameter</code> or <code>Module</code> to an attribute registers it (via <code>__setattr__</code>), so <code>model.parameters()</code> can walk the tree. Week 5's descriptor lesson shows why this works.</li>
<li><strong>Optimizers</strong> that hold references to the same Parameter objects. That's why <code>load_state_dict</code> must copy values <em>into</em> existing arrays instead of replacing the Parameters (a test checks <code>m.weight is w</code> after loading).</li>
</ul>
<pre><code># the training loop you'll write a hundred times
for xb, yb in DataLoader(X, y, batch_size=128, shuffle=True, seed=epoch):
    opt.zero_grad()
    loss = cross_entropy(model(Tensor(xb)), yb)
    loss.backward()
    opt.step()</code></pre>
<div class="callout prod"><b>In production</b>MLPs are everywhere in production systems: the towers of the Week 8 recommender, the feed-forward half of every transformer block (two-thirds of a GPT's parameters), heads on top of embeddings. "An MLP on good features" is the deep-learning equivalent of "logistic regression first".</div>
`,
      quiz: [
        { q: "$Z_\\ell = H_{\\ell-1}W_\\ell$ with $H_{\\ell-1}: N\\times 256$, $W_\\ell: 256\\times 128$. Given $\\delta_\\ell: N\\times 128$, what is $\\nabla W_\\ell$?", options: ["$\\delta_\\ell^\\top H_{\\ell-1}$", "$H_{\\ell-1}^\\top\\delta_\\ell$", "$\\delta_\\ell W_\\ell^\\top$", "$H_{\\ell-1}\\delta_\\ell^\\top$"], answer: 1, why: "$(256\\times N)(N\\times 128) = 256\\times 128$ ✓." },
        { q: "A 20-layer MLP with sigmoid activations trains its last layers but the first layers barely change. Why?", options: ["Learning rate too high", "Each layer multiplies the gradient by $\\sigma' \\le 0.25$ (times weights), so it vanishes geometrically", "Sigmoid is not differentiable", "Too many parameters"], answer: 1, why: "The backward recurrence multiplies by $\\phi'$ at every layer; $0.25^{19}$ is about $10^{-11}$." },
        { q: "Roughly how many FLOPs does one training step cost relative to one forward pass?", options: ["1×", "2×", "3×", "10×"], answer: 2, why: "Forward (1×) + backward (2×: weight grads and input grads)." },
      ],
      explain: "Explain the backward recurrence for an MLP layer (the delta equations) and use it to explain why sigmoid networks suffer from vanishing gradients.",
      resources: [
        { title: "Karpathy — Building makemore part 3/4 (activations, gradients, BatchNorm; backprop ninja)", url: "https://www.youtube.com/@AndrejKarpathy/videos", note: "matches this week almost exactly" },
        { title: "Goodfellow, Bengio, Courville — Deep Learning ch. 6", url: "https://www.deeplearningbook.org/", note: "feedforward networks" },
        { title: "Nielsen — Neural Networks and Deep Learning ch. 2, 4, 5", url: "http://neuralnetworksanddeeplearning.com/", note: "backprop, universality, vanishing gradients" },
      ],
      cards: [
        { f: "MLP backward recurrence for $\\delta_{\\ell-1}$", b: "$(\\delta_\\ell W_\\ell^\\top)\\odot\\phi'(Z_{\\ell-1})$" },
        { f: "Why must nonlinearities exist in an MLP?", b: "Composition of linear maps is linear; depth would add nothing." },
        { f: "First debugging step for a new model", b: "Overfit a single small batch to ~0 loss." },
      ],
    },
    {
      id: "w3l2", title: "Initialization and activations", minutes: 70,
      summary: "Derive the weight scale that keeps signals alive through depth, and see what each activation function does to it.",
      keypoints: [
        "For $y = \\sum_i^n w_ix_i$ with independent zero-mean terms, $\\mathrm{Var}(y) = n\\,\\mathrm{Var}(w)\\,\\mathrm{Var}(x)$.",
        "Xavier/Glorot (tanh, linear): $\\mathrm{Var}(w) = 1/n_{in}$ (or $2/(n_{in}+n_{out})$).",
        "He/Kaiming (ReLU): $\\mathrm{Var}(w) = 2/n_{in}$, because ReLU zeroes half the signal.",
        "Too-small init: activations and gradients vanish. Too large: tanh saturates / ReLU explodes.",
        "ReLU can die (units that never activate); GELU/SiLU are smooth alternatives used in transformers.",
      ],
      body: R`
<h2>The variance argument</h2>
<p>Consider one pre-activation $z = \sum_{i=1}^n w_ix_i$ with weights and inputs independent, zero-mean, each with variances $\mathrm{Var}(w)$ and $\mathrm{Var}(x)$. Since $\mathrm{Var}(w_ix_i) = \mathbb{E}[w^2]\mathbb{E}[x^2] = \mathrm{Var}(w)\mathrm{Var}(x)$ and variances of independent terms add:</p>
$$\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathrm{Var}(x).$$
<p>To keep the signal's scale constant from layer to layer we need $n\,\mathrm{Var}(w) = 1$, i.e. $\mathrm{Var}(w) = 1/n$. With $\mathrm{Var}(w)=1$ (the "obvious" <code>randn</code>), each layer multiplies the variance by $n$: with width 128, after 20 layers that's $128^{20}$. With $0.01^2$, it shrinks by $0.0128$ per layer. Same argument backward: gradients scale by $n_{out}\mathrm{Var}(w)$ per layer.</p>
<ul>
<li><strong>Xavier/Glorot</strong> (2010): for tanh/linear (≈ identity near 0), $\mathrm{Var}(w) = 1/n_{in}$, or the compromise $2/(n_{in}+n_{out})$ to balance forward and backward.</li>
<li><strong>He/Kaiming</strong> (2015): ReLU sets half the inputs to zero, so $\mathbb{E}[\mathrm{relu}(z)^2] = \frac12\mathrm{Var}(z)$. Compensate with a factor 2: $\mathrm{Var}(w) = 2/n_{in}$. Your <code>Linear</code> and <code>Conv2d</code> use this (for conv, $n_{in}$ = in_channels × k × k), and the tests check the std.</li>
</ul>
<div class="widget" data-widget="init"></div>
<p>Try: ReLU with N(0, 0.01²) vanishes, N(0,1) explodes, He stays flat. Switch to tanh with N(0,1): the signal doesn't explode (tanh is bounded) but saturates at ±1 where $\tanh' \approx 0$, so gradients die instead. Then turn on per-layer normalization: everything becomes healthy regardless of init. That's next lesson's tool, and why transformers are so robust.</p>
<p>The same argument explains the $\sqrt{d}$ in attention (Week 4): $q^\top k = \sum_i^d q_ik_i$ has variance $d$ if the components have unit variance, so we divide by $\sqrt d$. And the GPT-2 trick of scaling residual projections by $1/\sqrt{2L}$: the residual stream sums $2L$ contributions, so each must be smaller for the sum to keep unit variance.</p>

<h2>Activation functions</h2>
<table>
<tr><th></th><th>formula</th><th>derivative</th><th>notes</th></tr>
<tr><td>sigmoid</td><td>$1/(1+e^{-x})$</td><td>$s(1-s)\le 1/4$</td><td>saturates both ends; not zero-centered. Use for output probabilities, gates.</td></tr>
<tr><td>tanh</td><td>$\frac{e^x-e^{-x}}{e^x+e^{-x}}$</td><td>$1-t^2$</td><td>zero-centered, still saturates. RNN classics.</td></tr>
<tr><td>ReLU</td><td>$\max(0,x)$</td><td>$\mathbb 1[x>0]$</td><td>no saturation for $x>0$, cheap, sparse. Units can "die" if pushed negative forever.</td></tr>
<tr><td>GELU</td><td>$x\,\Phi(x)$</td><td>smooth</td><td>default in transformers (BERT, GPT). Smooth ReLU weighted by how "positive" $x$ is.</td></tr>
<tr><td>SiLU/Swish</td><td>$x\,\sigma(x)$</td><td>smooth</td><td>Llama's SwiGLU MLP uses it with a gate.</td></tr>
</table>
<div class="callout pitfall"><b>Dead ReLUs</b>A large negative bias or a big gradient step can push a unit's pre-activation below zero for every input. Its gradient is then zero forever: it's dead. Symptoms: a growing fraction of exactly-zero activations. Causes: learning rate too high, bad init. Leaky ReLU or GELU mitigate it.</div>
<div class="callout prod"><b>In production</b>When a new architecture won't train, plot per-layer activation and gradient statistics (mean, std, fraction saturated or zero) for the first few hundred steps, exactly what this widget shows. It's the fastest way to find a bad init, a missing normalization or a residual branch that's too large.</div>
`,
      quiz: [
        { q: "A ReLU layer has fan-in 512. What std should He initialization use?", options: ["$1/512$", "$\\sqrt{1/512} \\approx 0.044$", "$\\sqrt{2/512} = 0.0625$", "$2/512$"], answer: 2, why: "Variance $2/n_{in}$, so std $\\sqrt{2/512} = 1/16$." },
        { q: "Why does ReLU need twice the weight variance of tanh?", options: ["ReLU outputs are larger", "ReLU zeroes half its inputs, halving the second moment", "ReLU is not differentiable at 0", "Tradition"], answer: 1, why: "For symmetric $z$, $\\mathbb{E}[\\mathrm{relu}(z)^2] = \\frac12\\mathbb{E}[z^2]$; doubling Var(w) compensates." },
        { q: "The dot product of two random $d$-dim vectors with unit-variance i.i.d. components has variance…", options: ["1", "$\\sqrt d$", "$d$", "$d^2$"], answer: 2, why: "A sum of $d$ independent unit-variance products. Hence attention divides by $\\sqrt d$." },
      ],
      explain: "Derive why Var(w) = 2/fan_in keeps activations stable through a deep ReLU network, and explain what goes wrong with Var(w) = 1.",
      resources: [
        { title: "He et al. — Delving Deep into Rectifiers (Kaiming init)", url: "https://arxiv.org/abs/1502.01852", note: "section 2.2" },
        { title: "Glorot & Bengio — Understanding the difficulty of training deep feedforward networks", url: "https://proceedings.mlr.press/v9/glorot10a.html", note: "" },
        { title: "deeplearning.ai — Initializing neural networks (interactive)", url: "https://www.deeplearning.ai/ai-notes/initialization/", note: "" },
      ],
      cards: [
        { f: "Variance of $z=\\sum_i^n w_ix_i$ (independent, zero mean)", b: "$n\\,\\mathrm{Var}(w)\\,\\mathrm{Var}(x)$" },
        { f: "He initialization variance", b: "$2/n_{in}$ (for ReLU)" },
        { f: "Xavier initialization variance", b: "$1/n_{in}$ or $2/(n_{in}+n_{out})$" },
        { f: "What is a dead ReLU?", b: "A unit whose pre-activation is negative for all inputs, so its gradient is always zero." },
      ],
    },
    {
      id: "w3l3", title: "Normalization, residuals and regularization", minutes: 85,
      summary: "BatchNorm and LayerNorm (with the backward pass derived), residual connections as gradient highways, and dropout done right.",
      keypoints: [
        "BatchNorm normalizes each feature over the batch, then applies learnable scale γ and shift β.",
        "Train: batch statistics; eval: running averages (momentum 0.1; running var uses the unbiased estimate).",
        "BN backward couples examples: $\\bar x = \\frac{\\gamma}{\\sigma}(\\bar{\\hat x} - \\mathrm{mean}(\\bar{\\hat x}) - \\hat x\\,\\mathrm{mean}(\\bar{\\hat x}\\hat x))$.",
        "LayerNorm normalizes each example over its features: no batch dependence, same in train and eval.",
        "Residual $x + F(x)$ has Jacobian $I + \\partial F/\\partial x$: gradients flow even when F's are small.",
        "Inverted dropout scales kept units by $1/(1-p)$ in training so eval is the identity.",
      ],
      body: R`
<h2>Batch normalization</h2>
<p>For a batch $X\in\mathbb{R}^{N\times C}$, per feature $c$:</p>
$$\mu_c = \frac1N\sum_n x_{nc},\quad \sigma_c^2 = \frac1N\sum_n(x_{nc}-\mu_c)^2,\quad \hat x_{nc} = \frac{x_{nc}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\quad y_{nc} = \gamma_c\hat x_{nc} + \beta_c.$$
<p>Every feature enters the next layer with mean 0 and variance 1 (before $\gamma,\beta$), whatever the weights did. The learnable $\gamma,\beta$ let the network undo it if useful. Effects: much higher usable learning rates, less sensitivity to init, and a mild regularizing noise (each example's output depends on its batch-mates).</p>
<h3>Train vs eval</h3>
<p>At inference you may have one example, so batch statistics are meaningless. During training keep running averages, PyTorch-style:</p>
$$\mu_{run} \leftarrow (1-m)\mu_{run} + m\mu_B,\qquad \sigma^2_{run}\leftarrow(1-m)\sigma^2_{run} + m\,\tfrac{N}{N-1}\sigma_B^2,$$
<p>with $m=0.1$. Note the asymmetry the test checks: normalization uses the <em>biased</em> variance, but the running variance stores the <em>unbiased</em> one. In eval mode, normalize with the running statistics. Forgetting <code>model.eval()</code> at test time is the classic "my accuracy drops at inference" bug; the BN+Dropout MNIST test's error message is a hint for exactly that.</p>
<h3>The backward pass</h3>
<p>If you build BN from autograd ops (mean, subtract, multiply, power), your Week 1 engine derives the gradient for free, and that's what the reference does. It's still worth deriving once, because it shows how normalization couples examples. With $\bar{\hat x} = \bar y\,\gamma$ (per feature, dropping the $c$ index), and $\sigma = \sqrt{\sigma^2+\epsilon}$:</p>
$$\bar x_n = \frac{1}{\sigma}\Big(\bar{\hat x}_n - \frac1N\sum_m\bar{\hat x}_m - \hat x_n\frac1N\sum_m\bar{\hat x}_m\hat x_m\Big),\qquad \bar\gamma = \sum_n\bar y_n\hat x_n,\quad\bar\beta = \sum_n\bar y_n.$$
<p>Read it: the gradient is projected to remove its mean and its component along $\hat x$. Normalization makes the output invariant to shifting and scaling the input, so those directions get zero gradient. Karpathy's "backprop ninja" walks through every intermediate; do it once on paper.</p>

<h2>Layer normalization</h2>
<p>LayerNorm normalizes each example over its own features (the last dimension): $\mu,\sigma$ are per row, $\gamma,\beta$ per feature. No dependence on the batch, so train and eval are identical and it works with batch size 1 and variable-length sequences. That's why transformers use it (your GPT next week uses <code>nn.LayerNorm</code>; here you build it on forge). RMSNorm (Llama) drops the mean subtraction: $x/\sqrt{\mathrm{mean}(x^2)+\epsilon}\cdot\gamma$.</p>

<h2>Residual connections</h2>
<p>A residual block computes $y = x + F(x)$. Its Jacobian is $I + \partial F/\partial x$, so the backward pass is $\bar x = \bar y + \bar y\,\partial F/\partial x$: <mark>the gradient reaches earlier layers through the identity path unattenuated</mark>, even if $F$'s Jacobian is tiny. That's how ResNets train with 150+ layers and how every transformer works. It also means each block only has to learn a small correction to the identity, which is easy to initialize (zero-init the last layer of $F$, or scale it down like GPT-2's $1/\sqrt{2L}$).</p>
<p>Pre-norm vs post-norm: GPT-2 and later use $x + F(\mathrm{LN}(x))$ ("pre-LN"), keeping the residual stream itself un-normalized, which trains more stably than the original transformer's $\mathrm{LN}(x + F(x))$.</p>

<h2>Regularization for neural nets</h2>
<ul>
<li><strong>Dropout</strong>: during training zero each activation with probability $p$. <em>Inverted</em> dropout divides the survivors by $1-p$ so the expected activation is unchanged and eval needs no rescaling (identity). Interpretation: training an exponential ensemble of thinned networks that share weights. The test checks the drop fraction, the $1/(1-p)$ scale and the eval identity.</li>
<li><strong>Weight decay</strong>: L2 / AdamW decay from Week 1.</li>
<li><strong>Early stopping</strong>: stop when validation loss stops improving; a regularizer on training time.</li>
<li><strong>Data augmentation</strong>: random crops, flips, noise: encode invariances you know the task has.</li>
</ul>
<div class="callout prod"><b>In production</b>Normalization layers are a deployment detail too: BN's running stats must be computed on data representative of serving, and BN can be folded into the preceding conv/linear weights at export time for free speed. LayerNorm statistics are computed per request, which is one reason LLM serving is batch-size independent.</div>
`,
      quiz: [
        { q: "Training accuracy is 99%, but in eval mode with batch size 1 predictions are garbage. Most likely cause?", options: ["Overfitting", "BatchNorm still using batch statistics (forgot eval mode / running stats not used)", "Dropout rate too low", "Learning rate too high"], answer: 1, why: "Batch statistics of a single example are degenerate (variance 0). Eval must use running statistics." },
        { q: "With inverted dropout $p=0.2$, a kept activation of 1.0 is output as…", options: ["1.0", "0.8", "1.25", "0.2"], answer: 2, why: "Divide by $1-p = 0.8$ during training; eval is then the identity." },
        { q: "Why is LayerNorm preferred over BatchNorm in transformers?", options: ["It's faster", "It doesn't depend on the batch, so it works with variable lengths, batch size 1 and identical train/eval behavior", "It has more parameters", "BatchNorm can't be backpropagated"], answer: 1, why: "Per-example statistics avoid batch coupling and the train/eval discrepancy." },
        { q: "For $y = x + F(x)$, the gradient $\\bar x$ equals…", options: ["$\\bar y\\,\\partial F/\\partial x$", "$\\bar y + \\bar y\\,\\partial F/\\partial x$", "$\\bar y$", "$\\bar y - \\bar y\\,\\partial F/\\partial x$"], answer: 1, why: "Jacobian $I + \\partial F/\\partial x$: the identity term guarantees gradient flow." },
      ],
      explain: "Explain what BatchNorm does differently in training vs evaluation and why, then explain why residual connections let very deep networks train.",
      resources: [
        { title: "Ioffe & Szegedy — Batch Normalization", url: "https://arxiv.org/abs/1502.03167", note: "Algorithm 1 and 2" },
        { title: "Ba, Kiros, Hinton — Layer Normalization", url: "https://arxiv.org/abs/1607.06450", note: "" },
        { title: "He et al. — Deep Residual Learning", url: "https://arxiv.org/abs/1512.03385", note: "" },
        { title: "Srivastava et al. — Dropout", url: "https://jmlr.org/papers/v15/srivastava14a.html", note: "" },
      ],
      cards: [
        { f: "BatchNorm running variance update (PyTorch)", b: "$(1-m)\\,var_{run} + m\\cdot\\frac{N}{N-1}\\sigma_B^2$ — unbiased in the running stat, biased for normalization." },
        { f: "LayerNorm vs BatchNorm: which axis?", b: "BN: per feature across the batch. LN: per example across features." },
        { f: "Why do residual connections help gradient flow?", b: "Jacobian $I + \\partial F/\\partial x$: the identity path passes gradients unchanged." },
        { f: "Inverted dropout scaling", b: "Multiply kept units by $1/(1-p)$ during training; identity at eval." },
      ],
    },
    {
      id: "w3l4", title: "Convolutions, im2col and training diagnostics", minutes: 90,
      summary: "Convolution as a sparse, weight-shared linear map; im2col to make it a matmul; col2im for the backward; and how to read a training run.",
      keypoints: [
        "Conv = locally connected + weight sharing ⇒ translation equivariance with few parameters.",
        "Output size $\\lfloor(H + 2p - k)/s\\rfloor + 1$; parameters $C_{out}C_{in}k^2 + C_{out}$.",
        "im2col turns conv into one matmul: patches $(N H_o W_o)\\times(C k^2)$ times weights $(Ck^2)\\times C_{out}$.",
        "Backward: $\\bar W$ = patchesᵀ × grad; $\\bar X$ = col2im(grad × W), which scatter-adds overlapping patches.",
        "Receptive field grows with depth (and stride/pooling): a 3×3 stack of $L$ layers sees $(2L+1)^2$ pixels.",
      ],
      body: R`
<h2>Why convolution</h2>
<p>A dense layer on a 28×28 image connects every pixel to every unit: 784 weights per unit, and it has to learn "edge at position (3,5)" and "edge at (3,6)" separately. Images have structure: nearby pixels matter together (locality) and a cat is a cat wherever it is (translation). A convolution bakes both in:</p>
<ul>
<li><strong>Local connectivity</strong>: each output looks at a $k\times k$ patch.</li>
<li><strong>Weight sharing</strong>: the same kernel slides over every position.</li>
</ul>
<p>The result is <strong>translation equivariance</strong> (shift the input, the feature map shifts) and drastically fewer parameters: a 3×3 conv from 16 to 32 channels has $32\cdot16\cdot9 + 32 = 4{,}640$ parameters regardless of image size.</p>
<p>Deep learning "convolution" is actually cross-correlation (no kernel flip):</p>
$$Y[n,o,i,j] = b_o + \sum_{c}\sum_{u,v} W[o,c,u,v]\;X_{pad}[n,c,\,i s+u,\,j s+v].$$
<p>Output size: $H_{out} = \lfloor (H + 2p - k)/s\rfloor + 1$. "Same" padding for stride 1 is $p = (k-1)/2$.</p>

<h2>im2col: convolution as one matmul</h2>
<p>Loops over output positions in Python are hopeless (the test's naive reference does it, slowly, on tiny inputs). The trick every framework used before cuDNN: unfold each $C\times k\times k$ patch into a row. For $N$ images with $H_oW_o$ positions:</p>
$$\underbrace{\mathrm{cols}}_{(N H_o W_o)\times(C k^2)}\ \times\ \underbrace{W_{mat}^\top}_{(C k^2)\times C_{out}}\ =\ \underbrace{Y}_{(N H_o W_o)\times C_{out}}\ \to\ \text{reshape to } (N, C_{out}, H_o, W_o).$$
<p>Build the patches without copying through Python: <code>np.lib.stride_tricks.sliding_window_view(Xp, (k, k), axis=(2, 3))</code> gives a view of shape $(N, C, H', W', k, k)$; take every $s$-th window with slicing, transpose to $(N, H_o, W_o, C, k, k)$ and reshape (this reshape makes the one real copy).</p>
<h3>The backward pass</h3>
<p>It's the matmul backward from Week 1 plus undoing the unfold. With $G$ the upstream gradient reshaped to $(NH_oW_o)\times C_{out}$:</p>
<ul>
<li>$\bar W_{mat} = G^\top\,\mathrm{cols}$, reshaped to $(C_{out}, C, k, k)$. $\bar b = $ sum of $G$ over $N, H_o, W_o$.</li>
<li>$\overline{\mathrm{cols}} = G\,W_{mat}$, shape $(NH_oW_o)\times(Ck^2)$. Each row holds gradients for one patch. Patches overlap, so <strong>col2im</strong> must scatter-<em>add</em> them back into the padded input shape, then crop the padding.</li>
</ul>
<p>col2im efficiently: loop over the $k^2$ kernel offsets $(u,v)$ only (9 iterations for 3×3), and for each add a strided slab: <code>dXp[:, :, u:u+s*Ho:s, v:v+s*Wo:s] += dcols[u, v]</code>, where <code>dcols</code> has been rearranged to $(k, k, N, C, H_o, W_o)$ and made contiguous. Memory layout matters a lot here: slicing <code>dcols[..., u, v]</code> out of a $(N, H_o, W_o, C, k, k)$ array strides through memory and can cost a several-fold slowdown. Week 5 is where you'll profile it.</p>
<p>Alternative design: build conv from your existing ops (pad + fancy-index gather + reshape + matmul), and the backward comes from autograd. Correct, but <code>np.add.at</code> in the gather's backward is slow. Either passes the W3 speed test (fwd+bwd of a 64×16×28×28 batch in &lt;5 s); Week 5 then asks for the fast version.</p>
<h3>Pooling</h3>
<p>Max-pooling takes the max of each window: a little translation invariance and a cheap downsample. Backward routes the gradient only to the argmax of each window (record it in the forward pass); with ties, send it to exactly one element (the test feeds an all-ones window).</p>
<h3>Receptive field</h3>
<p>Stacking two 3×3 convs gives each output a 5×5 view of the input, with fewer parameters and an extra nonlinearity compared with one 5×5 conv (VGG's insight). Pooling/stride multiplies the growth. Your MNIST CNN: conv3 → pool2 → conv3 → pool2 gives each final feature a 10×10 receptive field.</p>

<h2>Reading a training run</h2>
<ul>
<li><strong>Loss at step 0</strong> should be about $\ln(\text{classes})$ = 2.30 for MNIST. Much higher ⇒ init too large.</li>
<li><strong>Loss flat from the start</strong> ⇒ learning rate far too low, or gradients not flowing (detached graph, dead units). Check gradient norms.</li>
<li><strong>Loss decreases then explodes to NaN</strong> ⇒ learning rate too high; add warmup or clipping.</li>
<li><strong>Train ↓, val ↑</strong> ⇒ overfitting: regularize, augment, stop early.</li>
<li><strong>LR range test</strong>: increase the LR exponentially over a few hundred steps and plot loss vs LR; pick ~1/10 of the LR where loss is lowest (Smith's "LR finder").</li>
</ul>
<div class="callout prod"><b>In production</b>Convolutions remain the workhorse for images, audio spectrograms and on-device models, and the im2col/GEMM view is still how many accelerators run them. ResNets and ConvNeXts are standard backbones; Vision Transformers replace convolution with patch embedding + attention but keep the "patches as rows" idea you just implemented.</div>
`,
      quiz: [
        { q: "Input 28×28, conv k=3, stride 2, padding 1. Output size?", options: ["28", "14", "13", "15"], answer: 1, why: "$\\lfloor(28+2-3)/2\\rfloor + 1 = 13 + 1 = 14$." },
        { q: "Parameters of Conv2d(in=16, out=32, k=3) with bias?", options: ["4,608", "4,640", "144", "14,336"], answer: 1, why: "$32\\cdot16\\cdot9 = 4608$ weights + 32 biases." },
        { q: "Why does col2im need scatter-ADD rather than assignment?", options: ["For numerical stability", "Patches overlap: one input pixel appears in several patches and receives gradient from each", "To handle padding", "Assignment is slower"], answer: 1, why: "Multivariable chain rule again: sum over all uses of each input pixel." },
        { q: "Initial training loss on 10-class MNIST is 9.7. What's the likely problem?", options: ["Learning rate too low", "Logits too large at init (bad weight scale)", "Dataset is unbalanced", "Nothing, it's normal"], answer: 1, why: "A sane init gives ≈ ln 10 = 2.30. Big initial loss means confident wrong predictions: weights too large." },
      ],
      explain: "Explain how im2col turns a convolution into a matrix multiply, and what col2im must do in the backward pass and why it adds.",
      resources: [
        { title: "CS231n — Convolutional Neural Networks notes", url: "https://cs231n.github.io/convolutional-networks/", note: "includes the im2col explanation" },
        { title: "Dumoulin & Visin — A guide to convolution arithmetic", url: "https://arxiv.org/abs/1603.07285", note: "output sizes, transposed conv" },
        { title: "Smith — Cyclical Learning Rates (LR range test)", url: "https://arxiv.org/abs/1506.01186", note: "" },
        { title: "Karpathy — A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/", note: "read this one fully" },
      ],
      cards: [
        { f: "Conv output size", b: "$\\lfloor(H+2p-k)/s\\rfloor + 1$" },
        { f: "im2col matrix shape", b: "$(N H_o W_o)\\times(C_{in}k^2)$" },
        { f: "Conv weight gradient via im2col", b: "$\\bar W_{mat} = G^\\top\\,\\mathrm{cols}$, reshaped to $(C_{out},C_{in},k,k)$." },
        { f: "Expected initial loss for a K-class classifier", b: "$\\ln K$ (2.30 for 10 classes)." },
      ],
    },
  ],
  project: {
    title: "forge.nn — a PyTorch-style library on your autograd",
    dir: "forge/nn/",
    pitch: "Build Module/Parameter registration, Linear, BatchNorm, LayerNorm, Dropout, Conv2d with im2col, MaxPool, losses and a DataLoader. Then train an MLP and a CNN on MNIST with numpy alone.",
    test: "pytest tests/w3 -q",
    slow: "pytest tests/w3 -m slow -s",
    overview: R`
<p>Two test tiers: the fast tests (34) check registration, state dicts, forward values against formulas or naive loops, and gradients with the grader's own finite differences (layers are cast to float64 with <code>.to(np.float64)</code> first). The slow tests train real models on MNIST: MLP 5 epochs ≥97.2%, BN+Dropout MLP 3 epochs ≥97.2%, CNN 3 epochs ≥97.8%. The reference finishes all three in about 80 seconds.</p>
<p>If something in Week 1 was shaky (dtype handling, broadcasting of float32 params with float64 inputs, getitem backward), it will surface here. Fix it in <code>forge/autograd.py</code> and rerun <code>pytest tests/w1</code> too.</p>
`,
    milestones: [
      { id: "w3m1", core: true, title: "Parameter and Module: registration, buffers, train/eval, to(), state_dict", test: "pytest tests/w3/test_module.py -q",
        detail: R`<p>Exact <code>named_parameters()</code> order (own params first, then children in assignment order), dedup of shared parameters, buffers exposed as attributes, strict <code>load_state_dict</code> that copies in place.</p>`,
        hints: [
          R`<p>In <code>__init__</code>, create the three dicts with <code>object.__setattr__</code> (your own <code>__setattr__</code> would recurse because the dicts don't exist yet).</p>`,
          R`<p><code>__setattr__</code>: first remove <code>name</code> from params/modules dicts (re-assignment), then register by type; if <code>name</code> is a buffer, update the buffer dict instead. Always also do <code>object.__setattr__</code> so normal attribute access works.</p>`,
          R`<p>Write one generator <code>_named(prefix, which_dict)</code> that yields own items then recurses into children with <code>prefix + name + "."</code>. Dedup params by <code>id()</code> in <code>named_parameters</code>.</p>`,
        ] },
      { id: "w3m2", core: true, title: "Linear, activations, Flatten, cross_entropy, mse_loss", test: "pytest tests/w3/test_layers.py -q -k \"linear or flatten or cross_entropy or mse\"",
        detail: R`<p>He init, float32 default, 3-D inputs for Linear (batched sequences, needed in Week 4's mental model).</p>`,
        hints: [
          R`<p><code>x @ self.weight.T + self.bias</code> works for any leading dims if your matmul supports batched @ 2-D.</p>`,
          R`<p>cross_entropy is one line on top of your log_softmax and fancy indexing.</p>`,
          R`<p>If the gradcheck on Linear fails only for the bias, your unbroadcast isn't summing the batch axis.</p>`,
        ] },
      { id: "w3m3", core: true, title: "Dropout, BatchNorm1d, LayerNorm", test: "pytest tests/w3/test_layers.py -q -k \"batchnorm or layernorm\"; pytest tests/w3/test_module.py -q -k dropout",
        detail: R`<p>Running stats with unbiased variance; eval uses them; LN over the last axis.</p>`,
        hints: [
          R`<p>Build BN from autograd ops: <code>mean = x.mean(axis=0)</code>, <code>xc = x - mean</code>, <code>var = (xc*xc).mean(axis=0)</code>, <code>xhat = xc / (var + eps) ** 0.5</code>. The backward comes for free and the gradcheck passes.</p>`,
          R`<p>Update running stats with <code>.data</code> (numpy, outside the graph): <code>self.running_var = (1-m)*self.running_var + m*var.data*n/(n-1)</code>. Keep the buffer's dtype.</p>`,
          R`<p>Dropout mask: <code>(rng.random(x.shape) >= p) / (1 - p)</code> as a constant array multiplied into the tensor.</p>`,
        ] },
      { id: "w3m4", core: true, title: "Conv2d (im2col) and MaxPool2d", test: "pytest tests/w3/test_layers.py -q -k \"conv or maxpool\"",
        detail: R`<p>Forward vs a naive loop for 5 stride/padding/kernel combos, gradchecks, ties in max-pool, and a speed check (&lt;5 s).</p>`,
        hints: [
          R`<p>Decide: custom op (fast, write col2im) or composition of ops (slower, free backward). For a custom op you need a way to create a Tensor with your own <code>_backward</code>: reuse the internal helper your Week 1 ops use.</p>`,
          R`<p>Forward: pad with <code>np.pad</code>, <code>sliding_window_view(Xp, (k,k), axis=(2,3))[:, :, ::s, ::s]</code>, transpose to (N,Ho,Wo,C,k,k), reshape to rows, matmul with <code>W.reshape(O, -1).T</code>, reshape and transpose to (N,O,Ho,Wo).</p>`,
          R`<p>MaxPool: reshape windows to (..., k*k), <code>argmax</code> on the last axis; in backward convert the flat argmax to (di, dj) with <code>np.divmod(arg, k)</code> and <code>np.add.at</code> into the right input coordinates.</p>`,
        ] },
      { id: "w3m5", core: true, title: "DataLoader", test: "pytest tests/w3/test_module.py -q -k dataloader",
        detail: R`<p>New permutation every epoch from one seeded generator, drop_last, len().</p>`,
        hints: [R`<p>Create <code>self.rng = np.random.default_rng(seed)</code> once in <code>__init__</code>; call <code>self.rng.permutation(n)</code> inside <code>__iter__</code> (make it a generator function with <code>yield</code>).</p>`] },
      { id: "w3m6", core: true, title: "Train on MNIST", test: "pytest tests/w3 -m slow -s",
        detail: R`<p>The tests define the recipes; your framework has to be correct and fast enough. Watch the per-epoch prints.</p>`,
        hints: [
          R`<p>Inputs are float32; keep parameters float32 too, or numpy will silently upcast every matmul to float64 (2× slower).</p>`,
          R`<p>If accuracy is near 10%, check that gradients reach the first layer (print norms) and that <code>zero_grad</code> runs every step.</p>`,
          R`<p>If the CNN epoch takes many minutes, profile one batch: <code>python -m cProfile -s tottime</code>. Usual suspects: np.add.at in the conv backward, Python loops over output positions, float64.</p>`,
        ] },
      { id: "w3m7", core: false, title: "Stretch: BatchNorm2d, a residual block, LR finder",
        detail: R`<p>Add <code>BatchNorm2d</code> (statistics over N, H, W per channel) and a <code>ResidualBlock</code> (conv-BN-ReLU-conv-BN + skip). Train a small ResNet on MNIST or Fashion-MNIST. Implement an LR range test and plot loss vs LR with matplotlib.</p>`,
        hints: [R`<p>BatchNorm2d = transpose to (N,H,W,C), reshape to (-1, C), reuse BatchNorm1d's math, reshape back.</p>`] },
    ],
  },
});
