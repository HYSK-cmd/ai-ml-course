COURSE.weeks.push({
  n: 2, id: "w2", title: "Machine Learning",
  tagline: "Generalization, linear models, trees and boosting, clustering and EM. The algorithms still running most production ML, built on your own autograd.",
  hours: { learn: 6.5, project: 7 },
  goals: [
    "Derive the bias–variance decomposition and use it to diagnose a model from its train/validation curves",
    "Explain ROC-AUC as a ranking probability and compute it in O(n log n) with ties",
    "Derive gradient boosting as gradient descent in function space, including the Newton leaf values for log-loss",
    "Derive EM for Gaussian mixtures and prove that each iteration can't decrease the likelihood",
    "Ship <code>forge.ml</code>, matching scikit-learn on accuracy, AUC, inertia and PCA to tight tolerances",
  ],
  schedule: [
    ["Mon", "L1 Generalization, bias–variance, validation"],
    ["Tue", "L2 Linear and logistic models, classification metrics"],
    ["Wed", "L3 Trees, bagging and gradient boosting"],
    ["Thu", "L4 k-means, Gaussian mixtures and EM (+ L5 SVMs if time)"],
    ["Fri", "Flashcards; derive the Newton leaf value on paper"],
    ["Sat", "Milestones 1–4: preprocessing/metrics/CV, linear, logistic, trees"],
    ["Sun", "Milestones 5–8: boosting, k-means, GMM, PCA"],
  ],
  lessons: [
    {
      id: "w2l1", title: "Generalization, bias–variance and validation", minutes: 70,
      summary: "What it means to learn rather than memorize, the decomposition that explains under- and overfitting, and how to measure it honestly.",
      keypoints: [
        "We minimize empirical risk but care about expected risk on new data; the gap is generalization error.",
        "Expected test MSE = bias² + variance + irreducible noise.",
        "Regularization trades a little bias for a lot of variance.",
        "Validation data must never influence training, including preprocessing statistics.",
        "k-fold CV estimates performance with less variance than a single split; temporal data needs time-ordered splits.",
      ],
      body: R`
<p>A model that scores 100% on its training data has learned nothing you can trust yet. This lesson defines what we actually want from learning, splits the error into parts you can diagnose separately, and sets the rules for measuring it without fooling yourself.</p>

<h2>Empirical risk minimization</h2>
<p>Data come from an unknown distribution $\mathcal{D}$ over $(x,y)$. For a model $f$ and loss $\ell$, the quantity we care about is the <strong>expected risk</strong> $R(f) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y)]$. We can't compute it, so we minimize the <strong>empirical risk</strong> on $n$ training samples:</p>
$$\hat R(f) = \frac1n\sum_{i=1}^n\ell(f(x_i),y_i),\qquad \hat f = \arg\min_{f\in\mathcal{F}}\hat R(f).$$
<p>$\hat R(\hat f)$ is optimistically biased: $\hat f$ was chosen <em>because</em> it does well on these points. The gap $R(\hat f) - \hat R(\hat f)$ grows with the flexibility of the model class $\mathcal{F}$ and shrinks with $n$. Learning theory (VC dimension, Rademacher complexity) makes this precise; the working intuition is: <mark>more capacity or less data ⇒ larger gap</mark>.</p>

<h2>The bias–variance decomposition</h2>
<p>Take regression with $y = f(x) + \varepsilon$, $\mathbb{E}\varepsilon = 0$, $\mathrm{Var}\,\varepsilon = \sigma^2$. Train on a random dataset $D$ to get $\hat f_D$. At a fixed test point $x$, average over both the training set and the test noise. Let $\bar f(x) = \mathbb{E}_D[\hat f_D(x)]$ be the "average model":</p>
$$\mathbb{E}_{D,\varepsilon}\big[(y - \hat f_D(x))^2\big] = \underbrace{(f(x) - \bar f(x))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat f_D(x) - \bar f(x))^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}.$$
<p><strong>Derivation.</strong> Write $y - \hat f = (f - \bar f) + (\bar f - \hat f) + \varepsilon$. Square and take expectations. The cross terms vanish: $\varepsilon$ is independent of $D$ with mean zero, and $\mathbb{E}_D[\bar f - \hat f_D] = 0$ by definition of $\bar f$, while $f - \bar f$ is a constant. What's left are the three squares. Do this on paper: it's the template for many ML proofs.</p>
<ul>
<li><strong>Bias</strong>: error of the average model; the model class can't represent $f$ (a line fit to a sine). Shows up as high training error.</li>
<li><strong>Variance</strong>: how much the fit moves when the training sample changes; the model chases noise. Shows up as a large train/validation gap.</li>
<li><strong>Noise</strong>: the floor. No model beats it.</li>
</ul>
<div class="widget" data-widget="biasvar"></div>
<p>Move the degree slider: low degree has high bias (both errors high), high degree has high variance (train error → 0, test error explodes). Then fix degree 15 and raise λ: ridge pulls the wiggly fit back, buying a little bias for a big drop in variance. That trade is what every regularizer does: weight decay, dropout, early stopping, tree depth limits, shrinkage in boosting.</p>
<div class="callout"><b>Reading learning curves</b>Train error high and close to validation error → underfitting (more capacity, better features, less regularization). Train error low, validation much higher → overfitting (more data, more regularization, simpler model). Both low and close → ship it. Deep nets complicate the U-curve ("double descent"), but the diagnostic still works at a fixed model size.</div>

<h2>Measuring honestly</h2>
<p><strong>Three splits.</strong> Train on train, choose hyperparameters on validation, report once on test. Every time you look at the test score and change something, test becomes validation and your estimate gets optimistic.</p>
<p><strong>k-fold cross-validation.</strong> Split into $k$ folds; train on $k-1$, score on the held-out one, rotate, average. Each point is used for validation exactly once, so the estimate has lower variance than one split, at $k\times$ the cost. Your <code>KFold</code> must give the first $n \bmod k$ folds one extra sample, exactly like sklearn, and <code>cross_val_score</code> must build a <em>fresh</em> model per fold (a model that keeps state between folds leaks).</p>
<div class="callout pitfall"><b>Leakage, the silent killer</b>
<ul>
<li><em>Preprocessing leakage</em>: fitting a scaler, PCA or vocabulary on all data before splitting lets validation statistics into training. Fit on the training fold only, then transform the rest. (The Week 2 tests fit <code>StandardScaler</code> on <code>Xtr</code> only.)</li>
<li><em>Temporal leakage</em>: random splits of time-ordered data train on the future. Week 8 uses per-user time-ordered splits and a test that checks it.</li>
<li><em>Group leakage</em>: the same patient or user in train and test lets the model memorize identities.</li>
<li><em>Target leakage</em>: a feature computed from the label (e.g. "refund issued" when predicting fraud).</li>
</ul>Leakage produces great offline numbers and a model that fails in production, so it's the first thing interviewers probe in ML system design.</div>

<div class="callout prod"><b>In production</b>The offline split is a simulation of deployment. Ask "at prediction time, what data would actually exist?" and make the split answer that question. A recommender evaluated with random splits can look 2× better than it will perform.</div>
`,
      quiz: [
        { q: "A model gets 2% train error and 19% validation error. The most likely issue and fix?", options: ["High bias: add features", "High variance: regularize or get more data", "Noise floor: nothing to do", "Leakage: the validation set is too easy"], answer: 1, why: "A large train/validation gap is the signature of variance (overfitting)." },
        { q: "In the bias–variance derivation, why does the cross term $2\\mathbb{E}[(f-\\bar f)(\\bar f - \\hat f_D)]$ vanish?", options: ["Because $f = \\bar f$", "Because $f-\\bar f$ is a constant w.r.t. $D$ and $\\mathbb{E}_D[\\bar f - \\hat f_D] = 0$", "Because the noise is Gaussian", "It doesn't vanish"], answer: 1, why: "$\\bar f$ is defined as $\\mathbb{E}_D\\hat f_D$, so the deviation has mean zero and the constant factor pulls out." },
        { q: "You standardize all features using the mean and std of the full dataset, then do 5-fold CV. What's wrong?", options: ["Nothing", "Validation folds' statistics leaked into training", "Standardization should be done per row", "CV needs 10 folds"], answer: 1, why: "Fit preprocessing on the training folds only. With many features or small data, this leakage measurably inflates scores." },
        { q: "Increasing the ridge penalty λ typically…", options: ["increases variance, decreases bias", "decreases variance, increases bias", "decreases both", "changes neither"], answer: 1, why: "Shrinking weights makes fits more stable across samples (less variance) but less able to match the truth (more bias)." },
      ],
      explain: "Explain the bias–variance decomposition to a classmate using the polynomial-degree example, and say what you would change if validation error were far above training error.",
      resources: [
        { title: "Stanford CS229 notes — Bias/variance and regularization", url: "https://cs229.stanford.edu/main_notes.pdf", note: "sections on generalization" },
        { title: "Hastie, Tibshirani, Friedman — Elements of Statistical Learning ch. 7", url: "https://hastie.su.domains/ElemStatLearn/", note: "model assessment and selection" },
        { title: "Kaufman et al. — Leakage in Data Mining", url: "https://dl.acm.org/doi/10.1145/2382577.2382579", note: "taxonomy of leakage" },
      ],
      cards: [
        { f: "Bias–variance decomposition of expected squared error", b: "bias² + variance + noise σ²" },
        { f: "Symptom of high variance", b: "Low training error, much higher validation error." },
        { f: "Rule for preprocessing inside cross-validation", b: "Fit scalers/encoders/PCA on the training fold only, then transform the held-out fold." },
      ],
    },
    {
      id: "w2l2", title: "Linear models and classification metrics", minutes: 80,
      summary: "Logistic and softmax regression as convex likelihood models, then the metrics that decide whether a classifier is any good.",
      keypoints: [
        "Logistic regression = Bernoulli MLE with $p = \\sigma(w^\\top x + b)$; its loss is convex.",
        "Softmax regression gradient: $\\nabla_W = X^\\top(P - Y)/N$.",
        "Standardize features: it improves conditioning and makes L2 penalties fair across features.",
        "ROC-AUC = P(random positive scores above random negative); compute via ranks in O(n log n).",
        "Accuracy misleads on imbalanced data; use precision/recall, PR-AUC, or cost-weighted metrics.",
      ],
      body: R`
<h2>From linear regression to logistic regression</h2>
<p>Linear regression predicts a real number. For a binary label we need a probability, so squash the linear score through the sigmoid: $p(y=1\mid x) = \sigma(z)$, $z = w^\top x + b$, $\sigma(z) = 1/(1+e^{-z})$. The model is linear in the <em>log-odds</em>: $\log\frac{p}{1-p} = z$. Each unit of $w_j x_j$ multiplies the odds by $e^{w_j}$, which is why logistic regression coefficients are interpretable.</p>
<p>Training is Bernoulli MLE (Week 1 L4): minimize the log-loss $-\frac1N\sum[y\log p + (1-y)\log(1-p)]$. Its gradient is $\frac1N X^\top(p - y)$, the same "prediction minus truth" as softmax CE. The loss is <strong>convex</strong> in $(w,b)$ (its Hessian $X^\top\mathrm{diag}(p(1-p))X$ is PSD), so gradient descent finds the global optimum. One catch: on <em>linearly separable</em> data the optimum is at infinity ($\|w\|\to\infty$ keeps reducing the loss), which is one reason to add an L2 penalty.</p>
<h3>Softmax regression</h3>
<p>For $K$ classes, $Z = XW + b$ with $W\in\mathbb{R}^{d\times K}$ and $P = \mathrm{softmax}(Z)$ row-wise. With one-hot $Y$, mean CE has gradient</p>
$$\nabla_W = \tfrac1N X^\top(P - Y),\qquad \nabla_b = \tfrac1N\mathbf 1^\top(P-Y).$$
<p>Your <code>LogisticRegression</code> builds exactly this graph with <code>forge.autograd</code> (<code>(X @ W + b).log_softmax(axis=1)[np.arange(n), y]</code>) and lets your autograd produce that gradient. The test monkeypatches <code>Tensor.backward</code> to verify you really use it.</p>
<p>Labels can be strings. Map them with <code>classes_, y_idx = np.unique(y, return_inverse=True)</code> and map predictions back with <code>classes_[argmax]</code>: a small detail that every production classifier needs.</p>
<div class="callout"><b>Why standardize</b>Gradient descent speed depends on the condition number of $X^\top X$ (Week 1 L5). Features measured in wildly different units make it huge. Standardizing ($z = (x-\mu)/\sigma$, fit on train) fixes the scale, and also makes an L2 penalty treat features equally: otherwise a feature measured in millimeters gets a tiny weight that's barely penalized.</div>

<h2>Classification metrics</h2>
<p>Threshold the score at $t$ and count the confusion matrix: TP, FP, FN, TN.</p>
<ul>
<li><strong>Precision</strong> $= \frac{TP}{TP+FP}$: of what I flagged, how much was right?</li>
<li><strong>Recall</strong> (TPR) $= \frac{TP}{TP+FN}$: of what was positive, how much did I catch?</li>
<li><strong>F1</strong> $= \frac{2PR}{P+R}$: harmonic mean, punishes imbalance between the two.</li>
<li><strong>FPR</strong> $= \frac{FP}{FP+TN}$.</li>
</ul>
<p>Define $0/0 = 0$ (the tests check this): a classifier that never predicts positive has precision and recall 0.</p>
<h3>ROC-AUC is a ranking probability</h3>
<p>The ROC curve plots TPR against FPR as the threshold sweeps from $+\infty$ to $-\infty$. Its area has a cleaner meaning: <mark>AUC = P(score of a random positive > score of a random negative)</mark>, counting ties as 1/2. That's the Mann–Whitney U statistic, so you never need to draw the curve:</p>
$$\mathrm{AUC} = \frac{\sum_{i\in\text{pos}}\mathrm{rank}_i - \frac{n_+(n_++1)}{2}}{n_+ n_-},$$
<p>where ranks are 1-based over all scores, and tied scores get the <em>average</em> of the ranks they span. Why it works: the rank of a positive counts how many items it beats (plus itself); subtracting $n_+(n_++1)/2$ removes positive-vs-positive comparisons. Sort once: $O(n\log n)$. The naive double loop over pairs is $O(n_+n_-)$, which is $10^{10}$ operations on the 400k-row speed test.</p>
<p>AUC is threshold-free and insensitive to class balance, which makes it great for comparing rankers and terrible for judging the operating point you'll deploy.</p>
<div class="callout pitfall"><b>Imbalance</b>With 1% fraud, "always predict legit" is 99% accurate. ROC-AUC can still look good while precision at any useful recall is poor, because FPR divides by the huge negative count. Use precision–recall curves (PR-AUC) and report precision at the recall you need.</div>
<div class="widget" data-widget="roc"></div>
<h3>Calibration and log-loss</h3>
<p>A classifier is <em>calibrated</em> if among examples it scores 0.8, 80% are positive. AUC ignores calibration (any monotone transform of scores has the same AUC); log-loss rewards it. Calibration matters whenever the probability is used downstream: expected-value decisions, bidding in ads, combining models. Your <code>log_loss</code> clips $p$ to $[\epsilon, 1-\epsilon]$ because $\log 0 = -\infty$ would make one confident mistake infinitely bad.</p>

<div class="callout prod"><b>In production</b>Logistic regression on good features is still a strong baseline for ads CTR, fraud and credit risk: fast, calibrated-ish, explainable, and easy to debug. Always build it first; if a deep model can't beat it, the problem is your data.</div>
`,
      quiz: [
        { q: "Scores for labels [0,0,1,1] are [0.1, 0.4, 0.35, 0.8]. What is the AUC?", options: ["1.0", "0.75", "0.5", "0.25"], answer: 1, why: "Positive–negative pairs: (0.35 vs 0.1 ✓), (0.35 vs 0.4 ✗), (0.8 vs 0.1 ✓), (0.8 vs 0.4 ✓): 3 of 4 = 0.75. This exact case is in the tests." },
        { q: "What happens to logistic regression without regularization on perfectly separable data?", options: ["It converges to a finite optimum", "Weights grow without bound as the loss keeps decreasing", "It can't fit the data", "The loss becomes non-convex"], answer: 1, why: "Scaling up a separating $w$ always lowers log-loss toward 0, so the minimizer is at infinity." },
        { q: "Applying $\\log$ to all of a model's scores changes its ROC-AUC by…", options: ["Nothing, if all scores are positive", "It always lowers AUC", "It always raises AUC", "Undefined"], answer: 0, why: "AUC depends only on the ordering of scores, and log is strictly increasing on positive numbers." },
        { q: "Why do tied scores get averaged ranks in the rank formula?", options: ["To make it faster", "Each positive–negative tie should count as 1/2", "sklearn requires it", "To avoid division by zero"], answer: 1, why: "Averaging ranks splits the credit for tied pairs evenly, matching the 'ties count 1/2' definition." },
      ],
      explain: "Explain why ROC-AUC equals the probability that a random positive is scored above a random negative, and why accuracy is misleading for a 1%-positive problem.",
      resources: [
        { title: "CS229 notes — Generalized linear models", url: "https://cs229.stanford.edu/main_notes.pdf", note: "logistic regression as a GLM" },
        { title: "Fawcett — An introduction to ROC analysis", url: "https://www.sciencedirect.com/science/article/pii/S016786550500303X", note: "the classic ROC paper" },
        { title: "scikit-learn user guide — Probability calibration", url: "https://scikit-learn.org/stable/modules/calibration.html", note: "" },
      ],
      cards: [
        { f: "Gradient of mean softmax CE w.r.t. $W$ for $Z = XW$", b: "$X^\\top(P-Y)/N$" },
        { f: "ROC-AUC as a probability", b: "P(score of random positive > score of random negative), ties count 1/2." },
        { f: "Precision and recall", b: "Precision = TP/(TP+FP); recall = TP/(TP+FN)." },
        { f: "AUC from ranks", b: "$(\\sum_{pos}\\text{rank} - n_+(n_++1)/2)/(n_+n_-)$ with average ranks for ties." },
      ],
    },
    {
      id: "w2l3", title: "Decision trees, bagging and gradient boosting", minutes: 95,
      summary: "How CART finds splits fast, why averaging trees reduces variance, and gradient boosting derived as gradient descent over functions, with Newton steps for classification.",
      keypoints: [
        "A split maximizes impurity decrease; sort each feature once and use prefix sums to score all thresholds in O(n).",
        "Bagging reduces variance: averaging B models with correlation ρ gives variance $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$.",
        "Boosting fits each new tree to the negative gradient of the loss (residuals for MSE, $y-p$ for log-loss).",
        "Newton leaf value for log-loss: $\\sum r / \\sum p(1-p)$ over the leaf.",
        "Learning rate (shrinkage) and subsampling regularize boosting.",
      ],
      body: R`
<p>Gradient-boosted trees (XGBoost, LightGBM, CatBoost) win most tabular-data competitions and power a huge amount of production ML: ranking, fraud, pricing. You'll build one from scratch, and you'll reuse it in Week 8 as the recommender's re-ranker.</p>

<h2>CART: finding a split</h2>
<p>A regression or classification tree recursively splits the data with axis-aligned questions "$x_j \le t$?". At each node, choose $(j,t)$ to maximize the <strong>impurity decrease</strong></p>
$$\Delta = N\,I(\text{node}) - N_L\,I(L) - N_R\,I(R).$$
<ul>
<li>Regression: $N\cdot I$ = sum of squared errors around the mean (SSE). A leaf predicts its mean.</li>
<li>Classification: Gini $I = 1-\sum_k p_k^2$ or entropy $I = -\sum_k p_k\log p_k$. A leaf predicts its class frequencies.</li>
</ul>
<h3>Doing it fast</h3>
<p>The naive search tries every threshold and recomputes impurity from scratch: $O(n^2)$ per feature per node. Instead, sort the node's values for feature $j$ once. Then moving the threshold past one sample moves that sample from right to left, so <mark>left/right statistics are prefix sums</mark>. For regression, with sorted targets $y_{(1)},\dots,y_{(n)}$ and prefix sums $S_i = \sum_{k\le i} y_{(k)}$, $Q_i = \sum_{k\le i} y_{(k)}^2$:</p>
$$\mathrm{SSE}_L(i) = Q_i - \frac{S_i^2}{i},\qquad \mathrm{SSE}_R(i) = (Q_n - Q_i) - \frac{(S_n - S_i)^2}{n-i}.$$
<p>Every candidate split is scored with a few vectorized numpy ops. For classification, use cumulative sums of one-hot labels to get class counts on each side. Two rules from the spec: a threshold is only valid <em>between distinct values</em> (midpoints), and both sides must have at least <code>min_samples_leaf</code> samples. Mask invalid positions with $+\infty$ cost and take the argmin. That's $O(n\log n)$ per feature per node; the speed test fits a depth-8 tree on 16,512 California-housing rows.</p>
<div class="callout"><b>Why split even with zero gain?</b>XOR: no single axis split reduces impurity at the root, yet two levels of splits classify perfectly. A greedy tree that refuses zero-gain splits stops at the root. sklearn splits whenever the node is impure; so do you (<code>test_tree_classifier_fits_xor_fully</code>).</div>
<p>Trees are low-bias, high-variance: a fully grown tree memorizes its training data, and a slightly different sample gives a very different tree. The two ensemble families attack this from opposite ends.</p>

<h2>Bagging and random forests: average away variance</h2>
<p>Train $B$ trees on bootstrap samples and average. If each has variance $\sigma^2$ and pairwise correlation $\rho$:</p>
$$\mathrm{Var}\Big(\frac1B\sum_b T_b\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$
<p>More trees kill the second term; the first term is a floor set by correlation. <strong>Random forests</strong> lower $\rho$ by letting each split consider only a random subset of features (default $\sqrt d$ for classification), so trees disagree more and the average improves. That's your stretch goal.</p>

<h2>Gradient boosting: gradient descent in function space</h2>
<p>Boosting goes the other way: start with a high-bias model and add small trees that fix what's left. Think of the predictions on the training set, $F = (F(x_1),\dots,F(x_n))$, as the parameters. To reduce $L(F) = \sum_i\ell(y_i, F(x_i))$, take a gradient step:</p>
$$F \leftarrow F - \eta\,\nabla_F L,\qquad (\nabla_F L)_i = \frac{\partial\ell(y_i,F_i)}{\partial F_i}.$$
<p>But we need a function that also works on new $x$, not just a vector of $n$ numbers. So <strong>fit a regression tree $h$ to the negative gradient</strong> $r_i = -\partial\ell/\partial F_i$ (the "pseudo-residuals") and step along it: $F \leftarrow F + \eta h$.</p>
<ul>
<li><strong>Squared loss</strong> $\ell = \frac12(y-F)^2$: $r_i = y_i - F_i$, the plain residual. Each tree fits what the ensemble still gets wrong. Start from $F_0 = \bar y$.</li>
<li><strong>Log-loss</strong> with $F$ = log-odds, $p = \sigma(F)$: $\ell = -[y\log p + (1-y)\log(1-p)]$ and $r_i = y_i - p_i$. Start from $F_0 = \log\frac{\bar y}{1-\bar y}$ (the test checks <code>init_</code>).</li>
</ul>
<div class="widget" data-widget="boost"></div>
<h3>Newton leaf values</h3>
<p>The tree's structure comes from fitting $r$ with squared error, but its leaf values can be chosen better. For leaf $R_j$, find the constant $\gamma$ minimizing $\sum_{i\in R_j}\ell(y_i, F_i + \gamma)$. Second-order Taylor expansion with gradient $g_i = p_i - y_i = -r_i$ and Hessian $h_i = p_i(1-p_i)$:</p>
$$\sum_{i\in R_j}\Big[\ell_i + g_i\gamma + \tfrac12 h_i\gamma^2\Big]\ \Rightarrow\ \gamma^* = -\frac{\sum g_i}{\sum h_i} = \frac{\sum_{i\in R_j} r_i}{\sum_{i\in R_j} p_i(1-p_i)}.$$
<p>A Newton step per leaf. Compared with the plain mean of residuals, it takes big steps where the model is confidently wrong-ish ($p(1-p)$ small) and small steps where it's uncertain. The test checks that log-loss drops below 0.35 within 10 stages, which plain residual means don't achieve. XGBoost's split gain and leaf weights are this same formula plus regularization: $w^* = -\frac{G}{H+\lambda}$, gain $= \frac12\big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\big]-\gamma$.</p>
<p>Since your tree owns its leaf storage, you'll need a way to overwrite leaf values after fitting: use <code>apply(X)</code> to see which leaf each sample lands in, compute the Newton value per leaf, write them back.</p>
<h3>Regularizing boosting</h3>
<ul>
<li><strong>Shrinkage</strong> $\eta\in[0.01, 0.3]$: smaller steps, more trees, better generalization.</li>
<li><strong>Shallow trees</strong> (depth 3–8): each tree captures low-order interactions.</li>
<li><strong>Subsampling</strong> rows per tree (stochastic GB): decorrelates trees and speeds training.</li>
<li><strong>Early stopping</strong> on validation loss: the number of trees is the main capacity knob.</li>
</ul>

<div class="callout prod"><b>In production</b>GBMs are the default for tabular problems: robust to feature scale, handle missing values and nonlinearity, fast to serve (a few hundred tree traversals). Typical pitfalls are leakage through target-encoded features and training/serving skew in feature computation. Week 8 uses your GBM as a ranker over two-tower candidates, the standard two-stage pattern at YouTube, Pinterest and others.</div>
`,
      quiz: [
        { q: "For squared loss, the pseudo-residuals boosting fits are…", options: ["$y - F$", "$(y-F)^2$", "$\\mathrm{sign}(y-F)$", "$F - y$"], answer: 0, why: "$-\\partial\\frac12(y-F)^2/\\partial F = y - F$." },
        { q: "A leaf contains 3 samples with $y = [1,0,1]$ and current $p = [0.5,0.5,0.5]$. What is its Newton leaf value?", options: ["$2/3$", "$1/3$", "$0.5$", "$4/3$"], answer: 0, why: "$r = y - p = [0.5,-0.5,0.5]$ so $\\sum r = 0.5$; $\\sum p(1-p) = 3 \\times 0.25 = 0.75$; $\\gamma = 0.5/0.75 = 2/3$. (The plain mean residual would be $1/6$: Newton steps are much bolder here.)" },
        { q: "Why do random forests consider a random subset of features per split?", options: ["To train faster only", "To decorrelate trees so averaging reduces variance more", "To reduce bias", "To handle missing values"], answer: 1, why: "Bagged-ensemble variance has a floor $\\rho\\sigma^2$; feature subsampling lowers $\\rho$." },
        { q: "Scoring all thresholds for one feature at a node with $n$ samples costs, with sort + prefix sums…", options: ["$O(n^2)$", "$O(n\\log n)$", "$O(n^3)$", "$O(\\log n)$"], answer: 1, why: "Sorting dominates; the prefix-sum sweep is $O(n)$." },
      ],
      explain: "Explain gradient boosting as 'gradient descent where each step is a tree', and why the leaf values for log-loss use sum(r)/sum(p(1-p)) instead of the mean residual.",
      resources: [
        { title: "Friedman — Greedy Function Approximation: A Gradient Boosting Machine", url: "https://jerryfriedman.su.domains/ftp/trebst.pdf", note: "the original; section 4.5 is two-class logistic" },
        { title: "Chen & Guestrin — XGBoost", url: "https://arxiv.org/abs/1603.02754", note: "section 2.2: second-order objective and split gain" },
        { title: "ESL ch. 9.2 (trees), 10 (boosting), 15 (random forests)", url: "https://hastie.su.domains/ElemStatLearn/", note: "" },
      ],
      cards: [
        { f: "Pseudo-residual for log-loss boosting with $F$ = log-odds", b: "$r = y - \\sigma(F) = y - p$" },
        { f: "Newton leaf value for log-loss", b: "$\\sum_{leaf} r_i / \\sum_{leaf} p_i(1-p_i)$" },
        { f: "Variance of an average of B correlated models", b: "$\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$" },
        { f: "Left SSE from prefix sums", b: "$Q_i - S_i^2/i$ where $S, Q$ are prefix sums of sorted $y$ and $y^2$." },
      ],
    },
    {
      id: "w2l4", title: "k-means, Gaussian mixtures and EM", minutes: 85,
      summary: "Clustering as optimization, k-means++ seeding, and the EM algorithm derived from a lower bound, with the monotonicity proof your test checks.",
      keypoints: [
        "Lloyd's algorithm alternates assignment and mean updates; each step can't increase inertia, so it converges (to a local optimum).",
        "k-means++ samples seeds with probability ∝ $D(x)^2$, giving an $O(\\log k)$-approximation in expectation.",
        "EM maximizes a lower bound (ELBO) on the log-likelihood; the E-step makes the bound tight, the M-step maximizes it.",
        "Therefore EM never decreases the log-likelihood.",
        "Compute responsibilities in log space with log-sum-exp; add reg_covar to keep covariances invertible.",
      ],
      body: R`
<h2>k-means as coordinate descent</h2>
<p>Objective (inertia): $J(c, \mu) = \sum_i\|x_i - \mu_{c_i}\|^2$ over assignments $c_i$ and centers $\mu_k$. Lloyd's algorithm alternates two exact minimizations:</p>
<ol>
<li><strong>Assign</strong>: with centers fixed, $c_i = \arg\min_k\|x_i-\mu_k\|^2$ minimizes $J$ over $c$.</li>
<li><strong>Update</strong>: with assignments fixed, $\mu_k = $ mean of its points minimizes $J$ over $\mu$ (set the gradient $-2\sum_{i:c_i=k}(x_i-\mu_k)$ to zero).</li>
</ol>
<p>Each step can only lower $J$, and there are finitely many assignments, so it converges, to a <em>local</em> minimum that depends on the start. Hence multiple restarts (<code>n_init</code>) keeping the lowest inertia.</p>
<p><strong>Vectorize distances.</strong> $\|x-\mu\|^2 = \|x\|^2 - 2x^\top\mu + \|\mu\|^2$ turns all $n\times k$ distances into one matmul. Clip at zero: rounding can make tiny distances negative.</p>
<div class="widget" data-widget="kmeans"></div>
<h3>k-means++</h3>
<p>Pick the first center uniformly from the data; each next center is a data point sampled with probability proportional to $D(x)^2$, its squared distance to the nearest center chosen so far. Far-away points are likely picks, so seeds spread across clusters. Arthur & Vassilvitskii proved the expected inertia is within $O(\log k)$ of optimal before any Lloyd iterations. The test plants one point at 100 among 999 at 0: the second seed must be the outlier (probability ≈ 1).</p>
<div class="callout pitfall"><b>Empty clusters</b>A center can lose all its points. Its mean is then undefined (numpy gives NaN with a warning). Re-seed it, e.g. at the point currently farthest from its assigned center.</div>

<h2>Gaussian mixtures</h2>
<p>k-means makes hard, spherical assignments. A GMM models the density as $p(x) = \sum_k\pi_k\,\mathcal{N}(x;\mu_k,\Sigma_k)$, with soft assignments and full covariances (elliptical clusters). Direct maximum likelihood is hard: $\log p(x) = \log\sum_k(\cdots)$, a log of a sum, has no closed-form maximizer. EM handles it by introducing the latent cluster $z$.</p>
<h2>EM, derived</h2>
<p>For any distribution $q(z)$ over the latent cluster of a point, Jensen's inequality (log is concave) gives</p>
$$\log p(x) = \log\sum_z q(z)\frac{p(x,z)}{q(z)} \ \ge\ \sum_z q(z)\log\frac{p(x,z)}{q(z)} =: \mathcal{L}(q,\theta).$$
<p>The gap is exactly $\log p(x) - \mathcal{L} = \mathrm{KL}(q(z)\,\|\,p(z\mid x))\ge 0$. EM alternates:</p>
<ul>
<li><strong>E-step</strong>: set $q(z) = p(z\mid x;\theta^{old})$, the posterior. The KL becomes 0, so the bound touches the log-likelihood. For GMMs these are the <em>responsibilities</em> $\gamma_{ik} = \frac{\pi_k\mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)}$.</li>
<li><strong>M-step</strong>: maximize $\mathcal{L}$ over $\theta$ with $q$ fixed. For GMMs it's weighted MLE in closed form: $N_k = \sum_i\gamma_{ik}$, $\pi_k = N_k/n$, $\mu_k = \frac1{N_k}\sum_i\gamma_{ik}x_i$, $\Sigma_k = \frac1{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$.</li>
</ul>
<p><strong>Why the likelihood never decreases</strong>: $\log p(x;\theta^{new}) \ge \mathcal{L}(q,\theta^{new}) \ge \mathcal{L}(q,\theta^{old}) = \log p(x;\theta^{old})$. The first inequality is the bound, the second is the M-step, the equality is the tight E-step. <code>test_gmm_em_monotone_and_recovers_means</code> asserts exactly this on your <code>lower_bound_history_</code>. If yours ever decreases, you have a bug (usually responsibilities computed with stale parameters, or a covariance missing the weights).</p>
<p>k-means is the limit of EM for a GMM with shared covariance $\sigma^2 I$ as $\sigma\to 0$: responsibilities become one-hot.</p>
<h3>Numerics</h3>
<ul>
<li>Compute $\log\pi_k + \log\mathcal{N}(x_i;\mu_k,\Sigma_k)$ for all $i,k$, then normalize with log-sum-exp. A point at $(10^4, -10^4)$ has density $e^{-10^8}$ under every component: in linear space that's 0/0 = NaN; in log space it's fine (the test checks).</li>
<li>Gaussian log-density via Cholesky $\Sigma = LL^\top$: $\log\mathcal{N} = -\frac12\big(\|L^{-1}(x-\mu)\|^2 + d\log 2\pi + 2\sum\log L_{jj}\big)$. No explicit inverse, and the log-determinant comes for free.</li>
<li>Add <code>reg_covar</code>$\cdot I$ to each covariance: a component that collapses onto one point would otherwise have $\det\Sigma\to0$ and likelihood $\to\infty$.</li>
</ul>

<div class="callout prod"><b>In production</b>k-means quietly runs inside systems you'll use: vector quantization in ANN indexes (your Week 8 IVF index trains coarse centroids with YOUR k-means), codebooks in product quantization, customer segmentation. EM is the template for every latent-variable model, and the ELBO you just derived is the objective of VAEs and diffusion models.</div>
`,
      quiz: [
        { q: "Why does Lloyd's algorithm always terminate?", options: ["Gradient descent converges", "Each step weakly decreases inertia and there are finitely many assignments", "k-means++ guarantees it", "It doesn't always terminate"], answer: 1, why: "A non-increasing objective over a finite set of assignment configurations can't cycle forever (with consistent tie-breaking)." },
        { q: "In EM, the E-step sets $q(z) = p(z\\mid x;\\theta)$ because…", options: ["it maximizes the likelihood directly", "it makes the lower bound equal to the log-likelihood (KL gap = 0)", "it's the only tractable choice", "it initializes the M-step"], answer: 1, why: "The gap between log-likelihood and ELBO is $\\mathrm{KL}(q\\|p(z|x))$, zero exactly at the posterior." },
        { q: "Your GMM's log-likelihood goes down between two iterations. Most likely…", options: ["Normal EM behavior", "A bug, e.g. responsibilities or covariances computed inconsistently", "Too few components", "Learning rate too high"], answer: 1, why: "Correct EM is monotone; a decrease means an implementation error. (EM has no learning rate.)" },
        { q: "k-means++ picks the next seed with probability proportional to…", options: ["$D(x)$", "$D(x)^2$", "uniform", "$1/D(x)$"], answer: 1, why: "Squared distance to the nearest existing center." },
      ],
      explain: "Explain the EM algorithm for a Gaussian mixture: what the E and M steps compute, and the three-line argument for why the likelihood never decreases.",
      resources: [
        { title: "Bishop PRML ch. 9 — Mixture models and EM", url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", note: "the clearest EM derivation" },
        { title: "Arthur & Vassilvitskii — k-means++", url: "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf", note: "" },
        { title: "CS229 notes — EM algorithm", url: "https://cs229.stanford.edu/main_notes.pdf", note: "" },
      ],
      cards: [
        { f: "GMM E-step: responsibility $\\gamma_{ik}$", b: "$\\pi_k\\mathcal{N}(x_i;\\mu_k,\\Sigma_k)/\\sum_j\\pi_j\\mathcal{N}(x_i;\\mu_j,\\Sigma_j)$, computed in log space." },
        { f: "Why EM is monotone", b: "$\\log p(\\theta^{new}) \\ge \\mathcal{L}(q,\\theta^{new}) \\ge \\mathcal{L}(q,\\theta^{old}) = \\log p(\\theta^{old})$" },
        { f: "Vectorized squared distances between rows of X and C", b: "$\\|x\\|^2 - 2XC^\\top + \\|c\\|^2$ (clip at 0)." },
      ],
    },
    {
      id: "w2l5", title: "SVMs and the kernel trick (short)", minutes: 45,
      summary: "Maximum-margin classifiers, hinge loss as regularized ERM, and kernels as implicit feature maps.",
      keypoints: [
        "Margin of a separating hyperplane is $1/\\|w\\|$ when the closest points satisfy $|w^\\top x + b| = 1$.",
        "Soft-margin SVM = hinge loss + L2 penalty: $\\min \\frac\\lambda2\\|w\\|^2 + \\frac1n\\sum\\max(0, 1-y_i(w^\\top x_i+b))$.",
        "Only support vectors (points on or inside the margin) determine the solution.",
        "Kernel trick: replace dot products $x^\\top x'$ with $k(x,x')$ to work in a feature space you never build.",
      ],
      body: R`
<p>SVMs are less common in production now, but the ideas recur everywhere: margins (contrastive losses), hinge losses (ranking losses), and kernels (attention can be read as a kernel smoother).</p>
<h2>Maximum margin</h2>
<p>With labels $y\in\{-1,+1\}$ and separable data, many hyperplanes $w^\top x + b = 0$ separate the classes. Pick the one farthest from the nearest points. The distance from $x$ to the plane is $|w^\top x+b|/\|w\|$. Rescale $(w,b)$ so the closest points have $y(w^\top x + b) = 1$; then the margin is $1/\|w\|$ and maximizing it is</p>
$$\min_{w,b}\ \tfrac12\|w\|^2\quad\text{s.t. } y_i(w^\top x_i + b)\ge 1\ \forall i.$$
<p>A convex quadratic program. At the solution, only the points with active constraints matter: the <strong>support vectors</strong>.</p>
<h2>Soft margin = hinge loss + L2</h2>
<p>Allow violations with slack and you get the unconstrained form</p>
$$\min_{w,b}\ \frac\lambda2\|w\|^2 + \frac1n\sum_i\max\big(0,\,1 - y_i(w^\top x_i+b)\big).$$
<p>Compare with logistic regression: same L2 penalty, but the log-loss $\log(1+e^{-yz})$ is replaced by the hinge $\max(0,1-yz)$. The hinge is exactly zero for points correctly classified beyond the margin, so they don't affect the solution (sparsity in the data). Logistic loss never reaches zero, so every point matters a little, and outputs are probabilities. You can train a linear SVM with your autograd and SGD: the hinge's subgradient is $-y x$ when $yz<1$, else 0.</p>
<h2>Kernels</h2>
<p>The dual of the SVM problem only touches data through dot products $x_i^\top x_j$. Replace them with a kernel $k(x_i,x_j) = \phi(x_i)^\top\phi(x_j)$ and you've trained a linear classifier in the feature space $\phi$ without ever computing $\phi$. The RBF kernel $k(x,x') = \exp(-\gamma\|x-x'\|^2)$ corresponds to an infinite-dimensional $\phi$. Cost: the $n\times n$ kernel matrix, which is why kernel methods stall beyond ~100k samples and why neural nets (which learn $\phi$) took over at scale.</p>
<div class="callout"><b>Connection forward</b>Attention computes $\sum_j \mathrm{softmax}_j(q^\top k_j)\,v_j$: a weighted average of values with weights from a (exponentiated) dot-product kernel between query and keys. Linear-attention variants literally replace it with a kernel feature map to drop the $O(T^2)$ cost.</div>
`,
      quiz: [
        { q: "The hinge loss $\\max(0, 1 - yz)$ is zero when…", options: ["$yz \\ge 1$", "$yz \\ge 0$", "$z = y$", "never"], answer: 0, why: "Correct and beyond the margin. Points there aren't support vectors and don't move the solution." },
        { q: "Why do kernel SVMs scale poorly to millions of examples?", options: ["The hinge loss is non-convex", "They need the n×n kernel matrix (memory and time ≥ quadratic in n)", "Kernels can't be computed on GPUs", "They need one-hot labels"], answer: 1, why: "Kernel methods work with pairwise similarities; $n^2$ entries for $n=10^6$ is $10^{12}$." },
      ],
      explain: "Compare logistic regression and a linear soft-margin SVM as 'L2 penalty + a loss on the margin yz'. When does each one ignore a training point?",
      resources: [
        { title: "CS229 notes — Support vector machines", url: "https://cs229.stanford.edu/main_notes.pdf", note: "Lagrange duality and kernels" },
        { title: "Shalev-Shwartz & Ben-David — Understanding Machine Learning, ch. 15–16", url: "https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/", note: "free" },
      ],
      cards: [
        { f: "Soft-margin SVM objective", b: "$\\frac\\lambda2\\|w\\|^2 + \\frac1n\\sum\\max(0,1-y_i(w^\\top x_i+b))$" },
        { f: "What is a support vector?", b: "A training point on or inside the margin; only these determine the SVM solution." },
      ],
    },
  ],
  project: {
    title: "forge.ml — classical ML from scratch",
    dir: "forge/ml/",
    pitch: "Implement the classical toolkit with sklearn-compatible APIs and match sklearn to tight tolerances: closed-form ridge, softmax regression on your autograd, vectorized CART, Newton-boosted trees, k-means++, EM for GMMs, PCA via SVD, AUC, CV.",
    test: "pytest tests/w2 -q",
    slow: "pytest tests/w2 -q -m stretch",
    overview: R`
<p>Every estimator follows the sklearn conventions: hyperparameters in <code>__init__</code> (store them, do nothing else), <code>fit(X, y)</code> returns <code>self</code>, learned attributes end with an underscore. scikit-learn is installed only so the tests can compare against it: <strong>never import it in <code>forge/</code></strong>.</p>
<p>Suggested order: metrics and model selection first (they're small and other tests use them), then linear models, trees, boosting, then the unsupervised trio. The reference solution passes all 33 tests in about 10 seconds; if your tree tests take minutes, your split search isn't vectorized yet.</p>
`,
    milestones: [
      { id: "w2m1", core: true, title: "StandardScaler, metrics, train_test_split, KFold, cross_val_score", test: "pytest tests/w2/test_linear_metrics.py -q -k \"scaler or roc or metrics or kfold or split or cross\"",
        detail: R`<p>AUC must handle ties exactly like sklearn and be O(n log n) (400k rows in under 3 s).</p>`,
        hints: [
          R`<p>Average ranks for ties: sort scores, walk runs of equal values, and give every element of a run the mean of its 1-based positions. <code>scipy.stats.rankdata</code> does this, but write it yourself.</p>`,
          R`<p>KFold sizes: <code>sizes = np.full(k, n // k); sizes[: n % k] += 1</code>, then slice consecutive chunks of the (possibly permuted) index array.</p>`,
          R`<p>confusion_matrix with explicit <code>labels</code>: build a <code>{label: row}</code> dict and skip pairs whose labels aren't in it, as sklearn does.</p>`,
        ] },
      { id: "w2m2", core: true, title: "LinearRegression (ridge, closed form)", test: "pytest tests/w2/test_linear_metrics.py -q -k linear_regression",
        detail: R`<p>Must equal sklearn's <code>LinearRegression</code>/<code>Ridge</code> coefficients to 1e-6 and survive exactly collinear columns.</p>`,
        hints: [
          R`<p>Center $X$ and $y$; solve for $w$ on centered data; then $b = \bar y - \bar x^\top w$.</p>`,
          R`<p>l2 &gt; 0: <code>np.linalg.solve(Xc.T @ Xc + l2 * I, Xc.T @ yc)</code>. l2 == 0: <code>np.linalg.lstsq(Xc, yc, rcond=None)[0]</code> (minimum-norm solution when collinear).</p>`,
          R`<p>If coefficients differ from Ridge slightly, check that you're not penalizing the intercept and that you didn't divide the penalty by n.</p>`,
        ] },
      { id: "w2m3", core: true, title: "LogisticRegression (softmax) on forge.autograd", test: "pytest tests/w2/test_linear_metrics.py -q -k logistic",
        detail: R`<p>Full-batch Adam for <code>max_iter</code> steps; ≥95% on digits and breast cancer; string labels.</p>`,
        hints: [
          R`<p>Parameters: <code>W = Tensor(np.zeros((d, K)), requires_grad=True)</code>, <code>b = Tensor(np.zeros(K), requires_grad=True)</code>. Zeros are fine here: the loss is convex.</p>`,
          R`<p>Loss: <code>-(X @ W + b).log_softmax(axis=1)[np.arange(n), y_idx].mean() + 0.5 * l2 * (W * W).sum()</code>.</p>`,
          R`<p><code>predict_proba</code> doesn't need autograd: compute softmax in numpy from <code>coef_</code>/<code>intercept_</code> (stable: subtract the row max).</p>`,
        ] },
      { id: "w2m4", core: true, title: "CART: DecisionTreeClassifier and DecisionTreeRegressor", test: "pytest tests/w2/test_trees_boosting.py -q -k tree_",
        detail: R`<p>The 1-D regression test demands sklearn-identical predictions, depth and leaf count. Constraints <code>min_samples_leaf</code>, <code>min_samples_split</code>, <code>max_depth=0</code> are tested.</p>`,
        hints: [
          R`<p>Store the tree as parallel lists/arrays (feature, threshold, left, right, value), built with an explicit stack of (indices, depth, parent). <code>apply</code> then walks all rows at once: keep a vector of current node ids and advance the ones not at a leaf.</p>`,
          R`<p>Per feature: <code>order = argsort(x)</code>; valid split positions are where <code>xs[i+1] > xs[i]</code> and both sides have ≥ min_samples_leaf. Cost arrays from prefix sums (regression) or cumulative one-hot counts (classification), set invalid to inf, argmin.</p>`,
          R`<p>To match sklearn exactly: threshold = midpoint of the two neighboring distinct values; break ties between equal-cost splits by taking the first (lowest feature index, then lowest threshold); split while the node is impure even if the gain is 0.</p>`,
        ] },
      { id: "w2m5", core: true, title: "Gradient boosting (regression + Newton-step classification)", test: "pytest tests/w2/test_trees_boosting.py -q -k gbm",
        detail: R`<p>RMSE &lt; 0.60 on California housing (and &lt; 0.85× linear regression). Classifier: loss &lt; 0.35 after 10 stages.</p>`,
        hints: [
          R`<p>Regressor: <code>F = full(n, y.mean())</code>; each stage fit a depth-3 tree to <code>y - F</code> (on the subsample) and do <code>F += lr * tree.predict(X)</code>. Record train MSE each stage.</p>`,
          R`<p>Classifier: give your regressor tree a method like <code>set_leaf_values({leaf_id: value})</code>. After fitting to <code>r = y - p</code>, compute per leaf <code>r[m].sum() / (p[m]*(1-p[m])).sum()</code> using <code>apply</code>.</p>`,
          R`<p>Use the same random generator across stages for subsampling (seeded once in fit). <code>predict_proba</code> columns are [1-p, p]; check that log(P1/P0) equals your decision function.</p>`,
        ] },
      { id: "w2m6", core: true, title: "k-means and k-means++", test: "pytest tests/w2/test_unsupervised.py -q -k kmeans",
        detail: R`<p>ARI &gt; 0.99 on blobs, inertia within 1% of sklearn, 50k×16 with 20 clusters in under 20 s.</p>`,
        hints: [
          R`<p>Distances: <code>(X*X).sum(1)[:,None] - 2*X@C.T + (C*C).sum(1)[None]</code>, clipped at 0.</p>`,
          R`<p>k-means++: keep a running <code>d2 = min(d2, dist_to_new_center)</code> instead of recomputing distances to all centers each round.</p>`,
          R`<p>Use one RNG for all n_init runs; keep the run with the lowest final inertia (recompute it after the last update).</p>`,
        ] },
      { id: "w2m7", core: true, title: "Gaussian mixture via EM", test: "pytest tests/w2/test_unsupervised.py -q -k gmm",
        detail: R`<p>Monotone <code>lower_bound_history_</code>, means within 0.2, exact log-density vs scipy, no NaN for far points.</p>`,
        hints: [
          R`<p>Initialize from your KMeans labels: one-hot responsibilities into the M-step.</p>`,
          R`<p>Log density with Cholesky: <code>L = cholesky(S)</code>, <code>z = solve(L, (X - mu).T)</code>, <code>-0.5*((z*z).sum(0) + d*log(2π) + 2*log(diag(L)).sum())</code>.</p>`,
          R`<p>Loop: E-step (log responsibilities and mean log-likelihood, append to history, check tol vs previous) then M-step. Appending before the convergence check makes the history consistent with the test.</p>`,
        ] },
      { id: "w2m8", core: true, title: "PCA via SVD", test: "pytest tests/w2/test_unsupervised.py -q -k pca",
        detail: R`<p>Components, explained variance and transform match sklearn up to sign.</p>`,
        hints: [
          R`<p><code>U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)</code>; components = first rows of Vt.</p>`,
          R`<p>explained_variance_ = S²/(n-1); the ratio divides by the sum over ALL singular values, not just the kept ones.</p>`,
          R`<p>inverse_transform: <code>Z @ components_ + mean_</code>.</p>`,
        ] },
      { id: "w2m9", core: false, title: "Stretch: random forest", test: "pytest tests/w2 -q -m stretch",
        detail: R`<p>Bootstrap rows + <code>max_features</code> per split; must beat a single fully grown tree on a held-out split.</p>`,
        hints: [R`<p>Your tree needs its own RNG for the per-node feature subset; pass each tree a different seed drawn from the forest's RNG.</p>`] },
    ],
  },
});
