COURSE.weeks.push({
  n: 1, id: "w1", title: "Math for ML",
  tagline: "Linear algebra as geometry, matrix calculus, probability and optimization: exactly the math that autograd, every loss function and every optimizer are made of.",
  hours: { learn: 6.5, project: 7 },
  goals: [
    "Read any matrix expression as a geometric operation and check its shapes in your head",
    "Derive <code>dL/dW</code> for a linear layer and <code>dL/dlogits</code> for softmax cross-entropy by hand",
    "Explain why MSE and cross-entropy are maximum-likelihood objectives, and what L2 regularization assumes",
    "Explain why gradient descent zig-zags, what momentum and Adam fix, and why AdamW decouples weight decay",
    "Ship <code>forge.autograd</code>: a reverse-mode autodiff engine over numpy that later weeks train real networks with",
  ],
  schedule: [
    ["Mon", "L1 Linear algebra as geometry + L2 Eigen, SVD, PCA (skim L2 if short on time)"],
    ["Tue", "L3 Matrix calculus and reverse-mode autodiff — the most important lesson this week"],
    ["Wed", "L4 Probability: MLE, MAP, entropy, KL"],
    ["Thu", "L5 Optimization: GD, momentum, Adam, schedules"],
    ["Fri", "Flashcards + re-derive the softmax-CE gradient on paper without notes"],
    ["Sat", "Project milestones 1–4: Tensor, broadcasting, matmul/reductions, stable log-softmax"],
    ["Sun", "Project milestones 5–7: no_grad, gradcheck, optimizers; run the full grader"],
  ],
  lessons: [
    {
      id: "w1l1", title: "Linear algebra as geometry", minutes: 75,
      summary: "Vectors, matrices as linear maps, rank, projections, and least squares derived from one picture.",
      keypoints: [
        "A matrix is a function: its columns are where the basis vectors land.",
        "$Ax$ is a weighted sum of $A$'s columns; the column space is everything $A$ can output.",
        "rank = number of independent columns = dimension of the output space actually reached.",
        "Least squares: the residual $y - A\\hat w$ must be orthogonal to the column space, which gives $A^\\top A \\hat w = A^\\top y$.",
        "Never invert $A^\\top A$ explicitly: it squares the condition number. Use QR/SVD (np.linalg.lstsq).",
      ],
      body: R`
<p>Almost every object in machine learning is a vector or a matrix: a data point, a weight matrix, a batch of embeddings, a gradient. If you can see these as <em>geometry</em> rather than as grids of numbers, most of the later derivations become obvious. This lesson builds that picture and ends with least squares, the first model you will implement (Week 2's <code>LinearRegression</code>).</p>

<h2>Vectors and the dot product</h2>
<p>A vector $x \in \mathbb{R}^d$ is an arrow from the origin, or equivalently a point. Two operations define almost everything: addition (tip to tail) and scaling. The <strong>dot product</strong> $a^\top b = \sum_i a_i b_i$ has a geometric meaning:</p>
$$a^\top b = \|a\|\,\|b\|\cos\theta.$$
<p>So $a^\top b / \|b\|$ is the length of $a$'s shadow on $b$'s direction. Two consequences you will use constantly:</p>
<ul>
<li><strong>Similarity.</strong> Cosine similarity $\frac{a^\top b}{\|a\|\|b\|}$ is how embedding models compare items. In Week 8 you L2-normalize embeddings so the dot product <em>is</em> the cosine.</li>
<li><strong>Orthogonality.</strong> $a^\top b = 0$ means "no shadow": the vectors carry independent information. Least squares below is nothing but an orthogonality condition.</li>
</ul>
<p>The <strong>projection</strong> of $a$ onto the line spanned by $b$ is $\mathrm{proj}_b(a) = \frac{a^\top b}{b^\top b}\, b$. The leftover $a - \mathrm{proj}_b(a)$ is orthogonal to $b$. Check it: $b^\top(a - \frac{a^\top b}{b^\top b} b) = a^\top b - a^\top b = 0$.</p>

<h2>A matrix is a linear map</h2>
<p>Write $A \in \mathbb{R}^{m\times n}$ by columns, $A = [a_1\ a_2\ \cdots\ a_n]$. Then</p>
$$Ax = x_1 a_1 + x_2 a_2 + \cdots + x_n a_n.$$
<p>Read that twice. <mark>$Ax$ is a weighted combination of the columns of $A$, with weights $x$.</mark> In particular $Ae_j = a_j$: column $j$ is where the $j$-th basis vector lands. Knowing where the basis goes tells you where everything goes, because the map is linear: $A(\alpha x + \beta y) = \alpha Ax + \beta Ay$.</p>
<p>The same product has a second reading, row by row: $(Ax)_i = r_i^\top x$, the dot product of row $i$ with $x$. A linear layer $y = Wx$ therefore computes $m$ "feature detectors" (rows of $W$), each measuring how aligned the input is with it. Both readings are correct; pick whichever makes the current problem easy.</p>
<p><strong>Matrix multiplication is composition.</strong> $(AB)x = A(Bx)$: first apply $B$, then $A$. That's why shapes must chain: $A\in\mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$ gives $AB\in\mathbb{R}^{m\times n}$, and why $AB \ne BA$ in general (rotating then scaling is not scaling then rotating). A deep network with no nonlinearity, $W_3W_2W_1x$, is just one linear map; the nonlinearities are what make depth worth anything.</p>

<div class="callout"><b>Shape discipline</b>In code you will almost always work with <em>batches as rows</em>: $X \in \mathbb{R}^{N\times d}$, one example per row. A linear layer is then $Y = XW^\top + b$ with $W\in\mathbb{R}^{d_{out}\times d_{in}}$ (PyTorch's convention, and yours in Week 3). Say the shapes out loud before writing any line: "$N{\times}d_{in}$ times $d_{in}{\times}d_{out}$ gives $N{\times}d_{out}$". Most bugs in this course will be shape bugs that numpy broadcasting silently "fixed".</div>

<h2>Column space, null space, rank</h2>
<p>The <strong>column space</strong> $\mathcal{C}(A) = \{Ax\}$ is the set of all outputs. The <strong>null space</strong> $\mathcal{N}(A) = \{x : Ax = 0\}$ is the set of inputs that get crushed to zero. The <strong>rank</strong> is the dimension of the column space: the number of linearly independent columns (which always equals the number of independent rows).</p>
<p>The rank–nullity theorem says $\mathrm{rank}(A) + \dim\mathcal{N}(A) = n$: every input dimension either survives into the output or gets crushed. Why you care:</p>
<ul>
<li>If your feature matrix has two collinear columns (say "price in USD" and "price in EUR"), $X$ is rank-deficient, the null space is nontrivial, and least squares has infinitely many solutions. Week 2's test <code>test_linear_regression_collinear_no_crash</code> checks exactly this.</li>
<li>A low-rank matrix $W = UV^\top$ with $U\in\mathbb{R}^{m\times r}$, $V\in\mathbb{R}^{n\times r}$ stores $r(m+n)$ numbers instead of $mn$. This is LoRA fine-tuning, and it's also what a two-tower recommender learns: a rank-$d$ factorization of the user×item matrix.</li>
</ul>

<h2>Least squares from a picture</h2>
<p>We want weights $w$ so that $Aw \approx y$, where $A \in \mathbb{R}^{N\times d}$ (data, $N \gg d$) and $y\in\mathbb{R}^N$ (targets). Usually no exact solution exists: $y$ is not in the column space. So we choose the point $A\hat w$ in the column space closest to $y$:</p>
$$\hat w = \arg\min_w \|y - Aw\|^2.$$
<p>Geometrically the closest point is the <strong>orthogonal projection</strong> of $y$ onto $\mathcal{C}(A)$: the residual $r = y - A\hat w$ must be perpendicular to every column of $A$. That's $d$ dot products equal to zero:</p>
$$A^\top (y - A\hat w) = 0 \quad\Longrightarrow\quad A^\top A\,\hat w = A^\top y.$$
<p>These are the <strong>normal equations</strong> ("normal" = perpendicular). If $A$ has full column rank, $A^\top A$ is invertible and $\hat w = (A^\top A)^{-1}A^\top y$; the projection matrix is $P = A(A^\top A)^{-1}A^\top$, which satisfies $P^2 = P$ (projecting twice changes nothing).</p>
<p>You'll derive the same equation with calculus in L3 (set the gradient $-2A^\top(y - Aw)$ to zero). Two routes, one answer: that's a good sign you understand it.</p>

<h3>The intercept and why centering works</h3>
<p>With an intercept, $\min_{w,b}\|y - Xw - b\mathbf{1}\|^2$. Setting the derivative in $b$ to zero gives $b = \bar y - \bar x^\top w$. Substitute back and the problem becomes plain least squares on centered data $X_c = X - \bar x^\top$, $y_c = y - \bar y$. That's the trick the Week 2 docstring hints at, and it's also why ridge regression doesn't penalize the intercept: shifting all targets by a constant shouldn't change the slope.</p>

<h3>Ridge regression</h3>
<p>Adding $\lambda\|w\|^2$ gives $(A^\top A + \lambda I)\hat w = A^\top y$. The $+\lambda I$ lifts every eigenvalue of $A^\top A$ by $\lambda$, so the system is always invertible and better conditioned. L4 shows it's also a Gaussian prior on $w$.</p>

<div class="callout pitfall"><b>Pitfall: don't form $(A^\top A)^{-1}$</b>The <em>condition number</em> $\kappa(A) = \sigma_{max}/\sigma_{min}$ measures how much relative error in the input can be amplified in the output. Forming $A^\top A$ squares it: $\kappa(A^\top A) = \kappa(A)^2$. With $\kappa(A)=10^6$ you lose about 12 of float64's 16 digits. <code>np.linalg.lstsq</code> uses an SVD-based solver that works on $A$ directly; <code>np.linalg.solve(A.T@A + lam*I, A.T@y)</code> is fine for ridge because $\lambda$ bounds the conditioning. <code>np.linalg.inv</code> is almost never the right call.</div>

<h2>Norms you'll meet</h2>
<table>
<tr><th>norm</th><th>formula</th><th>where it shows up</th></tr>
<tr><td>L2</td><td>$\|x\|_2 = \sqrt{\sum x_i^2}$</td><td>distances, weight decay, gradient clipping</td></tr>
<tr><td>L1</td><td>$\|x\|_1 = \sum|x_i|$</td><td>lasso (sparsity), robust losses</td></tr>
<tr><td>Frobenius</td><td>$\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2}$</td><td>matrix weight decay; $=\sqrt{\sum \sigma_i^2}$</td></tr>
<tr><td>spectral</td><td>$\|A\|_2 = \sigma_{max}$</td><td>how much a layer can stretch an input; Lipschitz bounds</td></tr>
</table>

<div class="callout prod"><b>In production</b>Retrieval systems (Week 8) are giant dot-product machines: score = user vector · item vector, for millions of items. Everything about ANN indexes comes down to computing a few dot products instead of all of them. And every "why is my model NaN" investigation starts by checking norms of activations and gradients layer by layer.</div>
`,
      quiz: [
        { q: "For $A\\in\\mathbb{R}^{3\\times 2}$ with columns $a_1, a_2$, what is $A\\begin{bmatrix}2\\\\-1\\end{bmatrix}$?", options: ["$2a_1 - a_2$", "the dot product of the first row with $(2,-1)$", "$2a_2 - a_1$", "It is undefined: shapes don't match"], answer: 0, why: "$Ax$ is the combination of columns weighted by $x$: $2a_1 + (-1)a_2$. (Each entry of the result is a row·x dot product; both views agree.)" },
        { q: "Your design matrix has 5 columns but one is exactly twice another. What is its rank (assuming the rest are independent), and what happens to least squares?", options: ["rank 5; unique solution", "rank 4; infinitely many minimizers, $X^\\top X$ is singular", "rank 3; no minimizer exists", "rank 4; the minimum error becomes larger"], answer: 1, why: "One column is redundant, so rank 4 and the null space is 1-dimensional: adding any null-space vector to a solution gives the same predictions. The minimum error is unchanged; only uniqueness is lost. lstsq returns the minimum-norm solution." },
        { q: "The least-squares residual $r = y - A\\hat w$ satisfies…", options: ["$r = 0$", "$A r = 0$", "$A^\\top r = 0$", "$r^\\top y = 0$"], answer: 2, why: "The residual is orthogonal to every column of $A$, i.e. $A^\\top r = 0$: these are the normal equations." },
        { q: "Why does ridge regression penalize $w$ but not the intercept $b$?", options: ["The intercept has no gradient", "Penalizing $b$ would make predictions depend on where the target's zero is, which is arbitrary", "It's a numpy limitation", "$b$ is always zero after centering"], answer: 1, why: "Shifting all targets by a constant (e.g. Celsius vs Kelvin offset) should shift predictions by the same constant. A penalty on $b$ would fight that shift." },
      ],
      explain: "Explain to a friend, without formulas first and then with them, why least squares leads to the equation A^T A w = A^T y.",
      resources: [
        { title: "3Blue1Brown — Essence of Linear Algebra (ch. 1–4, 7)", url: "https://www.3blue1brown.com/topics/linear-algebra", note: "the geometric picture, animated" },
        { title: "Deisenroth et al., Mathematics for Machine Learning, ch. 2–3", url: "https://mml-book.github.io/", note: "free PDF; ch. 3.8 is projections" },
        { title: "Strang, 18.06 lecture 15–16 (projections, least squares)", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", note: "" },
      ],
      cards: [
        { f: "What does column $j$ of a matrix $A$ represent geometrically?", b: "Where the $j$-th basis vector lands: $Ae_j = a_j$." },
        { f: "State the normal equations and the geometric condition behind them.", b: "$A^\\top A\\hat w = A^\\top y$: the residual $y - A\\hat w$ is orthogonal to the column space." },
        { f: "Why avoid forming $(A^\\top A)^{-1}$?", b: "It squares the condition number, losing roughly twice as many digits. Use lstsq/QR/SVD." },
        { f: "rank–nullity theorem", b: "$\\mathrm{rank}(A) + \\dim \\mathcal{N}(A) = $ number of columns." },
      ],
    },
    {
      id: "w1l2", title: "Eigenvectors, SVD and PCA", minutes: 80,
      summary: "Every matrix is a rotation, a stretch and a rotation. PCA is that fact applied to a data covariance.",
      keypoints: [
        "Eigenvector: a direction the map only scales, $Av = \\lambda v$. Symmetric matrices have orthonormal eigenvectors and real eigenvalues.",
        "SVD: $A = U\\Sigma V^\\top$ for ANY matrix; it maps the unit sphere to an ellipsoid with axes $\\sigma_i u_i$.",
        "Best rank-$k$ approximation = keep the top $k$ singular values (Eckart–Young).",
        "PCA directions maximize projected variance; they are the top eigenvectors of the covariance = right singular vectors of centered $X$.",
        "Explained variance of component $i$ is $\\sigma_i^2/(n-1)$.",
      ],
      body: R`
<p>Last lesson treated a matrix as a map. This lesson asks: <em>what does a map do, stripped to its essentials?</em> The answer, the singular value decomposition, underlies PCA (you'll implement it in Week 2), low-rank compression, the condition number that decides whether your solver is accurate, and the intuition for why deep networks' activations explode or vanish (Week 3).</p>

<h2>Eigenvectors: directions that don't turn</h2>
<p>For a square matrix $A$, a nonzero $v$ with $Av = \lambda v$ is an <strong>eigenvector</strong> with eigenvalue $\lambda$: along $v$ the map is just scaling. If $A$ has a full set of eigenvectors, $A = Q\Lambda Q^{-1}$, and then $A^k = Q\Lambda^kQ^{-1}$: repeated application is dominated by the largest $|\lambda|$. That is the entire story of exploding and vanishing signals in a deep linear network or an RNN: multiply by $W$ fifty times and every component along an eigenvector with $|\lambda| > 1$ blows up, every one with $|\lambda| < 1$ dies.</p>
<p><strong>Symmetric matrices are the friendly case</strong> (spectral theorem): if $S = S^\top$ then all eigenvalues are real and the eigenvectors can be chosen orthonormal, so $S = Q\Lambda Q^\top$ with $Q^\top Q = I$. Covariance matrices, Hessians and $A^\top A$ are all symmetric, which is why the spectral theorem appears everywhere in ML. If also all $\lambda_i \ge 0$, $S$ is <em>positive semi-definite</em>: $x^\top S x \ge 0$ for all $x$. Every covariance is PSD, because $x^\top \Sigma x = \mathrm{Var}(x^\top z) \ge 0$.</p>

<h2>The SVD: rotate, stretch, rotate</h2>
<p>Eigen-decompositions need square matrices and can be complex. The SVD works for any $A\in\mathbb{R}^{m\times n}$:</p>
$$A = U\Sigma V^\top,\qquad U^\top U = I,\ V^\top V = I,\ \Sigma = \mathrm{diag}(\sigma_1\ge\sigma_2\ge\cdots\ge 0).$$
<p>Read it right to left as a pipeline applied to $x$: $V^\top$ rotates (changes basis to the "input axes" $v_i$), $\Sigma$ stretches axis $i$ by $\sigma_i$, $U$ rotates into the "output axes" $u_i$. So <mark>every linear map sends the unit sphere to an ellipsoid whose semi-axes are $\sigma_i u_i$</mark>, and $Av_i = \sigma_i u_i$. Drag the sliders below and watch the singular vectors stay perpendicular on both sides.</p>
<div class="widget" data-widget="svd"></div>
<p>Connection to eigenvalues: $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V\Sigma^2V^\top$. So the $v_i$ are eigenvectors of $A^\top A$ with eigenvalues $\sigma_i^2$, and likewise $u_i$ for $AA^\top$. Facts that fall out immediately:</p>
<ul>
<li>rank$(A)$ = number of nonzero $\sigma_i$.</li>
<li>$\|A\|_2 = \sigma_1$ (the most any unit vector can be stretched) and $\|A\|_F^2 = \sum_i\sigma_i^2$.</li>
<li>condition number $\kappa = \sigma_1/\sigma_r$; $|\det A| = \prod\sigma_i$ for square $A$.</li>
<li>The pseudo-inverse $A^+ = V\Sigma^+U^\top$ (invert nonzero $\sigma$s) gives the minimum-norm least-squares solution $A^+y$: this is what <code>lstsq</code> computes when columns are collinear.</li>
</ul>

<h3>Low-rank approximation</h3>
<p>Write $A = \sum_i \sigma_i u_i v_i^\top$, a sum of rank-1 "layers" ordered by importance. The <strong>Eckart–Young theorem</strong>: truncating to the top $k$ terms, $A_k = \sum_{i\le k}\sigma_iu_iv_i^\top$, gives the best rank-$k$ approximation in both spectral and Frobenius norm, with error $\|A - A_k\|_F^2 = \sum_{i>k}\sigma_i^2$. Image compression, LoRA, and matrix-factorization recommenders are all this idea.</p>

<h2>PCA, derived</h2>
<p>Data $X\in\mathbb{R}^{n\times d}$, centered so each column has mean 0. Sample covariance $S = \frac{1}{n-1}X^\top X$. We want the unit direction $w$ along which the projected data $Xw$ has maximum variance:</p>
$$\max_{w}\ \frac{1}{n-1}\|Xw\|^2 = w^\top S w\quad\text{s.t. } w^\top w = 1.$$
<p>Lagrangian: $\mathcal{L} = w^\top Sw - \lambda(w^\top w - 1)$. Gradient: $2Sw - 2\lambda w = 0$, so $Sw = \lambda w$: <strong>$w$ must be an eigenvector of $S$</strong>, and the variance achieved is $w^\top S w = \lambda$. To maximize, take the eigenvector with the largest eigenvalue. The second component maximizes variance subject to being orthogonal to the first, which gives the second eigenvector, and so on.</p>
<p>Now use the SVD of the centered data, $X = U\Sigma V^\top$: then $S = V\frac{\Sigma^2}{n-1}V^\top$, so</p>
<ul>
<li>principal directions = rows of $V^\top$ (<code>components_</code>),</li>
<li>explained variance $\lambda_i = \sigma_i^2/(n-1)$ (<code>explained_variance_</code>, note the $n-1$),</li>
<li>the projected coordinates $XV_k = U_k\Sigma_k$.</li>
</ul>
<div class="callout pitfall"><b>Why SVD of $X$, not eigh of $X^\top X$</b>Same squaring problem as last lesson: forming $X^\top X$ squares the condition number, and tiny eigenvalues get swamped by rounding. The SVD of $X$ is computed stably. The Week 2 docstring asks you to "know why": this is why. (Also: always center first. PCA on uncentered data finds the direction of the mean, not of the variance.)</div>
<div class="callout pitfall"><b>Sign ambiguity</b>If $v$ is a singular vector, so is $-v$. Your PCA and sklearn's may differ by signs, so the Week 2 test compares "up to sign". Downstream code must never depend on the sign of a component.</div>

<div class="callout prod"><b>In production</b>PCA is still the first thing to try for visualizing embeddings, de-noising features, or compressing vectors before an ANN index (FAISS's OPQ/PCA transforms). And the singular values of your weight matrices are a cheap health check: a layer whose $\sigma_1$ is growing without bound during training is heading for divergence.</div>
`,
      quiz: [
        { q: "For $A = U\\Sigma V^\\top$, what is $Av_2$?", options: ["$\\sigma_2 v_2$", "$\\sigma_2 u_2$", "$u_2$", "$\\sigma_2^2 v_2$"], answer: 1, why: "$Av_i = U\\Sigma V^\\top v_i = U\\Sigma e_i = \\sigma_i u_i$. Input axis $v_i$ maps to output axis $u_i$ stretched by $\\sigma_i$." },
        { q: "The PCA optimization $\\max_{\\|w\\|=1} w^\\top S w$ is solved by…", options: ["the eigenvector of $S$ with the largest eigenvalue", "the column of $X$ with largest norm", "the mean of the data", "any vector with $Sw=0$"], answer: 0, why: "The Lagrange condition gives $Sw = \\lambda w$ and the objective equals $\\lambda$, so pick the top eigenvector." },
        { q: "Singular values of a centered $X$ ($n=101$ rows) are $[20, 10, 1]$. What fraction of total variance does the first component explain?", options: ["$20/31 \\approx 0.65$", "$400/501 \\approx 0.80$", "$20/101$", "$1/3$"], answer: 1, why: "Variances are $\\sigma_i^2/(n-1)$; the ratio is $400/(400+100+1)$. The $n-1$ cancels in the ratio." },
        { q: "A $50\\times 50$ weight matrix is applied 30 times (a deep linear net). Its largest singular value is 1.1. The output norm can grow by up to about…", options: ["$1.1\\times$", "$33\\times$", "$17\\times$", "$1.1^{30} \\approx 17\\times$, and it compounds per layer"], answer: 3, why: "Norms multiply: at most $\\sigma_1^{30} = 1.1^{30} \\approx 17.4$. This compounding is why initialization scale matters so much (Week 3)." },
      ],
      explain: "Explain what the SVD says a matrix does to the unit circle, and then explain why the principal components of a dataset are the right singular vectors of the centered data matrix.",
      resources: [
        { title: "Gilbert Strang — The SVD (MIT 18.065 lecture 6)", url: "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", note: "" },
        { title: "MML book ch. 4 (decompositions) and ch. 10 (PCA)", url: "https://mml-book.github.io/", note: "PCA derived three ways" },
        { title: "Shlens — A Tutorial on Principal Component Analysis", url: "https://arxiv.org/abs/1404.1100", note: "short and clear" },
      ],
      cards: [
        { f: "Geometric meaning of the SVD $A = U\\Sigma V^\\top$", b: "Rotate by $V^\\top$, stretch axis $i$ by $\\sigma_i$, rotate by $U$. Unit sphere → ellipsoid with semi-axes $\\sigma_iu_i$." },
        { f: "Relationship between singular values of $A$ and eigenvalues of $A^\\top A$", b: "$A^\\top A = V\\Sigma^2V^\\top$: eigenvalues are $\\sigma_i^2$, eigenvectors are the right singular vectors $v_i$." },
        { f: "Explained variance of principal component $i$ from the SVD of centered $X$ ($n$ rows)", b: "$\\sigma_i^2/(n-1)$" },
        { f: "Eckart–Young theorem", b: "Truncated SVD $A_k=\\sum_{i\\le k}\\sigma_iu_iv_i^\\top$ is the best rank-$k$ approximation; Frobenius error$^2 = \\sum_{i>k}\\sigma_i^2$." },
      ],
    },
    {
      id: "w1l3", title: "Matrix calculus and reverse-mode autodiff", minutes: 100,
      summary: "Gradients, Jacobians and the chain rule, then the algorithm that computes all of them at once: backprop.",
      keypoints: [
        "Convention: the gradient of a scalar w.r.t. a tensor has the tensor's shape.",
        "Linear layer $Y = XW$: $\\partial L/\\partial W = X^\\top G$ and $\\partial L/\\partial X = GW^\\top$ where $G = \\partial L/\\partial Y$. Check with shapes.",
        "Broadcasting in the forward pass becomes summation in the backward pass (unbroadcast).",
        "Softmax + cross-entropy: $\\partial L/\\partial z = p - y$ (per example; divide by $N$ for the mean).",
        "Reverse mode computes the gradient w.r.t. ALL inputs in one backward pass costing about as much as the forward pass; each node only needs its local vector-Jacobian product.",
      ],
      body: R`
<p>This is the lesson your whole project rests on. By the end you should be able to derive the backward pass of any operation you add to <code>forge.autograd</code>, and understand why the backward sweep must run in reverse topological order.</p>

<h2>Derivatives of vector functions</h2>
<p>For a scalar function $f:\mathbb{R}^n\to\mathbb{R}$, the <strong>gradient</strong> $\nabla f \in\mathbb{R}^n$ collects the partials $\partial f/\partial x_i$; to first order, $f(x+\delta) \approx f(x) + \nabla f^\top\delta$. It points in the direction of steepest increase. For a vector function $g:\mathbb{R}^n\to\mathbb{R}^m$, the <strong>Jacobian</strong> $J\in\mathbb{R}^{m\times n}$ has $J_{ij} = \partial g_i/\partial x_j$, and $g(x+\delta)\approx g(x) + J\delta$. The chain rule is Jacobians multiplying:</p>
$$\frac{\partial (f\circ g)}{\partial x} = \frac{\partial f}{\partial g}\,\frac{\partial g}{\partial x}\qquad(1\times m)\cdot(m\times n).$$

<h3>The one convention that prevents most bugs</h3>
<p>In ML the thing at the end is always a scalar loss $L$. We store $\bar X := \partial L/\partial X$ with <mark>the same shape as $X$</mark>, whatever $X$'s shape is. Every identity below can be sanity-checked by shapes alone: if your formula for $\bar W$ doesn't have $W$'s shape, it's wrong.</p>

<h2>Deriving the linear-layer gradients</h2>
<p>Take $Y = XW$ with $X\in\mathbb{R}^{N\times d}$, $W\in\mathbb{R}^{d\times k}$, $Y\in\mathbb{R}^{N\times k}$, and suppose we already have $G = \partial L/\partial Y$ (shape $N\times k$). Use index notation, which never lies:</p>
$$Y_{nk} = \sum_j X_{nj}W_{jk}\quad\Rightarrow\quad \frac{\partial L}{\partial W_{jk}} = \sum_{n}\frac{\partial L}{\partial Y_{nk}}\frac{\partial Y_{nk}}{\partial W_{jk}} = \sum_n G_{nk}X_{nj} = (X^\top G)_{jk}.$$
$$\frac{\partial L}{\partial X_{nj}} = \sum_k G_{nk}W_{jk} = (GW^\top)_{nj}.$$
<p>Shape check: $X^\top G$ is $(d\times N)(N\times k) = d\times k$ ✓, $GW^\top$ is $(N\times k)(k\times d) = N\times d$ ✓. A useful mnemonic: <em>the gradient of a matmul input is the upstream gradient multiplied by the other input, transposed, on the side that makes the shapes work.</em> For batched matmul $(\ldots,n,k)@(\ldots,k,m)$ the same formulas hold with <code>swapaxes(-1,-2)</code> as the transpose, plus unbroadcasting over batch dims.</p>

<h3>Elementwise ops are diagonal Jacobians</h3>
<p>For $y = \sigma(x)$ applied elementwise, the Jacobian is diagonal, so the vector-Jacobian product is just a multiply: $\bar x = \bar y \odot \sigma'(x)$. Useful derivatives: $\tanh' = 1-\tanh^2$, $\mathrm{sigmoid}' = s(1-s)$, $\exp' = \exp$, $\log' = 1/x$, $\mathrm{relu}' = \mathbb{1}[x>0]$. Notice several are cheapest written in terms of the <em>output</em>: save the output in the forward pass.</p>

<h3>Reductions and broadcasting are each other's adjoints</h3>
<p>If $s = \sum_i x_i$ then $\bar x_i = \bar s$ for every $i$: the backward of a <strong>sum is a broadcast</strong>. Conversely, if the forward pass broadcast a bias $b\in\mathbb{R}^{k}$ across $N$ rows ($Y = Z + b$), then $b$ influenced every row, and its gradient is the <strong>sum over the broadcast axis</strong>: $\bar b = \sum_n \bar Y_{n,:}$. This is <code>unbroadcast</code>, and you'll use it in every binary op. The rule: sum out leading axes that were added, then sum (keepdims) over axes where the original size was 1.</p>
<div class="callout pitfall"><b>The classic silent bug</b>Forget to unbroadcast and <code>b.grad</code> has shape $(N,k)$ instead of $(k,)$. numpy then happily broadcasts it in the optimizer update, and your model trains... slightly wrong. The Week 1 test checks <code>grad.shape == data.shape</code> for every op for exactly this reason.</div>

<h2>Softmax and cross-entropy</h2>
<p>Logits $z\in\mathbb{R}^K$, $p = \mathrm{softmax}(z)$, $p_i = e^{z_i}/\sum_j e^{z_j}$. Its Jacobian:</p>
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)\qquad\Longleftrightarrow\qquad J = \mathrm{diag}(p) - pp^\top.$$
<p>Cross-entropy with a one-hot target $y$: $L = -\sum_i y_i\log p_i = -\log p_c$. Rather than chaining through that Jacobian, use $\log p_i = z_i - \log\sum_j e^{z_j}$ (log-softmax):</p>
$$L = -z_c + \log\sum_j e^{z_j}\quad\Rightarrow\quad \frac{\partial L}{\partial z_j} = -\delta_{jc} + \frac{e^{z_j}}{\sum_k e^{z_k}} = p_j - y_j.$$
<p><mark>Prediction minus truth.</mark> For a batch mean, divide by $N$. This is tested explicitly (<code>test_softmax_cross_entropy_gradient_formula</code>) and you'll see the same $p-y$ in logistic regression, GBM classification residuals and the two-tower model.</p>
<p>For log-softmax as a standalone op, the backward is $\bar z = \bar\ell - p\,(\mathbf 1^\top\bar\ell)$ (per row), where $\ell = \log\mathrm{softmax}(z)$. Derive it: $\partial\ell_i/\partial z_j = \delta_{ij} - p_j$.</p>
<div class="callout pitfall"><b>Numerical stability: log-sum-exp</b>$e^{1000}$ overflows. Since softmax is shift-invariant, compute $\mathrm{lse}(z) = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$. Every exponent is then $\le 0$. The max is treated as a constant in the backward pass (the gradient of lse doesn't depend on $m$). The test feeds $[1000, 0, -1000]$.</div>

<h2>Reverse-mode autodiff</h2>
<p>A computation is a DAG: leaves (parameters, inputs), internal nodes (ops), one scalar output $L$. The chain rule for a node $v$ that feeds several children $c$:</p>
$$\bar v = \sum_{c\,\in\,\text{children}(v)} \bar c\,\frac{\partial c}{\partial v}.$$
<p>So <strong>gradients accumulate</strong> (<code>+=</code>) when a value is used more than once, as in <code>x*x + x</code>. And $\bar v$ can only be computed once all its children's $\bar c$ are final: we must visit nodes in <strong>reverse topological order</strong>. Step through it:</p>
<div class="widget" data-widget="backprop"></div>
<h3>Why reverse and not forward?</h3>
<p><em>Forward mode</em> pushes a tangent $\dot x$ forward (Jacobian-vector products) and gives the derivative of all outputs with respect to one input direction per pass. <em>Reverse mode</em> pulls an adjoint $\bar L = 1$ backward (vector-Jacobian products) and gives the derivative of one output with respect to all inputs per pass. A loss has one output and millions of parameters, so reverse mode wins by a factor of millions. Its cost: it must store the forward values it needs (the "activation memory" that dominates GPU memory in training).</p>
<p>Each op therefore needs exactly two things: its forward computation, and a closure computing its VJP, i.e. given $\bar{\text{out}}$, add the contribution to each input's grad. That closure is the <code>_backward</code> in your Tensor.</p>

<h3>Topological order without recursion</h3>
<p>The textbook implementation (micrograd) builds the order with a recursive DFS. Python's default recursion limit is 1000, and a 5,000-step chain (an RNN, a long training graph) crashes it; the Week 1 test does exactly this. Use an explicit stack: push <code>(node, expanded=False)</code>; when popped unexpanded, push it back as expanded then push its unvisited parents; when popped expanded, append it to the order. That's a post-order DFS, which is a valid topological order.</p>

<pre><code># pseudocode — you write the real thing
order = post_order_dfs(root)          # parents before children
seed root.grad = 1
for node in reversed(order):          # children before parents
    node.backward_fn()                # pushes node.grad into parents (+=)</code></pre>

<div class="callout prod"><b>In production</b>PyTorch's autograd is this algorithm with a C++ engine, a thread pool and one VJP per op (<code>derivatives.yaml</code>). Every "custom op" in real ML codebases (FlashAttention, fused kernels) is a forward plus a hand-derived backward, and the first thing reviewers ask for is a gradcheck. You're learning the exact skill.</div>
`,
      quiz: [
        { q: "$Y = XW$ with $X: 32\\times 10$, $W: 10\\times 4$. Upstream $G = \\partial L/\\partial Y$ has shape $32\\times 4$. Which is $\\partial L/\\partial W$?", options: ["$GX^\\top$", "$X^\\top G$", "$G^\\top X$", "$W^\\top G$"], answer: 1, why: "$X^\\top G$ is $(10\\times 32)(32\\times 4) = 10\\times 4$ = shape of $W$. $G^\\top X$ is $4\\times 10$, the transpose, a classic mistake." },
        { q: "Forward: $Y = Z + b$ where $Z: N\\times k$ and $b: (k,)$. The gradient of $b$ is…", options: ["$\\bar Y$", "$\\bar Y$ summed over axis 0", "$\\bar Y$ averaged over axis 0", "$\\bar Y[0]$"], answer: 1, why: "$b$ was broadcast across $N$ rows and influenced each, so contributions add: sum over the broadcast axis (unbroadcast)." },
        { q: "Mean cross-entropy over a batch of $N$ with logits $Z$ and one-hot $Y$. $\\partial L/\\partial Z = $", options: ["$P - Y$", "$(P - Y)/N$", "$-Y/P$", "$(Y - P)/N$"], answer: 1, why: "Per example the gradient is $p - y$; the mean divides by $N$." },
        { q: "In <code>y = x*x + x</code>, why must the backward pass use <code>+=</code> into <code>x.grad</code>?", options: ["For numerical stability", "Because $x$ feeds three edges; the total derivative is the sum over all paths", "Because numpy requires in-place ops", "It shouldn't; use assignment"], answer: 1, why: "Multivariable chain rule: sum contributions from every use of $x$. Assignment would keep only the last one. The correct gradient is $2x + 1$." },
        { q: "Why is reverse mode preferred over forward mode for training?", options: ["It uses less memory", "One backward pass gives the gradient w.r.t. all parameters of a scalar loss", "It's more numerically stable", "Forward mode can't handle matmul"], answer: 1, why: "Reverse mode cost is ~one pass per OUTPUT; a loss is one scalar. Forward mode needs one pass per INPUT direction. (Reverse mode actually uses MORE memory: it stores activations.)" },
      ],
      explain: "Explain, as if teaching someone, why the gradient of the loss w.r.t. softmax logits is p - y, and why the backward pass must visit nodes in reverse topological order.",
      resources: [
        { title: "Karpathy — The spelled-out intro to neural networks and backprop (micrograd)", url: "https://www.youtube.com/watch?v=VMj-3S1tku0", note: "scalar version of this week's project" },
        { title: "CS231n — Backpropagation notes & 'Derivatives, Backpropagation, and Vectorization'", url: "https://cs231n.stanford.edu/handouts/derivatives.pdf", note: "matrix gradients carefully" },
        { title: "Baydin et al. — Automatic Differentiation in Machine Learning: a Survey", url: "https://arxiv.org/abs/1502.05767", note: "forward vs reverse mode" },
        { title: "The Matrix Cookbook", url: "https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf", note: "reference, not reading" },
      ],
      cards: [
        { f: "$Y = XW$. Give $\\bar X$ and $\\bar W$ in terms of $G = \\bar Y$.", b: "$\\bar X = GW^\\top$, $\\bar W = X^\\top G$." },
        { f: "Jacobian of softmax", b: "$\\partial p_i/\\partial z_j = p_i(\\delta_{ij} - p_j)$, i.e. $\\mathrm{diag}(p) - pp^\\top$." },
        { f: "Gradient of softmax cross-entropy w.r.t. logits", b: "$p - y$ (divide by $N$ for the batch mean)." },
        { f: "What is the backward of a broadcast? Of a sum?", b: "Broadcast ↔ sum: the backward of broadcasting is summing over the broadcast axes; the backward of a sum is broadcasting." },
        { f: "Stable log-sum-exp", b: "$m + \\log\\sum_j e^{z_j - m}$ with $m = \\max_j z_j$." },
      ],
    },
    {
      id: "w1l4", title: "Probability: likelihood, entropy and KL", minutes: 75,
      summary: "Why MSE and cross-entropy are the losses they are, what regularization assumes, and the information-theory vocabulary of modern ML.",
      keypoints: [
        "Maximum likelihood = minimize negative log-likelihood (NLL).",
        "Gaussian noise ⇒ NLL is MSE; Bernoulli/categorical ⇒ NLL is (binary) cross-entropy.",
        "MAP with a Gaussian prior on weights = L2 regularization; Laplace prior = L1.",
        "Cross-entropy $H(p,q) = H(p) + \\mathrm{KL}(p\\|q)$, so minimizing CE over $q$ minimizes KL.",
        "KL is ≥ 0, zero iff $p=q$, and not symmetric.",
      ],
      body: R`
<p>Loss functions are not arbitrary. Each standard loss is the negative log-likelihood of a probabilistic model, and each regularizer is a prior. Knowing which model you're implicitly assuming tells you when a loss is wrong for your data.</p>

<h2>The minimum probability you need</h2>
<ul>
<li><strong>Expectation</strong> $\mathbb{E}[X] = \sum_x x\,p(x)$ is linear: $\mathbb{E}[aX+bY] = a\mathbb{E}X + b\mathbb{E}Y$ always, independence not required.</li>
<li><strong>Variance</strong> $\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}X^2 - \mu^2$. For independent $X,Y$: $\mathrm{Var}(X+Y) = \mathrm{Var}X + \mathrm{Var}Y$. So a sum of $n$ independent unit-variance terms has variance $n$: this one fact explains Xavier/He initialization (Week 3) and the $\sqrt{d}$ in attention (Week 4).</li>
<li><strong>Bayes</strong>: $p(\theta\mid D) = \dfrac{p(D\mid\theta)\,p(\theta)}{p(D)}$, i.e. posterior ∝ likelihood × prior.</li>
<li><strong>Distributions</strong>: Bernoulli$(\pi)$, Categorical$(p_1..p_K)$, Gaussian $\mathcal{N}(\mu,\sigma^2)$ with $\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} - \frac12\log(2\pi\sigma^2)$.</li>
</ul>

<h2>Maximum likelihood estimation</h2>
<p>Given i.i.d. data $D = \{(x_i,y_i)\}$ and a model $p_\theta(y\mid x)$, MLE picks $\theta$ that makes the observed data most probable. Products of tiny probabilities underflow, and log is monotone, so we minimize the <strong>negative log-likelihood</strong>:</p>
$$\hat\theta = \arg\min_\theta\ -\sum_i \log p_\theta(y_i\mid x_i).$$

<h3>MSE is Gaussian MLE</h3>
<p>Assume $y = f_\theta(x) + \varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. Then</p>
$$-\log p_\theta(y\mid x) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \tfrac12\log(2\pi\sigma^2).$$
<p>With $\sigma$ fixed, the second term is a constant and the first is squared error: <mark>minimizing MSE = maximum likelihood under Gaussian noise.</mark> This tells you when MSE is a bad choice: heavy-tailed noise (outliers dominate the square; a Laplace noise model gives L1/MAE instead) or targets that are counts, probabilities or strictly positive.</p>

<h3>Cross-entropy is Bernoulli / categorical MLE</h3>
<p>Binary: the model outputs $\pi = \sigma(z)$ and $y\in\{0,1\}$ with $p(y) = \pi^y(1-\pi)^{1-y}$:</p>
$$-\log p(y) = -\big[y\log\pi + (1-y)\log(1-\pi)\big]\qquad\text{(log-loss / BCE)}.$$
<p>Multiclass: $p(y=c) = \mathrm{softmax}(z)_c$, so $-\log p(y) = -\log\mathrm{softmax}(z)_y$, exactly the loss from L3, whose gradient is $p - y$. Logistic regression (Week 2), your CNN (Week 3) and your GPT's next-token loss (Week 4) are all this one formula.</p>

<h2>MAP: regularization is a prior</h2>
<p>Maximum a posteriori adds $\log p(\theta)$: minimize $-\sum_i\log p(y_i\mid x_i,\theta) - \log p(\theta)$. With a Gaussian prior $w_j\sim\mathcal{N}(0,\tau^2)$, $-\log p(w) = \frac{1}{2\tau^2}\|w\|^2 + \text{const}$: <strong>L2 regularization</strong> with $\lambda = \sigma^2/\tau^2$ (for the MSE case). A Laplace prior $p(w)\propto e^{-|w|/b}$ gives <strong>L1</strong>, whose sharp peak at zero is why lasso produces exact zeros. A stronger prior (smaller $\tau$) means more regularization; more data makes the likelihood term dominate and the prior matter less.</p>

<h2>Information theory in four definitions</h2>
<p><strong>Entropy</strong> $H(p) = -\sum_x p(x)\log p(x)$: the expected surprise, the average number of nats (bits, with $\log_2$) needed to encode samples of $p$ with an optimal code. Uniform over $K$ outcomes gives the maximum, $\log K$. At initialization your GPT should output nearly uniform predictions, so its loss should be $\approx\log(\text{vocab size})$: the Week 4 test <code>test_forward_shapes_and_initial_loss</code> checks exactly that.</p>
<p><strong>Cross-entropy</strong> $H(p,q) = -\sum_x p(x)\log q(x)$: the average code length if data come from $p$ but you use a code designed for $q$.</p>
<p><strong>KL divergence</strong> $\mathrm{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)$: the extra cost of using the wrong distribution. Properties: $\mathrm{KL}\ge 0$ (Jensen's inequality on the convex $-\log$), equality iff $p=q$, and <em>not symmetric</em>.</p>
$$H(p,q) = H(p) + \mathrm{KL}(p\|q).$$
<p>Since $H(p)$ of the data doesn't depend on the model, <mark>minimizing cross-entropy over $q$ is minimizing $\mathrm{KL}(p_{\text{data}}\|q_\theta)$</mark>, which is the same as MLE. Three names, one objective.</p>
<div class="callout"><b>Perplexity</b>Language models report $\exp(\text{mean CE})$. A perplexity of 4.5 means the model is, on average, as uncertain as a uniform choice among 4.5 tokens. Your char-level Shakespeare GPT will reach a validation loss near 1.5 nats, perplexity $e^{1.5}\approx 4.5$.</div>
<p>Where the asymmetry of KL matters: $\mathrm{KL}(p\|q)$ (forward) punishes $q$ for putting low mass where $p$ has mass, so $q$ spreads to cover every mode ("mean-seeking"). $\mathrm{KL}(q\|p)$ (reverse, used in variational inference and RLHF's KL penalty) punishes $q$ for mass where $p$ has little, so $q$ picks one mode ("mode-seeking").</p>

<div class="callout prod"><b>In production</b>Drift monitoring (Week 6) uses PSI, which is a symmetrized KL between binned feature distributions: $\sum(a-e)\ln(a/e) = \mathrm{KL}(a\|e) + \mathrm{KL}(e\|a)$. Calibration, label smoothing, distillation (KL between teacher and student) and the "logQ correction" in Week 8 all live in this vocabulary.</div>
`,
      quiz: [
        { q: "Your regression targets have rare, huge outliers. Under the likelihood view, switching from MSE to MAE (L1) corresponds to assuming noise that is…", options: ["Gaussian with bigger variance", "Laplace (heavier tails)", "Uniform", "Bernoulli"], answer: 1, why: "$-\\log$ of a Laplace density is $|y-f|/b$ + const. Heavier tails make big residuals less surprising, so they don't dominate the fit." },
        { q: "L2 weight decay $\\lambda\\|w\\|^2$ corresponds to what prior?", options: ["Uniform", "Laplace", "Zero-mean Gaussian", "Bernoulli"], answer: 2, why: "$-\\log\\mathcal{N}(w;0,\\tau^2) = \\|w\\|^2/(2\\tau^2)$ + const." },
        { q: "A freshly initialized 65-character language model has loss 4.17 nats. Is that plausible?", options: ["No, it should be 0", "Yes: $\\ln 65 \\approx 4.17$, the loss of a uniform prediction", "No, it should be $\\log_2 65$", "Only if the data are uniform"], answer: 1, why: "Near-uniform outputs give CE $= \\ln K$. A loss far above $\\ln K$ at init means the logits are too large (bad init scale)." },
        { q: "Which is always true?", options: ["$\\mathrm{KL}(p\\|q) = \\mathrm{KL}(q\\|p)$", "$H(p,q) \\ge H(p)$", "$H(p) \\ge H(p,q)$", "KL can be negative for continuous distributions"], answer: 1, why: "$H(p,q) = H(p) + \\mathrm{KL}(p\\|q)$ and KL ≥ 0. KL is not symmetric, and it's non-negative for continuous distributions too." },
      ],
      explain: "Explain why training a classifier with cross-entropy is the same as maximum likelihood, and also the same as minimizing KL divergence to the data distribution.",
      resources: [
        { title: "MML book ch. 6 (probability) and 8.3 (MLE, MAP)", url: "https://mml-book.github.io/", note: "" },
        { title: "Chris Olah — Visual Information Theory", url: "https://colah.github.io/posts/2015-09-Visual-Information/", note: "entropy and KL with pictures" },
        { title: "Bishop — Pattern Recognition and ML, §1.2 and §1.6", url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", note: "free PDF" },
      ],
      cards: [
        { f: "Which noise model makes MSE the maximum-likelihood loss?", b: "Additive Gaussian noise with fixed variance." },
        { f: "MAP with a Laplace prior gives which regularizer?", b: "L1 (lasso)." },
        { f: "Decompose cross-entropy $H(p,q)$", b: "$H(p) + \\mathrm{KL}(p\\|q)$." },
        { f: "Expected initial loss of a K-class classifier with near-uniform outputs", b: "$\\ln K$ nats." },
      ],
    },
    {
      id: "w1l5", title: "Optimization: GD, momentum, Adam", minutes: 85,
      summary: "Why gradient descent zig-zags on ill-conditioned problems, how momentum and adaptive methods fix it, and the exact update rules you'll implement.",
      keypoints: [
        "GD is stable on a quadratic only for $\\eta < 2/\\lambda_{max}$; progress along flat directions is set by $\\lambda_{min}$, so speed ∝ $1/\\kappa$.",
        "Momentum averages gradients: oscillating components cancel, consistent components accumulate.",
        "Adam = momentum on $g$ + per-parameter scaling by $\\sqrt{\\mathbb{E}g^2}$, with bias correction for the zero init.",
        "AdamW decouples weight decay from the adaptive scaling; L2-in-the-loss is NOT equivalent under Adam.",
        "Warmup then cosine decay is the default schedule for transformers.",
      ],
      body: R`
<p>You'll implement four optimizers this week and use AdamW to train a GPT in Week 4. This lesson explains what each term in their update rules is for, using the one model problem that captures most of the behavior: a quadratic bowl.</p>

<h2>Gradient descent on a quadratic</h2>
<p>Near a minimum, any smooth loss looks like $L(w) \approx \frac12 (w-w^*)^\top H (w-w^*)$, with $H$ the Hessian (symmetric, PSD). Diagonalize $H = Q\Lambda Q^\top$ and work in eigen-coordinates $u = Q^\top(w - w^*)$; the loss decouples into independent 1-D problems $\frac12\lambda_i u_i^2$. Gradient descent $w\leftarrow w - \eta\nabla L$ becomes</p>
$$u_i \leftarrow (1 - \eta\lambda_i)\,u_i.$$
<p>Each coordinate shrinks by the factor $|1-\eta\lambda_i|$ per step. Two consequences:</p>
<ul>
<li><strong>Stability:</strong> need $|1-\eta\lambda_i| < 1$ for all $i$, i.e. $\eta < 2/\lambda_{max}$. Above that, the steepest direction oscillates with growing amplitude and training diverges (loss → NaN).</li>
<li><strong>Speed:</strong> with the largest safe $\eta\approx 1/\lambda_{max}$, the flattest direction shrinks by $1 - \lambda_{min}/\lambda_{max} = 1 - 1/\kappa$ per step. Ill-conditioned problems ($\kappa\gg 1$) need $O(\kappa)$ steps: the path zig-zags across the steep valley walls while crawling along the valley floor.</li>
</ul>
<div class="widget" data-widget="gd"></div>
<p>Try it: on the bowl ($\kappa = 12$), plain GD with lr near $10^{-1}$ creeps; push lr past $2/12\approx 0.17$ and it diverges. Momentum at the same lr converges in a fraction of the steps. On Rosenbrock, try Adam.</p>
<p>This is also why feature standardization matters for linear models (Week 2): features on wildly different scales make $X^\top X$ ill-conditioned, and GD crawls.</p>

<h2>Momentum</h2>
<p>Heavy-ball momentum keeps a velocity: $v\leftarrow\beta v + g$, $w\leftarrow w - \eta v$ (PyTorch's convention, which the Week 1 test uses; on step 1 $v = g$). Unrolled, $v_t = \sum_k \beta^k g_{t-k}$: an exponentially weighted sum of past gradients. Components that flip sign each step (the zig-zag) cancel; components that point the same way every step accumulate, up to $1/(1-\beta)$ times larger (10× for $\beta=0.9$). On a quadratic, well-tuned momentum needs $O(\sqrt\kappa)$ steps instead of $O(\kappa)$.</p>
<p><strong>Weight decay in SGD</strong> is $g\leftarrow g + \lambda w$ before the momentum update. For plain SGD this is exactly the gradient of $\frac\lambda2\|w\|^2$.</p>

<h2>Adam, term by term</h2>
<p>Adam (Kingma & Ba, 2015) keeps two exponential moving averages per parameter:</p>
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad w \leftarrow w - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
<ul>
<li>$m$ is momentum (a running mean of the gradient).</li>
<li>$v$ is a running mean of the <em>squared</em> gradient; dividing by $\sqrt v$ normalizes each parameter's step to roughly $\eta$ regardless of its gradient scale. It's a cheap diagonal approximation of preconditioning by curvature, which attacks the conditioning problem per coordinate.</li>
<li><strong>Bias correction.</strong> Both averages start at 0, so early on they underestimate: $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$ if $g$ is stationary. Dividing by $1-\beta^t$ removes that bias. Without it, the first steps would be tiny for $m$ and, worse, $v$ would be ~1000× too small with $\beta_2 = 0.999$, making early steps huge.</li>
<li>$t$ is the step count, shared by all parameters, incremented once per <code>step()</code>.</li>
</ul>
<div class="callout pitfall"><b>Why AdamW exists</b>Adding $\frac\lambda2\|w\|^2$ to the loss puts $\lambda w$ into $g$, which then gets divided by $\sqrt{v}$: parameters with large gradients are decayed less. That's not what "shrink weights toward zero" should mean. Loshchilov & Hutter's AdamW applies decay directly to the weights, outside the adaptive scaling: $w\leftarrow w - \eta\lambda w$, then the Adam step. It regularizes better and the optimal $\lambda$ becomes independent of $\eta$. Your <code>AdamW</code> does the decay first, then the Adam update, matching PyTorch. And in Week 4 you'll exclude biases and LayerNorm gains from decay.</div>

<h2>Stochastic gradients, batch size, schedules</h2>
<p>With mini-batches, $g$ is a noisy but unbiased estimate of the full gradient; its variance shrinks like $1/B$. Some noise helps generalization and escapes saddle points; too much stalls convergence near the minimum, which is why we <strong>decay the learning rate</strong>. The modern default for transformers:</p>
<ul>
<li><strong>Linear warmup</strong> for the first few hundred steps: Adam's $v$ estimates are unreliable early and the network's early gradients are large, so start small.</li>
<li><strong>Cosine decay</strong> from the peak to a floor: $\eta_t = \eta_{min} + \frac12(1+\cos(\pi r))(\eta_{max}-\eta_{min})$ where $r$ is the fraction of the decay phase completed. You implement exactly this in Week 4's <code>lr_at</code>.</li>
</ul>
<p><strong>Gradient clipping</strong> (rescale $g$ if $\|g\| > c$) guards against rare huge gradients (a bad batch, attention spikes). It's in your Week 4 training loop.</p>
<h3>Convexity, briefly</h3>
<p>$f$ is convex if $f(\theta x + (1-\theta)y)\le\theta f(x) + (1-\theta)f(y)$; equivalently its Hessian is PSD everywhere. Then every local minimum is global, and GD with a suitable step converges. Linear and logistic regression (Week 2) are convex. Neural networks are not, yet GD works remarkably well on them; most critical points in high dimensions are saddles, not bad local minima, and SGD noise helps escape them.</p>

<div class="callout prod"><b>In production</b>When a training run diverges, the checklist is this lesson in reverse: learning rate too high for the current curvature (add warmup, lower peak lr), a gradient spike (clip), bad conditioning (normalize inputs, use LayerNorm), or a numerical overflow (log-sum-exp, bf16 range). Watching the gradient norm curve is the fastest diagnostic you have.</div>
`,
      quiz: [
        { q: "On $L = \\frac12(w_1^2 + 100w_2^2)$, what is the largest learning rate for which plain GD converges?", options: ["$2$", "$0.02$", "$0.01$", "$0.2$"], answer: 1, why: "Need $\\eta < 2/\\lambda_{max} = 2/100$." },
        { q: "In Adam, what goes wrong if you drop the bias correction on $v$ with $\\beta_2 = 0.999$?", options: ["Nothing important", "Early $v$ is ~1000× too small, so early steps are ~30× too large", "Early steps are too small", "Momentum stops working"], answer: 1, why: "$v_1 = 0.001g^2$, so $\\sqrt{v_1} \\approx 0.03|g|$ and the step is $\\eta m/\\sqrt v \\approx 3\\eta$ (with the $m$ bias also uncorrected), i.e. far larger than intended. Correction makes the first step ≈ $\\eta\\,\\mathrm{sign}(g)$." },
        { q: "Why is L2-in-the-loss not the same as weight decay under Adam?", options: ["It is the same", "The $\\lambda w$ term gets divided by $\\sqrt{v}$, so decay strength varies per parameter", "Adam ignores the loss term", "Weight decay only applies to biases"], answer: 1, why: "Coupled decay is rescaled by the adaptive denominator; AdamW applies $w \\mathrel{-}= \\eta\\lambda w$ separately." },
        { q: "Momentum with $\\beta = 0.9$ and a constant gradient $g$: the velocity converges to…", options: ["$g$", "$0.9g$", "$10g$", "$g/0.9$"], answer: 2, why: "$v = \\beta v + g$ has fixed point $v = g/(1-\\beta) = 10g$." },
      ],
      explain: "Explain to a teammate why gradient descent zig-zags in a long narrow valley, and how momentum and Adam each address it differently.",
      resources: [
        { title: "Goh — Why Momentum Really Works (Distill)", url: "https://distill.pub/2017/momentum/", note: "interactive; read the quadratic analysis" },
        { title: "Kingma & Ba — Adam", url: "https://arxiv.org/abs/1412.6980", note: "Algorithm 1 is one page" },
        { title: "Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)", url: "https://arxiv.org/abs/1711.05101", note: "" },
        { title: "PyTorch docs — torch.optim.SGD / Adam / AdamW pseudocode", url: "https://pytorch.org/docs/stable/optim.html", note: "the exact semantics the tests use" },
      ],
      cards: [
        { f: "Stability condition for GD on a quadratic with Hessian $H$", b: "$\\eta < 2/\\lambda_{max}(H)$" },
        { f: "Adam update with bias correction", b: "$m=\\beta_1m+(1-\\beta_1)g$, $v=\\beta_2v+(1-\\beta_2)g^2$, $\\hat m=m/(1-\\beta_1^t)$, $\\hat v=v/(1-\\beta_2^t)$, $w\\mathrel{-}=\\eta\\hat m/(\\sqrt{\\hat v}+\\epsilon)$" },
        { f: "What does AdamW change relative to Adam + L2?", b: "Decay is applied directly, $w\\mathrel{-}=\\eta\\lambda w$, outside the adaptive $1/\\sqrt{v}$ scaling." },
        { f: "Why warm up the learning rate?", b: "Early gradients are large and Adam's second-moment estimates are unreliable; small early steps avoid divergence." },
      ],
    },
  ],
  project: {
    title: "forge.autograd — a reverse-mode autodiff engine",
    dir: "forge/autograd.py, forge/optim.py, forge/gradcheck.py",
    pitch: "Build the engine that everything else in the course runs on: a numpy Tensor that records a graph and backpropagates through broadcasting, matmuls, reductions, fancy indexing and a stable log-softmax, plus gradcheck and four PyTorch-exact optimizers.",
    test: "pytest tests/w1 -q",
    overview: R`
<p>Read the module docstring of <code>forge/autograd.py</code> first: it is the contract the tests rely on. Then work milestone by milestone, running only the relevant tests with <code>-k</code>. The full grader has 61 tests; the reference solution passes all of them in about 1.5 seconds.</p>
<p><strong>Design you'll need to settle early:</strong> each op creates an output Tensor that stores its parents (<code>_prev</code>) and a closure (<code>_backward</code>) that, when called, adds the op's vector-Jacobian product into each parent's <code>.grad</code>. Write a small helper that does "make output, attach parents and closure only if grad mode is on and some parent requires grad". Write another that accumulates a gradient into a tensor after unbroadcasting it. Every op then becomes 5–10 lines.</p>
<p>Leaf vs interior: <code>.grad</code> must accumulate across <code>backward()</code> calls on leaves (the optimizer zeroes it), but interior nodes should start fresh each backward. Decide how you'll handle that before you write <code>backward()</code>.</p>
`,
    milestones: [
      { id: "w1m1", core: true, title: "Tensor basics, add/mul, and an iterative backward()", test: "pytest tests/w1/test_autograd.py -q -k \"dtype or scalar or reused or diamond or accumulate or leaf or nonscalar or deep\"",
        detail: R`<p>Constructor dtype rules, <code>__add__/__mul__</code> and their reflected versions, <code>__neg__/__sub__/__truediv__/__pow__</code>, and <code>backward()</code> with a non-recursive topological sort. The deep-graph test builds a 10,000-node chain.</p>`,
        hints: [
          R`<p>Express <code>-x</code>, <code>x - y</code> and <code>x / y</code> in terms of <code>*</code>, <code>+</code> and <code>**</code>. Fewer backward rules means fewer bugs.</p>`,
          R`<p>Iterative post-order DFS: a stack of <code>(node, expanded)</code> pairs plus a visited set keyed on <code>id(node)</code>. Pop; if expanded, append to order; otherwise push <code>(node, True)</code> then each unvisited parent with <code>False</code>.</p>`,
          R`<p>In <code>backward()</code>: build the order; reset interior nodes' <code>.grad</code> to None; seed the root; then <code>for node in reversed(order): if node._backward and node.grad is not None: node._backward()</code>. The closure for <code>a*b</code> accumulates <code>out.grad * b.data</code> into <code>a</code> and <code>out.grad * a.data</code> into <code>b</code>.</p>`,
        ] },
      { id: "w1m2", core: true, title: "Broadcasting and unbroadcast", test: "pytest tests/w1/test_autograd.py -q -k \"unbroadcast or bcast\"",
        detail: R`<p>Implement <code>unbroadcast(grad, shape)</code> and route every binary op's gradient through it. The test checks the adjoint identity $\langle g, \mathrm{broadcast}(v)\rangle = \langle\mathrm{unbroadcast}(g), v\rangle$.</p>`,
        hints: [
          R`<p>Two phases: first remove extra leading axes (numpy prepends 1s when ranks differ), then handle axes where the target size is 1 but the gradient's isn't.</p>`,
          R`<p>Phase 1: <code>while grad.ndim > len(shape): grad = grad.sum(axis=0)</code>. Phase 2: for each axis <code>i</code> with <code>shape[i] == 1</code> and <code>grad.shape[i] != 1</code>, sum with <code>keepdims=True</code>.</p>`,
          R`<p>Finish with <code>grad.reshape(shape)</code> so the <code>shape == ()</code> case returns a 0-d array. Call it inside your accumulate helper, not in each op.</p>`,
        ] },
      { id: "w1m3", core: true, title: "matmul, reductions, reshape/transpose, fancy indexing", test: "pytest tests/w1/test_autograd.py -q -k op_gradients",
        detail: R`<p>The parametrized <code>test_op_gradients</code> covers 28 cases, including batched matmul, <code>sum</code> over tuples of negative axes, and repeated fancy indices.</p>`,
        hints: [
          R`<p>matmul backward: $\bar A = G B^\top$, $\bar B = A^\top G$ with <code>np.swapaxes(x, -1, -2)</code> as the transpose. For batched cases the gradient for a 2-D operand comes out with a batch axis: your unbroadcast handles it.</p>`,
          R`<p>sum backward: if <code>keepdims=False</code>, re-insert the reduced axes with <code>np.expand_dims</code> (normalize negative axes first), then <code>np.broadcast_to(g, self.shape)</code>. mean = sum times 1/count. transpose backward uses the inverse permutation <code>np.argsort(axes)</code>.</p>`,
          R`<p>getitem backward: <code>full = np.zeros_like(self.data); np.add.at(full, idx, g)</code>. Plain <code>full[idx] += g</code> is wrong when indices repeat: it applies only one of the duplicate updates.</p>`,
        ] },
      { id: "w1m4", core: true, title: "exp/log/relu/tanh, stable sigmoid and log_softmax", test: "pytest tests/w1/test_autograd.py -q -k \"stability or softmax or mlp\"",
        detail: R`<p>No overflow warnings for inputs of ±1000. The MLP end-to-end test is your first real network's gradient.</p>`,
        hints: [
          R`<p>Stable sigmoid: for $x \ge 0$ use $1/(1+e^{-x})$; for $x<0$ use $e^{x}/(1+e^{x})$. <code>np.where</code> evaluates both branches, so compute $e^{-|x|}$ once and build both formulas from it.</p>`,
          R`<p>log_softmax forward: subtract the row max (as a constant), then subtract <code>log(sum(exp(...)))</code>. Backward per row: $\bar z = \bar\ell - p\sum_j\bar\ell_j$ where $p = e^{\ell}$.</p>`,
          R`<p>Implement log_softmax as ONE op with its own backward (not composed from exp/sum/log). It's faster and it's where stability lives.</p>`,
        ] },
      { id: "w1m5", core: true, title: "no_grad and detach", test: "pytest tests/w1/test_autograd.py -q -k \"no_grad or detach\"",
        detail: R`<p>A context manager that restores the previous state even on exceptions and nests properly.</p>`,
        hints: [
          R`<p><code>@contextlib.contextmanager</code> with a <code>global _GRAD_ENABLED</code>; save the previous value and restore it in <code>finally</code>.</p>`,
          R`<p>Your "make output" helper checks the flag: when disabled, the output gets no parents and <code>requires_grad=False</code>.</p>`,
          R`<p>detach: <code>Tensor(self.data)</code>. Sharing data is fine; the new tensor just has no graph.</p>`,
        ] },
      { id: "w1m6", core: true, title: "numerical_grad and gradcheck", test: "pytest tests/w1/test_optim.py -q -k \"numerical or gradcheck\"",
        detail: R`<p>Central differences $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$ have error $O(\epsilon^2)$ vs $O(\epsilon)$ for one-sided. The test also checks you restore inputs.</p>`,
        hints: [
          R`<p>Iterate over a flat view (<code>a.reshape(-1)</code> shares memory with a contiguous array) and perturb one element at a time.</p>`,
          R`<p>Wrap fresh Tensors around the arrays for every evaluation, and read <code>float(out.data)</code>.</p>`,
          R`<p>gradcheck: analytic pass with <code>Tensor(x.copy(), requires_grad=True)</code>, then compare each <code>.grad</code> to the numerical grad with <code>np.allclose(atol, rtol)</code>; return False if any grad is None.</p>`,
        ] },
      { id: "w1m7", core: true, title: "SGD (momentum, weight decay), Adam, AdamW", test: "pytest tests/w1/test_optim.py -q",
        detail: R`<p>The expected numbers were generated with PyTorch. Match its formulas exactly, including "velocity starts as g on the first step" and the per-optimizer step counter.</p>`,
        hints: [
          R`<p>Store per-parameter state in lists aligned with <code>self.params</code> (materialize the iterable in <code>__init__</code>; the test passes a generator).</p>`,
          R`<p>Update in place: <code>p.data -= lr * update</code>. Rebinding <code>p.data = p.data - ...</code> fails the in-place test.</p>`,
          R`<p>AdamW = Adam with one extra line before the Adam update: <code>p.data -= lr * wd * p.data</code>. Subclass and override a hook, or just override <code>step</code>.</p>`,
        ] },
      { id: "w1m8", core: false, title: "Stretch: graph visualizer and extra ops",
        detail: R`<p>Add <code>Tensor.to_dot()</code> that emits Graphviz DOT for the graph (label nodes with op name and shape) and render the MLP's graph. Add <code>max(axis)</code> (gradient to the argmax), <code>where(cond, a, b)</code> and <code>concatenate</code> with gradchecks.</p>`,
        hints: [R`<p>For max with ties, split the gradient equally among the tied maxima or send it to one of them; document your choice and test it.</p>`] },
    ],
  },
});
