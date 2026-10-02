/* Interactive widgets. Each lesson body places <div class="widget" data-widget="name"></div>; mount() builds them. */
(() => {
  "use strict";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const colors = () => ({ ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), rule: css("--rule"), accent: css("--accent"), ok: css("--ok"), bad: css("--bad"), warn: css("--warn"), paper: css("--paper"), card: css("--card"), mark: css("--mark") });
  const PALETTE = ["#2346c8", "#d9480f", "#2b8a3e", "#ae3ec9", "#e67700", "#0c8599", "#c2255c", "#5c940d", "#495057", "#1864ab"];
  const MONO = '12px "JetBrains Mono", ui-monospace, monospace';

  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v); }
    kids.forEach(k => e.append(k));
    return e;
  }
  let uid = 0;
  function slider(label, min, max, step, value, fmt = (v) => v) {
    const id = "wg" + (++uid);
    const input = el("input", { type: "range", min, max, step, value, id });
    const out = el("span", { class: "readout" });
    const wrap = el("label", { for: id }, el("span", { text: label }), input, out);
    const upd = () => (out.textContent = fmt(+input.value));
    input.addEventListener("input", upd); upd();
    return { wrap, input, get: () => +input.value };
  }
  function select(label, options, value) {
    const id = "wg" + (++uid);
    const s = el("select", { id });
    options.forEach(([v, t]) => { const o = el("option", { value: v, text: t }); if (v === value) o.selected = true; s.append(o); });
    s.style.width = "auto";
    return { wrap: el("label", { for: id }, el("span", { text: label }), s), input: s, get: () => s.value };
  }
  function toggle(label, checked) {
    const id = "wg" + (++uid);
    const c = el("input", { type: "checkbox", id });
    c.checked = checked;
    return { wrap: el("label", { for: id }, c, el("span", { text: label })), input: c, get: () => c.checked };
  }
  function canvas(w, h) {
    const c = el("canvas");
    c.width = w * 2; c.height = h * 2; c.style.aspectRatio = `${w} / ${h}`;
    const ctx = c.getContext("2d");
    ctx.scale(2, 2);
    return { c, ctx, w, h };
  }
  function frame(root, title) {
    root.replaceChildren();
    root.append(el("div", { class: "widget-title", text: title }));
    const controls = el("div", { class: "controls" });
    const readout = el("div", { class: "readout" });
    return { controls, readout };
  }
  function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // ------------------------------------------------------------------ 1. gradient descent
  function gd(root) {
    const { controls, readout } = frame(root, "Optimizers on a loss surface — click the plot to move the start point");
    const fn = select("surface", [["quad", "ill-conditioned bowl"], ["rosen", "Rosenbrock valley"]], "quad");
    const opt = select("optimizer", [["sgd", "gradient descent"], ["mom", "momentum"], ["adam", "Adam"]], "sgd");
    const lr = slider("log10 lr", -3.5, 0, 0.05, -1.2, (v) => (10 ** v).toExponential(1));
    const beta = slider("β", 0, 0.99, 0.01, 0.9, (v) => v.toFixed(2));
    controls.append(fn.wrap, opt.wrap, lr.wrap, beta.wrap);
    const cv = canvas(640, 360);
    root.append(controls, cv.c, readout);
    let start = null;
    const F = {
      quad: { f: (x, y) => 0.5 * (x * x + 12 * y * y), g: (x, y) => [x, 12 * y], box: [-3, 3, -1.7, 1.7], s: [-2.6, 1.2], min: [0, 0] },
      rosen: { f: (x, y) => (1 - x) ** 2 + 20 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 80 * x * (y - x * x), 40 * (y - x * x)], box: [-2, 2, -1, 3], s: [-1.6, 2.4], min: [1, 1] },
    };
    cv.c.addEventListener("pointerdown", (e) => {
      const r = cv.c.getBoundingClientRect(), P = F[fn.get()];
      const [x0, x1, y0, y1] = P.box;
      start = [x0 + (e.clientX - r.left) / r.width * (x1 - x0), y1 - (e.clientY - r.top) / r.height * (y1 - y0)];
      draw();
    });
    fn.input.addEventListener("change", () => { start = null; draw(); });
    function draw() {
      const C = colors(), P = F[fn.get()], { ctx, w, h } = cv;
      const [x0, x1, y0, y1] = P.box;
      const X = (x) => (x - x0) / (x1 - x0) * w, Y = (y) => (y1 - y) / (y1 - y0) * h;
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      // contours via log-spaced level sets on a grid
      const N = 90, vals = [];
      for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) vals.push(Math.log(1e-3 + P.f(x0 + i / N * (x1 - x0), y1 - j / N * (y1 - y0))));
      const lo = Math.min(...vals), hi = Math.max(...vals);
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
      for (let L = 1; L < 14; L++) {
        const lev = lo + (hi - lo) * L / 14;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const a = vals[j * (N + 1) + i], b = vals[j * (N + 1) + i + 1], c = vals[(j + 1) * (N + 1) + i];
          if ((a - lev) * (b - lev) < 0 || (a - lev) * (c - lev) < 0) { ctx.fillStyle = C.ink3; ctx.globalAlpha = 0.35; ctx.fillRect(i / N * w, j / N * h, 1.4, 1.4); ctx.globalAlpha = 1; }
        }
      }
      // run the optimizer
      let [x, y] = start || P.s, v = [0, 0], m = [0, 0], s2 = [0, 0];
      const eta = 10 ** lr.get(), b = beta.get(), path = [[x, y]];
      let diverged = false, steps = 0;
      for (let t = 1; t <= 300; t++) {
        const g = P.g(x, y);
        if (opt.get() === "sgd") { x -= eta * g[0]; y -= eta * g[1]; }
        else if (opt.get() === "mom") { v = [b * v[0] + g[0], b * v[1] + g[1]]; x -= eta * v[0]; y -= eta * v[1]; }
        else {
          m = [b * m[0] + (1 - b) * g[0], b * m[1] + (1 - b) * g[1]];
          s2 = [0.999 * s2[0] + 0.001 * g[0] ** 2, 0.999 * s2[1] + 0.001 * g[1] ** 2];
          const mh = [m[0] / (1 - b ** t), m[1] / (1 - b ** t)], vh = [s2[0] / (1 - 0.999 ** t), s2[1] / (1 - 0.999 ** t)];
          x -= eta * mh[0] / (Math.sqrt(vh[0]) + 1e-8); y -= eta * mh[1] / (Math.sqrt(vh[1]) + 1e-8);
        }
        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 1e6) { diverged = true; break; }
        path.push([x, y]);
        steps = t;
        if (Math.hypot(x - P.min[0], y - P.min[1]) < 1e-3) break;
      }
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath();
      path.forEach(([px, py], k) => { const cx = Math.max(-50, Math.min(w + 50, X(px))), cy = Math.max(-50, Math.min(h + 50, Y(py))); k ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy); });
      ctx.stroke();
      ctx.fillStyle = C.accent; path.forEach(([px, py]) => { ctx.beginPath(); ctx.arc(X(px), Y(py), 1.8, 0, 7); ctx.fill(); });
      ctx.fillStyle = C.ok; ctx.beginPath(); ctx.arc(X(P.min[0]), Y(P.min[1]), 5, 0, 7); ctx.fill();
      ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(path[0][0]), Y(path[0][1]), 5, 0, 7); ctx.fill();
      const last = path[path.length - 1];
      readout.textContent = diverged ? `Diverged after ${steps} steps: the learning rate is above 2/λ_max for this curvature.`
        : `${steps} steps · final loss ${P.f(last[0], last[1]).toExponential(2)} · distance to minimum ${Math.hypot(last[0] - P.min[0], last[1] - P.min[1]).toFixed(4)}`;
    }
    [lr, beta, opt].forEach(c => c.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 2. SVD of a 2x2 matrix
  function svd(root) {
    const { controls, readout } = frame(root, "A matrix maps the unit circle to an ellipse — its axes are the singular vectors");
    const a = slider("a", -2, 2, 0.05, 1.5, v => v.toFixed(2)), b = slider("b", -2, 2, 0.05, 0.8, v => v.toFixed(2));
    const c = slider("c", -2, 2, 0.05, 0.3, v => v.toFixed(2)), d = slider("d", -2, 2, 0.05, 0.9, v => v.toFixed(2));
    controls.append(a.wrap, b.wrap, c.wrap, d.wrap);
    const cv = canvas(640, 340);
    root.append(controls, cv.c, readout);
    function draw() {
      const C = colors(), { ctx, w, h } = cv;
      const A = [[a.get(), b.get()], [c.get(), d.get()]];
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const S = 55, halves = [[w * 0.25, h / 2], [w * 0.72, h / 2]];
      halves.forEach(([cx, cy]) => { ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(cx - 140, cy); ctx.lineTo(cx + 140, cy); ctx.moveTo(cx, cy - 150); ctx.lineTo(cx, cy + 150); ctx.stroke(); });
      // A^T A eigen-decomposition -> V, sigma
      const ata = [[A[0][0] ** 2 + A[1][0] ** 2, A[0][0] * A[0][1] + A[1][0] * A[1][1]], [0, A[0][1] ** 2 + A[1][1] ** 2]]; ata[1][0] = ata[0][1];
      const tr = ata[0][0] + ata[1][1], det = ata[0][0] * ata[1][1] - ata[0][1] ** 2, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
      const vec = (l) => { const x = ata[0][1], y = l - ata[0][0]; const n = Math.hypot(x, y); return n < 1e-9 ? (l === l1 ? [1, 0] : [0, 1]) : [x / n, y / n]; };
      let v1 = vec(l1), v2 = [-v1[1], v1[0]];
      const s1 = Math.sqrt(l1), s2 = Math.sqrt(l2);
      const mul = (p) => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
      const circle = (cx, cy, f, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let k = 0; k <= 120; k++) { const t = k / 120 * 2 * Math.PI; const p = f([Math.cos(t), Math.sin(t)]); k ? ctx.lineTo(cx + S * p[0], cy - S * p[1]) : ctx.moveTo(cx + S * p[0], cy - S * p[1]); } ctx.stroke(); };
      const arrow = (cx, cy, p, col, label) => { ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + S * p[0], cy - S * p[1]); ctx.stroke(); ctx.beginPath(); ctx.arc(cx + S * p[0], cy - S * p[1], 3.5, 0, 7); ctx.fill(); ctx.font = MONO; ctx.fillText(label, cx + S * p[0] + 6, cy - S * p[1] - 6); };
      circle(...halves[0], p => p, C.ink3);
      arrow(...halves[0], v1, PALETTE[0], "v₁"); arrow(...halves[0], v2, PALETTE[1], "v₂");
      circle(...halves[1], mul, C.ink3);
      arrow(...halves[1], mul(v1), PALETTE[0], "σ₁u₁"); arrow(...halves[1], mul(v2), PALETTE[1], "σ₂u₂");
      ctx.fillStyle = C.ink2; ctx.font = MONO; ctx.fillText("input space: unit circle", 12, 18); ctx.fillText("output space: A · circle", w / 2 + 12, 18);
      const detA = A[0][0] * A[1][1] - A[0][1] * A[1][0];
      readout.textContent = `σ₁ = ${s1.toFixed(3)}   σ₂ = ${s2.toFixed(3)}   |det A| = σ₁σ₂ = ${Math.abs(detA).toFixed(3)}   rank = ${s2 < 1e-3 ? (s1 < 1e-3 ? 0 : 1) : 2}   condition number σ₁/σ₂ = ${s2 < 1e-9 ? "∞" : (s1 / s2).toFixed(2)}`;
    }
    [a, b, c, d].forEach(s => s.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 3. backprop stepper
  function backprop(root) {
    const { controls, readout } = frame(root, "Reverse-mode autodiff, one node at a time: L = (tanh(w·x + b) − t)²");
    const fwdBtn = el("button", { type: "button", class: "primary", text: "Next step" });
    const resetBtn = el("button", { type: "button", text: "Reset" });
    controls.append(fwdBtn, resetBtn);
    const cv = canvas(640, 250);
    root.append(controls, cv.c, readout);
    const vals = { w: 0.5, x: 2.0, b: -0.3, t: 0.8 };
    const nodes = [
      { id: "w", x: 50, y: 50, label: "w", v: vals.w }, { id: "x", x: 50, y: 130, label: "x", v: vals.x },
      { id: "m", x: 190, y: 90, label: "×", from: ["w", "x"] }, { id: "b", x: 190, y: 190, label: "b", v: vals.b },
      { id: "z", x: 320, y: 140, label: "+", from: ["m", "b"] }, { id: "y", x: 430, y: 140, label: "tanh", from: ["z"] },
      { id: "t", x: 430, y: 220, label: "t", v: vals.t }, { id: "L", x: 570, y: 170, label: "(·−t)²", from: ["y", "t"] },
    ];
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
    const m = vals.w * vals.x, z = m + vals.b, y = Math.tanh(z), L = (y - vals.t) ** 2;
    const fwd = { m, z, y, L };
    const dL = 1, dy = 2 * (y - vals.t), dz = dy * (1 - y * y), dm = dz, db = dz, dw = dm * vals.x, dx = dm * vals.w;
    const steps = [
      ["m", "fwd", `forward: m = w·x = ${m.toFixed(3)}`], ["z", "fwd", `forward: z = m + b = ${z.toFixed(3)}`],
      ["y", "fwd", `forward: y = tanh(z) = ${y.toFixed(4)}`], ["L", "fwd", `forward: L = (y − t)² = ${L.toFixed(5)}`],
      ["L", "bwd", "backward: seed ∂L/∂L = 1"], ["y", "bwd", `∂L/∂y = 2(y − t) = ${dy.toFixed(4)}`],
      ["z", "bwd", `∂L/∂z = ∂L/∂y · (1 − y²) = ${dz.toFixed(4)}   (local derivative of tanh)`],
      ["m", "bwd", `∂L/∂m = ∂L/∂z · 1 = ${dm.toFixed(4)}   (+ passes gradient through unchanged)`],
      ["b", "bwd", `∂L/∂b = ∂L/∂z · 1 = ${db.toFixed(4)}`],
      ["w", "bwd", `∂L/∂w = ∂L/∂m · x = ${dw.toFixed(4)}   (× swaps: each input gets the other's value)`],
      ["x", "bwd", `∂L/∂x = ∂L/∂m · w = ${dx.toFixed(4)}`],
    ];
    const grads = { L: dL, y: dy, z: dz, m: dm, b: db, w: dw, x: dx };
    let k = 0;
    fwdBtn.addEventListener("click", () => { if (k < steps.length) k++; draw(); });
    resetBtn.addEventListener("click", () => { k = 0; draw(); });
    function draw() {
      const C = colors(), { ctx, w, h } = cv;
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const shownF = new Set(["w", "x", "b", "t"]), shownB = new Set();
      steps.slice(0, k).forEach(([id, kind]) => (kind === "fwd" ? shownF : shownB).add(id));
      ctx.lineWidth = 1.5;
      nodes.forEach(n => (n.from || []).forEach(f => { const s = byId[f]; ctx.strokeStyle = shownB.has(f) && shownB.has(n.id) ? C.bad : C.ink3; ctx.beginPath(); ctx.moveTo(s.x + 28, s.y); ctx.lineTo(n.x - 28, n.y); ctx.stroke(); }));
      nodes.forEach(n => {
        const cur = k > 0 && steps[k - 1][0] === n.id;
        ctx.fillStyle = C.card; ctx.strokeStyle = cur ? C.accent : C.ink3; ctx.lineWidth = cur ? 3 : 1.5;
        ctx.beginPath(); ctx.roundRect(n.x - 28, n.y - 16, 56, 32, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.font = MONO; ctx.textAlign = "center"; ctx.fillText(n.label, n.x, n.y + 4);
        const v = n.v != null ? n.v : fwd[n.id];
        if (shownF.has(n.id)) { ctx.fillStyle = C.accent; ctx.fillText(v.toFixed(3), n.x, n.y - 22); }
        if (shownB.has(n.id)) { ctx.fillStyle = C.bad; ctx.fillText("∂ " + grads[n.id].toFixed(3), n.x, n.y + 32); }
        ctx.textAlign = "start";
      });
      ctx.fillStyle = C.ink2; ctx.font = MONO; ctx.fillText("blue = forward value · red = gradient ∂L/∂node", 10, h - 8);
      readout.textContent = k === 0 ? "Press Next step. Leaves hold w=0.5, x=2, b=−0.3, target t=0.8." : `${k}/${steps.length}  ${steps[k - 1][2]}`;
      fwdBtn.disabled = k === steps.length;
    }
    return draw;
  }

  // ------------------------------------------------------------------ 4. bias-variance
  function biasvar(root) {
    const { controls, readout } = frame(root, "Underfitting vs overfitting: polynomial regression on 18 noisy points");
    const deg = slider("degree", 0, 15, 1, 3), lam = slider("log10 λ (ridge)", -8, 1, 0.1, -8, v => v <= -7.95 ? "0" : (10 ** v).toExponential(0));
    controls.append(deg.wrap, lam.wrap);
    const cv = canvas(640, 300), cv2 = canvas(640, 150);
    root.append(controls, cv.c, cv2.c, readout);
    const r = rng(7), f = (x) => Math.sin(2.6 * x) + 0.3 * x;
    const tr = Array.from({ length: 18 }, (_, i) => { const x = -1 + 2 * (i + r() * 0.8) / 18; return [x, f(x) + 0.28 * gauss(r)]; });
    const te = Array.from({ length: 300 }, () => { const x = -1 + 2 * r(); return [x, f(x) + 0.28 * gauss(r)]; });
    const feats = (x, d) => { const out = [1, x]; for (let k = 2; k <= d; k++) out.push(((2 * k - 1) * x * out[k - 1] - (k - 1) * out[k - 2]) / k); return out.slice(0, d + 1); }; // Legendre basis
    function fit(d, l) {
      const n = d + 1, A = Array.from({ length: n }, () => new Array(n).fill(0)), bb = new Array(n).fill(0);
      tr.forEach(([x, y]) => { const p = feats(x, d); for (let i = 0; i < n; i++) { bb[i] += p[i] * y; for (let j = 0; j < n; j++) A[i][j] += p[i] * p[j]; } });
      for (let i = 1; i < n; i++) A[i][i] += l;
      for (let i = 0; i < n; i++) A[i][i] += 1e-10;
      for (let c = 0; c < n; c++) { let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i; [A[c], A[p]] = [A[p], A[c]]; [bb[c], bb[p]] = [bb[p], bb[c]]; for (let i = c + 1; i < n; i++) { const m = A[i][c] / A[c][c]; for (let j = c; j < n; j++) A[i][j] -= m * A[c][j]; bb[i] -= m * bb[c]; } }
      const wv = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = bb[i]; for (let j = i + 1; j < n; j++) s -= A[i][j] * wv[j]; wv[i] = s / A[i][i]; }
      return (x) => feats(x, d).reduce((s, p, i) => s + p * wv[i], 0);
    }
    const mse = (g, data) => data.reduce((s, [x, y]) => s + (g(x) - y) ** 2, 0) / data.length;
    function draw() {
      const C = colors(), l = lam.get() <= -7.95 ? 0 : 10 ** lam.get(), d = deg.get(), g = fit(d, l);
      let { ctx, w, h } = cv;
      const X = (x) => (x + 1.05) / 2.1 * w, Y = (y) => h / 2 - y * h / 4.4;
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = C.ok; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -1 + 2 * i / 200; i ? ctx.lineTo(X(x), Y(f(x))) : ctx.moveTo(X(x), Y(f(x))); } ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.2; ctx.beginPath(); for (let i = 0; i <= 300; i++) { const x = -1 + 2 * i / 300; const yy = Math.max(-3, Math.min(3, g(x))); i ? ctx.lineTo(X(x), Y(yy)) : ctx.moveTo(X(x), Y(yy)); } ctx.stroke();
      ctx.fillStyle = C.ink; tr.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 3.5, 0, 7); ctx.fill(); });
      ctx.font = MONO; ctx.fillStyle = C.ink2; ctx.fillText("dots: training data · dashed: true function · line: your fit", 10, 16);
      // error curves
      ({ ctx, w, h } = cv2);
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const errs = Array.from({ length: 16 }, (_, k) => { const gk = fit(k, l); return [mse(gk, tr), mse(gk, te)]; });
      const maxE = 0.9, EX = (k) => 40 + k / 15 * (w - 60), EY = (e) => h - 22 - Math.min(e, maxE) / maxE * (h - 40);
      [[0, C.ink2, "train"], [1, C.bad, "test"]].forEach(([ix, col, name]) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); errs.forEach((e, k) => k ? ctx.lineTo(EX(k), EY(e[ix])) : ctx.moveTo(EX(k), EY(e[ix]))); ctx.stroke(); ctx.fillStyle = col; ctx.fillText(name + " MSE", EX(15) - 70, EY(errs[15][ix]) - 6); });
      ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.moveTo(EX(d), 8); ctx.lineTo(EX(d), h - 20); ctx.stroke();
      ctx.fillStyle = C.ink3; for (let k = 0; k <= 15; k += 3) ctx.fillText(String(k), EX(k) - 4, h - 6);
      ctx.fillText("degree →", w - 70, h - 6);
      readout.textContent = `degree ${d}: train MSE ${mse(g, tr).toFixed(3)} · test MSE ${mse(g, te).toFixed(3)} · noise floor σ² ≈ 0.078`;
    }
    [deg, lam].forEach(s => s.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 5. initialization and activation statistics
  function init(root) {
    const { controls, readout } = frame(root, "Activation scale through a 20-layer MLP (width 128) at initialization");
    const act = select("activation", [["tanh", "tanh"], ["relu", "ReLU"]], "relu");
    const ini = select("init", [["small", "N(0, 0.01²)"], ["unit", "N(0, 1)"], ["xavier", "Xavier: N(0, 1/n)"], ["he", "He: N(0, 2/n)"]], "small");
    const bn = toggle("normalize each layer (LayerNorm-like)", false);
    controls.append(act.wrap, ini.wrap, bn.wrap);
    const cv = canvas(640, 240);
    root.append(controls, cv.c, readout);
    function draw() {
      const C = colors(), { ctx, w, h } = cv, n = 128, B = 48, L = 20, r = rng(3);
      let H = Array.from({ length: B }, () => Float64Array.from({ length: n }, () => gauss(r)));
      const std = { small: 0.01, unit: 1, xavier: Math.sqrt(1 / n), he: Math.sqrt(2 / n) }[ini.get()];
      const stds = [];
      for (let l = 0; l < L; l++) {
        const Wm = Array.from({ length: n }, () => Float64Array.from({ length: n }, () => gauss(r) * std));
        H = H.map(row => { const out = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; const wj = Wm[j]; for (let i = 0; i < n; i++) s += wj[i] * row[i]; out[j] = act.get() === "relu" ? Math.max(0, s) : Math.tanh(s); } return out; });
        if (bn.get()) H = H.map(row => { const m = row.reduce((a, b) => a + b, 0) / n; const v = row.reduce((a, b) => a + (b - m) ** 2, 0) / n; return row.map(x => (x - m) / Math.sqrt(v + 1e-5)); });
        let s = 0, c = 0; H.forEach(row => row.forEach(x => { s += x * x; c++; }));
        stds.push(Math.sqrt(s / c));
      }
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const lo = -8, hi = 4, Y = (v) => h - 26 - (Math.max(lo, Math.min(hi, Math.log10(v + 1e-30))) - lo) / (hi - lo) * (h - 44);
      ctx.font = MONO; ctx.fillStyle = C.ink3;
      [-8, -4, 0, 4].forEach(e => { ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(40, Y(10 ** e)); ctx.lineTo(w - 8, Y(10 ** e)); ctx.stroke(); ctx.fillText("1e" + e, 2, Y(10 ** e) + 4); });
      const bw = (w - 60) / L;
      stds.forEach((s, l) => { const good = s > 0.2 && s < 5; ctx.fillStyle = good ? C.ok : C.bad; ctx.fillRect(46 + l * bw, Y(s), bw - 4, h - 26 - Y(s)); });
      ctx.fillStyle = C.ink2; ctx.fillText("layer 1 → 20 · bar = RMS activation (log scale)", 46, h - 8);
      const fin = stds[L - 1];
      readout.textContent = `RMS activation at layer 20: ${fin.toExponential(2)} — ` + (fin < 1e-3 ? "signals vanished: gradients will too." : fin > 1e3 ? "signals exploded: tanh saturates / ReLU overflows." : "healthy: the variance is preserved layer to layer.");
    }
    [act, ini, bn].forEach(s => s.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 6. attention
  function attention(root) {
    const { controls, readout } = frame(root, "Scaled dot-product attention weights for one head");
    const causal = toggle("causal mask", true), scale = toggle("divide by √d", true);
    const dim = slider("head dim d", 4, 256, 4, 64), temp = slider("temperature", 0.2, 3, 0.1, 1, v => v.toFixed(1));
    controls.append(causal.wrap, scale.wrap, dim.wrap, temp.wrap);
    const cv = canvas(640, 360);
    root.append(controls, cv.c, readout);
    const toks = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "warm"];
    function draw() {
      const C = colors(), { ctx, w, h } = cv, d = dim.get(), T = toks.length, r = rng(11);
      const emb = {}; const vec = (s) => emb[s] || (emb[s] = Array.from({ length: 256 }, () => gauss(r)));
      toks.forEach(t => vec(t.toLowerCase()));
      const Wq = Array.from({ length: 256 }, () => gauss(r) / 16), Wk = Array.from({ length: 256 }, () => gauss(r) / 16);
      const q = toks.map((t, i) => Array.from({ length: d }, (_, j) => vec(t.toLowerCase())[j] * (1 + Wq[j]) + 0.3 * Math.sin((i + 1) * (j + 1) / 7)));
      const k = toks.map((t, i) => Array.from({ length: d }, (_, j) => vec(t.toLowerCase())[j] * (1 + Wk[j]) + 0.3 * Math.cos((i + 1) * (j + 1) / 7)));
      const A = q.map((qi, i) => {
        const s = k.map((kj, j) => { if (causal.get() && j > i) return -Infinity; let dot = 0; for (let m = 0; m < d; m++) dot += qi[m] * kj[m]; return dot / (scale.get() ? Math.sqrt(d) : 1) / temp.get(); });
        const mx = Math.max(...s), e = s.map(x => Math.exp(x - mx)), z = e.reduce((a, b) => a + b, 0);
        return e.map(x => x / z);
      });
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const left = 80, top = 60, cell = Math.min((w - left - 20) / T, (h - top - 10) / T);
      ctx.font = MONO;
      toks.forEach((t, i) => { ctx.fillStyle = C.ink2; ctx.textAlign = "right"; ctx.fillText(t, left - 8, top + i * cell + cell / 2 + 4); ctx.save(); ctx.translate(left + i * cell + cell / 2, top - 8); ctx.rotate(-Math.PI / 4); ctx.textAlign = "left"; ctx.fillText(t, 0, 0); ctx.restore(); });
      ctx.textAlign = "start";
      A.forEach((row, i) => row.forEach((p, j) => { ctx.fillStyle = C.accent; ctx.globalAlpha = 0.06 + 0.94 * p; ctx.fillRect(left + j * cell + 1, top + i * cell + 1, cell - 2, cell - 2); ctx.globalAlpha = 1; if (causal.get() && j > i) { ctx.fillStyle = C.rule; ctx.fillRect(left + j * cell + 1, top + i * cell + 1, cell - 2, cell - 2); } }));
      ctx.fillStyle = C.ink3; ctx.fillText("rows = queries (who is looking) · columns = keys (what they look at)", 8, 16);
      const H = A.map(row => -row.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0));
      const maxRow = A.map(row => Math.max(...row));
      readout.textContent = `mean row entropy ${(H.reduce((a, b) => a + b, 0) / T).toFixed(2)} nats · mean max weight ${(maxRow.reduce((a, b) => a + b, 0) / T).toFixed(2)}` + (!scale.get() && d >= 64 ? " — unscaled dot products grow like √d, so softmax saturates to one-hot and gradients vanish." : "");
    }
    [causal, scale, dim, temp].forEach(s => s.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 7. dynamic batching simulator
  function batching(root) {
    const { controls, readout } = frame(root, "Dynamic batching: Poisson arrivals into one GPU worker");
    const rate = slider("arrivals/s", 5, 400, 5, 120), mb = slider("max batch", 1, 64, 1, 16), wait = slider("max wait ms", 0, 50, 1, 5);
    controls.append(rate.wrap, mb.wrap, wait.wrap);
    const note = el("div", { class: "readout", text: "GPU cost model: one forward pass takes 18 ms + 1.5 ms per request in the batch." });
    const cv = canvas(640, 220);
    root.append(controls, note, cv.c, readout);
    function simulate(lam, B, W) {
      const r = rng(5), N = 3000, arr = []; let t = 0;
      for (let i = 0; i < N; i++) { t += -Math.log(1 - r()) / lam; arr.push(t); }
      // Mirrors serve/batcher.py: take the oldest request, then collect until the batch is full or
      // max_wait has passed since that moment; the GPU runs one batch at a time.
      const lat = [], sizes = []; let free = 0, i = 0;
      while (i < N) {
        const startAt = Math.max(free, arr[i]);
        const deadline = startAt + W / 1000;
        let j = i + 1;
        while (j < N && j - i < B && arr[j] <= deadline) j++;
        const batchN = j - i;
        const launch = batchN === B ? Math.max(startAt, arr[j - 1]) : deadline;
        const done = launch + (18 + 1.5 * batchN) / 1000;
        for (let k = i; k < j; k++) lat.push((done - arr[k]) * 1000);
        sizes.push(batchN); free = done; i = j;
      }
      lat.sort((a, b) => a - b);
      return { thr: N / (free - arr[0]), p50: lat[Math.floor(0.5 * N)], p99: lat[Math.floor(0.99 * N)], avgB: N / sizes.length, lat };
    }
    function draw() {
      const C = colors(), { ctx, w, h } = cv;
      const s = simulate(rate.get(), mb.get(), wait.get()), base = simulate(rate.get(), 1, 0);
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const maxL = Math.max(200, Math.min(2000, s.lat[s.lat.length - 1])), bins = 40, hist = new Array(bins).fill(0);
      s.lat.forEach(l => hist[Math.min(bins - 1, Math.floor(l / maxL * bins))]++);
      const mx = Math.max(...hist), bw = (w - 40) / bins;
      hist.forEach((c, k) => { ctx.fillStyle = C.accent; ctx.fillRect(30 + k * bw, h - 24 - c / mx * (h - 50), bw - 2, c / mx * (h - 50)); });
      ctx.font = MONO; ctx.fillStyle = C.ink3; ctx.fillText("latency histogram (ms)", 30, 14);
      [0, 0.5, 1].forEach(f => ctx.fillText(String(Math.round(f * maxL)), 26 + f * (w - 60), h - 8));
      const sat = rate.get() > 1000 / 19.5;
      readout.textContent = `batched: ${s.thr.toFixed(0)} req/s served · avg batch ${s.avgB.toFixed(1)} · p50 ${s.p50.toFixed(0)} ms · p99 ${s.p99.toFixed(0)} ms   |   batch=1: p50 ${base.p50.toFixed(0)} ms · p99 ${base.p99.toFixed(0)} ms${sat ? " (queue grows without bound: 1 req per 19.5 ms is the ceiling)" : ""}`;
    }
    [rate, mb, wait].forEach(c => c.input.addEventListener("input", draw));
    return draw;
  }

  // ------------------------------------------------------------------ 8. consistent hash ring
  function ring(root) {
    const { controls, readout } = frame(root, "Consistent hashing: each key goes to the next node point clockwise");
    const nodesS = slider("nodes", 2, 10, 1, 4), vn = slider("virtual nodes each", 1, 200, 1, 1);
    const add = el("button", { type: "button", text: "Add one node and count moved keys" });
    controls.append(nodesS.wrap, vn.wrap, add);
    const cv = canvas(640, 320);
    root.append(controls, cv.c, readout);
    const hash = (s) => { let h1 = 2166136261; for (let i = 0; i < s.length; i++) { h1 ^= s.charCodeAt(i); h1 = Math.imul(h1, 16777619); } h1 ^= h1 >>> 13; h1 = Math.imul(h1, 0x5bd1e995); h1 ^= h1 >>> 15; return (h1 >>> 0) / 4294967296; };
    const keys = Array.from({ length: 2000 }, (_, i) => hash("key:" + i));
    function build(n, v) { const pts = []; for (let i = 0; i < n; i++) for (let j = 0; j < v; j++) pts.push([hash(`node${i}#${j}`), i]); pts.sort((a, b) => a[0] - b[0]); return pts; }
    function owner(pts, k) { let lo = 0, hi = pts.length; while (lo < hi) { const m = (lo + hi) >> 1; if (pts[m][0] < k) lo = m + 1; else hi = m; } return pts[lo % pts.length][1]; }
    let moved = null;
    add.addEventListener("click", () => { const n = nodesS.get(), v = vn.get(); const a = build(n, v), b = build(n + 1, v); let m = 0; keys.forEach(k => { if (owner(a, k) !== owner(b, k)) m++; }); moved = { frac: m / keys.length, ideal: 1 / (n + 1) }; draw(); });
    function draw() {
      const C = colors(), { ctx, w, h } = cv, n = nodesS.get(), v = vn.get(), pts = build(n, v);
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const cx = 170, cy = h / 2, R = 125;
      ctx.strokeStyle = C.rule; ctx.lineWidth = 8; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
      const load = new Array(n).fill(0);
      keys.forEach(k => load[owner(pts, k)]++);
      keys.slice(0, 400).forEach(k => { const a = k * 2 * Math.PI - Math.PI / 2; ctx.fillStyle = PALETTE[owner(pts, k) % 10]; ctx.globalAlpha = 0.5; ctx.fillRect(cx + (R - 16) * Math.cos(a) - 1, cy + (R - 16) * Math.sin(a) - 1, 2.5, 2.5); ctx.globalAlpha = 1; });
      pts.forEach(([p, i]) => { const a = p * 2 * Math.PI - Math.PI / 2; ctx.strokeStyle = PALETTE[i % 10]; ctx.lineWidth = v > 50 ? 1 : 3; ctx.beginPath(); ctx.moveTo(cx + (R - 6) * Math.cos(a), cy + (R - 6) * Math.sin(a)); ctx.lineTo(cx + (R + 10) * Math.cos(a), cy + (R + 10) * Math.sin(a)); ctx.stroke(); });
      const mean = keys.length / n, maxL = Math.max(...load), bx = 360, bw = (w - bx - 20) / n;
      ctx.font = MONO; ctx.fillStyle = C.ink3; ctx.fillText("keys per node (dashed = perfect balance)", bx, 18);
      load.forEach((c, i) => { const bh = c / (2.2 * mean) * (h - 60); ctx.fillStyle = PALETTE[i % 10]; ctx.fillRect(bx + i * bw + 3, h - 30 - Math.min(bh, h - 60), bw - 6, Math.min(bh, h - 60)); });
      const my = h - 30 - mean / (2.2 * mean) * (h - 60); ctx.strokeStyle = C.ink2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(bx, my); ctx.lineTo(w - 20, my); ctx.stroke(); ctx.setLineDash([]);
      const sd = Math.sqrt(load.reduce((s, c) => s + (c - mean) ** 2, 0) / n);
      readout.textContent = `max/mean load ${(maxL / mean).toFixed(2)} · coefficient of variation ${(sd / mean).toFixed(3)}` + (moved ? ` · adding node ${n + 1} moved ${(100 * moved.frac).toFixed(1)}% of keys (ideal 1/${n + 1} = ${(100 * moved.ideal).toFixed(1)}%; mod-N hashing would move ~${(100 * n / (n + 1)).toFixed(0)}%)` : "");
    }
    [nodesS, vn].forEach(s => s.input.addEventListener("input", () => { moved = null; draw(); }));
    return draw;
  }

  // ------------------------------------------------------------------ 9. A/B test power
  function abpower(root) {
    const { controls, readout } = frame(root, "A/B test sample size and power (two-sided two-proportion z-test)");
    const p = slider("baseline rate", 0.01, 0.5, 0.005, 0.1, v => (100 * v).toFixed(1) + "%"), mde = slider("MDE (absolute)", 0.001, 0.05, 0.001, 0.01, v => "+" + (100 * v).toFixed(1) + " pts");
    const alpha = select("α", [["0.05", "0.05"], ["0.01", "0.01"], ["0.1", "0.10"]], "0.05"), pow = select("power", [["0.8", "80%"], ["0.9", "90%"]], "0.8");
    controls.append(p.wrap, mde.wrap, alpha.wrap, pow.wrap);
    const cv = canvas(640, 220);
    root.append(controls, cv.c, readout);
    const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
    const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
    const inv = (q) => { let lo = -10, hi = 10; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (Phi(m) < q) lo = m; else hi = m; } return (lo + hi) / 2; };
    function draw() {
      const C = colors(), { ctx, w, h } = cv, p1 = p.get(), d = mde.get(), p2 = p1 + d, a = +alpha.get(), pw = +pow.get();
      const pb = (p1 + p2) / 2, za = inv(1 - a / 2), zb = inv(pw);
      const n = Math.ceil((za * Math.sqrt(2 * pb * (1 - pb)) + zb * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / d ** 2);
      const powerAt = (m) => { const se0 = Math.sqrt(2 * pb * (1 - pb) / m), se1 = Math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / m); return Phi((d - za * se0) / se1); };
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
      const maxN = n * 2.5, X = (m) => 40 + m / maxN * (w - 60), Y = (v) => h - 26 - v * (h - 46);
      ctx.strokeStyle = C.rule; [0, 0.5, 0.8, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(40, Y(v)); ctx.lineTo(w - 20, Y(v)); ctx.stroke(); });
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.2; ctx.beginPath(); for (let i = 1; i <= 200; i++) { const m = i / 200 * maxN; i > 1 ? ctx.lineTo(X(m), Y(powerAt(m))) : ctx.moveTo(X(m), Y(powerAt(m))); } ctx.stroke();
      ctx.strokeStyle = C.ok; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(n), Y(0)); ctx.lineTo(X(n), Y(1)); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = MONO; ctx.fillStyle = C.ink3; ctx.fillText("power vs users per arm", 44, 14); ["0", "50%", "80%", "100%"].forEach((t, i) => ctx.fillText(t, 2, Y([0, 0.5, 0.8, 1][i]) + 4));
      ctx.fillText(Math.round(maxN).toLocaleString(), w - 70, h - 8);
      readout.textContent = `${n.toLocaleString()} users per arm (${(2 * n).toLocaleString()} total) to detect ${(100 * p1).toFixed(1)}% → ${(100 * p2).toFixed(1)}% with α=${a}, power ${Math.round(pw * 100)}%. Halving the MDE roughly quadruples n.`;
    }
    [p, mde, alpha, pow].forEach(c => c.input.addEventListener("input", draw));
    return draw;
  }

  const REGISTRY = { gd, svd, backprop, biasvar, init, attention, batching, ring, abpower };
  const live = new Set();
  window.ForgeWidgets = {
    mount(scope) {
      live.clear();
      scope.querySelectorAll(".widget[data-widget]").forEach(node => {
        const make = REGISTRY[node.dataset.widget];
        if (!make) return;
        try { const draw = make(node); live.add(draw); draw(); } catch (e) { node.textContent = "This widget failed to load."; }
      });
    },
  };
  const redraw = () => live.forEach(d => { try { d(); } catch (e) {} });
  window.addEventListener("forge-theme", () => setTimeout(redraw, 0));
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw); } catch (e) {}
})();
