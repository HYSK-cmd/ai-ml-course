COURSE.weeks.push({
  n: 5, id: "w5", title: "Python Engineering",
  tagline: "The language machinery under every ML framework, then the craft that turns your forge into a real package: types, tests, profiling, concurrency, CI.",
  hours: { learn: 6, project: 7 },
  goals: [
    "Explain operator dispatch, the descriptor protocol and attribute lookup well enough to read framework source",
    "Use generators, context managers and decorators idiomatically",
    "Type-check a numeric codebase with mypy --strict and test it with property-based tests",
    "Profile before optimizing and pick threads, processes or asyncio correctly for a workload",
    "Ship forge as an installable package with a CLI, lint/type gates and CI",
  ],
  schedule: [
    ["Mon", "L1 The Python data model (dunders, descriptors, generators, decorators)"],
    ["Tue", "L2 Types and API design + L3 Testing"],
    ["Wed", "L4 Performance and concurrency"],
    ["Thu", "L5 Packaging and tooling; record your conv benchmark baseline"],
    ["Fri", "Flashcards; profile your Week 3 conv"],
    ["Sat", "Milestones 1–3: drills, CLI, packaging"],
    ["Sun", "Milestones 4–7: mypy/ruff, property tests, CI, performance write-up"],
  ],
  lessons: [
    {
      id: "w5l1", title: "The Python data model", minutes: 85,
      summary: "How Python decides what a + b, obj.attr and for x in obj mean, and how frameworks hook into each decision.",
      keypoints: [
        "a + b tries a.__add__(b); if that returns NotImplemented, b.__radd__(a). Never raise TypeError yourself for unknown types.",
        "Objects that compare equal must hash equal; mutable objects with value equality should be unhashable.",
        "Attribute lookup: data descriptors on the class, then the instance __dict__, then non-data descriptors / class attributes, then __getattr__.",
        "property, methods, classmethod and dataclass fields are all descriptors.",
        "Generators are lazy, single-pass iterators; code before the first yield runs on the first next().",
        "A decorator is a function from function to function; use functools.wraps.",
      ],
      body: R`
<p>You've already used this machinery: <code>Tensor.__radd__</code> and <code>__array_priority__</code> so that <code>2 * x</code> and <code>arr * x</code> work, <code>Module.__setattr__</code> for parameter registration, <code>__getattr__</code> for buffers, <code>contextlib.contextmanager</code> for <code>no_grad</code>. This lesson explains the rules behind them.</p>

<h2>Operator dispatch</h2>
<p>For <code>a + b</code> Python calls <code>type(a).__add__(a, b)</code>. If that returns the singleton <code>NotImplemented</code> (not raise!), it tries <code>type(b).__radd__(b, a)</code>. If both decline, <em>then</em> Python raises <code>TypeError</code>. Special case: if <code>type(b)</code> is a subclass of <code>type(a)</code> that overrides the reflected method, it gets tried first, so subclasses can take control.</p>
<p>This is why your <code>Vector.__mul__</code> must <em>return</em> <code>NotImplemented</code> for a Vector argument: the test expects <code>Vector * Vector</code> to raise <code>TypeError</code>, and Python only raises it after giving the other operand its chance. numpy arrays are greedy: <code>ndarray.__mul__</code> happily treats an unknown object as a 0-d object array. <code>__array_priority__</code> (or <code>__array_ufunc__ = None</code>) tells numpy to return <code>NotImplemented</code> instead, which is how <code>arr * tensor</code> reaches <code>Tensor.__rmul__</code>.</p>
<h3>Equality and hashing</h3>
<p>Contract: if <code>a == b</code> then <code>hash(a) == hash(b)</code>. Dicts and sets rely on it. Defining <code>__eq__</code> sets <code>__hash__ = None</code> (unhashable) unless you define <code>__hash__</code> too, which is correct for mutable objects: if a value used as a dict key changes, it's lost in the wrong bucket. Your immutable <code>Vector</code> hashes its component tuple. Also: <code>__eq__</code> with a foreign type should return <code>NotImplemented</code>, so <code>Vector(1,2) == (1.0, 2.0)</code> falls back to identity and is <code>False</code>.</p>
<h3>Immutability and __slots__</h3>
<p><code>__slots__ = ("_c",)</code> removes the per-instance <code>__dict__</code> (less memory, faster attribute access, and no accidental new attributes). To make the object immutable, also override <code>__setattr__</code> to raise, and set the initial value with <code>object.__setattr__(self, "_c", ...)</code> inside <code>__init__</code>.</p>

<h2>Attribute lookup and descriptors</h2>
<p><code>obj.x</code> runs <code>type(obj).__getattribute__(obj, "x")</code>, which roughly does:</p>
<ol>
<li>Find <code>x</code> on the class (walking the MRO). If it's a <strong>data descriptor</strong> (defines <code>__set__</code> or <code>__delete__</code>), call its <code>__get__(obj, type)</code>.</li>
<li>Otherwise, if <code>x</code> is in <code>obj.__dict__</code>, return it.</li>
<li>Otherwise, if the class attribute is a non-data descriptor (only <code>__get__</code>, like a function), call its <code>__get__</code> (that's how a function becomes a bound method).</li>
<li>Otherwise return the class attribute, or call <code>__getattr__</code> if defined, or raise <code>AttributeError</code>.</li>
</ol>
<p>A <strong>descriptor</strong> is any object with <code>__get__</code>/<code>__set__</code> stored as a class attribute. <code>property</code> is one; so are methods, <code>classmethod</code>, <code>staticmethod</code>, and dataclass fields at definition time. Your <code>Typed</code> descriptor validates on <code>__set__</code> and stores the value in the <em>instance's</em> <code>__dict__</code> under a private name it learns from <code>__set_name__(owner, name)</code>, which Python calls when the class body is executed. Accessed on the class (<code>obj is None</code> in <code>__get__</code>), return the descriptor itself. This is precisely how Django model fields, SQLAlchemy columns and pydantic's validation hooks work.</p>
<p><code>__setattr__</code> is the other hook: it runs on every assignment. <code>forge.nn.Module</code> uses it to sniff the type of the value and register Parameters and child Modules. <code>__getattr__</code> runs only when normal lookup <em>fails</em>, which makes it a safe place for fallbacks such as buffers. Make it raise <code>AttributeError</code> for unknown names, or <code>hasattr</code>, <code>copy</code> and <code>pickle</code> will misbehave.</p>

<h2>Iterators and generators</h2>
<p>An <em>iterable</em> has <code>__iter__</code> returning an <em>iterator</em>; an iterator has <code>__next__</code> raising <code>StopIteration</code> at the end. A generator function (one containing <code>yield</code>) returns an iterator whose body runs lazily, pausing at each <code>yield</code>. Consequences:</p>
<ul>
<li>Lazy pipelines over infinite or huge streams: <code>chunked(itertools.count(), 2)</code> must work.</li>
<li>Single pass: a generator is exhausted after one loop. Your <code>DataLoader.__iter__</code> is a generator function, so each <code>for</code> loop calls it afresh and reshuffles.</li>
<li><strong>Deferred execution trap</strong>: code before the first <code>yield</code> doesn't run until the first <code>next()</code>. If <code>chunked</code> validates <code>n</code> inside the generator, <code>chunked(x, 0)</code> doesn't raise until iteration. Fix: a normal function that validates, then returns an inner generator. The test checks this.</li>
</ul>
<h2>Context managers and decorators</h2>
<p><code>with cm as v:</code> calls <code>cm.__enter__()</code> (its return value is <code>v</code>) and guarantees <code>cm.__exit__(exc_type, exc, tb)</code> on the way out; returning a truthy value from <code>__exit__</code> <em>swallows</em> the exception, which is almost never what you want (your <code>Timer</code> must not). <code>@contextlib.contextmanager</code> builds one from a generator: code before <code>yield</code> is enter, the <code>finally</code> after it is exit.</p>
<p>A decorator is just <code>f = deco(f)</code>. A decorator <em>factory</em> like <code>@retry(times=3)</code> is a function returning a decorator: three levels of nesting. Wrap with <code>functools.wraps(fn)</code> so the result keeps <code>__name__</code>, <code>__doc__</code> and gets <code>__wrapped__</code> (tools like pytest and FastAPI introspect these).</p>
<pre><code>def retry(times, exceptions, backoff, sleep):     # factory
    def deco(fn):                                 # decorator
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):             # replacement function
            ...                                   # you write this part
        return wrapper
    return deco</code></pre>
<div class="callout prod"><b>In production</b>Reading framework source is a superpower: FastAPI's dependency injection is decorators plus signature introspection, pydantic is descriptors and metaclasses, pytest fixtures are decorators plus a registry, PyTorch's <code>nn.Module</code> is the <code>__setattr__</code> trick you wrote. When something "magic" breaks, these rules tell you where to look.</div>
`,
      quiz: [
        { q: "<code>Vector.__mul__(self, other)</code> receives a Vector. What should it do so that <code>v * w</code> raises TypeError?", options: ["raise TypeError", "return NotImplemented", "return None", "return 0"], answer: 1, why: "Returning NotImplemented lets Python try other.__rmul__; when that also declines, Python raises TypeError itself." },
        { q: "A class defines <code>__eq__</code> but not <code>__hash__</code>. Its instances are…", options: ["hashable by id", "hashable by value", "unhashable", "hashable only if frozen"], answer: 2, why: "Defining __eq__ sets __hash__ to None." },
        { q: "<code>gen = chunked(range(3), 0)</code> where chunked is a generator function that checks n inside. When is ValueError raised?", options: ["At the call", "On the first next(gen)", "Never", "At import time"], answer: 1, why: "A generator body doesn't run until iteration starts. Validate eagerly in an outer function." },
        { q: "In attribute lookup, which wins: a data descriptor on the class or an entry in the instance __dict__?", options: ["Instance __dict__", "Data descriptor", "Whichever was defined last", "__getattr__"], answer: 1, why: "Data descriptors take precedence; that's how property setters intercept assignment." },
      ],
      explain: "Explain how Python evaluates a * b when a is a numpy array and b is your Tensor, step by step, including NotImplemented and __array_priority__.",
      resources: [
        { title: "Python docs — Data model", url: "https://docs.python.org/3/reference/datamodel.html", note: "emulating numeric types; __slots__" },
        { title: "Python docs — Descriptor HowTo Guide", url: "https://docs.python.org/3/howto/descriptor.html", note: "read the 'pure Python equivalents'" },
        { title: "Luciano Ramalho — Fluent Python, 2nd ed.", url: "https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/", note: "ch. 1, 16 (operators), 17 (iterators), 23 (descriptors)" },
      ],
      cards: [
        { f: "What does Python do when a.__add__(b) returns NotImplemented?", b: "Tries b.__radd__(a); if that also returns NotImplemented, raises TypeError." },
        { f: "Equality/hash contract", b: "a == b ⇒ hash(a) == hash(b). Defining __eq__ alone makes instances unhashable." },
        { f: "Attribute lookup precedence", b: "Data descriptor on type > instance __dict__ > non-data descriptor/class attr > __getattr__." },
        { f: "When does a generator's body start running?", b: "At the first next(), not at the call." },
      ],
    },
    {
      id: "w5l2", title: "Types and API design", minutes: 55,
      summary: "Gradual typing with mypy --strict, Protocols for structural interfaces, dataclasses, and API conventions that make code hard to misuse.",
      keypoints: [
        "Type hints are checked statically by mypy; they don't change runtime behavior.",
        "Protocol = structural typing: any class with the right methods matches, no inheritance needed.",
        "Use field(default_factory=...) for mutable or computed dataclass defaults.",
        "TypeVar/generics keep input and output types connected; Self for fluent methods.",
        "Design APIs so illegal states can't be expressed: narrow inputs, explicit errors, consistent conventions.",
      ],
      body: R`
<h2>What --strict buys you</h2>
<p><code>mypy --strict</code> requires annotations on every function, forbids implicit <code>Any</code> and checks every call. In numeric code it catches the bugs tests miss: returning <code>float | None</code> where callers assume <code>float</code>, passing a <code>Tensor</code> where an <code>ndarray</code> is expected, forgetting a branch returns nothing. Annotate <code>forge.autograd</code> so that ops return <code>Tensor</code>, <code>backward</code> returns <code>None</code>, and closures have types (<code>Callable[[], None]</code>). numpy arrays are <code>np.ndarray</code> (or <code>numpy.typing.NDArray[np.float64]</code> if you want dtypes); numpy calls sometimes return <code>Any</code>, which strict mode flags when you return it from a typed function. Wrap with <code>float(...)</code>, <code>np.asarray(...)</code> or a typed local variable.</p>
<h2>Protocols: interfaces without inheritance</h2>
<p><code>serve/engine.py</code> defines</p>
<pre><code>class Engine(Protocol):
    def generate_batch(self, reqs: list[GenerateRequest]) -> list[str]: ...
    def stream(self, req: GenerateRequest) -> Iterator[str]: ...</code></pre>
<p>Any class with those two methods <em>is</em> an <code>Engine</code> to mypy: <code>GPTEngine</code>, the test's <code>FakeEngine</code>, a future vLLM adapter. No base class, no registration. That's dependency inversion in Python: the HTTP layer depends on the protocol, not on PyTorch, which is exactly why the Week 6 app tests run without a GPU.</p>
<h2>Dataclasses</h2>
<p><code>@dataclass</code> generates <code>__init__</code>, <code>__repr__</code> and <code>__eq__</code> from annotations; <code>frozen=True</code> makes instances immutable and hashable (<code>GenerateRequest</code> is frozen: it's used to group requests). Mutable defaults are a classic trap: <code>x: list = []</code> would share one list across instances, so dataclasses forbid it; use <code>field(default_factory=list)</code>. <code>TrainConfig.device</code> uses <code>default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"</code> so the check happens per instance, not once at import.</p>
<h2>Generics</h2>
<p><code>def chunked(iterable: Iterable[T], n: int) -> Iterator[list[T]]</code> tells mypy that chunking strings yields lists of strings. <code>retry</code>'s signature uses a <code>TypeVar R</code> so the decorated function keeps its return type (for full fidelity including arguments, use <code>ParamSpec</code>). <code>typing.Self</code> types methods that return their own instance, like <code>fit</code> returning <code>self</code> and <code>Timer.__enter__</code>.</p>
<h2>API design that resists misuse</h2>
<ul>
<li><strong>Consistent conventions</strong>: sklearn's <code>fit</code> returns <code>self</code>, learned attributes end in <code>_</code>, constructor only stores hyperparameters. Users learn it once.</li>
<li><strong>Fail loudly at the boundary</strong>: <code>GPT.forward</code> raises <code>ValueError</code> for sequences longer than <code>block_size</code> instead of producing silent garbage from an out-of-range position embedding.</li>
<li><strong>Make illegal states unrepresentable</strong>: a frozen request object can't be mutated after it's queued; a cache object knows its own length so callers can't pass the wrong offset.</li>
<li><strong>Return values, not flags</strong>: <code>recommend()</code> returns <code>(items, used_fallback)</code> rather than writing to a global.</li>
</ul>
<div class="callout prod"><b>In production</b>Type checking in CI is cheap insurance on large ML codebases where a tensor-shape or Optional bug can silently corrupt a training run. Protocols and frozen dataclasses are the standard tools for clean service boundaries, and the same ideas carry to pydantic models at API edges (Week 6).</div>
`,
      quiz: [
        { q: "Why does <code>@dataclass class C: items: list = []</code> fail?", options: ["Lists can't be annotated", "A mutable default would be shared by all instances; use field(default_factory=list)", "Dataclasses can't contain lists", "It needs frozen=True"], answer: 1, why: "Dataclasses raise ValueError for mutable defaults to prevent the shared-default bug." },
        { q: "FakeEngine in the tests doesn't inherit from Engine. Does mypy accept it where Engine is expected?", options: ["No, it must subclass", "Yes, Protocol uses structural typing", "Only with a cast", "Only at runtime"], answer: 1, why: "Matching method signatures are enough for a Protocol." },
      ],
      explain: "Explain what a typing.Protocol is and why the Week 6 server depends on an Engine protocol instead of importing your GPT class directly.",
      resources: [
        { title: "mypy docs — Protocols and structural subtyping", url: "https://mypy.readthedocs.io/en/stable/protocols.html", note: "" },
        { title: "PEP 484 / PEP 544 / PEP 673 (Self)", url: "https://peps.python.org/pep-0544/", note: "" },
        { title: "Python docs — dataclasses", url: "https://docs.python.org/3/library/dataclasses.html", note: "" },
      ],
      cards: [
        { f: "Protocol in one sentence", b: "Structural interface: any class with matching methods satisfies it, no inheritance required." },
        { f: "Mutable dataclass default", b: "field(default_factory=...)" },
        { f: "Type for a method returning its own instance", b: "typing.Self" },
      ],
    },
    {
      id: "w5l3", title: "Testing numerical and ML code", minutes: 65,
      summary: "pytest fixtures and parametrization, property-based testing with hypothesis, and the oracle strategies this course's grader is built on.",
      keypoints: [
        "Fixtures provide setup with scopes (function, module); tmp_path and monkeypatch are built in.",
        "parametrize turns one test into a table of cases; markers select subsets (slow, stretch).",
        "Property-based tests assert invariants over generated inputs; hypothesis shrinks failures to minimal examples.",
        "Numerical oracles: finite differences, a trusted reference implementation, closed-form cases, invariants.",
        "Compare floats with tolerances; seed every RNG; keep slow tests out of the default run.",
      ],
      body: R`
<h2>pytest essentials</h2>
<ul>
<li><strong>Fixtures</strong>: a function decorated with <code>@pytest.fixture</code>, requested by naming it as a test argument. Scope controls reuse: <code>scope="module"</code> builds the tiny GPT checkpoint once for all Week 6 engine tests, and the Week 8 pipeline fixture trains once for three tests. Built-ins you've seen: <code>tmp_path</code> (fresh directory), <code>monkeypatch</code> (temporarily replace attributes; used to count <code>Tensor.backward</code> calls), <code>capsys</code> (capture stdout for CLI tests).</li>
<li><strong>Parametrize</strong>: <code>@pytest.mark.parametrize("stride,pad,k", [...])</code> runs one test body per row; Week 1's 28 op gradchecks are one function.</li>
<li><strong>Markers</strong>: <code>slow</code> and <code>stretch</code> are deselected by default via <code>addopts</code> in pyproject; <code>-m slow</code> selects them. <code>-k expr</code> selects by name (and module name, which is why some milestone filters end in an underscore).</li>
<li><code>pytest.raises(E)</code> passes for subclasses of <code>E</code> too. <code>NotImplementedError</code> is a subclass of <code>RuntimeError</code>, so a stub would "pass" a <code>raises(RuntimeError)</code> test. The course grader uses a custom exception in such cases.</li>
</ul>
<h2>Oracles for numerical code</h2>
<p>The hardest part of testing ML code is knowing the right answer. This course's grader uses every standard strategy, and you should use them in your own work:</p>
<ol>
<li><strong>A different algorithm for the same quantity</strong>: finite differences for gradients (the grader's own <code>numgrad</code>, not yours), a naive loop for conv and pooling.</li>
<li><strong>A trusted reference implementation</strong> (differential testing): sklearn for trees, AUC, PCA; scipy for KS and Gaussian densities; PyTorch for optimizer trajectories.</li>
<li><strong>Hand-computed cases</strong>: the BPE Wikipedia example, the PSI example, the A/B sample size.</li>
<li><strong>Invariants</strong>: EM never decreases likelihood; adding a node to a hash ring only moves keys to it; cached decoding equals full decoding; causal logits don't depend on the future.</li>
<li><strong>End-to-end thresholds</strong>: MNIST accuracy, Shakespeare val loss, recall vs popularity. Calibrated by running a reference and leaving margin.</li>
</ol>
<p>Every threshold in this course's tests was validated against a private reference solution before you got the repo, so a failing test means your code, not the test, unless you've found a real bug in the grader (tell the tutor why you think so).</p>
<h2>Property-based testing</h2>
<p>Instead of hand-picking inputs, describe the space of inputs and a property that must hold for all of them. hypothesis generates hundreds of cases, including nasty ones (empty arrays, huge and tiny floats, repeated values), and when one fails it <em>shrinks</em> it to a minimal counterexample.</p>
<pre><code>@given(arrays(np.float64, st.integers(1, 20), elements=st.floats(-10, 10)), st.floats(-10, 10))
def test_log_softmax_is_shift_invariant(x, c):
    np.testing.assert_allclose(Tensor(x + c).log_softmax().data,
                               Tensor(x).log_softmax().data, atol=1e-8)</code></pre>
<p>Good autograd properties: linearity of gradients, shift invariance of log-softmax, rows of softmax summing to 1, gradient accumulation doubling, the adjoint identity for unbroadcast. Your milestone: at least five. Keep element ranges bounded (exp overflows) and use tolerances.</p>
<div class="callout pitfall"><b>Flaky tests</b>Tests that depend on unseeded randomness or wall-clock timing fail intermittently and train people to ignore red builds. Seed every generator, compare with tolerances, and give timing assertions generous margins (the course's speed tests are set 3–10× above the reference timings).</div>
<div class="callout prod"><b>In production</b>ML teams test data pipelines (schemas, null rates, leakage checks), training code (overfit-one-batch tests, gradient checks for custom ops), and serving (contract tests against a fake model, exactly like the Week 6 app tests). Model quality itself is gated by evaluation thresholds in CI, the same pattern as the slow tests.</div>
`,
      quiz: [
        { q: "A test asserts <code>with pytest.raises(RuntimeError): f()</code> and f's stub raises NotImplementedError. The test…", options: ["fails", "passes, because NotImplementedError subclasses RuntimeError", "errors", "is skipped"], answer: 1, why: "raises() matches subclasses. Use a specific exception type when the stub's error could satisfy the check." },
        { q: "Which is a property (invariant) test rather than an example test?", options: ["assert add(2, 3) == 5", "for all arrays x and constants c, log_softmax(x + c) == log_softmax(x)", "assert model accuracy ≥ 0.97 on MNIST", "assert the file exists"], answer: 1, why: "It quantifies over generated inputs." },
      ],
      explain: "Describe four different 'oracle' strategies for testing numerical code and give an example of each from this course's grader.",
      resources: [
        { title: "pytest docs — fixtures, parametrize, markers", url: "https://docs.pytest.org/en/stable/how-to/fixtures.html", note: "" },
        { title: "Hypothesis docs — quickstart and numpy strategies", url: "https://hypothesis.readthedocs.io/en/latest/quickstart.html", note: "" },
        { title: "Breck et al. — The ML Test Score", url: "https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/", note: "production ML testing rubric" },
      ],
      cards: [
        { f: "What does hypothesis do when a property fails?", b: "Shrinks the failing input to a minimal counterexample and reports it." },
        { f: "Five oracle strategies for numerical code", b: "Alternative algorithm, reference implementation, hand-computed cases, invariants, end-to-end thresholds." },
        { f: "Fixture scope for an expensive shared setup", b: "scope=\"module\" (or session)." },
      ],
    },
    {
      id: "w5l4", title: "Performance and concurrency", minutes: 80,
      summary: "Measure first, then fix what the profiler shows: vectorization, memory layout and copies; then the GIL and when to use threads, processes or asyncio.",
      keypoints: [
        "Profile before optimizing: cProfile for where time goes, timeit for micro-benchmarks, py-spy for live processes.",
        "numpy speed comes from doing work in C on contiguous memory; Python loops and strided access are the enemies.",
        "The GIL lets one thread run Python bytecode at a time; numpy and I/O release it.",
        "Threads for I/O-bound work, processes for CPU-bound Python, asyncio for many concurrent I/O tasks.",
        "Never block the event loop: run blocking calls with asyncio.to_thread; bound concurrency with a Semaphore.",
      ],
      body: R`
<h2>Measure first</h2>
<p>Intuition about performance is wrong often enough that you should never optimize without a measurement. Tools:</p>
<ul>
<li><code>python -m cProfile -s tottime script.py</code>: per-function time. <em>tottime</em> excludes callees (where work actually happens); <em>cumtime</em> includes them (which call tree is expensive).</li>
<li><code>timeit</code> / <code>%timeit</code>: micro-benchmarks with repetition; report the median, not a single run.</li>
<li><code>py-spy top --pid N</code>: sampling profiler for a running process (training jobs, servers), no code changes.</li>
<li>Your <code>benchmarks/bench_conv.py</code>: a fixed workload, warm-up run, median of 5. Record the baseline before changing anything.</li>
</ul>
<p><strong>Amdahl's law</strong>: if a part taking fraction $f$ of the runtime is sped up by $s$, the total speedup is $1/((1-f) + f/s)$. Making a 10% component infinitely fast saves at most 10%. The profiler tells you $f$.</p>
<div class="callout pitfall"><b>Benchmark hygiene</b>Background load changes results dramatically. While calibrating this course, the same numpy matmul measured 3 ms and 1.9 s depending on whether a GPU training job was running at the same time (most likely BLAS threads contending with the training process for cores), and the conv benchmark dropped from ~530 ms to ~100 ms once the job finished. Benchmark on a quiet machine, repeat, and compare medians.</div>
<h2>Making numpy fast</h2>
<ul>
<li><strong>Vectorize</strong>: replace Python loops over elements with whole-array operations. The CART split search and the k-means distance matrix are the canonical examples.</li>
<li><strong>Memory layout</strong>: arrays are row-major (C order) by default. Operations along contiguous memory are fast; slicing the last axis of a big array with a stride pulls a cache line for every element. Transposing to put the axis you iterate over first, then <code>np.ascontiguousarray</code> once, can be several times faster than repeated strided access (the col2im loop).</li>
<li><strong>Avoid temporaries</strong>: <code>a * b + c</code> allocates intermediates; for huge arrays use <code>out=</code> or in-place ops.</li>
<li><strong>Use float32</strong> when precision allows: half the memory traffic, and BLAS is faster.</li>
<li><strong>Avoid <code>np.add.at</code> in hot paths</strong>: correct for repeated indices but unbuffered and slow. When indices have structure (like convolution windows), restructure as a few strided slab additions.</li>
</ul>
<h2>The GIL and concurrency models</h2>
<p>CPython's Global Interpreter Lock lets only one thread execute Python bytecode at a time. C extensions (numpy, PyTorch, I/O syscalls) release it while working.</p>
<table>
<tr><th>workload</th><th>tool</th><th>why</th></tr>
<tr><td>Waiting on network/disk, a few dozen tasks</td><td><code>ThreadPoolExecutor</code></td><td>GIL released while blocked</td></tr>
<tr><td>CPU-bound pure Python</td><td><code>ProcessPoolExecutor</code> / multiprocessing</td><td>separate interpreters, separate GILs; arguments are pickled</td></tr>
<tr><td>Thousands of concurrent I/O tasks (a server)</td><td><code>asyncio</code></td><td>one thread, cooperative scheduling, tiny per-task cost</td></tr>
<tr><td>Big numpy/torch ops</td><td>just call them</td><td>already parallel in C (BLAS threads, CUDA)</td></tr>
</table>
<p>Your <code>parallel_map</code> uses processes, so <code>fn</code> must be picklable (a module-level function, not a lambda), and on Windows child processes re-import your module, which is why the test's helper is top-level. Python 3.13's experimental free-threaded build removes the GIL, but the ecosystem still assumes it.</p>
<h2>asyncio in one page</h2>
<p>An <code>async def</code> function returns a coroutine; <code>await</code> suspends it until the awaited thing is ready, letting the event loop run other coroutines. Everything runs on one thread, so:</p>
<ul>
<li><strong>Never block the loop.</strong> A <code>time.sleep(0.05)</code> or a 50 ms GPU forward inside a coroutine freezes every other request. Use <code>await asyncio.to_thread(fn, ...)</code>. Your Week 6 batcher must do this for synchronous <code>process_batch</code> functions; a test measures that the loop keeps ticking.</li>
<li><strong>Bound concurrency.</strong> <code>asyncio.gather</code> over 10,000 coroutines opens 10,000 connections at once. An <code>asyncio.Semaphore(limit)</code> gives backpressure: your <code>gather_limited</code>.</li>
<li><strong>Futures connect producers and consumers</strong>: a caller awaits a <code>Future</code>, a background task later calls <code>set_result</code>. That's the core of the dynamic batcher.</li>
</ul>
<div class="callout prod"><b>In production</b>Model servers are asyncio front-ends (FastAPI/uvicorn) handing work to GPU threads or processes; data loaders use worker processes to escape the GIL; and the most common performance bug in ML services is a synchronous call blocking the event loop, which shows up as p99 latency spikes under load.</div>
`,
      quiz: [
        { q: "A function taking 20% of runtime is made 4× faster. Overall speedup?", options: ["4×", "1.18×", "1.8×", "1.25×"], answer: 1, why: "1/((1−0.2) + 0.2/4) = 1/0.85 ≈ 1.18." },
        { q: "You need to compute a pure-Python scoring function over 1M items on 8 cores. Best tool?", options: ["threads", "asyncio", "ProcessPoolExecutor", "a bigger batch size"], answer: 2, why: "CPU-bound Python holds the GIL; processes run truly in parallel." },
        { q: "Inside an async request handler you call model(x) which takes 40 ms on the GPU synchronously. Effect on a server with 100 concurrent requests?", options: ["None", "Every other request waits: the event loop is blocked for 40 ms each call", "The GPU runs them in parallel automatically", "uvicorn spawns threads automatically"], answer: 1, why: "Blocking calls stall the single-threaded loop. Use asyncio.to_thread or a batcher with a worker." },
      ],
      explain: "Explain the GIL and give a concrete example from this course of when you'd use threads, processes, and asyncio.",
      resources: [
        { title: "Python docs — The Python Profilers", url: "https://docs.python.org/3/library/profile.html", note: "" },
        { title: "Jake VanderPlas — Losing your Loops (talk)", url: "https://www.youtube.com/watch?v=EEUXKG97YRw", note: "numpy vectorization" },
        { title: "Python docs — asyncio (tasks, synchronization, to_thread)", url: "https://docs.python.org/3/library/asyncio-task.html", note: "" },
        { title: "Gorelick & Ozsvald — High Performance Python", url: "https://www.oreilly.com/library/view/high-performance-python/9781492055013/", note: "" },
      ],
      cards: [
        { f: "Amdahl's law", b: "Speedup = 1/((1−f) + f/s)" },
        { f: "Threads vs processes vs asyncio", b: "Threads: I/O-bound. Processes: CPU-bound Python. asyncio: many concurrent I/O tasks on one thread." },
        { f: "How to call blocking code from a coroutine", b: "await asyncio.to_thread(fn, *args)" },
        { f: "tottime vs cumtime in cProfile", b: "tottime excludes time in callees; cumtime includes it." },
      ],
    },
    {
      id: "w5l5", title: "Packaging and tooling", minutes: 50,
      summary: "pyproject.toml, editable installs and console scripts, lint/format/type gates, and a CI pipeline that runs them on every push.",
      keypoints: [
        "pyproject.toml declares metadata, dependencies, optional extras and tool config in one file.",
        "pip install -e . makes your source importable everywhere; edits take effect without reinstalling.",
        "[project.scripts] creates a console command bound to a function.",
        "ruff (lint + format) and mypy run locally via pre-commit and remotely in CI.",
        "CI should be fast: run fast tests on every push; slow/GPU tests nightly or manually.",
      ],
      body: R`
<h2>pyproject.toml</h2>
<p>Your repo already has one: <code>[build-system]</code> (setuptools), <code>[project]</code> (name <code>forge-ml</code>, version, <code>dependencies</code>), <code>[project.optional-dependencies]</code> (extras: <code>pip install -e ".[dev,w4]"</code>), and <code>[tool.*]</code> sections that pytest, ruff and mypy read. This week you add:</p>
<pre><code>[project.scripts]
forge = "forge.cli:main"

[tool.ruff]
line-length = 120

[tool.mypy]
python_version = "3.13"</code></pre>
<p>A console script is a tiny generated executable that imports <code>forge.cli</code> and calls <code>main()</code>, using its return value as the exit code. That's why <code>main(argv=None) -> int</code> returns codes instead of calling <code>sys.exit</code>: testable, and composable.</p>
<h3>Editable installs</h3>
<p><code>pip install -e .</code> registers your source directory with the environment, so <code>import forge</code> works from any directory and <code>importlib.metadata.version("forge-ml")</code> reports your version, while edits take effect immediately. Until now the repo-root <code>conftest.py</code> made imports work only under pytest.</p>
<h2>argparse CLIs</h2>
<p>Subcommands via <code>add_subparsers(dest="cmd")</code>; each subparser has its own arguments. argparse calls <code>sys.exit(2)</code> on usage errors, so catch <code>SystemExit</code> inside <code>main</code> and return its code (the test for an unknown command expects 2). Keep heavy imports (torch) inside the subcommand handlers so <code>forge --version</code> is instant.</p>
<h2>Quality gates</h2>
<ul>
<li><strong>ruff</strong>: linter and formatter in one fast tool. Your ruff version's default rules include pyupgrade-style modernizations (<code>collections.abc</code> imports, unquoted annotations) and import sorting. <code>ruff check --fix</code> handles most automatically.</li>
<li><strong>mypy --strict</strong> on the core modules (L2).</li>
<li><strong>pre-commit</strong> hooks run ruff and mypy on staged files before each commit (stretch).</li>
</ul>
<h2>CI with GitHub Actions</h2>
<pre><code>name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.13"}
      - run: pip install -e ".[dev]"
      - run: ruff check forge pyeng
      - run: mypy --strict forge/autograd.py forge/optim.py forge/nn
      - run: pytest -q</code></pre>
<p>This runs fast tests only (slow tests are deselected by default; there's no GPU or dataset on the runner). A real project adds dependency caching, a Python version matrix, and a nightly job for slow tests. Push the repo to GitHub to see it run: a green CI badge on a from-scratch ML stack is a strong portfolio signal.</p>
<div class="callout prod"><b>In production</b>Reproducible environments are an ML problem as much as a software one: pin dependency versions (a lock file via uv or pip-tools), record the exact package set with each trained model, and build the serving container from the same lock (Week 6's Dockerfile).</div>
`,
      quiz: [
        { q: "Why should <code>main()</code> return an int instead of calling sys.exit?", options: ["sys.exit is deprecated", "It makes main testable and lets the console-script wrapper use the return value as the exit code", "argparse requires it", "It's faster"], answer: 1, why: "Tests can call main([...]) and check the code without catching SystemExit." },
        { q: "What does <code>pip install -e .</code> change?", options: ["Copies the code into site-packages", "Links the source tree into the environment so imports resolve to your working copy", "Compiles the package", "Creates a virtualenv"], answer: 1, why: "Editable installs point the environment at your source directory." },
      ],
      explain: "Explain what happens when you type `forge --version` after an editable install: how the shell finds the command, how it reaches your code, and where the exit code comes from.",
      resources: [
        { title: "Python Packaging User Guide — Writing pyproject.toml", url: "https://packaging.python.org/en/latest/guides/writing-pyproject-toml/", note: "" },
        { title: "ruff docs", url: "https://docs.astral.sh/ruff/", note: "" },
        { title: "GitHub Actions — Building and testing Python", url: "https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python", note: "" },
      ],
      cards: [
        { f: "pyproject section for a CLI command", b: "[project.scripts] name = \"package.module:function\"" },
        { f: "argparse exit code on a usage error", b: "2 (via SystemExit)" },
      ],
    },
  ],
  project: {
    title: "Productionize forge",
    dir: "pyeng/, forge/cli.py, pyproject.toml, .github/, benchmarks/, docs/",
    pitch: "Implement the Python drills, then turn your codebase into a real package: CLI, editable install, mypy --strict, ruff, five hypothesis properties, CI, and a profiling-driven speedup of your conv block with a written analysis.",
    test: "pytest tests/w5 -q",
    overview: R`
<p>Two halves. The drills (<code>pyeng/drills.py</code>, 11 tests) exercise the language features from L1 and L4. The engineering checks (<code>tests/w5/test_engineering.py</code>) verify deliverables in your repo: pyproject entries, an editable install, a working CLI, clean mypy/ruff runs, a CI workflow, your own property tests, and a conv-block benchmark under 400 ms (the reference is ~100 ms) with a write-up.</p>
<p>Before touching your conv code, record the baseline: <code>python benchmarks/bench_conv.py --record baseline</code>. Optimize, then <code>--record optimized</code>, and explain what the profiler showed in <code>docs/w5_perf.md</code> (template in <code>docs/templates/perf.md</code>). If your Week 3 conv is already under 400 ms, profile anyway, pick the biggest remaining hotspot (often MaxPool's <code>np.add.at</code>), and document that improvement.</p>
`,
    milestones: [
      { id: "w5m1", core: true, title: "Drills: Vector, Typed, chunked, sliding_window, Timer, retry", test: "pytest tests/w5/test_drills.py -q -k \"vector or typed or chunked or sliding or timer or retry\"",
        detail: R`<p>Data model, descriptor, generator laziness, context manager and decorator factory.</p>`,
        hints: [
          R`<p>Vector: <code>__slots__ = ("_c",)</code>, set it with <code>object.__setattr__</code> in <code>__init__</code>, override <code>__setattr__</code> to raise AttributeError. Return NotImplemented for wrong operand types.</p>`,
          R`<p>Typed: <code>__set_name__(self, owner, name)</code> stores a private key like <code>"_typed_" + name</code>; <code>__get__</code> returns <code>self</code> when <code>obj is None</code>, else reads <code>obj.__dict__</code> (KeyError → AttributeError); <code>__set__</code> validates then writes.</p>`,
          R`<p>chunked: validate n, then <code>return gen()</code> where the inner generator loops <code>while chunk := list(itertools.islice(it, n)): yield chunk</code>. sliding_window: a <code>deque(maxlen=n)</code>.</p>`,
        ] },
      { id: "w5m2", core: true, title: "Drills: gather_limited and parallel_map", test: "pytest tests/w5/test_drills.py -q -k \"gather or parallel\"",
        detail: R`<p>Exactly <code>limit</code> coroutines in flight, results in input order, errors propagate; work in child processes.</p>`,
        hints: [
          R`<p>Wrap each factory: <code>async with sem: return await f()</code>, then <code>asyncio.gather(*wrapped)</code> preserves order.</p>`,
          R`<p><code>with ProcessPoolExecutor(workers) as ex: return list(ex.map(fn, items))</code>.</p>`,
        ] },
      { id: "w5m3", core: true, title: "CLI and packaging", test: "pip install -e \".[dev]\"; pytest tests/w5/test_engineering.py -q -k \"cli or pyproject_declares or installed\"",
        detail: R`<p><code>forge --version</code>, <code>train-mnist</code>, <code>train-gpt</code>, <code>generate</code>; usage errors return 2.</p>`,
        hints: [
          R`<p>Wrap <code>ap.parse_args(argv)</code> in <code>try/except SystemExit as e: return int(e.code or 0)</code>.</p>`,
          R`<p>Dispatch with a dict from subcommand name to handler function; import torch/numpy inside handlers.</p>`,
          R`<p>If <code>importlib.metadata.version("forge-ml")</code> fails after installing, check you're running the venv's Python (<code>where python</code>).</p>`,
        ] },
      { id: "w5m4", core: true, title: "mypy --strict and ruff", test: "pytest tests/w5/test_engineering.py -q -k \"mypy or ruff or tool_config\"",
        detail: R`<p>Annotate <code>forge/autograd.py</code>, <code>forge/optim.py</code> and <code>forge/nn</code> until <code>mypy --strict</code> passes; add <code>[tool.mypy]</code> and <code>[tool.ruff]</code>.</p>`,
        hints: [
          R`<p>Run mypy directly to iterate: <code>mypy --strict forge/autograd.py</code>. Fix module by module, starting with autograd (everything imports it).</p>`,
          R`<p>Closures: <code>self._backward: Callable[[], None] | None = None</code>. For numpy results typed as Any, assign to an annotated local first.</p>`,
          R`<p><code>ruff check --fix forge pyeng</code> clears most lint automatically; read what remains, don't blanket-ignore.</p>`,
        ] },
      { id: "w5m5", core: true, title: "Five property-based tests", test: "pytest tests/w5/test_properties.py -q",
        detail: R`<p>Add at least five <code>@given</code> tests to <code>tests/w5/test_properties.py</code> (one example is there).</p>`,
        hints: [R`<p>Ideas are in the file's docstring. Use <code>hypothesis.extra.numpy.arrays</code> with bounded float elements, and <code>assert_allclose</code> with tolerances.</p>`] },
      { id: "w5m6", core: true, title: "CI workflow", test: "pytest tests/w5/test_engineering.py -q -k ci_workflow",
        detail: R`<p>Create <code>.github/workflows/ci.yml</code> that installs, lints, type-checks and tests. Push to GitHub to see it run (optional but recommended).</p>`,
        hints: [R`<p>Copy the YAML from L5 and adapt; YAML indentation errors are the usual failure.</p>`] },
      { id: "w5m7", core: true, title: "Profile and speed up the conv block", test: "pytest tests/w5/test_engineering.py -q -k performance",
        detail: R`<p>Record baseline, profile, optimize, record optimized, write <code>docs/w5_perf.md</code>. Target &lt; 400 ms.</p>`,
        hints: [
          R`<p><code>python -m cProfile -s tottime benchmarks/bench_conv.py | head -25</code>. Is the time in your conv backward, maxpool backward, or autograd overhead?</p>`,
          R`<p>Common wins: custom conv op instead of fancy-index composition; col2im as k² strided slab adds over a contiguous (k,k,N,C,Ho,Wo) array; float32 end to end.</p>`,
          R`<p>MaxPool with stride == kernel and divisible sizes: reshape to (N,C,H/k,k,W/k,k) and reduce; the backward becomes a broadcasted mask multiply instead of <code>np.add.at</code>.</p>`,
        ] },
      { id: "w5m8", core: false, title: "Stretch: pre-commit, coverage gate, numba",
        detail: R`<p>Add a <code>.pre-commit-config.yaml</code> with ruff and mypy; add <code>pytest-cov</code> and fail CI under 85% coverage for <code>forge/</code>; try <code>numba.njit</code> on the tree split search and compare.</p>`,
        hints: [R`<p>Coverage on stubs-turned-code is a good way to find untested branches in your autograd (e.g. keepdims paths).</p>`] },
    ],
  },
});
