"""
Per-request streaming metrics for OpenVINO GenAI's ContinuousBatchingPipeline.

All prompts are submitted at t=0. Each gets a live tqdm bar that updates as
tokens arrive during decode, so you can watch the batch progress in real time
and see which requests finish early. Per-request metrics print after all bars
close.

Requires: pip install tqdm
"""

import time
from dataclasses import dataclass

import openvino_genai as ov_genai
from openvino_genai.py_openvino_genai import GenerationHandle
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

MODEL_PATH = (
    "/mnt/Ironwolf-4TB/Models/OpenVINO/Deepseek/"
    "DeepSeek-R1-0528-Qwen3-8B-OpenVINO/"
    "DeepSeek-R1-0528-Qwen3-8B-int4_asym-ov/"
)
DEVICE = "GPU.0"
MAX_NEW_TOKENS = 8192

PROMPTS = [
    # ~80 tokens — short reasoning
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Show your reasoning step by step and "
    "double-check the final answer by plugging it back into the original "
    "constraints.",
 
    # ~120 tokens — code explanation
    "Explain what the following Python expression evaluates to and why: "
    "`sorted([{'a': 3}, {'a': 1}, {'a': 2}], key=lambda d: d['a'])[-1]['a']`. "
    "Walk through the evaluation order: first the lambda key function, then "
    "how sorted applies it, then the list indexing, then the dict lookup. "
    "Mention what would happen if one of the dicts were missing the 'a' key.",
 
    # ~150 tokens — technical comparison
    "Compare and contrast continuous batching and static batching for LLM "
    "inference. Cover the following points: how each handles requests "
    "arriving at different times, the impact on time-to-first-token for a "
    "request arriving mid-batch, GPU utilization under bursty traffic, "
    "memory management implications (particularly KV cache), and the "
    "implementation complexity tradeoff. Conclude with a recommendation for "
    "a production serving system handling ~100 concurrent users with highly "
    "variable prompt lengths.",
 
    # ~200 tokens — math problem with setup
    "A factory produces widgets on three machines, A, B, and C. Machine A "
    "produces 40% of the widgets, B produces 35%, and C produces 25%. The "
    "defect rates are 2% for A, 3% for B, and 5% for C. A widget is selected "
    "at random from the day's production and found to be defective. What is "
    "the probability that it came from machine C? Use Bayes' theorem, show "
    "the full computation including the law of total probability for the "
    "denominator, and express the final answer as both a fraction and a "
    "percentage rounded to two decimal places. Also compute the corresponding "
    "probabilities for machines A and B so the three sum to 1, as a sanity "
    "check on the arithmetic.",
 
    # ~250 tokens — algorithm design
    "Design an algorithm to find the k-th smallest element in the union of "
    "two sorted arrays of sizes m and n, without merging them. Your solution "
    "should run in O(log(min(m, n))) time. Walk through the intuition first: "
    "why this is essentially a binary search over partition points rather "
    "than over values. Then describe the invariants your algorithm "
    "maintains at each step, specifically what it means for a partition of "
    "the two arrays to be 'correct' for the k-th element. Provide pseudocode "
    "with clear variable names. Discuss the edge cases your code must "
    "handle: k = 1, k = m + n, one array being empty, and the case where "
    "all elements of one array are smaller than all elements of the other. "
    "Finally, sketch how you would test the implementation with a small "
    "example, choosing arrays and a k value that exercises a non-trivial "
    "partition.",
 
    # ~300 tokens — code review
    "Review this Python function for correctness, efficiency, and style. "
    "Identify any bugs, suggest improvements, and rewrite it in a cleaner "
    "form.\n\n"
    "```python\n"
    "def find_duplicates(lst):\n"
    "    duplicates = []\n"
    "    for i in range(len(lst)):\n"
    "        for j in range(len(lst)):\n"
    "            if i != j and lst[i] == lst[j]:\n"
    "                if lst[i] not in duplicates:\n"
    "                    duplicates.append(lst[i])\n"
    "    return duplicates\n"
    "```\n\n"
    "In your review, address: (1) the time complexity of the current "
    "implementation and what it should be, (2) whether the function handles "
    "unhashable elements correctly and whether it needs to, (3) whether "
    "order of the output matters and how the current code handles it, (4) "
    "what happens with inputs like an empty list, a list with no "
    "duplicates, or a list where every element is the same. Then provide "
    "a rewritten version using collections.Counter, and a second version "
    "using a set-based single-pass approach. Compare the two rewrites and "
    "explain when you'd prefer each.",
 
    # ~350 tokens — systems debugging
    "You are debugging a production issue where a Python web service's "
    "p99 latency has jumped from 50ms to 800ms over the past week, while "
    "p50 latency is unchanged at around 15ms and CPU utilization across "
    "the fleet has actually decreased slightly. Request volume is flat. "
    "The service is a relatively simple HTTP API backed by PostgreSQL "
    "and a Redis cache, deployed across 12 instances behind a load "
    "balancer. No code has been deployed in two weeks. Walk through your "
    "investigation methodology. What hypotheses would you generate from "
    "this symptom pattern, and in what order would you test them? "
    "Specifically address: why p50 being stable but p99 spiking is "
    "informative, what the CPU utilization clue tells you about whether "
    "the bottleneck is compute-bound, which observability signals you "
    "would pull first (database metrics, GC pauses, network latency, "
    "lock contention, downstream service latencies), and how you would "
    "rule each candidate cause in or out. Conclude with what a runbook "
    "entry for this incident pattern should contain for the next on-call "
    "engineer who sees similar symptoms.",
 
    # ~400 tokens — physics explanation
    "Explain why the sky is blue, going deeper than the usual 'Rayleigh "
    "scattering' one-liner. Build the explanation in layers. First, "
    "describe what scattering means physically: an incoming "
    "electromagnetic wave drives bound electrons in air molecules into "
    "oscillation, and those oscillating charges re-radiate. Second, "
    "derive (or at least motivate) the inverse-fourth-power wavelength "
    "dependence of Rayleigh scattering — why short wavelengths are "
    "scattered so much more strongly than long ones. Third, address the "
    "follow-up question this immediately raises: if violet is scattered "
    "even more strongly than blue, why doesn't the sky look violet? "
    "Cover both the solar spectrum (less violet emission than blue) and "
    "the response of human cone cells. Fourth, explain why sunsets are "
    "red using the same framework — the geometry of light traveling "
    "through more atmosphere at low sun angles. Finally, describe what "
    "the sky looks like on Mars and why it differs (hint: it's not just "
    "the thinner atmosphere; the dominant scatterers are different in "
    "kind, not just in density). Throughout, keep the explanation "
    "accessible to someone with high-school physics but don't shy away "
    "from quantitative claims when they sharpen the picture.",
 
    # ~450 tokens — open-ended analysis
    "Analyze the tradeoffs between monolithic and microservices "
    "architectures for a hypothetical mid-stage startup: roughly 50 "
    "engineers, $20M ARR, growing 80% year-over-year, currently running "
    "a single Ruby on Rails monolith that has become difficult to deploy "
    "(deploys take 45 minutes, deploy failures are common, and engineers "
    "frequently step on each other's changes). The CTO is considering a "
    "migration to microservices and has asked you to write a memo. Cover "
    "the following in your memo. First, identify what problems "
    "microservices actually solve and which of the company's current "
    "pain points would and would not be addressed by such a migration. "
    "Second, identify the new problems that microservices would "
    "introduce — distributed tracing complexity, network reliability "
    "concerns, data consistency across services, increased operational "
    "burden, the difficulty of refactoring across service boundaries — "
    "and how much of an organization at this size and growth rate would "
    "need to invest to handle them well. Third, present an alternative "
    "intermediate path: keeping the monolith but adopting modular "
    "boundaries, trunk-based development with feature flags, parallel "
    "test execution, and better deploy tooling. Make a recommendation "
    "with explicit reasoning about which approach you would advocate "
    "for given the company's stage, and explicitly name the conditions "
    "that would change your recommendation. Be concrete: cite specific "
    "patterns and tools where relevant. The memo should be persuasive "
    "but balanced — assume the CTO is technically sophisticated and "
    "will push back on glib answers.",
 
    # ~500 tokens — creative writing with constraints
    "Write the opening chapter of a literary science fiction novel set "
    "in a generation ship 200 years into its 800-year voyage. The "
    "chapter should be approximately 1200 words and should accomplish "
    "the following: introduce the protagonist, a botanist in her "
    "mid-thirties named Inez who has lived her entire life aboard the "
    "ship; establish through small concrete details rather than "
    "exposition that the original mission's purpose has become "
    "ambiguous to the current generation; show without telling that the "
    "ship's biosphere is subtly failing in ways the leadership is not "
    "openly discussing; introduce a secondary character through "
    "Inez's eyes who will serve as her antagonist later in the book, "
    "but make the initial impression sympathetic rather than ominous; "
    "and end on a moment of small but unsettling discovery — something "
    "Inez notices in the agricultural deck that she cannot immediately "
    "explain. Stylistically, the prose should be quiet and observational, "
    "favor concrete sensory detail over abstract reflection, and avoid "
    "any infodump about how the ship works or how Inez came to be there. "
    "The reader should feel slightly disoriented at the start and slowly "
    "orient themselves through context. Resist the temptation to "
    "explicitly state the chapter's themes; let them emerge from "
    "juxtaposition and detail. Avoid sci-fi cliches like blinking "
    "consoles, dramatic alarms, or characters monologuing about Earth. "
    "Names of plants, equipment, and ship locations should feel "
    "lived-in rather than designed for the reader's benefit — they're "
    "things Inez has known her whole life and would not pause to "
    "explain to herself. The point of view is close third, present "
    "tense.",
 
    # ~550 tokens — multi-step technical task
    "I want to set up a local development environment for fine-tuning "
    "open-weights language models on a single workstation with one "
    "consumer GPU (24GB VRAM). Walk me through the full setup, "
    "explaining the reasoning behind each choice rather than just "
    "listing commands. Specifically, address the following in order. "
    "First, the choice of base framework: compare using "
    "transformers + peft + trl directly versus using a higher-level "
    "wrapper like Axolotl or unsloth, and recommend one for a user "
    "who wants to understand what's happening but doesn't want to "
    "reinvent training loops. Second, the choice of fine-tuning "
    "technique: full fine-tuning is off the table at this VRAM "
    "budget for any model above ~3B parameters, so we're choosing "
    "among LoRA, QLoRA, and DoRA — explain the practical tradeoffs "
    "and recommend a default. Third, the environment setup itself: "
    "Python version, virtual environment tool, the specific torch "
    "build that matches the user's CUDA version, and a strategy for "
    "pinning versions so the environment is reproducible six months "
    "from now. Fourth, a minimal end-to-end smoke test: fine-tune a "
    "1B-parameter base model on a small instruction-following "
    "dataset (say, a 1000-row subset of something publicly "
    "available) for a single epoch, with the goal of confirming the "
    "pipeline works rather than producing a good model. Specify "
    "exact hyperparameters and explain why each was chosen for a "
    "smoke test rather than a real training run. Fifth, how to "
    "evaluate whether the smoke test succeeded — what should the "
    "loss curve look like, what should generations look like before "
    "and after, what are the common failure modes (NaN losses, "
    "OOM on the first backward pass, the model producing only the "
    "EOS token) and how to diagnose each. Sixth, what to read or "
    "study next to go from 'I ran a smoke test' to 'I can "
    "intelligently fine-tune for a real use case'. Throughout, "
    "prefer concrete defaults over open-ended choices — the reader "
    "is a competent engineer but new to fine-tuning.",
 
    # ~600 tokens — philosophical / interpretive
    "There's a recurring debate in the philosophy of mind about "
    "whether large language models can be said to 'understand' "
    "anything, or whether they only perform sophisticated pattern "
    "matching that mimics understanding without instantiating it. "
    "Steelman both sides of this debate as carefully as you can, "
    "then offer your own analysis of where you think the debate "
    "actually turns. Specifically, do the following. First, present "
    "the strongest version of the position that LLMs do not "
    "understand: include the Chinese Room argument, the symbol "
    "grounding problem, and the more recent observation that "
    "next-token prediction is in principle a purely syntactic "
    "operation. Don't strawman these — present them as their best "
    "proponents would. Second, present the strongest version of the "
    "opposing view: that the dichotomy between 'real understanding' "
    "and 'mere pattern matching' may not survive scrutiny, that "
    "human cognition is also implemented in physical substrate "
    "performing what could be described as pattern matching at "
    "different levels, and that 'understanding' may be better "
    "thought of as a graded functional property than a binary "
    "metaphysical one. Cover the relevant empirical observations "
    "about LLM behavior — both the failure cases that suggest "
    "shallow processing and the success cases that suggest "
    "something more is going on. Third, identify what you think "
    "the debate actually turns on. Is it a substantive empirical "
    "disagreement about what's happening inside these systems? Is "
    "it a conceptual disagreement about what 'understanding' even "
    "means? Is it a disagreement about which intuitions to trust "
    "when our pre-theoretic concept of understanding is applied to "
    "an entity quite different from a human? Be willing to take a "
    "position, but be honest about the parts of your position that "
    "feel less than fully resolved. Finally, propose an experiment "
    "or observation that, if its result came out one way versus "
    "another, would actually move you on this question — and if no "
    "such experiment exists, explain why and what that tells us "
    "about the nature of the debate. The response should treat the "
    "reader as a thoughtful interlocutor capable of holding "
    "multiple views in mind simultaneously, not as someone who "
    "needs to be convinced of a predetermined conclusion.",

    
]


# --------------------------------------------------------------------------- #
# Metrics data                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class SequenceTiming:
    first_token_time: float | None = None
    last_token_time: float | None = None
    num_output_tokens: int = 0

    def record(self, token_count: int) -> None:
        if token_count == 0:
            return
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.num_output_tokens += token_count

    @property
    def has_tokens(self) -> bool:
        return self.first_token_time is not None and self.last_token_time is not None


@dataclass
class GenerationMetrics:
    input_tokens: int
    new_tokens: int
    ttft_ms: float
    tpot_ms: float
    prefill_throughput: float
    decode_throughput: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.new_tokens

    def format(self, request_id: int) -> str:
        return (
            f"Request {request_id}: "
            f"ttft={self.ttft_ms:.1f}ms "
            f"tpot={self.tpot_ms:.1f}ms/tok "
            f"prefill={self.prefill_throughput:.1f}tok/s "
            f"decode={self.decode_throughput:.1f}tok/s "
            f"in={self.input_tokens} new={self.new_tokens} total={self.total_tokens}"
        )


# --------------------------------------------------------------------------- #
# Per-request tracker (now with a live progress bar)                           #
# --------------------------------------------------------------------------- #

class RequestTracker:
    """Wraps a GenerationHandle, polls tokens, and renders a live tqdm bar."""

    def __init__(
        self,
        handle: GenerationHandle,
        input_len: int,
        max_new_tokens: int,
        position: int,
    ):
        self.handle = handle
        self.input_len = input_len
        self.max_new_tokens = max_new_tokens
        self.start_time = time.perf_counter()
        self.sequences: dict[int, SequenceTiming] = {}
        self.active = True
        self.bar = tqdm(
            total=max_new_tokens,
            desc=f"req {position:>2} (in={input_len:>4})",
            position=position,
            leave=True,
            unit="tok",
            dynamic_ncols=True,
            bar_format=(
                "{desc} [{bar}] {n_fmt}/{total_fmt} "
                "{rate_fmt} elapsed={elapsed}"
            ),
        )

    def can_read(self) -> bool:
        return self.handle.can_read()

    def is_finished(self) -> bool:
        return self.handle.get_status() == ov_genai.GenerationStatus.FINISHED

    def poll(self) -> None:
        """Pull newly-generated tokens off the handle, update timing + bar."""
        delta_total = 0
        for seq_id, output in self.handle.read().items():
            seq = self.sequences.setdefault(seq_id, SequenceTiming())
            token_count = len(output.generated_ids)
            seq.record(token_count)
            delta_total += token_count
        if delta_total > 0:
            self.bar.update(delta_total)

    def close_bar(self) -> None:
        self.bar.refresh()
        self.bar.close()

    def compute_metrics(self) -> GenerationMetrics:
        completed = [s for s in self.sequences.values() if s.has_tokens]
        if not completed:
            return GenerationMetrics(self.input_len, 0, 0.0, 0.0, 0.0, 0.0)

        first_token_time = min(s.first_token_time for s in completed)
        last_token_time = max(s.last_token_time for s in completed)
        new_tokens = sum(s.num_output_tokens for s in completed)

        ttft_s = first_token_time - self.start_time
        decode_s = last_token_time - first_token_time
        decode_tokens = max(0, new_tokens - 1)

        return GenerationMetrics(
            input_tokens=self.input_len,
            new_tokens=new_tokens,
            ttft_ms=ttft_s * 1000.0,
            tpot_ms=(decode_s * 1000.0 / decode_tokens) if decode_tokens else 0.0,
            prefill_throughput=(self.input_len / ttft_s) if ttft_s > 0 else 0.0,
            decode_throughput=(decode_tokens / decode_s) if decode_s > 0 else 0.0,
        )


# --------------------------------------------------------------------------- #
# Pipeline setup                                                               #
# --------------------------------------------------------------------------- #

def build_scheduler_config() -> ov_genai.SchedulerConfig:
    config = ov_genai.SchedulerConfig()
    config.max_num_batched_tokens = 3072
    config.max_num_seqs = 16
    config.cache_size = 12
    config.dynamic_split_fuse = True
    config.enable_prefix_caching = True
    return config


def build_generation_config(max_new_tokens: int = MAX_NEW_TOKENS) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    return config


def count_tokens(tokenizer: ov_genai.Tokenizer, prompt: str) -> int:
    return int(tokenizer.encode(prompt).input_ids.shape[-1])


# --------------------------------------------------------------------------- #
# Batch execution                                                              #
# --------------------------------------------------------------------------- #

def submit_prompts(
    pipe: ov_genai.ContinuousBatchingPipeline,
    prompts: list[str],
    generation_config: ov_genai.GenerationConfig,
) -> list[RequestTracker]:
    tokenizer = pipe.get_tokenizer()
    trackers = []
    for req_id, prompt in enumerate(prompts):
        handle = pipe.add_request(req_id, prompt, generation_config)
        trackers.append(
            RequestTracker(
                handle=handle,
                input_len=count_tokens(tokenizer, prompt),
                max_new_tokens=generation_config.max_new_tokens,
                position=req_id,
            )
        )
    return trackers


def drive_pipeline(
    pipe: ov_genai.ContinuousBatchingPipeline,
    trackers: list[RequestTracker],
) -> None:
    """Step the pipeline until every request finishes, polling each tick."""
    while pipe.has_non_finished_requests():
        pipe.step()
        for tracker in trackers:
            if not tracker.active:
                continue
            if tracker.can_read():
                tracker.poll()
            if tracker.is_finished():
                tracker.active = False
    for tracker in trackers:
        tracker.close_bar()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    pipe = ov_genai.ContinuousBatchingPipeline(
        MODEL_PATH,
        device=DEVICE,
        scheduler_config=build_scheduler_config(),
    )

    trackers = submit_prompts(pipe, PROMPTS, build_generation_config())
    drive_pipeline(pipe, trackers)

    # Bars are closed; safe to print metrics below them.
    print("\n")
    for i, tracker in enumerate(trackers):
        print(tracker.compute_metrics().format(i))


if __name__ == "__main__":
    main()
