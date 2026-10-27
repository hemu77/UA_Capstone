# Method Comparison: Rationale And Unequal Computation

## Status And Scope

October 27, 2026. This explanation was added **after main collection began**.
It documents existing frozen choices, not a new preregistration or an experiment
chosen from main-study results. It changes no prompts, parser, analysis, model,
schedule, budget, or running job. The authoritative design is
[Final Collection](FINAL_COLLECTION.md); execution is recorded in
[Hoffman2 Execution](HOFFMAN2_EXECUTION.md).

## Plain-Language Explanation

The four methods perform different tasks. Global asks for a whole network in
one response. Local asks each person to nominate friends. Sequential builds the
network through ordered decisions with access to evolving network information.
Iterative starts with local nominations, then revises ties through three rounds.
Giving iterative only one call would no longer implement that procedure.

| Method | Planned decisions for one 50-person graph, before correction attempts |
| --- | --- |
| Global | 1 whole-network response |
| Local | 50 person-level nomination decisions |
| Sequential | 50 ordered person-level nomination decisions |
| Iterative | 50 initial nominations plus up to 3 x 50 x 2 add/drop decisions: up to 350 |

These are decision counts, not promised billed-request totals. Iterative skips
an add when no eligible new friend exists and skips a drop when only one friend
remains. Correction attempts and authorized transport replacements can add
requests; a terminal failure can end a graph early. Tokens, cache use, response
length and model prices also differ, so request count alone is not compute cost.
See the [generation procedure](../generate_networks.py) and the
[three-round invocation and correction handling](../calibration_v6.py).

Every model/method/setting has **eight planned repetitions**, with **224 graph
attempts per method and 896 total**. A graph's hundreds of decisions are not
hundreds of independent observations. The frozen primary analysis uses eight
matched schedule blocks, not request counts, personas, or model rows as extra
replications. Missingness can reduce the usable blocks for a comparison.

## What The Comparison Can Establish

We compare **the complete procedures as implemented**, including their network
information, decision order, nomination constraints and refinement rounds. The
three-round setting was fixed before the main run; it is not established as an
optimal or universally fair setting. We will not tune rounds after inspecting
the main results and then present the tuned result as confirmation.

The primary country/instruction-language contrasts are made **within each
method** under the existing analysis. Model-specific comparisons and homophily
summaries retain their stated descriptive status. Within-method comparisons
hold the algorithm and nominal round allocation fixed, but realized calls can
still vary with graph state, corrections and failures. They do not establish
effects on real national populations or translation-independent language effects.

Global's primary endpoint is density; the quota-constrained methods use average
local clustering, with their density descriptive. These different endpoints
must not be turned into a single cross-method performance ranking. No additional
confirmatory hypotheses or significance tests are introduced by this note.

## Limitation And Required Reporting

**Equal repetitions do not mean equal computation.** Iterative receives more
decision opportunities and feedback from the evolving network. A cross-method
difference cannot isolate an algorithmic advantage from those extra resources
or the other procedural differences. Better agreement with a real-network
metric alone would not establish intrinsic superiority, efficiency, or human
realism. Cost-normalized summaries describe a trade-off; they do not remove
this confounding or substitute for an equal-budget experiment.

The final report must include, by method/model/setting:

- [ ] Planned attempts, valid graphs, valid empty graphs and terminal failures.
- [ ] Actual request counts, correction/transport replacement counts, and input,
  output and available cache-token usage, including unsuccessful attempts.
- [ ] Recorded usage costs and retained uncertain reservations separately;
  totals are not an independently reconciled provider invoice.
- [ ] Total cost per attempted graph and total cost per valid graph, charging
  failed attempts to the procedure rather than hiding their cost. A zero-valid
  denominator is reported as undefined, not zero.
- [ ] Recorded API elapsed time separately from collection wall time, including
  the effect of persistence, verification, waits and scheduler continuations.
- [ ] Network effects and uncertainty under the frozen analysis, alongside
  these resource summaries. No method is declared a winner from one metric.

These are pending reporting requirements, not claims that final tables exist.
The current study does **not** include an equal-token, equal-cost, or
equal-request experiment. A reviewer may reasonably request one, or a
refinement-round sensitivity study. Such work needs a separately declared
design and authorization; we must not silently alter or pool it with this run.

## Manuscript-Ready Explanation

> We compare four complete network-generation procedures rather than methods
> constrained to identical computational budgets. Global generation requests
> a network jointly, local and sequential generation make person-level choices,
> and iterative generation adds three predefined refinement rounds to a local
> initialization. All conditions receive eight planned repetitions, but the
> procedures differ in decision count, information exposure and opportunity to
> revise ties. Our primary contrasts therefore assess sensitivity to the tested
> instructions within each procedure; cross-procedure differences do not isolate
> a compute-matched algorithmic advantage. We report resource use and failures
> alongside structural outcomes. Equal-budget and refinement-round comparisons
> are not included and remain limitations of the present design.

Use the final reporting sentence only once those resource/failure tables have
actually been produced and checked. Transparent scope reduces avoidable
misinterpretation; it cannot guarantee reviewer agreement or acceptance.
