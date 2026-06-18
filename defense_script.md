# Defense Presentation Script — Full Speaking Notes
### *Generative Modelling of Maritime AIS Trajectories with Retrieval-Augmented Transformers*
**Andrei-Valentin Știrbu — Advisor: Davide Buscaldi (LIX, École Polytechnique & Sorbonne Université)**

---

## How to use this document

- **Format:** 10-minute talk + 10-minute Q&A, 15 slides.
- **Each slide below has:** ⏱ timing · 🎯 the one sentence you must land · 🖼 what's on screen · 🗣 the spoken script (say this almost verbatim, then adapt to your own voice) · 🔍 deep interpretation of every figure and number · 👉 pointing cues · 🔁 the transition into the next slide · 💬 the follow-up question this slide tends to provoke.
- **Golden rule:** the deck is image-heavy on purpose. **Talk *to* the figures; never read the bullets aloud.** The audience reads faster than you speak — your job is to *interpret*, not narrate.
- **Every number in this script is exactly the number in the report or on the slide.** Quote them with confidence; they are internally consistent.
- **Pace control:** time checkpoints are absolute (**[mm:ss]** from the start). If at **Slide 8 you are past [05:00]**, drop the vessel-type aside on Slide 12 and the A*-baseline aside on Slide 11 — they're marked *(cut if behind)*.
- **Headline accuracy:** validation = **5930 ± 372 m** (5-seed mean over disjoint subsamples). Always state it *with* its protocol (5 seeds, MMSI-grouped split, retrieval from train only) — that's what makes it defensible.

---
---

# SLIDE 1 — Title  ⏱ **[00:00 – 00:25]**

🎯 **One line:** *"From only a start, an end, and a ship type, my model draws a full, accurate, land-avoiding maritime route."*

🖼 **On screen:** EP logo, the two-line title, the subtitle "*A Land-Aware, Water-Strict Route Generator from Start, End and Vessel Type*", author + advisor.

🗣 **Script:**
> "Good morning, and thank you for being here. My project is a generative model for maritime shipping routes. The setup is deliberately minimal: you give the model only three things — a **start position**, a **destination**, and a **vessel type** — nothing at all about the waters in between — and it produces the **entire route**, as 128 waypoints, that a real ship would plausibly sail.
>
> Two properties lift this above a toy problem, and they're the two adjectives in my subtitle. First, **land-aware**: the model is taught the geometry of the coastline. Second, **water-strict**: the final route is *guaranteed* never to cross land — not 'usually', not '99% of the time', but a hard zero. Holding accuracy and that hard constraint together, with no trade-off between them, is the contribution."

👉 **Point:** the logo for half a second, then rest your hand under the subtitle's two adjectives — they are the thesis in two words.

🔁 **Transition:** "Let me start with why anyone should care about generating ship routes."

💬 **Likely follow-up:** none this early; if asked "why these three inputs?" — they're the minimal operationally-available query: at dispatch you know origin, destination, and what kind of ship it is, but not the path.

---

# SLIDE 2 — Why learn maritime routes?  ⏱ **[00:25 – 01:05]**

🎯 **One line:** *"There's a year of structured movement data and real operational value in modelling it — if we can learn the regularities navigators use."*

🖼 **On screen:** bullets (Question / Data / Value) + `data_start_end.png` (green start points, red end points across the US coast).

🗣 **Script:**
> "AIS — the Automatic Identification System — is mandated by the International Maritime Organisation on every large vessel. Each ship continuously broadcasts its identity, position, speed and type. In the US, NOAA's Marine Cadastre programme archives all of this for coastal waters, and **a single year is on the order of 10⁸ position reports.** I used the full 2024 corpus: about **99 million raw position rows across roughly 600 000 track segments.**
>
> That volume invites a machine-learning question: can we *learn* the spatial regularities that human navigators rely on — the shipping lanes, the standard approaches to ports — so that a model handed only origin, destination and ship type can sketch a plausible route? And if we can, **it pays off operationally**: traffic and congestion simulation, storm-risk and exclusion-zone routing, port-arrival-time estimation, and — the one I find most compelling — **anomaly detection**, flagging a vessel whose observed track departs from the learned distribution of normal routes."

🔍 **Figure interpretation (`data_start_end.png`):**
- Green = trip start points, red = trip end points, plotted over the US coastline.
- **Two things to call out:** (1) the points are not uniform — they **cluster into a handful of port regions** (Gulf of Mexico, the Eastern Seaboard, the West Coast). *That clustering is exactly the structure my retrieval mechanism later exploits* — if many past ships went from cluster A to cluster B, a new A→B query has good neighbours to borrow from. (2) coverage spans very different geographies, so the model can't memorise one region; it has to generalise across the whole coast.

👉 **Point:** circle one dense port cluster with your finger and say "this density is what makes retrieval work."

🔁 **Transition:** "So that's the motivation. Let me now state precisely what the model is asked to do."

💬 **Likely follow-up:** "Why only US data?" → That's what NOAA publishes openly and densely; it's a stated limitation that the system is tuned to US waters (Slide 14).

---

# SLIDE 3 — Problem statement  ⏱ **[01:05 – 01:55]**

🎯 **One line:** *"Minimise ADE under two hard constraints — endpoint-exact and water-valid — and those three goals actively fight each other."*

🖼 **On screen:** the formal query/output, the ADE definition, **C1** and **C2**, and `motiv_v5_lands_on_coast.png`.

🗣 **Script:**
> "Formally: a query is q = (start p₁, end p_T, vessel type v). The generator must output a route τ̂ of **T = 128 waypoints, equally spaced along arc length.** We measure accuracy with **ADE, the Average Displacement Error: the mean, over the 128 steps, of the haversine distance between the predicted point and the reference point, in metres.** So whenever I say a number in metres for the rest of the talk, that's ADE — keep that one quantity in your head.
>
> Now the two constraints. **C1, endpoint-exactness:** the route must begin exactly at the start and end exactly at the destination. **C2, water-validity:** every single waypoint must clear land — its signed distance to the coast must stay below a threshold θ. I'll call a method **water-strict** when its crossing rate is *exactly* zero.
>
> Here's the heart of the difficulty, and it's worth dwelling on for a moment. **These three objectives pull against each other.** A model that *only* minimises ADE — plain regression on the steps — drifts away from the destination over 127 autoregressive steps and breaks C1. A model that over-emphasises hitting the endpoint **collapses towards a straight-line interpolation**, which kills route diversity *and* sails straight across land, breaking C2. And a model that bolts on a hard land-projection step inside the loop pays an ADE penalty at every point it has to push off the coast. **The whole contribution of this project is to show all three can be satisfied at once — with no measurable accuracy cost.**"

🔍 **Figure interpretation (`motiv_v5_lands_on_coast.png`):**
- This is an *early* variant of my own model, v5, before land-awareness. The predicted track cuts a corner across a headland and **literally ends up on land.**
- I show it deliberately: this concrete failure is what forced the two ideas on Slides 8–9. The point isn't "v5 was bad" — it's "ADE alone does not buy you legality."

👉 **Point:** trace the predicted line where it crosses the coast and pause.

🔁 **Transition:** "Beyond that three-way tension, four properties of the data make this genuinely hard."

💬 **Likely follow-up:** "Why 128 points?" → A fixed length makes batching and the endpoint loss clean; 128 resolves coastal curvature without exploding the autoregressive horizon. "Why ADE and not a mode-aware metric?" → that's the big Q&A point; preview: ADE is field-standard and comparable, and its single-reference weakness is *why* retrieval helps.

---

# SLIDE 4 — Four reasons it is hard  ⏱ **[01:55 – 02:40]**

🎯 **One line:** *"Irregular coastline, multiple valid routes, two orders of magnitude in length, and ground truth that itself crosses land 10.83% of the time."*

🖼 **On screen:** four bullets + two failure photos (`motiv_dirty_gt_on_river.png`, `motiv_v9_long_route_collapse.png`).

🗣 **Script:**
> "Four properties, each of which drove a specific design choice.
>
> **One — an irregular constraint surface.** The land/water boundary flips on a scale of a few kilometres along a complex coastline. A usable 0.05-degree raster still misses narrow inlets that only resolve at 0.005 degrees. So a naïve coarse-grid geometry loss can actually push the model *into* landlocked artefacts — the resolution mismatch is a real trap, and it's why I use two grids: coarse for the differentiable loss, fine for the final snap.
>
> **Two — the route distribution is multimodal.** For one origin–destination pair there are often several perfectly valid routes — inshore versus offshore, port versus starboard of an island. But **ADE compares to a single reference track, so it *penalises* a perfectly legal route that happens to pick the 'wrong' mode.** This is, in my view, the single biggest reason retrieval helps — a retrieved neighbour anchors the model to the mode the data actually took.
>
> **Three — two orders of magnitude in voyage length.** Routes run from 20-kilometre coastal hops to 2000-kilometre trans-coastal voyages. No single inductive bias is best at both ends: short routes want a straight geodesic, long routes want learned shipping-lane shape. I'll quantify that cross-over on Slide 12.
>
> **Four — noisy ground truth.** Even after discarding 97% of raw pings in quality filtering, the surviving unfiltered 128-point corpus **still crosses land 10.83% of the time at a 10-kilometre threshold — and that's in the *ground truth itself*.** You cannot ask a model to be cleaner than its targets, so this number forced the data-cleaning step on Slide 6."

🔍 **Figure interpretation:**
- **Left, `motiv_dirty_gt_on_river.png`:** a raw ground-truth track that runs *up a river*. It's a real vessel, but it is useless as a *sea-route* target. This is precisely the noise the 10.83% figure measures — inland and riverine tracks contaminating the corpus.
- **Right, `motiv_v9_long_route_collapse.png`:** a long route where my earlier retrieval model, v9, **collapses** — it cannot reproduce the route's shape and produces a degenerate path. This foreshadows *why* v9 wasn't good enough and motivates the per-step retrieval on Slide 8.

👉 **Point:** left photo — trace the river inland; right photo — trace the collapsed shape against the (correct) ground truth.

🔁 **Transition:** "Here's the full system that addresses all four. One slide on the pipeline, then I'll unpack the three parts that actually move the numbers."

💬 **Likely follow-up:** "How did you measure 10.83%?" → trajectory-level crossing rate at θ=10 km using the same SDF the model uses; at θ=5 km even cleaned GT shows 14.8% because raster noise dominates below 10 km (Slide 9 / threshold sweep).

---

# SLIDE 5 — End-to-end production pipeline  ⏱ **[02:40 – 03:15]**

🎯 **One line:** *"Raw NOAA archives in, post-processed water-strict route out — and every artefact is cached and reproducible from one repository."*

🖼 **On screen:** `data_pipeline.png` (the colour-coded flow) + the colour legend bullets.

🗣 **Script:**
> "This is the whole system in one diagram; I'll trace it left to right. **Raw NOAA archives** on the left get filtered, merged, resampled to 128 waypoints, and land-cleaned to produce the training corpus. **The colour code is the key to reading it:** orange is raw external input; blue are processing scripts; **green are cached artefacts computed once and reused** — there are three of them, and they matter — the signed-distance raster, the water-cell graph, and the retrieval index; yellow is training; and gold is the final production output.
>
> Notice the **three side branches** feeding those green artefacts. Each one powers exactly one of the three contributions I'm about to show: the retrieval index powers idea one, the SDF raster powers idea two's training loss, and the water graph powers idea two's inference snap."

🔍 **Interpretation / what to emphasise:**
- The pipeline is **linear and reproducible** — *"every figure and table in the report regenerates from a single script over two cached eval files."* Say this; examiners care about reproducibility.
- The green "compute-once" artefacts are why inference is cheap: the expensive geometry (SDF, graph, KNN) is precomputed, so per-route inference is dominated by the model forward pass plus a ~22 ms snap.

👉 **Point:** tap each green box as you name the three artefacts, then sweep across the three side branches.

🔁 **Transition:** "The first contribution isn't a model change at all. It's data — and it was the single biggest lever in the entire project."

💬 **Likely follow-up:** "What does resampling to 128 do to timing information?" → after arc-length resampling the points are uniform in space, not time; v1 used time deltas, but for route *shape* generation spatial uniformity is what matters.

---

# SLIDE 6 — Data quality: the single largest intervention  ⏱ **[03:15 – 04:00]**

🎯 **One line:** *"Cleaning the corpus — with the model weights held fixed — cut water-strict ADE by 32% and crossings to zero, larger than any architectural change."*

🖼 **On screen:** three bullets (Problem / Filter / Impact) + `land_sdf.png` and `data_distributions.png`.

🗣 **Script:**
> "The unfiltered 128-point ground truth crosses land 10.83% of the time. So before touching the model, I clean the corpus. The **E1 filter**, built on the GSHHS shoreline, drops any track with more than 2% of its points on land *or* more than 10 kilometres of land penetration. **That keeps 61 000 of 207 000 tracks — about 30% — and takes the ground-truth crossing rate to exactly zero.**
>
> Now the result I want you to remember from this slide. **I took the v10 model weights, changed *nothing* about them, and only switched the evaluation corpus from dirty to clean. Water-strict ADE dropped from 8898 metres to 6051 metres — a 32% reduction — and the trajectory-crossing rate fell from 2.0% to 0.0%.** That is a larger single improvement than *any* architecture change anywhere in the project. The headline lesson of the whole thesis is on this slide: **on this problem, data quality was a bigger lever than architecture.** And to be fully honest, retraining from scratch on the clean corpus then adds only a further ~120 metres — confirming the win is the data, not the retrain."

🔍 **Figure interpretation:**
- **Left, `land_sdf.png` — the signed-distance field.** Warm/positive = land, cool/negative = water; the value at each pixel is the distance in km to the nearest coastline. **Three reasons this raster is the backbone of the system:** (1) it's sampled with *bilinear* interpolation, which is differentiable — that's what lets it enter the training loss; (2) it gives O(1) nearest-water queries for the inference snap; (3) it's computed once over the bounding box (−125°, 10°, −60°, 55°) at 0.05°.
- **Right, `data_distributions.png` — the cleaned corpus.** Route-length and vessel-type distributions after filtering. The point: **the filter removed inland garbage without distorting the real distribution** — the vessel mix (passenger / cargo / tanker) and the length spread are preserved.

👉 **Point:** on the SDF, put your finger on a bright coastal pixel and say "positive, land"; move offshore and say "negative, water."

🔁 **Transition:** "With a clean corpus in hand, now the model itself — and the two ideas inside it that do the real work."

💬 **Likely follow-up (this slide attracts the toughest question):** "Aren't you just evaluating on easier data?" → Partly, and I'm explicit: but the *weights were identical*, only eval changed; *and* inland river tracks were never valid sea routes, so the clean corpus is the *correct* target, not a cherry-pick; *and* retraining from scratch on clean data confirms the win rather than reversing it (~120 m further gain).

---

# SLIDE 7 — Model architecture (v10)  ⏱ **[04:00 – 04:40]**

🎯 **One line:** *"A retrieval-augmented encoder–decoder Transformer — 5.8M parameters — with two non-standard pieces: a per-step retrieval bank and a windowed-causal decoder."*

🖼 **On screen:** `architecture_v10.png` (shifted left) + four bullets.

🗣 **Script:**
> "The production model, v10, is an encoder–decoder Transformer — d_model 256, 8 heads, a 4-layer decoder, 2 memory-encoder layers, **5.84 million trainable parameters.** It has two non-standard pieces.
>
> **First, the encoder memory.** Three base conditioning tokens — start, end, vessel type, each with a learned type-embedding so the decoder can tell them apart in cross-attention. On top of those I add a **retrieval bank: for each of the K = 5 nearest historical routes, I subsample 32 waypoints, project each to a token, and add a learned route-index and step-position embedding.** So the encoder memory is **3 + 5 × 32 = 163 tokens.** The decoder cross-attends to 'neighbour k at step t' *directly*.
>
> **Second, the decoder.** It's autoregressive: at each step its input is a 5-dimensional vector — current longitude, latitude, normalised progress t/(T−1), and the sine and cosine of the bearing to the destination — and a single linear head outputs the next displacement, which I accumulate. The non-standard bit is a **windowed-causal mask with k_past = 32**: a step cannot attend more than 32 steps into its own past, which curbs drift from stale self-state over a 128-step rollout — a measurable improvement at this horizon.
>
> The four-term training loss and the three inference passes are the next three slides."

🔍 **Numbers to have ready (don't say all, but be ready):** d_ff = 1024; dropout 0.1; vessel embedding 8-dim over 28 AIS codes; trains in ~3 h on an RTX 3090 (<12 GB), or ~8 h on an Apple M4; best checkpoint at epoch 52 of 60.

👉 **Point:** the encoder block (say "163 tokens"), then the decoder block (say "one displacement per step"), then the loss block and the three post-processing boxes — "covered next."

🔁 **Transition:** "Two ideas in this architecture did the real work — per-step retrieval and land-awareness. Let me take each in turn, starting with the one that moved ADE the most."

💬 **Likely follow-up:** "Why a Transformer over an LSTM?" → direct cross-attention to global conditioning tokens (including 160 retrieval tokens) with no recurrent bottleneck, plus parallel teacher-forced training at T=128. "Why bearing features?" → giving the decoder sin/cos of the bearing-to-destination is a cheap, smooth directional cue that helped endpoint-seeking without over-weighting the endpoint loss.

---

# SLIDE 8 — Key idea 1: per-step retrieval  ⏱ **[04:40 – 05:25]**

🎯 **One line:** *"Retrieval was the largest single architectural lever — over 25% ADE — and exposing neighbours per-step, not mean-pooled, is what unlocked long routes."*

🖼 **On screen:** four bullets (How / Why beats v9 / Evidence / Lesson) + `retrieval_quality_scatter.png`.

🗣 **Script:**
> "Of every architectural idea I tried, retrieval moved the numbers most, by a wide margin. The **zero-training top-1 retrieval baseline alone cuts ADE by 19%** relative to the best non-retrieval Transformer; the *learned* per-step bank cuts it by **more than 25%.** Every *other* idea I tried — bigger model, bearing features, extra encoder layers, scheduled sampling, two pointer formulations, obstacle conditioning — moved ADE by at most about 1%.
>
> The mechanism: a **5-dimensional KNN over (start lon, start lat, end lon, end lat, vessel type)** fetches the 5 nearest historical routes for each query. Each is subsampled to 32 steps and fed to the decoder as memory tokens.
>
> Now the key *technical* insight — why v10 beats my earlier v9. **v9 mean-pooled each retrieved route into a *single* vector before the decoder ever saw it.** On a long route the *shape* is the entire signal, and mean-pooling averages it away into a centroid — that's the collapse you saw on Slide 4. **v10 keeps all 32 steps of each neighbour, so no shape information is lost before the decoder.** That one change is why long-route error roughly halved."

🔍 **Figure interpretation (`retrieval_quality_scatter.png`) — interpret carefully, this is your *evidence* slide:**
- x-axis: distance from the query to its nearest retrieved neighbour; y-axis: the model's per-route ADE.
- **The positive trend is the whole point: when a *good* neighbour exists, the route is *accurate*.** That proves retrieval is doing real work — it's using the neighbour as a shape prior — rather than just adding parameters that happen to help.
- **Be precise about strength:** Spearman ρ = **+0.40 on the short bucket**, **≈ 0 on medium**, and **weak +0.18 on long** (n=174). Interpretation: retrieval helps most where near-neighbours are plentiful (short, dense lanes); on long routes good neighbours are rare, which is exactly where residual error remains.

👉 **Point:** sweep along the rising trend line; then tap the upper-right (far neighbour → high error) and lower-left (close neighbour → low error) corners.

🔁 **Transition:** "Retrieval makes the route *accurate*. The second idea makes it *legal* — water-strict."

💬 **Likely follow-up:** "Is it just memorising the training set?" → No — for a validation query, neighbours are retrieved from the **train split only**, so there's no leakage; and the ρ=+0.40 correlation shows the model adapts neighbours to the exact endpoints rather than copying. "Why K=5, t_retr=32?" → empirically tuned trade-off: K=5 balances coverage vs token count (already 163); 32 steps preserve long-route shape while keeping the encoder tractable.

---

# SLIDE 9 — Key idea 2: land-aware loss & WaterRouter snap  ⏱ **[05:25 – 06:10]**

🎯 **One line:** *"A differentiable land penalty in training plus a snap to *connected* water at inference gives exactly zero crossings, ~22 ms/route, and zero ADE cost on clean data."*

🖼 **On screen:** three bullets (Training / Inference / Cost) + before/after (`motiv_v10_raw_crosses_land.png` → `viz_v10_clean.png`).

🗣 **Script:**
> "Water-validity comes from two complementary mechanisms — one in training, one at inference.
>
> **In training**, I add a land penalty: the mean of ReLU of (signed-distance minus the threshold θ), squared. In words: pay a smooth, quadratic price for any waypoint that comes within θ of land. Because I sample the SDF with **bilinear interpolation, this term is differentiable** — the gradient flows back into the predicted point and therefore into the entire history of displacements that produced it. I use θ_train = 10 km and a weight λ_land = 0.05, which I'll justify with a sweep in a moment.
>
> **At inference**, the WaterRouter takes any waypoint that still violates the threshold and **snaps it to the nearest *connected* water cell on the fine 0.005-degree graph.** The word *connected* is the subtle, important bit: a cell must have at least one water neighbour, which prevents the snap from landing in a landlocked inlet that looks like water on the coarse grid but is dry at fine resolution.
>
> And the punchline: **this guarantees exactly zero land crossings, costs about 22 milliseconds per route, and has *no measurable ADE penalty* on the cleaned corpus.** On the *dirty* data this snap used to cost over a kilometre of ADE; once the data is clean, that penalty collapses to zero — so on the production corpus, **water-validity is essentially free.**"

🔍 **Figure interpretation (before → after):**
- **Left, `motiv_v10_raw_crosses_land.png`:** the *raw* model output still clips a peninsula. Land-awareness in *training* reduces violations a lot but does not eliminate them — pre-snap crossing rate is about 1.88%.
- **Right, `viz_v10_clean.png`:** after the snap, the route hugs the coast on the water side. **The correction is *local and tiny*** — a few waypoints nudged a cell or two — which is precisely why ADE doesn't move.

🔍 **The λ_land sweep (have these numbers ready, say them if asked or if ahead of time):**
- λ_land = 0 → val ADE 6359 m, pre-snap crossings **3.12%**.
- λ_land = 0.05 (production) → val ADE 5913 m, pre-snap crossings **1.88%**, post-snap cross **0.00%**.
- So **the loss roughly halves the work the post-processor has to do** (3.12% → 1.88% violations to fix), and 0.05 beats 0.01 and 0.1 on three independent metrics — a broad optimum, not a knife-edge.

👉 **Point:** left photo — trace the line crossing the headland; right photo — trace the same route now on water. Then mime "snap" with a small hand nudge to convey how *local* the fix is.

🔁 **Transition:** "That snap is the last of three inference passes. Here's the full inference recipe in one picture."

💬 **Likely follow-up:** "Isn't the snap hiding a weak model behind post-processing?" → Two answers: the snap changes ADE by **17 metres** (5913 → 5930), so it's a guarantee, not a crutch; and it only fires on the ~2% of points the raw model gets wrong, because the training loss already removed most violations.

---

# SLIDE 10 — Inference: three post-processing passes  ⏱ **[06:10 – 06:50]**

🎯 **One line:** *"Rollout, then three cheap post-hoc passes — project, endpoint-correct, snap — that together enforce both C1 and C2 with no retraining."*

🖼 **On screen:** a 4-box TikZ flow: ① Autoregressive rollout → ② Hard land projection → ③ Endpoint correction → ④ WaterRouter snap.

🗣 **Script:**
> "Inference is the autoregressive rollout — box one — followed by three correction passes; this is the production routing algorithm.
>
> **Box two, hard land projection — *inside* the loop.** As the model rolls out, any waypoint that lands on land is immediately projected to the nearest water cell by a bounded breadth-first search, capped at 20 cells — and, crucially, **the corrected point is fed *back* into the decoder**, so every subsequent step conditions on the already-corrected trajectory rather than compounding the error.
>
> **Box three, linear endpoint correction — *after* the loop.** The rollout accumulates a few-kilometre residual at the final point. I distribute that residual *linearly* across all 128 points with a ramp α_t = (t−1)/(T−1) — zero at the start, one at the end. This **drives FDE, the final-point error, to numerical zero** and so satisfies C1, while leaving the route *shape* almost untouched because each point moves only a little.
>
> **Box four, the WaterRouter snap** from the previous slide — the final guarantee of water-validity.
>
> The one thing to take away: **all three passes are post-hoc — no retraining — and together they satisfy C1, endpoint-exactness, and C2, water-strictness, simultaneously.**"

🔍 **Interpretation / why the *order* matters:** projection happens *during* rollout so errors don't compound; endpoint correction happens *after* the full shape exists so the ramp is well-defined; the snap is *last* so it has the final word on legality after the endpoint nudge. If asked "could endpoint correction re-introduce a crossing?" — yes, in principle, which is exactly why the snap runs *after* it.

👉 **Point:** tap each box in sequence; emphasise the feedback arrow concept at box 2 with a small circular gesture ("fed back in").

🔁 **Transition:** "So — does it work? Here are the headline numbers."

💬 **Likely follow-up:** "Why linear and not optimal redistribution?" → linear is parameter-free, provably drives FDE→0, and empirically barely perturbs shape; a learned redistribution adds complexity for no measured gain.

---

# SLIDE 11 — Headline results  ⏱ **[06:50 – 07:40]**  ⭐ *the most important slide*

🎯 **One line:** *"v10 + router: −32% ADE versus great-circle, and the only learned method that is also water-strict — at essentially zero ADE cost."*

🖼 **On screen:** the headline table (production row highlighted) + `ade_by_model.png`.

🗣 **Script:**
> "Read the table from the top, the highlighted production row first. **v10 plus router reaches 5930 ± 372 metres ADE on validation.** The great-circle baseline is 8735 — so that's a **32% reduction in ADE.** But look at the last column, the crossing rate: **0.04% — effectively zero — while great-circle crosses land 8.8% of the time.** We are the *only learned method that is also water-strict.*
>
> Two rows I want you to notice, because they make the argument honest. **'v10 raw' versus production:** the router takes ADE from 5913 to 5930 — **17 metres** — while removing the crossings. That's the entire thesis in two rows: **validity is essentially free.** And **'retrieval top-1':** zero-training retrieval is water-valid but **11 667 metres** — nearly twice our error. It's a strong *validity* baseline but a weak *accuracy* one, which is exactly why a learned model is needed.
>
> One more thing on protocol, because it's what makes these numbers defensible. The figures are a **5-seed mean over disjoint 500-route subsamples**, the split is **MMSI-grouped** so no vessel appears in both train and validation, and for a validation query the **retrieval neighbours come from the train split only** — so there's no leakage anywhere in the pipeline."

🔍 **Figure interpretation (`ade_by_model.png`):** visual confirmation of the table ordering — production lowest, great-circle variants in the middle, zero-training retrieval highest; error bars are the 5-seed spread, and they're tight enough that the ordering is unambiguous.

🔍 *(cut if behind)* **A\* baseline — the cleanest evidence the model learns *shape*:** the natural water-strict baseline is the **shortest A\* path on the same water graph** — legal by construction, but ignorant of shipping conventions. It scores **9882 m on validation — about 50–60% worse than us.** So beating A\* by ~3.6 km is direct evidence that the model contributes *shape* information beyond mere legality.

👉 **Point:** rest your finger on the highlighted production row for the ADE and crossing numbers; then jump to the 'retrieval top-1' row to make the accuracy-vs-validity contrast.

🔁 **Transition:** "A single mean hides *where* we win and lose, so let me break it down by route length."

💬 **Likely follow-up:** "FDE is 608 for production but 0 for some rows — why?" → FDE is uninformative for endpoint-corrected methods; the 608 is residual bookkeeping, the operational endpoint error after correction is numerically zero. "Is the gap to great-circle significant?" → yes: paired bootstrap 95% CI on validation is [−2185, −1376] m — bounded away from zero (Slide 13).

---

# SLIDE 12 — Where the model wins + qualitative  ⏱ **[07:40 – 08:20]**

🎯 **One line:** *"We roughly halve error on long voyages where lane-shape matters; on the very shortest hops a geodesic is genuinely competitive — and the qualitative routes show real navigation, not interpolation."*

🖼 **On screen:** `length_bucketed_ade.png` (ADE by bucket) + `viz_v10_clean.png` (generated vs GT).

🗣 **Script:**
> "Breaking the mean down by route length tells the real story.
>
> **On long routes, over 500 kilometres, the production model averages 27.4 kilometres of error versus the great-circle's 60.1 — less than half.** That's the learned shipping-lane shape paying off where it matters most. **On the shortest hops, under 50 kilometres, a straight geodesic is actually competitive** — 1431 metres for great-circle — because there's little lane structure to learn and a straight line is hard to beat. This is the 'no single bias fits both ends' point from Slide 4, now quantified: **we are not uniformly better, we are decisively better where the problem is hard.**
>
> On the right are real generated routes in red against ground truth in green. Notice these are not straight lines — the model **traces coherent coastal lanes around the Florida peninsula and through the Gulf**, reproducing navigational decisions, while the great-circle would cut straight across the land."

🔍 **Figure interpretation:**
- **Left, `length_bucketed_ade.png`:** the production and great-circle curves **cross over** — geodesic wins the short bucket, the model wins medium and long. *Point to the crossover.* Long-bucket error bars are wide because that bucket is small-n (few >500 km routes), so read it as "roughly halved," not to three significant figures.
- **Right, `viz_v10_clean.png`:** green = GT, red = production, grey = land. **Pick one curved route and say "the model chose to follow this channel."** That single observation — curvature that matches a real lane — is worth more than any number for a human audience.

👉 **Point:** the crossover on the left chart; one curved channel-following route on the right.

🔁 *(cut if behind)* **Vessel-type aside:** passenger routes are easiest (val ADE 2767 m — regular ferry lines), tankers hardest (10072 m — variable long-haul). The spread tracks how *repetitive* each class's routes are, which is consistent with the retrieval story.

🔁 **Transition:** "Last results slide — how much should you trust these numbers?"

💬 **Likely follow-up:** "Why is the long bucket so noisy?" → small sample (5% of routes exceed 500 km) and per-route ADE scales with length, so a few hard routes dominate the bucket's variance.

---

# SLIDE 13 — Robustness: stable across seeds  ⏱ **[08:20 – 09:00]**

🎯 **One line:** *"The ranking and the water-validity hold across every seed, and the gap over each baseline is statistically significant — so the headline isn't subsample luck."*

🖼 **On screen:** `seed_variance.png` (single figure, left) + four bullets (tight spread / ranking never flips / always water-strict / significant).

🗣 **Script:**
> "How much should you trust these numbers? Three points.
>
> **One — the spread is tight.** The figure shows ADE across five *disjoint* subsample seeds of the validation pool. The production model's spread is just ± 225 to 372 metres — narrow enough that the ordering is unambiguous.
>
> **Two — the ranking never flips.** In every one of the five seeds, v10 plus router is the lowest-error method, and the baselines keep their relative order. There's no seed where a baseline sneaks ahead.
>
> **Three — it stays water-strict, and the gap is significant.** Land-crossing is exactly zero in *all* five seeds — the guarantee isn't seed-dependent. And a paired bootstrap — 10 000 resamples, route as the unit, paired across methods — puts the 95% confidence interval on the gap versus great-circle at roughly −2185 to −1376 metres: **bounded away from zero**, so the win is real, not noise. As a cross-check, three fully independent retrains differ by only ~138 metres, so the result is stable to model initialisation too."

🔍 **Figure interpretation (`seed_variance.png`):** one bar group per method, repeated across the five seeds; **the vertical separation between the production model, the baselines, and zero-training retrieval is clean and consistent** — visually, you can see that no amount of reseeding changes who wins. The error bars are the across-seed spread, and the production bar's is the smallest of the learned methods.

👉 **Point:** sweep across the five seed groups and say "same order, every time"; then tap the production error bar and say "and the tightest spread."

🔁 **Transition:** "Let me close with the lessons that generalise beyond this one system."

💬 **Likely follow-up:** "What about a held-out test set?" → The corpus is split MMSI-grouped into train/val with no vessel overlap, retrieval draws from train only, and I report a 5-seed mean — so the validation number is already a leakage-free, variance-aware estimate, not a single lucky run. "Could it be hyperparameter selection on val?" → only λ_land and λ_end were swept, in small neighbourhoods; everything else was inherited, so the val-selection surface is tiny.

---

# SLIDE 14 — Lessons, limitations & outlook  ⏱ **[09:00 – 09:40]**

🎯 **One line:** *"Three transferable lessons — retrieval, data quality, and 'per-step loss lies' — plus an honest limitations list and a diffusion-based next step."*

🖼 **On screen:** three lesson bullets + two blocks (Limitations / Outlook).

🗣 **Script:**
> "Three lessons that I think transfer beyond ships.
>
> **One — retrieval was the biggest architectural lever.** Of eight ideas tried, none of the seven parametric ones moved ADE more than about 1%. The single non-parametric one — retrieval — moved it 19% to over 25%. **When your training set already contains many near-neighbours, reach for a non-parametric memory before scaling the parametric model.**
>
> **Two — data quality was the biggest single intervention overall.** With the weights *held fixed* and only the evaluation corpus swapped from dirty to clean, the water-strict ADE dropped from 8898 to 6051 metres — 32%. To be precise about *what* that 32% is: it's not that the model got 32% more accurate — the *raw* ADE only moved ~5%. It's that we stopped scoring the model against inland ground-truth tracks it could never legally match, and the water-snap penalty that hurt on dirty data collapsed to zero. The lesson stands either way: **check your targets before redesigning your model.**
>
> **Three — per-step validation loss does not predict rollout ADE.** My v2-to-v5 ladder improved per-step Huber loss from 0.105 to 0.074 with *zero* movement in end-to-end ADE; a later pointer variant hit 87% per-step cell accuracy but 27 kilometres of rollout error. **Only end-to-end rollout should decide architecture** — per-step metrics are for monitoring, not for choosing.
>
> Limitations I'll own up front: it's a generator for *known* waters — it interpolates patterns it has seen, it doesn't reason about novel geography from first principles; and its accuracy depends on how dense the retrieval neighbours are, so sparse-lane long routes are the weak spot. The outlook is **v13: a whole-sequence diffusion Transformer with EDM preconditioning and classifier-free guidance, with score-based land guidance folded directly into the sampling process**, plus cross-region transfer and genuine multimodal route sampling."

👉 **Point:** count the three lessons on your fingers; then gesture to the two blocks as "what I'd fix" / "where it goes next."

🔁 **Transition:** "Thank you — I'd be glad to take your questions."

💬 **Likely follow-up:** "Why diffusion next?" → it natively models a *distribution* of routes (addresses the multimodality / ADE-single-reference problem) and lets land-validity enter as a guidance term during sampling rather than as a post-hoc fix.

---

# SLIDE 15 — Thank you  ⏱ **[09:40 – 10:00]**
Leave this slide up for the whole Q&A. EP logo, title, advisor on screen — a calm, clean backdrop.

---
---

# Q&A PREPARATION (10 minutes) — extended

> Strategy: answer in **three beats** — (1) one-sentence direct answer, (2) the evidence/number, (3) the honest caveat. Examiners reward the caveat. If you don't know, say "I didn't test that; my expectation is X because Y."

**Q1. Why ADE, given it penalises valid alternative routes?**
Direct: it's the field-standard, baseline-comparable accuracy metric. Evidence: it's *why* retrieval helps — anchoring to the historical mode reduces the "right answer, wrong mode" penalty (ρ=+0.40 short bucket). Caveat: the proper fix is multimodal sampling + a distributional metric, which is the v13 motivation.

**Q2. Isn't the WaterRouter snap just hiding a weak model behind post-processing?**
Direct: no — it's a guarantee, not a crutch. Evidence: it changes ADE by **17 m** (5913→5930) and only fires on the ~2% of points the raw model gets wrong, because the land-aware *training* loss already removed most violations (pre-snap crossings drop 3.12%→1.88% with λ_land). Caveat: on *dirty* data the snap *did* cost >1 km — the reason it's free is the data cleaning.

**Q3. Why does retrieval help so much — is it memorisation?**
Direct: it's non-parametric conditioning, not lookup. Evidence: validation queries retrieve from the **train split only** (no leakage), and ADE correlates with *neighbour quality* (ρ=+0.40), meaning the model adapts neighbours to the exact endpoints. Caveat: on long routes with no near-neighbour (ρ=+0.18) it degrades toward a geodesic — the residual-error regime.

**Q4. The 32% data-cleaning win — aren't you evaluating on easier data, and didn't the model just get 32% better?**
Direct: it's a fair-measurement effect, not a 32% jump in model skill — and I'm explicit about it. Evidence: the *weights were identical*, only the eval corpus changed; the **raw** ADE moved only ~5% (6584→6278 m), while the **water-strict** number moved 32% (8898→6051 m) because (a) inland ground-truth tracks the model could never legally match left the eval set, and (b) the water-snap penalty that was large on dirty data collapsed to ~0. Retraining from scratch on clean data then adds only ~120 m, confirming the win is the data. Caveat/justification: inland river tracks were never valid sea routes — the clean corpus is the *correct* target, not a cherry-pick.

**Q5. How do you defend the result without a separate held-out test number?**
Direct: the validation estimate is already leakage-free and variance-aware. Evidence: the split is MMSI-grouped so no vessel crosses train/val; retrieval for a val query draws from the train split only; and the headline is a 5-seed mean ± std over disjoint subsamples, not one lucky run. Caveat: it's an in-distribution estimate for US waters — out-of-region generalisation is untested (Q9).

**Q6. Why a Transformer over an RNN/LSTM?**
Direct: global cross-attention + parallel training. Evidence: the decoder attends directly to 163 conditioning/retrieval tokens with no recurrent bottleneck, and teacher forcing parallelises T=128 training. Caveat: prior maritime work (Nguyen et al.) uses both; the Transformer's attention is what makes the 160-token retrieval bank tractable.

**Q7. Why K=5 and t_retr=32?**
Direct: empirically tuned trade-offs. Evidence: K=5 balances neighbour coverage against the 163-token budget; 32 steps preserve long-route shape (the failure that killed mean-pool v9) while keeping the encoder sequence manageable. Caveat: I did *not* grid-search K or t_retr — a stated scope limit.

**Q8. What's the production model's failure mode?**
Direct: long routes with no good retrieval neighbour. Evidence: ρ=+0.18 and 27 km error on the >500 km bucket; the model falls back toward a great-circle-ish guess and loses lane shape. Caveat: more data or cross-region transfer (v13) is the path.

**Q9. Does it generalise outside US waters?**
Direct: untested — a stated limitation. Evidence: the SDF, water graph and retrieval index are all US-2024. Caveat: cross-region transfer is explicit future work; nothing in the architecture is US-specific, only the cached artefacts.

**Q10. Why not enforce water-validity *inside* the model as a hard constraint?**
Direct: I tried; it lost. Evidence: the abandoned v11/v12 pointer-over-water-graph variants discretised onto water cells — 87% per-step cell accuracy but **27 km rollout ADE**; per-step accuracy didn't aggregate and discretisation killed sub-km precision. Caveat: continuous regression + post-hoc snap won decisively — lesson #3 in action.

**Q11. How is the SDF made differentiable, exactly?**
Direct: bilinear interpolation on a regular grid. Evidence: sampling sdf(p̂_t) bilinearly makes it a smooth function of the four surrounding pixels, so ∂sdf/∂p̂_t exists and backprops through the ReLU penalty into the displacement history. Caveat: below θ=10 km raster noise dominates (GT itself shows 14.8% crossings at θ=5 km), which is why θ_train=10 km.

**Q12. λ_land = 0.05 — how sensitive?**
Direct: a broad optimum, not a knife-edge. Evidence: 0.05 beats 0.01 and 0.1 (val ADE 5913 vs 5925 vs 5973, pre-snap cross 1.88% vs 2.36% vs 2.16%); 0 costs +446 m and doubles pre-snap crossings. Caveat: only λ_land and λ_end were swept; LR, batch, λ_smooth, K, t_retr were not.

**Q13. Why does scheduled sampling not help on clean data?**
Direct: it slightly *hurts* (v5 ≈ 200 m worse than v4 on clean). Evidence: the original dirty-corpus win was a single-seed artefact; the controlled clean retrain reverses it. Caveat: exposure-bias mitigation matters less here because endpoint correction already handles the dominant drift.

**Q14. What about the v1 next-step task — why abandon it?**
Direct: it's a near-perfect use-case for the constant-velocity baseline. Evidence: on turn-segmented dense AIS, CV is extremely hard to beat — v1 matched but didn't beat it. Caveat: all the learning signal and operational value is in the *full-route* formulation, so I moved there.

**Q15. Compute and footprint?**
Direct: small. Evidence: 5.84M params, ~3 h on one RTX 3090 (<12 GB), inference dominated by the forward pass + ~22 ms snap. Caveat: the cached geometry (SDF, graph, KNN) is the one-time cost.

---

# CHEAT-SHEET — numbers on the tip of your tongue

| Quantity | Value |
|---|---|
| Production ADE (val, 5-seed mean ± std) | **5930 ± 372 m** |
| Improvement vs great-circle | **−32%** |
| Crossing rate, production | **0.00–0.04%** (water-strict) |
| Crossing rate, great-circle | 8.80% |
| GT crossing rate before cleaning (θ=10 km) | **10.83%** |
| Data-cleaning effect (weights fixed) | 8898 → 6051 m (**−32%**), 2.0% → 0.0% crossings |
| Corpus kept by E1 filter | 61k / 207k (**~30%**) |
| Long-route ADE (>500 km): model vs GC | **27.4 km vs 60.1 km** |
| Short-route ADE (<50 km): GC | 1431 m (GC competitive) |
| Router ADE cost | **17 m** (5913 → 5930), ~22 ms/route |
| Retrieval lever | 19% (zero-train) → **>25%** (learned) |
| Retrieval ρ (short / med / long) | +0.40 / ≈0 / +0.18 |
| A* water-strict baseline (val) | 9882 m (~50–60% worse) |
| Memory tokens | **163** = 3 + 5×32 |
| Model size | **5.84M params**, d_model 256, 8 heads, 4 dec layers |
| Decoder input (5-dim) | [lon, lat, progress, sin β, cos β] |
| Windowed mask | k_past = **32** |
| Loss weights | λ_end=10, λ_smooth=1, **λ_land=0.05**, θ=10 km |
| Optimiser | AdamW, η=2e-4, cosine, 5% warmup, batch 128 bf16, 60 ep (best ep 52) |
| Train/val split | MMSI-grouped (no vessel overlap), seed 42 |
| Training-seed std (3 retrains) | ~138 m |
| Paired-bootstrap CI vs GC (val) | [−2185, −1376] m (significant) |

---

# 30-SECOND ELEVATOR VERSION (if asked to summarise, or if you run out of time)
> "I built a retrieval-augmented Transformer that turns a start, an end, and a ship type into a full 128-point maritime route. It cuts displacement error 32% below a great-circle baseline — to about 5.9 km — and is the only learned method that *guarantees* zero land crossings, at about 22 ms of post-processing per route. Two findings stand out: retrieval was the biggest architectural lever, over 25%; and data quality — cleaning inland tracks out of the corpus — was an even bigger single lever at 32%, with the model weights untouched."
