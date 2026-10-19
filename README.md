# llm-grounding-kit

A small library for one narrow job: take a free-text value that an LLM (or a
regex, or a human) pulled out of a document, and map it onto a fixed list of
canonical values. If it cannot do that with enough confidence, it says
Unknown instead of guessing.

```
"Port of Hamburg"  -> Hamburg     (alias)
"Istambul"         -> Istanbul    (fuzzy, 0.88)
"maize"            -> Corn        (embedding)
"Santos"           -> Unknown     (not in the reference; best guess kept for review)
```

I wrote this as a standalone version of the validation layer in a larger,
private project: an LLM pipeline that reads unstructured shipping emails and
extracts fields like load port and cargo type. The code here was rewritten
from scratch for publication, and the data is synthetic. No business logic or
real messages are included.

## The problem

LLM extractors are good at finding the right span and bad at returning it in
a consistent form. The same port comes back as "Constanța", "Constantza",
"CONSTANTA RO" or "Constantsa". Anything downstream that joins, filters or
prices on that field needs one canonical key.

The obvious fix is to ask the model to pick from the list. That works until
the list has a few thousand entries, or the value is not on the list at all.
In that case the model tends to pick the closest-sounding entry, and you get a
clean, well-formatted, wrong answer.

## How it works

```
raw string -> normalise -> alias index -> fuzzy -> embeddings (optional) -> key | Unknown
```

1. **Normalise.** Lowercase, strip accents, collapse punctuation, drop filler
   phrases configured per reference ("port of", "in bulk"). The same function
   is applied to the reference when the index is built, so the two sides always
   agree.
2. **Alias layer.** An exact dictionary lookup over every key, canonical name
   and alias. It costs nothing, and it cannot be wrong unless the data is wrong.
3. **Fuzzy layer.** `rapidfuzz.token_sort_ratio` against every surface form.
   This catches typos and word order ("urea prilled").
4. **Embedding layer.** Cosine similarity with a sentence-transformers model.
   It is the only layer that knows "maize" is corn. It is off unless you pass
   an encoder.

Each layer returns a candidate along with its own accept or reject verdict. The
resolver takes the first accepted candidate. If no layer accepts, the result
is `Unknown`, and the rejected candidates are attached to the result so a
reviewer can see what each layer thought.

## Design decisions

**Cheapest layer first.** Most real values are either exact or a known alias.
Running an embedding model on "Rotterdam" wastes time, and it adds a way to be
wrong where there was none. Each layer only sees what the cheaper layers could
not resolve.

**Unknown instead of the best guess.** In the original pipeline a wrong port
is worse than a missing one. A missing value goes to a review queue. A wrong
value quietly produces a wrong distance, a wrong price and a wrong match, and
nobody notices. So every layer has a threshold, and below it the answer is
Unknown. The "never abstain" row in the benchmark shows what happens without
this rule: 15 confident wrong answers out of 59.

**Margin over the runner-up, not just a threshold.** A score of 86 means
little if the second-best entry scored 85. Both the fuzzy and the embedding
layer require the winner to beat the best *different* entry by a margin
(3 points / 0.05 cosine by default).

**Short strings skip fuzzy matching.** For "TBA" or "SG", one edit is a third
of the string. Short codes belong in the alias list.

**Bad reference data fails at load time.** A duplicate key, or an alias that
maps to two entries, raises when the `Reference` is built. Otherwise it would
surface weeks later as an occasional wrong match.

**The encoder is injected.** `EmbeddingLayer` takes any
`texts -> (n, d) array` function. The tests use a fake one, so they run in
under a second with no model download. You can also swap in an API-based
embedding model without touching the layer.

**Immutable objects.** References, layers, candidates and matches are frozen
dataclasses. A resolver can be shared across threads, and nothing changes
between building it and using it.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate elsewhere
pip install -e ".[dev]"           # core + pytest
pip install -e ".[embeddings]"    # optional: sentence-transformers (pulls in torch)

pytest
python examples/quickstart.py
python benchmarks/run_benchmark.py            # add --errors to list every miss
python benchmarks/run_benchmark.py --no-embeddings
```

Using it on your own data:

```python
from grounding import Reference, Resolver, sentence_transformer_encoder

ref = Reference.from_json("my_reference.json")   # same shape as data/ports.json
resolver = Resolver.build(ref)                    # alias + fuzzy
# resolver = Resolver.build(ref, encode=sentence_transformer_encoder())

m = resolver.resolve("Constantsa")
m.key, m.canonical, m.method, m.score, m.suggestion
```

## Benchmark

`data/eval_set.jsonl` contains 59 hand-written queries against two small
references (22 ports, 18 bulk cargoes). 45 have a correct answer in the
reference. 14 deliberately do not: ports that are not in the list, regions,
"TBA", and cargo categories like "grain" that match several entries. A query
counts as correct when the returned key equals the label. For those 14, the
only correct answer is Unknown.

Defaults: fuzzy threshold 85, margin 3; embeddings `all-MiniLM-L6-v2`,
threshold 0.60, margin 0.05.

| configuration | accuracy | in-reference found | wrong accepts | abstained |
|---|---|---|---|---|
| alias only | 36/59 (61%) | 22/45 | 0 | 37 |
| alias + fuzzy, never abstain | 44/59 (75%) | 44/45 | 15 | 0 |
| alias + fuzzy | 52/59 (88%) | 38/45 | 0 | 21 |
| alias + fuzzy + embeddings | 55/59 (93%) | 42/45 | 1 | 16 |

Correct answers by query type:

| tag | n | alias only | fuzzy, never abstain | alias + fuzzy | + embeddings |
|---|---|---|---|---|---|
| exact | 12 | 12 | 12 | 12 | 12 |
| normalisation | 10 | 10 | 10 | 10 | 10 |
| noise ("Houston, TX") | 3 | 0 | 3 | 1 | 3 |
| typo | 14 | 0 | 14 | 11 | 11 |
| reorder | 4 | 0 | 4 | 4 | 4 |
| synonym ("maize") | 2 | 0 | 1 | 0 | 2 |
| out of reference | 14 | 14 | 0 | 14 | 13 |

What I take from this:

- Normalisation alone does a lot of work. Accents, punctuation and filler
  words cost nothing to handle, and the alias layer gets all of those right.
- The fuzzy layer recovers most typos without a single wrong accept, *because*
  of the threshold. Without the threshold it recovers every typo and also
  answers all 14 out-of-reference queries with something wrong
  ("Santos" -> Constanta, "Valletta" -> Valencia).
- Embeddings fix the cases fuzzy matching cannot (synonyms, trailing country
  names), but they bring the first confident mistake: "wood pellets" ->
  Iron Ore, because the reference contains "iron ore pellets". That is the
  reason the layer is optional and runs last.

Fuzzy threshold sweep (alias + fuzzy only):

| threshold | accuracy | in-reference found | wrong accepts |
|---|---|---|---|
| 70 | 56/59 | 43 | 1 |
| 75 | 56/59 | 43 | 1 |
| 80 | 56/59 | 42 | 0 |
| 85 | 52/59 | 38 | 0 |
| 90 | 47/59 | 33 | 0 |
| 95 | 42/59 | 28 | 0 |

On this set, 80 looks strictly better than the default of 85. I have kept 85
anyway. The set is 59 queries that I wrote myself, and picking the threshold
on the same data you report on is how you overfit an eval. If you have a
labelled sample of your own traffic, run the sweep on that and choose.

## Limitations

- The eval set is small and synthetic. It shows how the layers behave
  relative to each other. It is not an accuracy claim for any real
  workload.
- Fuzzy matching is a linear scan over all surface forms. That is fine for
  a few thousand entries. Beyond that you would want an n-gram index or
  `rapidfuzz.process.cdist` in batches.
- Scores from different layers are not comparable (edit ratio vs cosine), so
  the result keeps every layer's rejected candidate instead of picking a
  single "best" suggestion.
- Filler stripping is literal and per reference. "Port of Spain" would be
  normalised to "spain" if "port of" is a filler. Keep fillers conservative.
- The embedding layer encodes the whole reference when it is built. With a
  large reference, cache or precompute those vectors.
- Only single values are resolved. Finding the entity in a longer text is the
  extractor's job.

## Layout

```
src/grounding/
  normalize.py   fold() and Normalizer (filler phrases)
  reference.py   Entry, Reference (alias index, load-time checks)
  layers.py      AliasLayer, FuzzyLayer, EmbeddingLayer, encoder factory
  resolver.py    Resolver, Match
data/            synthetic references and eval set
benchmarks/      run_benchmark.py
examples/        quickstart.py
tests/           pytest suite (no model download needed)
```

## License

MIT
