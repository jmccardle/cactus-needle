# Needle constrained-decoding A/B: stock vs llguidance grammar + jump-forward

200 examples, 10 domains x 20. Model sees all 10 tools every call. CPU (jax cpu). Total wall 317s.

**Arms.** *stock* = unconstrained greedy argmax (Needle's raw output). *llg* = llguidance schema-union grammar with jump-forward (grammar-forced token runs injected in one pass).

**Metrics.** `valid` = well-formed call with correct tool name, keys, and every value semantically valid for its domain (real date, in-range int, valid state, etc.). `exact` = arguments equal the ground-truth. `passes` = model forward passes (≈ wall-time on the un-cached decode). Higher valid/exact, lower passes = better.


## Overall (stock / llg)

| scope | json% | name% | **valid%** | exact% | passes | ms | speedup |
|---|---|---|---|---|---|---|---|
| **ALL** | 98/100 | 84/92 | 38/92 | 30/48 | 22.0/11.8 | 932/405 | 1.9x |


## Per domain (stock / llg)

| domain | json% | name% | **valid%** | exact% | passes | ms | speedup |
|---|---|---|---|---|---|---|---|
| schedule_event | 100/100 | 100/100 | 80/100 | 45/60 | 23.6/12.8 | 1024/446 | 1.9x |
| convert_currency | 100/100 | 100/100 | 25/100 | 25/80 | 33.1/12.1 | 1341/399 | 2.7x |
| set_shipping_address | 90/100 | 85/95 | 10/95 | 5/20 | 31.1/12.4 | 1260/443 | 2.5x |
| rate_product | 100/100 | 90/90 | 50/90 | 45/75 | 16.9/5.7 | 736/188 | 3.0x |
| set_brightness | 100/100 | 85/100 | 80/100 | 75/75 | 13.0/8.0 | 634/258 | 1.6x |
| dial_phone | 100/100 | 70/70 | 10/70 | 10/15 | 22.9/12.6 | 946/424 | 1.8x |
| set_color | 100/100 | 80/85 | 60/85 | 35/60 | 16.8/10.1 | 742/339 | 1.7x |
| set_thermostat | 100/100 | 75/75 | 10/75 | 15/30 | 19.7/11.2 | 824/372 | 1.8x |
| set_waypoint | 100/100 | 70/100 | 50/100 | 50/50 | 18.9/18.9 | 807/686 | 1.0x |
| set_timer | 95/100 | 85/100 | 5/100 | 0/10 | 24.4/14.3 | 1003/492 | 1.7x |

## Headline

- **Validity**: 38% -> 92% (+54 pts) with the grammar.
- **Exact-match**: 30% -> 48% (+17 pts).
- **Forward passes**: 22.0 -> 11.8 mean (1.9x fewer) via jump-forward.
- Validity is a *hard guarantee* under the grammar; the remaining exact-match gap is the un-fine-tuned model picking a valid-but-wrong value (e.g. a valid US state code that isn't the intended one). Fine-tuning closes that gap; the grammar already closes the validity gap.
