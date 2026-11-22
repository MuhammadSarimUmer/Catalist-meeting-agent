# Eval report

## Method
- 13 questions (E01-E13) on teams that are **not** among the 25 graded questions, so nothing was tuned on them.
- Gold answers were written by reading the raw transcript turns directly (keyword search with `scripts/find.py` plus reading the turns in context), not by asking the agent. The supporting turn ids are stored in the `evidence` field of `evals/my_questions.jsonl`, so anyone can re-check them.
- The gold answers were prepared with an AI assistant's help, and I (the candidate) must re-read the evidence turns before relying on them. This is stated here so the method is not misrepresented.
- `python -m scripts.run_eval` runs the agent on each question and checks two things automatically: every citation is valid (exists, 6+ words, one turn), and the key facts (`must_contain`) appear. The verdict (right / partial / wrong) was set by reading the agent answer against the gold answer. Full detail is in `evals/results.json`.

## Results
| Measure | Result |
|---|---|
| Citations valid (exact, automatic) | 13 / 13 |
| Verdict: right | 8 / 13 |
| Verdict: partial | 3 / 13 |
| Verdict: wrong | 2 / 13 |
| Automatic key-fact match | 69% (crude: it misses "25" vs "twenty-five" and "battery" vs "batteries", so it under-counts) |

| ID | Team | Verdict | Note |
|---|---|---|---|
| E01 | ES2005 | partial | Rubber is right, but it misses the early plastic idea. |
| E02 | ES2012 | right | |
| E03 | ES2013 | right | |
| E04 | ES2015 | right | |
| E05 | IS1000 | right | |
| E06 | IS1001 | right | |
| E07 | IS1004 | right | |
| E08 | IS1004 | wrong | Said "palm-shaped"; the team chose a banana shape. |
| E09 | TS3005 | right | |
| E10 | TS3005 | right | |
| E11 | TS3008 | wrong | Said the power plan was settled; it was not. |
| E12 | ES2013 | partial | Missed the research promise and the "no costing information" report. |
| E13 | ES2013 | partial | Right final decision, wrong claim that nothing was decided earlier. |

## Three honest failures
1. **E08 (wrong shape).** The agent answered "palm-shaped", while the transcript shows a banana shape ("we're going for a stylish banana shape"). *Likely why (not confirmed by inspecting the retrieved evidence):* the question was about "shape", and the model picked up a turn about fitting the hand instead. Quotes were valid, but they did not support the claim. *Fix:* check each citation against the claim with a small entailment call, and retrieve using extra keywords from the notes (such as "banana").
2. **E11 (overconfident on a power plan that was never settled).** The agent said the team settled on one ordinary battery and ruled out kinetic power. In the last meeting a member still suggested doing "both, battery and kinetic". *Likely why (not confirmed):* the latest notes said "include one battery", and the "latest meeting wins" rule treats that as final. *Fix:* give notes a "tentative vs agreed" flag, and let the answer say "not clearly settled".
3. **E12 / E13 (right fact, missing context).** E12 gave the final cost but not the promise and the report-back; E13 said nothing was decided earlier, though meeting c already agreed on kinetic power if feasible. *Likely why (not confirmed):* retrieval keeps the best chunks with only 2 guaranteed per meeting, so small but important turns can be left out. *Fix:* retrieve more notes per meeting for "what changed" and "who promised" questions, and add a self-check step ("does the answer cover every meeting?").

## Other notes
- A bug found by this eval: model answers sometimes contained internal markers such as `[turn_id:4075, hint:...]` or `(Speaker B, 12442)`. `clean_text` in `services/citation_service.py` now removes them, and `answers.json` was checked and has none.
- The eval run took about 20 minutes, mostly waiting on the Groq tokens-per-minute limit (about one answer per minute).
