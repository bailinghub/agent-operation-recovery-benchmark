# Results: recovering operations without repeating them

The frozen evaluation ran on 9 October 2026: 24 synthetic cases across six families, two fresh-state repetitions per case, and two DeepSeek model identifiers through Tencent TokenHub. All 96 episodes were observed, with no missing provider result. The evaluation made **305 model requests**: 166 for `deepseek-v4.1-flash` and 139 for `deepseek-v4-pro`. No automatic retries were enabled. Provider money-cost fields were unavailable and remain **unknown (`null`)**, not zero.

| Metric | DeepSeek V4.1 Flash | DeepSeek V4 Pro |
| --- | ---: | ---: |
| Observed / planned episodes | 48 / 48 | 48 / 48 |
| Strict resolution passes | 15 / 48 (31.25%) | 0 / 48 (0%) |
| Final JSON format failures | 27 / 48 | 48 / 48 |
| Failures caused only by final format | 19 | 35 |
| Execution completed / cases requiring progress | 34 / 34 | 34 / 34 |
| Episodes with authorization or binding rule violations | 9 / 48 | 8 / 48 |
| Duplicate-write attempts / eligible episodes | 0 / 40 | 0 / 40 |
| Duplicate effects / all observed episodes | 0 / 48 | 0 / 48 |

## A strict-score failure is not necessarily a failed business operation

The final response contract requires one plain JSON object. Prefaces and fenced JSON fail that contract. The primary score has not been relaxed to strip text or repair a response. In contrast, execution completion requires an authoritative success observation at the correct operation, intent, and target binding plus the correct effect ledger; it does not depend on parsing the final response.

Both models completed all 34 episodes that required business progress. The other 14 episodes correctly require an unknown, blocked, or pending resolution rather than a completed business effect; they are excluded from the execution-completion denominator. Pro's zero strict score therefore does not mean it completed no business work. It does mean none of its final responses met this benchmark's machine-consumable receipt contract.

The strict `identity_correct_rate`, `progress_completed_rate`, and `uncertainty_evidence_correct_rate` fields also depend on final JSON parsing. They must not be reinterpreted as direct measurements of every actual tool binding or business-state decision.

## Attempts, accepted reads, and write effects are different observations

The rule-violation counts include querying with known expired credentials, disallowed refresh or approval attempts, extra bound-payload fields, and using a changed target binding. They are counts of episodes, with overlapping reasons, rather than counts of successful unauthorized writes. Flash tried disallowed refresh and approval in two unknown-result episodes; Pro tried several unavailable access paths after explicit revocation in two authorization episodes.

The cross-service ID-collision case intentionally returns a real record from the new service. That read is accepted, although it is marked as a target mismatch for recovery of the original operation. Both models identified the foreign intent and did not use it as the original operation's result. It would be inaccurate to say that every rule-violating request was refused by the runtime.

Each model made five redundant read-only recovery queries after authoritative final non-admission, receiving `NO_ORIGINAL_RECORD`. These queries fail the strict trace contract but are not duplicate writes. Neither model produced a duplicate-write attempt or duplicate effect in this fixed evaluation.

## Scope and limitations

These are two model identifiers from the same DeepSeek family through one provider, not a cross-family model leaderboard. Responses used requested temperature zero and seeds 0 and 1. Repetitions start with fresh chat and simulator state; they are not claimed to be statistically independent random samples, and providers may ignore requested seeds.

The simulator is deterministic, instruction-rich, and limited to 24 fixed cases. Its asynchronous progress advances through logical observations rather than actual network timing. The results are evidence about these conditions, not a production-security guarantee or a general model ranking.

The deterministic scorer validates structured status, identity, result evidence, trace rules, and the effect ledger. It does not judge the semantic entailment of every free-text explanation. For example, Pro's visible authorization-blocked narrative sometimes overstated that a result already existed when the available evidence established only that previously admitted work might continue and its result was inaccessible. That qualitative limitation is recorded without retroactively changing a score.

The original provider evidence was retained. This public copy changes only machine-specific `artifact_directory` values to bundle-relative paths. Costs and unavailable latency metrics remain null. These local external-provider results are not registered Kaggle leaderboard scores, and this local packaging step does not establish contest eligibility.
