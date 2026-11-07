# Compute study amendment A2 — completion budget only

2026-11-06 Bangkok. Parent protocol SHA256 `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`; [A1](T5-COMPUTE-DL-SPEC-A1.md) SHA256 `0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453`. JJ explicitly authorizes extending held GPU time to finish the pre-scheduled P arm, accepting the additional serving outage. **This amendment changes budget and permits one optional P restoration break, and adds nothing else.** Freeze its SHA/time before adopting the extension or reaching A1's admission cutoff.

## New cumulative limit

**Total operator-held GPU-time limit:36000s (10 hours). New admission cutoff:33900s (9h25m). Reserve:2100s (35 minutes)** for an in-flight request, drain and G restoration. These replace18000s/15900s only. Carry the complete existing held-time ledger forward without resetting/subtracting earlier boots, diagnostics, replays, pauses while locks are held or restorations. Offline CPU work and waiting with G restored remain outside held GPU time as before.

Sizing: at the reported decision point **49/240 P replies collected**,191 remain. At100–110s each, remaining generation alone is19100–21010s (**5h18m20s–5h50m10s**), before any new boot/restoration overhead. All240 P replies at110s cost26400s; adding the original3413.519s, remaining D/L and intervening boot/restoration/diagnostic costs, an optional break and reserve motivates the10-hour cumulative ceiling. This is headroom, not a promise to consume ten hours or a guarantee of completion. Actual counters, not these estimates, govern admission. Report the actual further serving outage and any divergence from the estimate.

After A2 freezes, Implementer A may update the running controller's budget configuration **only if it can safely adopt the new limits without resetting the ledger, changing request behavior, duplicating a request or running two collection controllers**. Append an audit record binding old/new limits, unchanged accumulated held seconds, actual adoption time and A2 SHA. Validate cached/configured limits agree. Prefer continuing the current qualified P boot; do not remove the request deadline or other safety checks to keep it alive.

Once cumulative held time reaches33900s, start **no new request or boot**. Drain the in-flight request under its unchanged270s deadline and restore G. If completion is still impossible, retain **INCOMPLETE** with scheduled missingness/solve bounds and unknown missing costs, as A1 requires. Safety restoration takes precedence if it overruns36000s; disclose the overrun. No autonomous further extension, cap change, cohort shrink or selected completed-subset efficacy claim.

## Optional restoration break

If safe continuation of the current controller/boot is unavailable, or Implementer A needs a break, permit **one G-restore break between P collection turns**. Stop admission, settle the current request, preserve all attempted/observed records, drain and restore G. Before resuming, seal the exact **unattempted P-key suffix** and original wire hashes against the fixed240-key P manifest. Preserve original request order. Never retry an observed correct/wrong/capped/error/policy-terminated reply; reconcile ambiguous in-flight attempts from durable/native evidence before scheduling anything. Fresh namespace and the inherited cold-boot/environment/diagnostic qualifications are mandatory. No concurrent second collector.

The break, new cold cache/boot and collection period are **disclosed confounds**, with boot/restore costs included in the same cumulative ledger. Report P before/after-break batch descriptors, without excluding a batch or changing the primary analysis. Waiting with G restored is excluded only under the original held-time definition. The extra P turn is continuation of existing keys, not a restart, replacement cohort or extra replicate.

## Outcome-informed decision and unchanged scope

The extension was authorized **after partial P outcomes were seen**, at the reported49 replies/11:53Z decision point; that cannot be called pre-outcome budgeting. Record the actual durable key count/mask at adoption, which may exceed49 as live collection continues. The reason is completion of JJ's ordinary-answer comparator, not a newly selected effect threshold. Do not inspect additional outcomes to choose another extension or a stopping point.

**Only the original720 pre-scheduled P/D/L keys may be completed.** Tasks, answers, seeds, prompts/declarations, image, strict primary extraction, secondary sensitivity, amended L/P primary, D/L mechanism contrast, verifier/clipping policy, activation/safety gates, sampling,8192 returned-ID cap,270s request deadline and analysis are unchanged. Every already collected reply remains. A1's post-partial-outcome inference limitations and accounting limits remain. T5 performs no GPU/queue action; this is the budget freeze requested by JJ.
