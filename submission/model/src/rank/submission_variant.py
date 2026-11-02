"""Frozen final-selection policy; generated deterministically."""
from __future__ import annotations

import numpy as np

VARIANT_ID = 'V18_T6_DE'


def _tt_only(pairs, score, queries, catalog, posthoc):
    from rank.pipeline import pick_best
    cfg = posthoc or {}
    tt = cfg.get("tt")
    if tt is None:
        return pick_best(pairs, score, catalog, len(queries))
    from rank.train_memory import tt_adjust
    penalty = tt.penalty(pairs, queries, catalog)
    adjusted = tt_adjust(np.asarray(score, dtype=np.float64), penalty, cfg.get("tt_weight", 10.0))
    return pick_best(pairs, adjusted, catalog, len(queries))


def _de_swap(queries, catalog, final, threshold=1.0, log=print):
    from common import is_hangul_name
    from rank.pipeline import DUAL_ENCODER
    encoders = [DUAL_ENCODER[key] for key in ("obj", "obj2", "obj3")
                if DUAL_ENCODER.get(key) is not None]
    if not encoders:
        return list(final)
    ids = catalog["vessel_id"].astype(str).to_numpy()
    positions = {value: index for index, value in enumerate(ids)}
    records = queries.to_dict("records")
    output, hangul_rows, swaps = list(final), 0, 0
    for index, row in enumerate(records):
        if not is_hangul_name(row.get("vessel_name")) or final[index] not in positions:
            continue
        hangul_rows += 1
        scores = None
        for encoder in encoders:
            query, catalog_matrix = encoder.query(row)
            value = catalog_matrix @ query
            scores = value if scores is None else scores + value
        current = positions[final[index]]
        current_score = float(scores[current])
        scores = scores.copy()
        scores[current] = -np.inf
        best = int(np.argmax(scores))
        if float(scores[best]) - current_score > threshold:
            output[index] = ids[best]
            swaps += 1
    log(f"submission variant de_swap t={threshold:g}: Hangul rows {hangul_rows}, swapped {swaps}")
    return output


def apply_submission_variant(pairs, score, queries, catalog, posthoc, original_final):
    if VARIANT_ID in ("V11_LATIN_TT", "V12_HANGUL_TT"):
        from common import is_hangul_name
        tt = _tt_only(pairs, score, queries, catalog, posthoc)
        hangul = queries["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
        if VARIANT_ID == "V11_LATIN_TT":
            return [original_final[i] if hangul[i] else tt[i] for i in range(len(tt))]
        return [tt[i] if hangul[i] else original_final[i] for i in range(len(tt))]
    if VARIANT_ID == "V18_T6_DE":
        return _de_swap(queries, catalog, original_final, 1.0)
    raise RuntimeError(f"unsupported packaged variant {VARIANT_ID}")
