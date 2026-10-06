import pandas as pd

from src.data.cleaner import clean_dataframe
from src.data.deduplicator import count_duplicated, find_conflicting_questions, max_similarity
from src.data.loader import standardize
from src.data.normalization import matching_key, normalize_text


def make(rows, cfg):
    return standardize(pd.DataFrame(rows, columns=["question", "answer"]), "train", cfg)


# ---------- normalization ------------------------------------------------------
def test_normalize_preserves_medical_content():
    text = "HbA1c ≥ 6.5% and BP 130/80 mmHg; give 500 mg (q.d.)."
    assert normalize_text(text) == text


def test_normalize_does_not_strip_less_than_values():
    assert normalize_text("Keep glucose <7 mmol/L") == "Keep glucose <7 mmol/L"


def test_normalize_removes_html_and_entities():
    assert normalize_text("<p>Fever &amp; cough<br/>Rest.</p>") == "Fever & cough\nRest."


def test_normalize_unicode():
    # full-width letters, NBSP, zero-width space, smart quotes
    assert normalize_text("Ａsthma is​ “common”") == 'Asthma is "common"'


def test_normalize_whitespace_and_boilerplate():
    assert normalize_text("  Gout   hurts.  Click here to subscribe to our newsletter. ") == "Gout hurts."
    assert normalize_text("Fact. © 2019 Site. All rights reserved.") == "Fact."


def test_boilerplate_with_irregular_whitespace():
    assert normalize_text("\tRead   more   » Major depression is common.") == "Major depression is common."
    assert normalize_text("Click\u00a0here  to subscribe. Gout hurts.") == "Gout hurts."


def test_lowercase_is_optional():
    assert normalize_text("HbA1c", lowercase=True) == "hba1c"
    assert normalize_text("HbA1c") == "HbA1c"


def test_matching_key_is_insensitive_to_formatting():
    assert matching_key("What is GOUT?") == matching_key("  what is  <b>gout</b> ")
    assert matching_key("Type 1 diabetes") != matching_key("Type 2 diabetes")


def test_missing_values_normalize_to_empty():
    assert normalize_text(None) == "" and normalize_text(float("nan")) == ""


# ---------- cleaner ------------------------------------------------------------
def test_removes_invalid_rows_with_reasons(cfg):
    df = make(
        [
            ["What is gout?", ""],                       # blank cell == missing (as pandas reads CSV)
            ["", "Gout is a type of arthritis."],         # empty question
            [None, None],                                 # null both
            ["What is gout?", "<p></p>"],                 # empty after markup removal
            ["test", "test"],                             # placeholder
            ["What is gout?", "Yes."],                    # too short
            ["What is gout?", "Gout is a type of inflammatory arthritis."],
        ],
        cfg,
    )
    res = clean_dataframe(df, cfg)
    assert len(res.cleaned) == 1
    reasons = res.log.set_index("question_id").loc[df["question_id"], "reason"].tolist()
    assert reasons[:6] == ["null_answer", "null_question", "null_question_and_answer", "empty_answer",
                           "malformed_question", "too_short_answer"]
    assert res.report["invalid_rows"] == 6
    assert res.report["empty_answers"] == 2 and res.report["null_rows"] == 3


def test_exact_and_normalized_duplicates_removed_first_kept(cfg):
    a = "Gout is a type of inflammatory arthritis."
    df = make([["What is gout?", a], ["What is gout?", a], ["WHAT IS GOUT ?", f"<b>{a}</b>"]], cfg)
    res = clean_dataframe(df, cfg)
    assert res.cleaned["question_id"].tolist() == [df["question_id"][0]]
    assert res.report["exact_duplicates"] == 1 and res.report["normalized_duplicates"] == 1
    assert set(res.log["duplicate_of"]) == {"", df["question_id"][0]}


def test_same_question_different_answer_is_kept_and_flagged(cfg):
    df = make([["What is gout?", "Gout is inflammatory arthritis."], ["What is gout?", "Gout is caused by uric acid crystals."]], cfg)
    res = clean_dataframe(df, cfg)
    assert len(res.cleaned) == 2
    assert (res.log["action"] == "kept_flagged").sum() == 2


def test_very_long_answer_is_kept_intact(cfg):
    long = "Osteoporosis weakens bones. " * 500
    res = clean_dataframe(make([["What is osteoporosis?", long]], cfg), cfg)
    assert res.cleaned["answer"][0] == long.strip()


def test_special_characters_and_unicode_survive(cfg):
    ans = "Use 5–10 µg/kg; Sjögren's syndrome α-blockers ≥ 2 doses."
    res = clean_dataframe(make([["Dosing for Sjögren's?", ans]], cfg), cfg)
    out = res.cleaned["answer"][0]
    # NFKC folds the micro sign U+00B5 into Greek mu U+03BC - same meaning, consistent form.
    assert "\u03bcg/kg" in out and "Sjögren's" in out and "α-blockers" in out and "≥" in out


def test_report_is_consistent(cfg):
    a = "Gout is a type of inflammatory arthritis."
    res = clean_dataframe(make([["What is gout?", a], ["What is gout?", a], ["", a]], cfg), cfg)
    r = res.report
    assert r["final_rows"] == r["total_rows"] - r["invalid_rows"] - r["exact_duplicates"] - r["normalized_duplicates"]
    assert len(res.log) == r["total_rows"]  # every row is logged


def test_semantic_duplicates_are_flagged_not_removed(cfg, embedder):
    df = make([["What are symptoms of type 1 diabetes?", "Thirst and weight loss are common."],
               ["What are symptoms of type 2 diabetes?", "Many people have no symptoms early."]], cfg)
    from dataclasses import replace
    res = clean_dataframe(df, replace(cfg, semantic_dup_threshold=0.8), embedder)
    assert len(res.cleaned) == 2
    assert len(res.semantic_duplicates) == 1


# ---------- dedup helpers --------------------------------------------------------
def test_count_duplicated():
    assert count_duplicated(pd.Series(["a", "a", "b", "a"])) == 2


def test_find_conflicting_questions():
    df = pd.DataFrame({"question_id": ["1", "2", "3"], "question": ["Q?", "q", "Other"], "answer": ["A", "B", "C"]})
    assert set(find_conflicting_questions(df)["question_id"]) == {"1", "2"}


def test_max_similarity_excludes_self():
    import numpy as np
    e = np.eye(3)
    e[2] = e[0]
    idx, score = max_similarity(e, e, exclude_self=True)
    assert idx[0] == 2 and score[0] == 1.0
