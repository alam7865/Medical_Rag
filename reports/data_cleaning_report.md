# Data Cleaning Report

Cleaning applied to the knowledge corpus (`train.csv`) only. The raw file is unchanged; every decision is in `cleaning_log.csv`.

## Summary

| Step | Rows |
|---|---:|
| Original rows | 539 |
| Invalid rows removed | 46 |
| &nbsp;&nbsp;null rows | 17 |
| &nbsp;&nbsp;empty questions (incl. null) | 8 |
| &nbsp;&nbsp;empty answers (incl. null) | 23 |
| &nbsp;&nbsp;malformed / placeholder | 4 |
| &nbsp;&nbsp;too short | 11 |
| Exact duplicate QA pairs removed | 65 |
| Normalized duplicate QA pairs removed | 29 |
| **Final rows** | **399** |

## Detected but not removed

- Exact duplicate questions (any answer): 68
- Normalized duplicate questions (any answer): 100
- Rows sharing a question but with a *different* answer: 12 (kept - they may be complementary or need expert review; see `conflicting_questions.csv`)
- Semantic near-duplicate question pairs (≥ 0.95): 9 (kept - e.g. type 1 vs type 2 diabetes are similar but medically distinct)

## Removed rows by reason

| Reason | Rows |
|---|---:|
| exact_duplicate_qa_pair | 65 |
| normalized_duplicate_qa_pair | 29 |
| empty_answer | 14 |
| too_short_answer | 11 |
| null_answer | 9 |
| null_question | 7 |
| malformed_question | 4 |
| null_question_and_answer | 1 |

## Normalization applied (rows changed, kept or removed)

| Operation | Rows |
|---|---:|
| boilerplate_removed | 82 |
| html_removed | 106 |
| unicode_normalized | 42 |
| whitespace_normalized | 113 |

Normalization is conservative: Unicode NFKC, HTML tag/entity removal, known web boilerplate, whitespace. Punctuation, numbers, units (mg/dL, mmHg) and medical terminology are preserved; lowercasing is OFF (case can carry meaning, e.g. `HbA1c`, `IgA`).

## Examples

- **too_short_answer**: Q='What are the signs of asthma?' A='See a doctor.'
- **too_short_answer**: Q='Is there a way to avoid varicella?' A='It depends.'
- **exact_duplicate_qa_pair** (duplicate of `train-9d741c1f1fb3-0`): Q='What reduces the risk of kidney stones?' A='Kidney stones can be prevented by drinking enough water to produce about 2.5 liters of ur…'
- **exact_duplicate_qa_pair** (duplicate of `train-6be13f3dd95d-0`): Q='How do I know if I have high blood pressure?' A='Hypertension usually causes no symptoms, which is why it is called the silent killer. Ver…'
- **null_answer**: Q='Can renal calculi be prevented?' A=''
- **null_answer**: Q='How can type 2 diabetes be confirmed?' A=''
- **normalized_duplicate_qa_pair** (duplicate of `train-ae0bc2f1823d-0`): Q='HOW DO I PREVENT RHEUMATOID ARTHRITIS?' A='<p>Click here to subscribe to our newsletter for more health tips. Rheumatoid arthritis c…'
- **normalized_duplicate_qa_pair** (duplicate of `train-58c5b2fff92a-0`): Q='WHICH MEDICATIONS ARE USED FOR IRON-DEFICIENCY ANEMIA?' A='<p>Iron-deficiency anemia is treated with oral iron supplements such as ferrous sulfate, …'
- **empty_answer**: Q='How do I prevent iron-deficiency anemia?' A='<p></p>'
- **empty_answer**: Q='What reduces the risk of major depressive disorder?' A='<p></p>'
- **null_question**: Q='' A='Hypothyroidism is diagnosed with blood tests showing a high TSH and low free T4. Thyroid …'
- **null_question**: Q='' A='Hyperthyroidism is treated with antithyroid drugs such as methimazole, radioactive iodine…'
- **malformed_question**: Q='???' A='asdf'
- **malformed_question**: Q='question' A='answer'
- **null_question_and_answer**: Q='' A=''
- **normalized** (unicode_normalized;whitespace_normalized): A='Ａsthmａ is mａnaged with inhaled corticosteroids for long-term control and short-acting bronchodilators such as…'
- **normalized** (whitespace_normalized): A='Chickenpox is usually diagnosed from the characteristic rash. PCR testing of blister fluid can confirm the vi…'
- **normalized** (whitespace_normalized): A='Rheumatoid arthritis is diagnosed from joint examination plus blood tests for rheumatoid factor, anti-CCP ant…'

## Semantic near-duplicate examples (kept)

- 0.980: 'What are the signs of osteoarthritis?' ↔ 'What are the symptoms of osteoarthritis?'
- 0.979: 'What are the symptoms of type 2 diabetes?' ↔ 'What are the signs of type 2 diabetes?'
- 0.959: 'What are the symptoms of chickenpox?' ↔ 'What are the signs of chickenpox?'
- 0.956: 'Which test shows if I have underactive thyroid?' ↔ 'Which test shows if I have overactive thyroid?'
- 0.980: 'What are the symptoms of urinary tract infection?' ↔ 'What are the signs of urinary tract infection?'
