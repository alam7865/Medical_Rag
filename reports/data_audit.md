# Data Audit

Raw splits as loaded (no cleaning applied).

## Schema

- **train**: columns=['Question', 'Answer', 'source', 'focus_area', 'qtype'] → question=`Question`, answer=`Answer`, id=`(generated)`, source=`source`, version=`482503f647648475`
- **validate**: columns=['Question', 'Answer', 'source', 'focus_area', 'qtype'] → question=`Question`, answer=`Answer`, id=`(generated)`, source=`source`, version=`029ccab9f2b91743`
- **test**: columns=['Question', 'Answer', 'source', 'focus_area', 'qtype'] → question=`Question`, answer=`Answer`, id=`(generated)`, source=`source`, version=`3b5cdc616dcf8f7a`

## Quality profile

| Check | train | validate | test |
|---|---:|---:|---:|
| rows | 539 | 132 | 139 |
| empty_or_null_questions | 8 | 0 | 0 |
| empty_or_null_answers | 10 | 0 | 1 |
| answers_empty_after_markup_removal | 14 | 0 | 0 |
| exact_duplicate_questions | 103 | 0 | 1 |
| exact_duplicate_qa_pairs | 65 | 0 | 0 |
| normalized_duplicate_questions | 142 | 0 | 1 |
| answers_with_html | 106 | 0 | 0 |
| answers_with_non_ascii | 66 | 0 | 0 |
| answer_len_mean | 222.2 | 238.6 | 244.4 |
| answer_len_max | 1778 | 1756 | 1756 |
| unique_sources | 3 | 2 | 2 |

## Sample records (train)

- `train-b07b718962ca-0` **Q:** 'How can wheezing lung disease be managed ?'  **A:** 'Ａsthmａ is mａnaged with inhaled corticosteroids for long-term control and short-acting bronchodilators such as albuterol for quick relief, a…'
- `train-4ac14dd331b2-0` **Q:** 'Which test shows if I have varicella ?'  **A:** 'Chickenpox is usually diagnosed from the characteristic rash. PCR testing of blister fluid can confirm the virus in unclear cases.'
- `train-75956b5bfc06-0` **Q:** 'How can rheumatoid arthritis be confirmed?'  **A:** 'Rheumatoid arthritis is diagnosed from joint examination plus blood tests for rheumatoid factor, anti-CCP antibodies, ESR and CRP, and imag…'
- `train-fba67a692d1f-0` **Q:** 'What is the best treatment for brittle bones?'  **A:** 'Osteoporosis is treated with bisphosphonates such as alendronate, denosumab or bone-building drugs like teriparatide, plus calcium, vitamin…'
- `train-a42fa5c3ee92-0` **Q:** 'How can renal calculi be managed?'  **A:** 'Small kidney stones often pass with fluids, pain relief and tamsulosin. Larger stones may need shock wave lithotripsy, ureteroscopy or perc…'
