# Error Analysis (test split)

Success = correct document within top 1. Up to 4 examples per category, most informative (largest rank change) first.

## raw_fail_clean_success (26)

- **Q:** What are the treatments for hyperthyroidism?  
  **Expected:** Hyperthyroidism is treated with antithyroid drugs such as methimazole, radioactive iodine therapy or thyroid surgery. B…  
  **Raw** rank=3 · top-1: 'How do I prevent hyperthyroidism?' → '&nbsp;' _(cleaning status: removed:empty_answer)_  
  **Cleaned** rank=1 · top-1: 'How is hyperthyroidism treated?' → 'Hyperthyroidism is treated with antithyroid drugs such as methimazole, radioact…'  
  _Why:_ 2 raw document(s) ranked above the correct one were noise that cleaning removed (1x empty_answer, 1x too_short_answer).
- **Q:** Which test shows if I have wheezing lung disease?  
  **Expected:** Asthma is diagnosed with a clinical history plus spirometry showing airflow obstruction that improves after a bronchodi…  
  **Raw** rank=3 · top-1: 'Which test shows if I have lung infection?' → 'It depends.' _(cleaning status: removed:too_short_answer)_  
  **Cleaned** rank=1 · top-1: 'How do doctors check for wheezing lung disease?' → 'Asthma is diagnosed with a clinical history plus spirometry showing airflow obs…'  
  _Why:_ 2 raw document(s) ranked above the correct one were noise that cleaning removed (1x too_short_answer, 1x empty_answer).
- **Q:** How can GERD be confirmed?  
  **Expected:** GERD is usually diagnosed from typical symptoms and response to acid-suppressing treatment. Upper endoscopy and esophag…  
  **Raw** rank=3 · top-1: 'How is GERD diagnosed?' → '<p></p>' _(cleaning status: removed:empty_answer)_  
  **Cleaned** rank=1 · top-1: 'How do doctors check for acid reflux?' → 'GERD is usually diagnosed from typical symptoms and response to acid-suppressin…'  
  _Why:_ 2 raw document(s) ranked above the correct one were noise that cleaning removed (1x empty_answer, 1x too_short_answer).
- **Q:** How is hypertension diagnosed?  
  **Expected:** Hypertension is diagnosed when repeated blood pressure readings are 130/80 mmHg or higher. Home or ambulatory monitorin…  
  **Raw** rank=3 · top-1: 'How is hypertension treated?' → '&nbsp;' _(cleaning status: removed:empty_answer)_  
  **Cleaned** rank=1 · top-1: 'How can hypertension be confirmed?' → 'Hypertension is diagnosed when repeated blood pressure readings are 130/80 mmHg…'  
  _Why:_ 2 raw document(s) ranked above the correct one were noise that cleaning removed (1x empty_answer, 1x null_answer).

## raw_success_clean_fail (2)

- **Q:** How do I know if I have chronic obstructive pulmonary disease?  
  **Expected:** COPD causes a chronic cough with mucus, shortness of breath that worsens with activity, wheezing and frequent chest inf…  
  **Raw** rank=1 · top-1: 'What does chronic obstructive pulmonary disease feel like?' → 'COPD \u200bcauses \u200ba chronic cough with mucus, shortness of breath that worsens with…' _(cleaning status: kept)_  
  **Cleaned** rank=2 · top-1: 'Which test shows if I have chronic obstructive pulmonary di…' → 'COPD is diagnosed by spirometry showing a post-bronchodilator FEV1/FVC ratio be…'  
  _Why:_ Cleaned ranking placed other documents above the correct one.
- **Q:** Explain severe recurring headaches in detail.  
  **Expected:** Migraine is a common medical condition. Migraine is a neurological disorder involving activation of the trigeminovascul…  
  **Raw** rank=1 · top-1: 'Tell me about severe recurring headaches.' → 'Migraine is a common medical condition. Migraine is a neurological disorder inv…' _(cleaning status: kept)_  
  **Cleaned** rank=2 · top-1: 'What is the cause of severe recurring headaches?' → 'Migraine is a neurological disorder involving activation of the trigeminovascul…'  
  _Why:_ Cleaned ranking placed other documents above the correct one.

## both_failed (41)

- **Q:** Is there a way to avoid juvenile diabetes?  
  **Expected:** There is currently no proven way to prevent type 1 diabetes. Screening relatives for autoantibodies can identify people…  
  **Raw** rank=2 · top-1: 'Can juvenile diabetes be prevented?' → 'There is currently no proven way to prevent type 1 diabetes. Consult a healthca…' _(cleaning status: kept_flagged)_  
  **Cleaned** rank=2 · top-1: 'Can juvenile diabetes be prevented?' → 'There is currently no proven way to prevent type 1 diabetes. Consult a healthca…'  
  _Why:_ Neither corpus ranks the answer in top-K: likely an embedding/paraphrase limitation, not data quality.
- **Q:** How does someone develop overactive thyroid?  
  **Expected:** Hyperthyroidism is most commonly caused by Graves' disease. Toxic nodular goiter, thyroiditis and excess iodine or thyr…  
  **Raw** rank=3 · top-1: 'How does someone develop underactive thyroid?' → "Hypothyroidism \u200bis \u200bmost often caused by Hashimoto's thyroiditis, an autoimmune…" _(cleaning status: kept)_  
  **Cleaned** rank=2 · top-1: 'How does someone develop underactive thyroid?' → "Hypothyroidism is most often caused by Hashimoto's thyroiditis, an autoimmune c…"  
  _Why:_ Neither corpus ranks the answer in top-K: likely an embedding/paraphrase limitation, not data quality.
- **Q:** How can severe recurring headaches be managed?  
  **Expected:** Migraine attacks are treated with NSAIDs, triptans or gepants. Frequent migraines may need preventive therapy such as b…  
  **Raw** rank=3 · top-1: 'Can severe recurring headaches be prevented?' → 'Migraine frequency can <b>be</b> reduced by keeping a headache diary, regular s…' _(cleaning status: kept)_  
  **Cleaned** rank=3 · top-1: 'Can severe recurring headaches be prevented?' → 'Migraine frequency can be reduced by keeping a headache diary, regular sleep an…'  
  _Why:_ Neither corpus ranks the answer in top-K: likely an embedding/paraphrase limitation, not data quality.
- **Q:** Can overactive thyroid be prevented?  
  **Expected:** Hyperthyroidism usually cannot be prevented, but avoiding excess iodine supplements and attending regular follow-up hel…  
  **Raw** rank=10 · top-1: 'How can overactive thyroid be managed?' → 'Click here to subscribe to our newsletter for more health tips. Hyperthyroidism…' _(cleaning status: kept)_  
  **Cleaned** rank=8 · top-1: 'How can overactive thyroid be managed?' → 'Hyperthyroidism is treated with antithyroid drugs such as methimazole, radioact…'  
  _Why:_ Neither corpus ranks the answer in top-K: likely an embedding/paraphrase limitation, not data quality.

## both_succeeded (62)

- **Q:** How is pneumonia diagnosed?  
  **Expected:** Pneumonia is diagnosed by physical examination and a chest X-ray showing infiltrates. Blood tests, pulse oximetry and s…  
  **Raw** rank=1 · top-1: 'How can pneumonia be confirmed?' → 'Pneumonia is diagnosed by physical examination and a chest X-ray showing infilt…' _(cleaning status: kept)_  
  **Cleaned** rank=1 · top-1: 'How can pneumonia be confirmed?' → 'Pneumonia is diagnosed by physical examination and a chest X-ray showing infilt…'  
  _Why:_ 
- **Q:** How is psoriasis diagnosed?  
  **Expected:** Psoriasis is usually diagnosed by examining the skin, scalp and nails. A skin biopsy is occasionally done to rule out o…  
  **Raw** rank=1 · top-1: 'How do doctors check for scaly skin plaques?' → 'Psoriasis is usually diagnosed by examining the skin, scalp and nails. A skin b…' _(cleaning status: kept)_  
  **Cleaned** rank=1 · top-1: 'How can psoriasis be confirmed?' → 'Psoriasis is usually diagnosed by examining the skin, scalp and nails. A skin b…'  
  _Why:_ 
- **Q:** Which medications are used for type 2 diabetes?  
  **Expected:** Type 2 diabetes treatment starts with healthy eating, weight loss and physical activity, usually with metformin. Other …  
  **Raw** rank=1 · top-1: 'What is the best treatment for adult-onset diabetes?' → 'Type 2 diabetes treatment starts with healthy eating, weight loss and physical …' _(cleaning status: kept)_  
  **Cleaned** rank=1 · top-1: 'What is the best treatment for adult-onset diabetes?' → 'Type 2 diabetes treatment starts with healthy eating, weight loss and physical …'  
  _Why:_ 
- **Q:** Can the flu be prevented?  
  **Expected:** The annual flu vaccine is the best prevention against influenza, together with hand washing, covering coughs and stayin…  
  **Raw** rank=1 · top-1: 'Is there a way to avoid the flu?' → 'The annual flu vaccine is the best prevention against influenza, together with …' _(cleaning status: kept)_  
  **Cleaned** rank=1 · top-1: 'Is there a way to avoid the flu?' → 'The annual flu vaccine is the best prevention against influenza, together with …'  
  _Why:_ 

