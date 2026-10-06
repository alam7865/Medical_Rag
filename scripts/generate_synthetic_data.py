"""Generate a SYNTHETIC medical QA dataset (train / validate / test).

This is placeholder data so the project runs end-to-end. Replace the files in
data/raw/ with your own dataset and re-run the pipeline - nothing else needs
to change (column names are auto-detected; see .env.example).

The content is general educational text written for this demo; it is NOT
medical advice.

Design
------
* Each (condition, aspect) pair is one "fact" with one canonical answer.
* Each fact has 5 question phrasings. They are shuffled per fact: 1 goes to
  test, 1 to validate, and the remaining 3 to train. So eval questions are
  paraphrases that never appear verbatim in train.
* Train then gets realistic noise injected (duplicates, HTML, boilerplate,
  Unicode artefacts, empty / junk rows, conflicting answers, long records).
* A few planted overlaps are added to test/validate so leakage detection has
  something to find.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config  # noqa: E402

# name, lay name, {aspect: answer}
KB: list[tuple[str, str, dict[str, str]]] = [
    ("type 1 diabetes", "juvenile diabetes", {
        "symptoms": "Type 1 diabetes often causes increased thirst, frequent urination, unexplained weight loss, extreme hunger, fatigue and blurred vision. Symptoms can develop quickly over a few weeks, and some people first present with diabetic ketoacidosis.",
        "causes": "Type 1 diabetes is an autoimmune disease in which the immune system destroys the insulin-producing beta cells of the pancreas. Genetic susceptibility and possibly environmental triggers such as viral infections are thought to play a role.",
        "diagnosis": "Type 1 diabetes is diagnosed with blood glucose tests such as fasting plasma glucose, HbA1c or a random glucose above 200 mg/dL with symptoms. Autoantibody tests and C-peptide levels help distinguish it from type 2 diabetes.",
        "treatment": "Type 1 diabetes is treated with lifelong insulin therapy delivered by multiple daily injections or an insulin pump, together with blood glucose monitoring, carbohydrate counting and regular exercise.",
        "prevention": "There is currently no proven way to prevent type 1 diabetes. Screening relatives for autoantibodies can identify people at high risk, and the drug teplizumab may delay onset in some of them.",
    }),
    ("type 2 diabetes", "adult-onset diabetes", {
        "symptoms": "Type 2 diabetes develops slowly and may cause increased thirst, frequent urination, tiredness, slow-healing sores, frequent infections and tingling in the hands or feet. Many people have no symptoms for years.",
        "causes": "Type 2 diabetes results from insulin resistance combined with a gradual decline in insulin production. Excess body weight, physical inactivity, family history and older age increase the risk.",
        "diagnosis": "Type 2 diabetes is diagnosed when HbA1c is 6.5% or higher, fasting plasma glucose is 126 mg/dL or higher, or a 2-hour oral glucose tolerance test reading is 200 mg/dL or higher.",
        "treatment": "Type 2 diabetes treatment starts with healthy eating, weight loss and physical activity, usually with metformin. Other options include SGLT2 inhibitors, GLP-1 receptor agonists and insulin.",
        "prevention": "Type 2 diabetes can often be prevented or delayed by losing 5 to 7 percent of body weight, exercising about 150 minutes per week and choosing a diet rich in fiber and low in refined sugar.",
    }),
    ("hypertension", "high blood pressure", {
        "symptoms": "Hypertension usually causes no symptoms, which is why it is called the silent killer. Very high readings can cause severe headache, chest pain, shortness of breath, nosebleeds or vision changes.",
        "causes": "Most hypertension is primary, linked to age, genetics, high salt intake, obesity, alcohol and inactivity. Secondary hypertension can be caused by kidney disease, sleep apnea, thyroid problems or some medications.",
        "diagnosis": "Hypertension is diagnosed when repeated blood pressure readings are 130/80 mmHg or higher. Home or ambulatory monitoring helps confirm the diagnosis and rule out white-coat hypertension.",
        "treatment": "Hypertension is treated with lifestyle changes such as reducing salt and alcohol, plus medicines including ACE inhibitors, angiotensin receptor blockers, calcium channel blockers and thiazide diuretics.",
        "prevention": "Hypertension risk can be lowered by keeping a healthy weight, limiting sodium, following the DASH diet, exercising regularly, limiting alcohol and not smoking.",
    }),
    ("asthma", "wheezing lung disease", {
        "symptoms": "Asthma causes episodes of wheezing, shortness of breath, chest tightness and coughing, often worse at night or early morning and triggered by exercise, allergens or cold air.",
        "causes": "Asthma is caused by chronic inflammation and hyperresponsiveness of the airways. Genetics, allergies, early respiratory infections and exposure to tobacco smoke or pollution increase the risk.",
        "diagnosis": "Asthma is diagnosed with a clinical history plus spirometry showing airflow obstruction that improves after a bronchodilator. Peak flow monitoring and FeNO testing can support the diagnosis.",
        "treatment": "Asthma is managed with inhaled corticosteroids for long-term control and short-acting bronchodilators such as albuterol for quick relief, along with avoiding known triggers and having a written action plan.",
        "prevention": "Asthma attacks can be prevented by taking controller medication as prescribed, avoiding triggers such as smoke and dust mites, getting the flu vaccine and monitoring peak flow.",
    }),
    ("COPD", "chronic obstructive pulmonary disease", {
        "symptoms": "COPD causes a chronic cough with mucus, shortness of breath that worsens with activity, wheezing and frequent chest infections. Symptoms progress slowly over years.",
        "causes": "COPD is mainly caused by long-term cigarette smoking. Occupational dust and chemicals, indoor smoke from cooking fuels and the genetic condition alpha-1 antitrypsin deficiency also contribute.",
        "diagnosis": "COPD is diagnosed by spirometry showing a post-bronchodilator FEV1/FVC ratio below 0.70 in a person with symptoms and risk factors. Chest imaging helps exclude other conditions.",
        "treatment": "COPD treatment includes stopping smoking, long-acting bronchodilators (LAMA and LABA inhalers), pulmonary rehabilitation, vaccinations and oxygen therapy for people with low blood oxygen.",
        "prevention": "The best way to prevent COPD is never to smoke or to quit smoking, and to reduce exposure to occupational dust, fumes and indoor air pollution.",
    }),
    ("migraine", "severe recurring headaches", {
        "symptoms": "Migraine causes moderate to severe throbbing headache, usually on one side, lasting 4 to 72 hours, with nausea and sensitivity to light and sound. Some people have a visual aura beforehand.",
        "causes": "Migraine is a neurological disorder involving activation of the trigeminovascular system. Common triggers include stress, hormonal changes, lack of sleep, skipped meals, alcohol and certain foods.",
        "diagnosis": "Migraine is diagnosed clinically based on the pattern of headaches and associated symptoms. Brain imaging is only needed when there are red-flag features such as a sudden severe headache or neurological deficits.",
        "treatment": "Migraine attacks are treated with NSAIDs, triptans or gepants. Frequent migraines may need preventive therapy such as beta blockers, topiramate, amitriptyline or CGRP monoclonal antibodies.",
        "prevention": "Migraine frequency can be reduced by keeping a headache diary, regular sleep and meals, staying hydrated, managing stress and avoiding identified triggers.",
    }),
    ("pneumonia", "lung infection", {
        "symptoms": "Pneumonia causes cough with phlegm, fever, chills, shortness of breath and sharp chest pain when breathing. Older adults may instead have confusion and a low body temperature.",
        "causes": "Pneumonia is an infection of the air sacs of the lungs, most often caused by bacteria such as Streptococcus pneumoniae, viruses such as influenza and RSV, or less commonly fungi.",
        "diagnosis": "Pneumonia is diagnosed by physical examination and a chest X-ray showing infiltrates. Blood tests, pulse oximetry and sputum cultures help judge severity and identify the organism.",
        "treatment": "Bacterial pneumonia is treated with antibiotics such as amoxicillin or a macrolide. Rest, fluids and fever control help recovery, and severe cases need hospital care and oxygen.",
        "prevention": "Pneumonia can be prevented with pneumococcal and influenza vaccines, hand washing, not smoking and good management of chronic illnesses.",
    }),
    ("influenza", "the flu", {
        "symptoms": "Influenza causes sudden fever, chills, muscle aches, headache, dry cough, sore throat and extreme tiredness. Most people recover in one to two weeks.",
        "causes": "Influenza is caused by influenza A and B viruses that spread through respiratory droplets when infected people cough, sneeze or talk, and through contaminated surfaces.",
        "diagnosis": "Influenza is usually diagnosed from symptoms during flu season. Rapid antigen tests or PCR on a nasal swab confirm the infection.",
        "treatment": "Influenza is treated with rest, fluids and fever reducers. Antiviral drugs such as oseltamivir work best when started within 48 hours and are recommended for high-risk patients.",
        "prevention": "The annual flu vaccine is the best prevention against influenza, together with hand washing, covering coughs and staying home when sick.",
    }),
    ("iron-deficiency anemia", "low iron", {
        "symptoms": "Iron-deficiency anemia causes fatigue, weakness, pale skin, shortness of breath on exertion, dizziness, cold hands and feet, brittle nails and sometimes cravings for ice.",
        "causes": "Iron-deficiency anemia results from blood loss such as heavy periods or gastrointestinal bleeding, low dietary iron, poor absorption as in celiac disease, or increased needs during pregnancy.",
        "diagnosis": "Iron-deficiency anemia is diagnosed with a complete blood count showing low hemoglobin and small red cells, plus low serum ferritin. The underlying cause of iron loss should be investigated.",
        "treatment": "Iron-deficiency anemia is treated with oral iron supplements such as ferrous sulfate, or intravenous iron if tablets are not tolerated, while treating the source of blood loss.",
        "prevention": "Iron-deficiency anemia can be prevented by eating iron-rich foods such as red meat, beans and leafy greens with vitamin C, and by iron supplementation during pregnancy when advised.",
    }),
    ("hypothyroidism", "underactive thyroid", {
        "symptoms": "Hypothyroidism causes tiredness, weight gain, feeling cold, constipation, dry skin, hair thinning, depression and slowed heart rate.",
        "causes": "Hypothyroidism is most often caused by Hashimoto's thyroiditis, an autoimmune condition. Other causes include thyroid surgery, radioactive iodine treatment and iodine deficiency.",
        "diagnosis": "Hypothyroidism is diagnosed with blood tests showing a high TSH and low free T4. Thyroid peroxidase antibodies suggest Hashimoto's thyroiditis.",
        "treatment": "Hypothyroidism is treated with daily levothyroxine, a synthetic thyroid hormone, with the dose adjusted according to TSH levels checked every 6 to 8 weeks.",
        "prevention": "Most hypothyroidism cannot be prevented, but adequate dietary iodine prevents the iodine-deficiency form, and screening allows early treatment.",
    }),
    ("hyperthyroidism", "overactive thyroid", {
        "symptoms": "Hyperthyroidism causes weight loss despite good appetite, rapid or irregular heartbeat, anxiety, tremor, sweating, heat intolerance and frequent bowel movements.",
        "causes": "Hyperthyroidism is most commonly caused by Graves' disease. Toxic nodular goiter, thyroiditis and excess iodine or thyroid hormone medication are other causes.",
        "diagnosis": "Hyperthyroidism is diagnosed with a low TSH and high free T4 or T3. Thyroid antibodies and a radioactive iodine uptake scan help identify the cause.",
        "treatment": "Hyperthyroidism is treated with antithyroid drugs such as methimazole, radioactive iodine therapy or thyroid surgery. Beta blockers control symptoms like palpitations.",
        "prevention": "Hyperthyroidism usually cannot be prevented, but avoiding excess iodine supplements and attending regular follow-up help detect relapse early.",
    }),
    ("osteoarthritis", "wear-and-tear arthritis", {
        "symptoms": "Osteoarthritis causes joint pain that worsens with use, stiffness after rest, reduced range of motion, swelling and a grating sensation, commonly in the knees, hips and hands.",
        "causes": "Osteoarthritis develops as the cartilage cushioning the joints breaks down over time. Age, obesity, joint injury, repetitive stress and genetics increase the risk.",
        "diagnosis": "Osteoarthritis is diagnosed from symptoms and examination, with X-rays showing joint space narrowing and bone spurs. Blood tests are used to rule out inflammatory arthritis.",
        "treatment": "Osteoarthritis is managed with exercise, weight loss, physical therapy, topical or oral NSAIDs and joint injections. Severe cases may need joint replacement surgery.",
        "prevention": "Osteoarthritis risk can be reduced by maintaining a healthy weight, staying active with low-impact exercise, strengthening muscles and avoiding joint injuries.",
    }),
    ("rheumatoid arthritis", "inflammatory joint disease", {
        "symptoms": "Rheumatoid arthritis causes painful, swollen, warm joints, usually symmetrical in the small joints of the hands and feet, with morning stiffness lasting more than an hour and fatigue.",
        "causes": "Rheumatoid arthritis is an autoimmune disease in which the immune system attacks the synovium lining the joints. Genetic factors, smoking and female sex increase the risk.",
        "diagnosis": "Rheumatoid arthritis is diagnosed from joint examination plus blood tests for rheumatoid factor, anti-CCP antibodies, ESR and CRP, and imaging to look for joint erosions.",
        "treatment": "Rheumatoid arthritis is treated early with disease-modifying drugs such as methotrexate, plus biologics like TNF inhibitors when needed, and short courses of steroids for flares.",
        "prevention": "Rheumatoid arthritis cannot be fully prevented, but not smoking lowers the risk and early treatment prevents permanent joint damage.",
    }),
    ("GERD", "acid reflux", {
        "symptoms": "GERD causes heartburn, a burning feeling in the chest after meals, regurgitation of sour liquid, difficulty swallowing, chronic cough and hoarseness.",
        "causes": "GERD occurs when the lower esophageal sphincter relaxes abnormally and stomach acid flows back into the esophagus. Obesity, hiatal hernia, pregnancy, smoking and large late meals contribute.",
        "diagnosis": "GERD is usually diagnosed from typical symptoms and response to acid-suppressing treatment. Upper endoscopy and esophageal pH monitoring are used when symptoms are atypical or severe.",
        "treatment": "GERD is treated with lifestyle changes, antacids, H2 blockers and proton pump inhibitors such as omeprazole. Surgery such as fundoplication is an option for refractory cases.",
        "prevention": "GERD symptoms can be prevented by losing excess weight, eating smaller meals, avoiding lying down within 3 hours of eating, raising the head of the bed and limiting trigger foods.",
    }),
    ("kidney stones", "renal calculi", {
        "symptoms": "Kidney stones cause severe sharp pain in the side and back that may spread to the lower abdomen and groin, blood in the urine, nausea, vomiting and frequent painful urination.",
        "causes": "Kidney stones form when urine contains more crystal-forming substances such as calcium oxalate or uric acid than the fluid can dilute. Dehydration, diet and some medical conditions contribute.",
        "diagnosis": "Kidney stones are diagnosed with a non-contrast CT scan or ultrasound, together with urine tests for blood and infection and blood tests for kidney function and calcium.",
        "treatment": "Small kidney stones often pass with fluids, pain relief and tamsulosin. Larger stones may need shock wave lithotripsy, ureteroscopy or percutaneous nephrolithotomy.",
        "prevention": "Kidney stones can be prevented by drinking enough water to produce about 2.5 liters of urine a day, reducing salt and animal protein, and getting normal amounts of dietary calcium.",
    }),
    ("urinary tract infection", "bladder infection", {
        "symptoms": "A urinary tract infection causes a burning sensation when urinating, a frequent urge to urinate, cloudy or strong-smelling urine and pelvic pain. Fever and back pain suggest a kidney infection.",
        "causes": "Urinary tract infections are usually caused by bacteria, most often Escherichia coli from the bowel, entering the urethra. Female anatomy, sexual activity and urinary catheters raise the risk.",
        "diagnosis": "A urinary tract infection is diagnosed with a urine dipstick test for nitrites and leukocytes and confirmed by a urine culture that identifies the bacteria.",
        "treatment": "Uncomplicated urinary tract infections are treated with a short course of antibiotics such as nitrofurantoin or trimethoprim-sulfamethoxazole, plus fluids.",
        "prevention": "Urinary tract infections can be prevented by drinking plenty of fluids, urinating after intercourse, wiping front to back and avoiding irritating feminine products.",
    }),
    ("gout", "uric acid arthritis", {
        "symptoms": "Gout causes sudden attacks of intense pain, redness, warmth and swelling in a joint, most often the base of the big toe, frequently starting at night.",
        "causes": "Gout is caused by high levels of uric acid in the blood that form urate crystals in joints. Red meat, seafood, alcohol, sugary drinks, obesity and diuretics increase the risk.",
        "diagnosis": "Gout is confirmed by finding urate crystals in joint fluid drawn with a needle. Serum uric acid levels and dual-energy CT or ultrasound support the diagnosis.",
        "treatment": "Gout flares are treated with NSAIDs, colchicine or corticosteroids. Long-term urate-lowering therapy with allopurinol or febuxostat prevents further attacks.",
        "prevention": "Gout attacks can be prevented by limiting alcohol, red meat and sugary drinks, staying hydrated, losing weight and taking urate-lowering medication when prescribed.",
    }),
    ("psoriasis", "scaly skin plaques", {
        "symptoms": "Psoriasis causes thick red patches of skin covered with silvery scales, commonly on the elbows, knees, scalp and lower back, which may itch or crack. Nail pitting is common.",
        "causes": "Psoriasis is a chronic immune-mediated disease that speeds up the skin cell life cycle. Genetics plays a role and flares can be triggered by stress, infections, skin injury and some medications.",
        "diagnosis": "Psoriasis is usually diagnosed by examining the skin, scalp and nails. A skin biopsy is occasionally done to rule out other conditions.",
        "treatment": "Psoriasis is treated with topical corticosteroids and vitamin D analogues, phototherapy, and for moderate to severe disease, methotrexate or biologic drugs targeting IL-17 or IL-23.",
        "prevention": "Psoriasis flares can be reduced by moisturizing the skin, managing stress, avoiding skin injury, limiting alcohol and not smoking.",
    }),
    ("major depressive disorder", "clinical depression", {
        "symptoms": "Major depressive disorder causes persistent sadness or loss of interest for at least two weeks, with changes in sleep and appetite, low energy, poor concentration, feelings of worthlessness and sometimes thoughts of death.",
        "causes": "Major depressive disorder arises from a combination of genetic vulnerability, brain chemistry, stressful life events, trauma, chronic illness and certain medications.",
        "diagnosis": "Major depressive disorder is diagnosed through a clinical interview using DSM-5 criteria. Questionnaires such as the PHQ-9 help screen and measure severity, and blood tests rule out thyroid disease.",
        "treatment": "Major depressive disorder is treated with psychotherapy such as cognitive behavioral therapy, antidepressants such as SSRIs, or both. Severe cases may need hospital care or electroconvulsive therapy.",
        "prevention": "The risk of depression relapse can be reduced by continuing treatment as advised, regular exercise, good sleep, social support and early help when symptoms return.",
    }),
    ("generalized anxiety disorder", "constant worrying", {
        "symptoms": "Generalized anxiety disorder causes excessive, hard-to-control worry on most days for at least six months, with restlessness, fatigue, irritability, muscle tension and trouble sleeping.",
        "causes": "Generalized anxiety disorder is linked to genetics, differences in brain chemistry, personality traits and stressful or traumatic experiences.",
        "diagnosis": "Generalized anxiety disorder is diagnosed by a clinician using DSM-5 criteria, often supported by the GAD-7 questionnaire, after excluding medical causes such as hyperthyroidism.",
        "treatment": "Generalized anxiety disorder is treated with cognitive behavioral therapy and medications such as SSRIs or SNRIs. Relaxation techniques and exercise also help.",
        "prevention": "Anxiety symptoms can be reduced by regular physical activity, limiting caffeine and alcohol, good sleep habits, stress management and seeking help early.",
    }),
    ("coronary artery disease", "clogged heart arteries", {
        "symptoms": "Coronary artery disease can cause chest pain or pressure (angina) during exertion, shortness of breath and fatigue. The first sign may be a heart attack.",
        "causes": "Coronary artery disease is caused by atherosclerosis, the buildup of cholesterol plaque in the heart's arteries. Smoking, high LDL cholesterol, hypertension and diabetes are major risk factors.",
        "diagnosis": "Coronary artery disease is diagnosed using an electrocardiogram, stress testing, coronary CT angiography and invasive coronary angiography.",
        "treatment": "Coronary artery disease is treated with lifestyle changes, statins, aspirin, beta blockers and nitrates, and with angioplasty and stenting or bypass surgery when arteries are severely narrowed.",
        "prevention": "Coronary artery disease can be prevented by not smoking, controlling blood pressure and cholesterol, managing diabetes, exercising regularly and eating a Mediterranean-style diet.",
    }),
    ("stroke", "brain attack", {
        "symptoms": "Stroke causes sudden face drooping, arm or leg weakness on one side, slurred speech, confusion, vision loss or a severe headache. Remember FAST: face, arms, speech, time to call emergency services.",
        "causes": "Most strokes are ischemic, caused by a clot blocking blood flow to the brain. Hemorrhagic strokes are caused by bleeding from a ruptured blood vessel. Hypertension and atrial fibrillation are major risk factors.",
        "diagnosis": "Stroke is diagnosed urgently with a CT or MRI scan of the brain to distinguish a clot from bleeding, along with blood tests and an ECG to look for atrial fibrillation.",
        "treatment": "Ischemic stroke is treated with clot-busting thrombolysis within 4.5 hours or mechanical thrombectomy. Hemorrhagic stroke management focuses on controlling bleeding and blood pressure.",
        "prevention": "Stroke can be prevented by controlling blood pressure, treating atrial fibrillation with anticoagulants, not smoking, managing cholesterol and diabetes and staying active.",
    }),
    ("osteoporosis", "brittle bones", {
        "symptoms": "Osteoporosis usually has no symptoms until a bone breaks. Signs can include back pain from a collapsed vertebra, loss of height, a stooped posture and fractures from minor falls.",
        "causes": "Osteoporosis occurs when bone loss outpaces new bone formation. Menopause, aging, low calcium and vitamin D, steroid use, smoking and low body weight increase the risk.",
        "diagnosis": "Osteoporosis is diagnosed with a DXA bone density scan; a T-score of -2.5 or lower confirms it. The FRAX tool estimates 10-year fracture risk.",
        "treatment": "Osteoporosis is treated with bisphosphonates such as alendronate, denosumab or bone-building drugs like teriparatide, plus calcium, vitamin D and fall prevention.",
        "prevention": "Osteoporosis can be prevented with weight-bearing exercise, adequate calcium and vitamin D, avoiding smoking and excess alcohol, and screening women over 65.",
    }),
    ("celiac disease", "gluten intolerance", {
        "symptoms": "Celiac disease can cause diarrhea, bloating, abdominal pain, weight loss and fatigue, as well as non-digestive problems such as anemia, an itchy blistering rash and bone loss.",
        "causes": "Celiac disease is an autoimmune reaction to gluten, a protein in wheat, barley and rye, that damages the lining of the small intestine in genetically susceptible people with HLA-DQ2 or DQ8.",
        "diagnosis": "Celiac disease is diagnosed with a tissue transglutaminase IgA blood test while still eating gluten, followed by a small bowel biopsy showing villous atrophy.",
        "treatment": "The only treatment for celiac disease is a strict lifelong gluten-free diet, which allows the intestine to heal. Nutritional deficiencies such as iron and vitamin D are corrected.",
        "prevention": "Celiac disease cannot be prevented, but complications are prevented by strict gluten avoidance, and screening of first-degree relatives allows early diagnosis.",
    }),
    ("chickenpox", "varicella", {
        "symptoms": "Chickenpox causes an itchy rash of red spots that become fluid-filled blisters and then crust over, along with fever, tiredness and loss of appetite.",
        "causes": "Chickenpox is caused by the varicella-zoster virus, which spreads easily through the air and by contact with blister fluid. The same virus can later reactivate as shingles.",
        "diagnosis": "Chickenpox is usually diagnosed from the characteristic rash. PCR testing of blister fluid can confirm the virus in unclear cases.",
        "treatment": "Chickenpox is treated with rest, fluids, calamine lotion and antihistamines for itching, and paracetamol for fever. Aciclovir is used for adults and high-risk patients. Aspirin should be avoided in children.",
        "prevention": "Chickenpox is prevented with two doses of the varicella vaccine. People who are not immune should avoid contact with infected individuals.",
    }),
]

TEMPLATES: dict[str, list[str]] = {
    "symptoms": [
        "What are the symptoms of {name}?",
        "What are the signs of {name}?",
        "How do I know if I have {lay}?",
        "Which symptoms does {name} cause?",
        "What does {lay} feel like?",
    ],
    "causes": [
        "What causes {name}?",
        "Why do people get {name}?",
        "What is the cause of {lay}?",
        "What are the risk factors for {name}?",
        "How does someone develop {lay}?",
    ],
    "diagnosis": [
        "How is {name} diagnosed?",
        "What tests are used to diagnose {name}?",
        "How do doctors check for {lay}?",
        "How can {name} be confirmed?",
        "Which test shows if I have {lay}?",
    ],
    "treatment": [
        "How is {name} treated?",
        "What are the treatments for {name}?",
        "What is the best treatment for {lay}?",
        "Which medications are used for {name}?",
        "How can {lay} be managed?",
    ],
    "prevention": [
        "How can {name} be prevented?",
        "How do I prevent {name}?",
        "Is there a way to avoid {lay}?",
        "What reduces the risk of {name}?",
        "Can {lay} be prevented?",
    ],
    "overview": [
        "What is {name}?",
        "Can you give me an overview of {name}?",
        "Tell me about {lay}.",
        "What should I know about {name}?",
        "Explain {lay} in detail.",
    ],
}

LONG_EXTRA = (
    "Living with {name} usually involves a long-term partnership with a healthcare team. Regular follow-up "
    "visits allow the treatment plan to be adjusted, side effects to be identified early and complications to be "
    "screened for. Patients are encouraged to keep a written record of their symptoms, medications and questions "
    "for each appointment, and to ask about reliable sources of education and local support groups. Family members "
    "can also benefit from learning about the condition. Any sudden worsening of symptoms should prompt urgent "
    "medical review rather than waiting for the next scheduled appointment."
)

BOILERPLATE = [
    "Click here to subscribe to our newsletter for more health tips.",
    "Advertisement",
    "© 2019 HealthInfoSite. All rights reserved.",
    "Share this article on Facebook.",
    "Read more »",
]
JUNK_ROWS = [("test", "test"), ("N/A", "N/A"), ("???", "asdf"), ("question", "answer"), ("lorem ipsum", "lorem ipsum")]
SHORT_ANSWERS = ["See a doctor.", "Yes.", "It depends.", "Ask your GP.", "Varies."]


def build_facts() -> list[dict]:
    facts = []
    for name, lay, aspects in KB:
        for aspect, answer in aspects.items():
            facts.append({"name": name, "lay": lay, "aspect": aspect, "answer": answer})
    # Long "overview" records (> CHUNK_SIZE characters) for the first 6 conditions to exercise chunking.
    for name, lay, aspects in KB[:6]:
        body = " ".join(aspects[a] for a in ("causes", "symptoms", "diagnosis", "treatment", "prevention"))
        answer = f"{name[0].upper() + name[1:]} is a common medical condition. {body} {LONG_EXTRA.format(name=name)}"
        facts.append({"name": name, "lay": lay, "aspect": "overview", "answer": answer})
    return facts


def render(template: str, fact: dict) -> str:
    q = template.format(name=fact["name"], lay=fact["lay"])
    return q[0].upper() + q[1:]


# --- noise functions (train only) ----------------------------------------------
def add_html(text: str, rng: random.Random) -> str:
    style = rng.choice(["p", "br", "entity", "bold"])
    if style == "p":
        return f"<p>{text}</p>"
    if style == "br":
        return text.replace(". ", ".<br/>", 1)
    if style == "entity":
        return text.replace(" and ", " &amp; ", 1).replace("'", "&#39;")
    words = text.split(" ")
    i = rng.randrange(len(words))
    words[i] = f"<b>{words[i]}</b>"
    return " ".join(words)


def add_unicode_noise(text: str, rng: random.Random) -> str:
    style = rng.choice(["nbsp", "zwsp", "quotes", "fullwidth"])
    if style == "nbsp":
        return text.replace(" ", " ", 3)
    if style == "zwsp":
        return text.replace(" ", " ​", 2)
    if style == "quotes":
        return '“' + text + '”'
    return text.replace("A", "Ａ").replace("a", "ａ", 2)  # full-width letters


def add_whitespace_noise(text: str, rng: random.Random) -> str:
    return rng.choice(["  ", "\t", "\n\n"]) + text.replace(" ", "   ", 2) + rng.choice(["   ", "\n", " \t"])


def make_noisy(answer: str, rng: random.Random) -> str:
    if rng.random() < 0.18:
        answer = add_html(answer, rng)
    if rng.random() < 0.18:
        bp = rng.choice(BOILERPLATE)
        answer = f"{bp} {answer}" if rng.random() < 0.5 else f"{answer} {bp}"
    if rng.random() < 0.10:
        answer = add_unicode_noise(answer, rng)
    if rng.random() < 0.15:
        answer = add_whitespace_noise(answer, rng)
    return answer


def generate(seed: int) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    facts = build_facts()
    train, validate, test = [], [], []

    for i, fact in enumerate(facts):
        templates = TEMPLATES[fact["aspect"]][:]
        rng.shuffle(templates)
        fact["train_templates"] = templates[2:]  # eval phrasings must never leak into train
        base = {"source": rng.choice(["SyntheticMedQA-A", "SyntheticMedQA-B"]), "focus_area": fact["name"], "qtype": fact["aspect"]}
        test.append({**base, "Question": render(templates[0], fact), "Answer": fact["answer"]})
        validate.append({**base, "Question": render(templates[1], fact), "Answer": fact["answer"]})
        for t in templates[2:]:
            train.append({**base, "Question": render(t, fact), "Answer": fact["answer"], "_fact": i})

    # ---- inject noise into train -------------------------------------------------
    clean_train = train[:]
    noisy: list[dict] = []
    for row in clean_train:
        row = {k: v for k, v in row.items() if k != "_fact"}
        row["Answer"] = make_noisy(row["Answer"], rng)
        if rng.random() < 0.08:
            row["Question"] = add_whitespace_noise(row["Question"], rng)
        noisy.append(row)

    extra: list[dict] = []
    # Exact duplicates (scraped twice): ~12% of rows, some 3 copies.
    for row in rng.sample(noisy, k=int(0.12 * len(noisy))):
        for _ in range(rng.choice([1, 1, 2])):
            extra.append(dict(row))
    # Normalized duplicates: same QA, different formatting.
    for row in rng.sample(noisy, k=int(0.08 * len(noisy))):
        dup = dict(row)
        dup["Question"] = rng.choice([dup["Question"].lower(), dup["Question"].upper(), dup["Question"].rstrip("?") + " ?"])
        dup["Answer"] = add_html(dup["Answer"], rng) if rng.random() < 0.5 else add_whitespace_noise(dup["Answer"], rng)
        extra.append(dup)
    # Empty-answer rows whose question is a real paraphrase (classic scraping failure) - strong distractors.
    for row in rng.sample(clean_train, k=int(0.06 * len(clean_train))):
        fact = facts[row["_fact"]]
        q = render(rng.choice(fact["train_templates"]), fact)
        extra.append({"Question": q, "Answer": rng.choice(["", "   ", None, "<p></p>", "&nbsp;"]),
                      "source": row["source"], "focus_area": row["focus_area"], "qtype": row["qtype"]})
    # Null questions with an answer.
    for row in rng.sample(clean_train, k=int(0.02 * len(clean_train))):
        extra.append({"Question": rng.choice([None, "", "  "]), "Answer": row["Answer"], "source": row["source"],
                      "focus_area": row["focus_area"], "qtype": row["qtype"]})
    # Too-short answers to real questions.
    for row in rng.sample(clean_train, k=int(0.03 * len(clean_train))):
        extra.append({"Question": row["Question"], "Answer": rng.choice(SHORT_ANSWERS), "source": row["source"],
                      "focus_area": row["focus_area"], "qtype": row["qtype"]})
    # Pure junk rows.
    for q, a in JUNK_ROWS:
        extra.append({"Question": q, "Answer": a, "source": "unknown", "focus_area": "", "qtype": ""})
    # Conflicting answers: same question, a different (outdated/partial) answer. Kept + flagged by the cleaner.
    for row in rng.sample(clean_train, k=4):
        first_sentence = row["Answer"].split(". ")[0] + ". Consult a healthcare provider for more information."
        extra.append({"Question": row["Question"], "Answer": first_sentence, "source": row["source"],
                      "focus_area": row["focus_area"], "qtype": row["qtype"]})

    train_rows = noisy + extra
    rng.shuffle(train_rows)

    # ---- planted overlaps / unusable rows in eval splits --------------------------
    leak_src = rng.sample(clean_train, k=6)
    for row in leak_src[:3]:  # exact overlap
        test.append({"Question": row["Question"], "Answer": row["Answer"], "source": row["source"],
                     "focus_area": row["focus_area"], "qtype": row["qtype"]})
    for row in leak_src[3:]:  # normalized overlap
        test.append({"Question": "  " + row["Question"].upper() + " ", "Answer": row["Answer"], "source": row["source"],
                     "focus_area": row["focus_area"], "qtype": row["qtype"]})
    validate.append({"Question": leak_src[0]["Question"], "Answer": leak_src[0]["Answer"], "source": "SyntheticMedQA-A",
                     "focus_area": leak_src[0]["focus_area"], "qtype": leak_src[0]["qtype"]})
    test.append({"Question": "What are the symptoms of measles?", "Answer": "Measles causes high fever, cough, runny nose, red watery eyes and a spreading rash.",
                 "source": "SyntheticMedQA-B", "focus_area": "measles", "qtype": "symptoms"})  # answer not in corpus
    test.append({"Question": "How is asthma treated?", "Answer": "", "source": "SyntheticMedQA-B",
                 "focus_area": "asthma", "qtype": "treatment"})  # unusable eval row
    rng.shuffle(test)
    rng.shuffle(validate)

    cols = ["Question", "Answer", "source", "focus_area", "qtype"]
    return {
        "train": pd.DataFrame(train_rows)[cols],
        "validate": pd.DataFrame(validate)[cols],
        "test": pd.DataFrame(test)[cols],
    }


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--out-dir", type=Path, default=cfg.raw_data_dir)
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in (args.out_dir / f for f in ("train.csv", "validate.csv", "test.csv")) if p.exists()]
    if existing and not args.force:
        print(f"Refusing to overwrite existing data: {[str(p) for p in existing]} (use --force)")
        sys.exit(1)

    for split, df in generate(args.seed).items():
        df.to_csv(args.out_dir / f"{split}.csv", index=False)
        print(f"{split:9s}: {len(df):4d} rows -> {args.out_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
