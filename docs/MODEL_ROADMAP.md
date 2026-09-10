# Model Roadmap — Maksimal kuchli log anomaly detection

> Maqsad: hozirgi loyihaning aniqlash modelini **state-of-the-art (SOTA)** darajaga chiqarish.
> Sana: 2026-09-07. Muallif: giyosiddnm@gmail.com uchun tayyorlangan.
> Bu hujjat internetdan (2022–2026 tadqiqotlar) chuqur o'rganib tuzilgan. Havolalar oxirida.

---

## 0. TL;DR (qisqa xulosa)

Hozirgi model **kuchsiz**, chunki:

1. **Har bir log qatorini alohida** ko'radi (point-wise). Haqiqiy anomaliyalar **ketma-ketlikda** (sequence) yashiringan — bitta qator normal ko'rinadi, lekin 50 qatorlik naqsh anomal.
2. **Semantik ma'noni** tashlab yuboradi. `template_id`, `path_length`, `status` kabi 19 ta qo'lda feature log **matnining ma'nosini** yo'qotadi. SOTA modellar log matnini **embedding** (word2vec / FastText / BERT) qiladi.
3. **Aniqlik hech qachon o'lchanmagan.** Precision/Recall/F1 yo'q, labeled benchmark yo'q. "Kuchli" ekanini isbotlab bo'lmaydi.

Kuchli model qilish yo'li = **3 ustun**:

| Ustun | Hozir | Kerak |
|-------|-------|-------|
| **Representation** | 19 ta qo'lda son | Semantic embedding + sequence |
| **Model** | Isolation Forest (point) | Sequence deep model → Transformer → LLM |
| **Evaluation** | Yo'q | Loghub benchmark, temporal split, F1/PR-AUC |

**Nishon ko'rsatkich:** HDFS F1 ≥ 0.98, BGL F1 ≥ 0.90 (SOTA daraja).

---

## 1. SOTA landshaft — internetdan o'rganilgan

Public benchmark (HDFS, BGL, Thunderbird, Liberty) dagi F1 natijalari:

| Model | Yondashuv | HDFS | BGL | O'rtacha | Izoh |
|-------|-----------|------|-----|----------|------|
| **LogLLM** (2024) | BERT (semantic) + Llama (decoder) + projector, QLoRA, 3-bosqich | **0.997** | **0.916** | **0.959** | Hozirgi eng kuchli. Qimmat (~1065 min train). |
| **NeuralLog** (2021) | Parsing YO'Q, pretrained embedding + Transformer | 0.979 | 0.835 | 0.893 | BGL/Thunderbird/Spirit da eng yaxshi klassik DL. |
| **LogRobust** (2019) | Bi-LSTM + attention + FastText, supervised | 0.980 | 0.810 | 0.771 | Unstable loglarda ham barqaror. |
| **LogAnomaly** | Semantic + quantitative LSTM, template2vec | ~0.95 | — | — | Balans yaxshi. |
| **DeepLog** | LSTM, keyingi event'ni bashorat qilish | ~0.96 | — | — | Unsupervised, sequence. Klassika. |
| **LogBERT** | Transformer, self-supervised MLM | SOTA-yaqin | — | — | Kuchli lekin qimmat, ba'zi benchmarkda past. |
| **PCA / SVM / RF** (Loglizer) | Klassik ML, count vector | ~0.9 | past | — | Arzon baseline. |
| **Isolation Forest (BIZDAGI)** | Point-wise, 19 qo'lda feature | **?** (o'lchanmagan) | — | — | Sequence yo'q, semantic yo'q. Kuchsiz. |

**Muhim tendensiyalar (2025–2026):**

- **Parser-free g'olib chiqmoqda.** NeuralLog va LogLLM Drain3 template o'rniga regex bilan o'zgaruvchilarni `<*>` ga almashtiradi. Parsing xatosi aniqlikni buzadi — buni chetlab o'tish barqarorroq.
- **LLM eng yuqori.** Fine-tuned transformer F1 = 0.96–0.99. Prompt-based zero-shot (label kerak emas) F1 = 0.82–0.91. **LoRA/QLoRA** bilan arzonlashtiriladi. Kichik variant: **LogTinyLLM**.
- **Deep learning har doim ham klassik ML'dan ustun emas** (ICSE'24 empirik tadqiqot). Yaxshi feature bilan klassik ML arzonroq va tez — shuning uchun **tiered (bosqichli)** arxitektura mantiqiy.
- **Semi-supervised kuchaymoqda.** To'liq label qimmat. PLELog (probabilistic label estimation), PU-learning, contrastive learning (LogEncoder, ContraLog) label kamligini yengadi.

---

## 2. Eng muhim qoida: EVALUATION AVVAL (aks holda hammasi ko'r ish)

Yangi model qurishdan **oldin** halol baholash tizimi bo'lishi shart. Internetdagi "How Far Are We?" (ICSE 2022) tadqiqotining asosiy saboqlari:

1. **Random split = DATA LEAKAGE.** Loglarni tasodifiy bo'lish F1'ni sun'iy oshiradi (kelajak train'ga sizib kiradi). **Xronologik (temporal) split** ishlat — vaqt bo'yicha, train o'tmish / test kelajak. Halol raqam faqat shunday chiqadi.
2. **Data grouping to'g'ri tanlan:**
   - **Session window** — biror ID bo'yicha guruh (HDFS `block_id`). Eng aniq.
   - **Fixed time window** — masalan 60s, maks 256 qator (BGL uslubi).
   - **Sliding window** — event ID ketma-ketligi ustidan (DeepLog uslubi).
3. **Class imbalance real.** Anomaliya 0.05%–15% oralig'ida. Accuracy yaramaydi — **Precision, Recall, F1, PR-AUC** ishlat. Oversampling / focal loss qo'lla.
4. **Parsing xatosini o'lchash kerak.** Parser noto'g'ri bo'lsa model ham buziladi.
5. **Early detection** — anomaliyani qancha erta topadi, shuni ham o'lchash foydali.

**Deliverable (Faza 1):** `evaluation/` moduli — Loghub datasetni yuklaydi, temporal split qiladi, Precision/Recall/F1/PR-AUC/ROC-AUC chiqaradi, modellarni bir jadvalda solishtiradi.

---

## 3. Data va representation (kuchning 50%'i shu yerda)

### 3.1 Datasetlar (labeled benchmark)
[Loghub](https://github.com/logpai/loghub) dan yuklab olinadi:

| Dataset | Hajm | Anomaliya % | Grouping |
|---------|------|-------------|----------|
| HDFS | 11.2M qator | 2.93% | session (block_id) |
| BGL | 4.7M qator | 7.34% | time window |
| Thunderbird | 10M (subset) | 0.049% | time window |
| Liberty / Spirit | 5M (subset) | 15–32% | time window |

Bulardan tashqari loyihaning o'z `data/sample_logs.log` (web-server) formatida ham label yig'ish kerak (hujum injection qilib sintetik anomaliya yaratish mumkin).

### 3.2 Representation bosqichma-bosqich kuchaytirish
1. **Bosqich A — Count/semantic vector (klassik):** template ketma-ketligini oyna ichida sanash (quantitative) + TF-IDF. Loglizer uslubi. Arzon baseline.
2. **Bosqich B — Semantic embedding:** har log matnini **FastText / word2vec** bilan vektor qilish, **TF-IDF og'irligi** bilan o'rtacha (LogRobust uslubi). Unstable/yangi loglarga chidamli, parsing xatosini yumshatadi.
3. **Bosqich C — Contextual embedding:** **BERT / Sentence-BERT** bilan log matnini embedding qilish (NeuralLog / LogLLM uslubi). Parsing kerak emas — regex bilan o'zgaruvchini `<*>` ga almashtir.
4. **Sequence formation:** embeddinglarni oynaga yig'ib (session yoki fixed 100 qator, step 100) ketma-ketlik hosil qil. Modelga shu ketma-ketlik kiradi.

---

## 4. Model arxitekturasi — TIERED (bosqichli, "juda kuchli")

Bitta modelga tayanmaslik. To'rt qavatli tizim: pastki qavat tez/arzon, yuqori qavat kuchli/qimmat. Real vaqtda arzon qavat filtrlaydi, shubhali holatlar yuqori qavatga uzatiladi.

```
Kiruvchi log oqimi
      |
[Tier 0] Rule + regex (SQLi/traversal/XSS)  --> aniq hujum = darhol alert
      |
[Tier 1] Klassik ML (Isolation Forest / PCA / RF)  --> tez, 10k+/sek
      |  (shubhali bo'lsa yuqoriga)
[Tier 2] Sequence deep model (DeepLog / LogAnomaly / LogRobust)  --> sequence anomaliya
      |
[Tier 3] LLM (LogLLM / fine-tuned BERT, QLoRA)  --> eng aniq, tasdiqlash + tushuntirish
      |
  Anomaliya event + tushuntirish + confidence
```

### Tier tafsiloti

- **Tier 0 — Rule/signature:** hozirgi `suspicious_patterns` shu yerga. `../`, `union select`, `<script>` = deterministik. Feature emas, alohida qavat.
- **Tier 1 — Klassik ML (mavjudni yaxshilash):** Isolation Forest'ni saqlab qol, lekin representation'ni Bosqich A/B ga ko'tar. Adaptive threshold (percentile/MAD/IQR — `utils/helpers.py` da tayyor, ulanmagan) wire qil. Tez filtr sifatida ishlaydi.
- **Tier 2 — Sequence deep learning (asosiy kuch):**
  - **DeepLog** (LSTM, unsupervised, keyingi event bashorati) — label kam bo'lsa.
  - **LogAnomaly** (semantic + quantitative LSTM).
  - **LogRobust** (Bi-LSTM + attention, supervised) — label bo'lsa, unstable loglarga chidamli.
  - PyTorch'da yoziladi (loyihada torch allaqachon bor).
- **Tier 3 — LLM (cho'qqi aniqlik):**
  - Boshlanish: **fine-tuned BERT** klassifikator (sequence embedding → normal/anomal).
  - Cho'qqi: **LogLLM** uslubi — BERT (semantic vektor) + kichik decoder LLM (Llama/Qwen) + projector, **QLoRA** (4-bit) bilan arzon fine-tune. 3-bosqichli train (avval javob shablonini o'rgatish — bu bosqichni tashlash F1'ni 29.7% tushiradi).
  - Alternativa: **prompt-based zero-shot** (label yo'q) — Claude API bilan shubhali sequence'ni tekshirish + inson o'qiydigan tushuntirish (loyihaning "explainable" va'dasi bilan mos).

**Ensemble:** Tier 1 + Tier 2 score'larini birlashtir (config'da `ensemble` bor, kodi yo'q). Bu robustlikni oshiradi.

---

## 5. Production — kuchli modelni jonli ushlab turish

Kuchli model bir marta train qilinib qolmaydi; loglar o'zgaradi (concept drift).

1. **Concept drift detection:** [Evidently AI](https://www.evidentlyai.com/) yoki Alibi Detect bilan input taqsimoti siljishini kuzat. Yangi template paydo bo'lsa ogohlantir.
2. **Incremental / online learning:** DeepLog online yangilanadi. Yoki rejalashtirilgan retraining (MLflow / Airflow) — masalan haftalik.
3. **Feedback loop (false positive kamaytirish):** operator "bu normal edi" desa, shu label bilan qayta o'rgat (semi-supervised, PLELog uslubi). Bu ishonchni oshiradi va yolg'on ogohlantirishni kamaytiradi.
4. **MLOps:** har model versiyasini MLflow'da registr qil (metrika + artefakt). Har deploy'dan oldin benchmark F1 avvalgidan past bo'lmasin (regression guard CI'da).
5. **Serving:** kuchli modellar sekin — Tier 1 bilan filtrlab, faqat shubhalilarni Tier 2/3 ga yubor (yuqoridagi kaskad). GPU ixtiyoriy (config `use_gpu`).

---

## 6. Fazalar bo'yicha reja (har fazada model kuchayadi)

| Faza | Nom | Natija | Model kuchi |
|------|-----|--------|-------------|
| **F1** | Evaluation harness | Loghub yuklash, temporal split, F1/PR-AUC jadval. Hozirgi IF baseline o'lchanadi. | Baseline aniqlanadi |
| **F2** | Representation upgrade | Semantic embedding (FastText+TF-IDF), sequence window. Tier 1 yangilanadi. | +katta sakrash |
| **F3** | Sequence deep model | DeepLog + LogAnomaly + LogRobust PyTorch'da. Benchmark solishtir. | SOTA-yaqin |
| **F4** | Transformer / LLM | Fine-tuned BERT → LogLLM (QLoRA). Explainable tushuntirish. | SOTA (F1 ~0.95+) |
| **F5** | Ensemble + kaskad serving | Tier 0–3 kaskad, ensemble score, real-time. | Kuchli + tez |
| **F6** | Drift + feedback + MLOps | Drift detection, retraining, MLflow registry, CI regression guard. | Jonli barqaror |

**Har fazada MAJBURIY:** benchmark F1 jadvalini yangilab borish. "Kuchli" = raqam bilan isbot.

---

## 7. Xatarlar va ehtiyot choralari

- **Data leakage** — random split ishlatmang, faqat temporal. Aks holda F1 yolg'on chiqadi.
- **Overfitting benchmark'ga** — HDFS'da 0.99 olish oson (juda tuzilgan). BGL/Thunderbird va o'z web-server data'da ham sinang.
- **LLM qimmat** — QLoRA/PEFT ishlating, kichik LLM (LogTinyLLM) dan boshlang. Har logga LLM emas — faqat kaskad cho'qqisi.
- **Label yo'qligi** — semi-supervised / unsupervised (DeepLog) dan boshlang; feedback loop bilan asta label yig'ing.
- **Parsing xatosi** — Drain3'ga to'la ishonmang; parser-free (regex `<*>`) variantni ham sinang.

---

## 8. Manbalar (internetdan o'rganilgan)

- LogLLM — LLM asosidagi eng kuchli model: https://arxiv.org/abs/2411.08561
- "Log-based Anomaly Detection with Deep Learning: How Far Are We?" (ICSE 2022) — baholash saboqlari: https://arxiv.org/pdf/2202.04301 · repo: https://github.com/LogIntelligence/LogADEmpirical
- NeuralLog — parser-free: https://arxiv.org/pdf/2108.01955
- LogRobust (robust/transferable): https://dl.acm.org/doi/pdf/10.1145/3588918
- LLM benchmark (fine-tuned vs prompt F1): https://arxiv.org/html/2604.12218v1
- LogTinyLLM — kichik LLM: https://arxiv.org/pdf/2507.11071
- Log representation effektivligi (embedding/TF-IDF): https://arxiv.org/pdf/2308.08736
- PLELog — semi-supervised probabilistic label: https://xgdsmileboy.github.io/files/paper/plelog-icse21.pdf
- LOGPAI / Loghub datasetlar + Loglizer/LogAI toolkit: https://github.com/logpai
- ML uslublari to'liq survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC12185583/
- Concept drift & retraining (MLOps): https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/

---

## Progress log

- **2026-09-07 — Faza 1 DONE.** Evaluation harness (`evaluation/`, `docs/EVALUATION.md`). Baseline IsolationForest on own web logs: PR-AUC 0.325, ROC 0.95, operating F1 0.356, best-F1 ceiling 0.61. Weak — confirms thesis.
- **2026-09-07 — Faza 2 DONE.** Semantic ensemble (`app/models/semantic_detector.py`, `docs/PHASE2_SEMANTIC.md`): OOV n-gram novelty + IsolationForest(SVD char-TF-IDF) + IsolationForest(numeric), z-fused on train stats. Same benchmark: **PR-AUC 0.325 → 0.691 (2.1×), operating F1 0.356 → 0.617 (+73%)**, reaches 90% precision. Residual misses = brute force + categorical → Faza 3 (sequence).
- **2026-09-10 — Faza 3 DONE.** Sequence tier: DeepLog (`app/models/deeplog.py`, LSTM next-event top-k) + Quantitative head (`app/models/quantitative.py`, count-vector IF) fused (LogAnomaly two-head). `docs/PHASE3_SEQUENCE.md`. Synthetic session benchmark: **fused F1 0.96, recall 0.97 — brute-force gap from Faza 2 CLOSED**. DeepLog: 0 false positives, catches order/novel; Quant catches volume bursts. Markov bigram near-perfect on first-order synthetic (caveat documented) — DeepLog's real edge needs HDFS. Real HDFS pipeline wired + tested (`run_hdfs_deeplog`, `--dataset hdfs --model deeplog`), runs on Loghub HDFS_v1 download. Tests: 26/26 in evaluation suite.
- **2026-09-10 — Faza 4 DONE.** Tier 3. LogTransformer (`app/models/log_transformer.py`, self-attention next-event) + higher-order long-range benchmark (`build_matched_sessions`) where bigram is structurally blind. `docs/PHASE4_TRANSFORMER.md`. Result: on the long-range benchmark **Transformer PR-AUC 0.856 > DeepLog 0.806 > Markov 0.756**; Markov recall ceiling 0.667 (never flags mismatched-close), Transformer → 1.0. This is the deep-vs-bigram demonstration Faza 3 lacked. LLM explainer (`app/models/llm_explainer.py`): zero-shot Claude classify+explain (Tier 3 peak / explainability), lazy SDK import, mock-tested, needs ANTHROPIC_API_KEY live. Full LogLLM (Llama+QLoRA) out of scope (no GPU). Evaluation suite 32/32.
- **2026-09-10 — Real HDFS run.** Full Loghub HDFS_v1 (11.2M lines, 575,061 sessions, Drain3 vocab=47) through the Phase-3 pipeline. `docs/HDFS_BENCHMARK.md`, `reports/hdfs_deeplog.md`. **DeepLog: F1 0.78 @ natural 2.93% (precision 0.92, recall 0.67, FPR 0.0018)**, top_k=15, 10k-session train. Honest gap vs literature 0.96 = recall: ~30% of HDFS anomalies are quantitative (next-event can't see them; DeepLog recall ceiling ~0.80 even flagging 70% of everything), the simple count head is too noisy to fuse (FPR 0.80), and RAM OOM-capped training at 10k sessions (40k killed). Reproduce: `python -m evaluation.run --dataset hdfs --model deeplog --hdfs-log HDFS.log --hdfs-label anomaly_label.csv`.
