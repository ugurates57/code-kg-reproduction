# CoDe-KG: Speaker Notes

All slide content and notes are in English. Main talk: 9 minutes 40 seconds, including two code switches. Slides 9–11 are hidden backups.

## Before presenting

Open the deck and editor side by side or in separate windows. Set the editor font to at least 20–22 pt. Pre-open `CoDe-KG/run_pipeline.py`, `CoDe-KG/llm_runtime.py`, `CoDe-KG/coref_stage.py` and the companion `walkthrough_evidence.json`. Use saved evidence. Do not start API inference or training during the talk. File paths below are relative to your repository root.

## Slide 1: CoDe-KG

**Time: 0:00–0:25**

Opening (25 seconds)
“I reproduced the CoDe-KG processing pipeline under a laptop and API budget constraint. I trained the sentence classifier locally, connected the generative stages to the university gateway, and inspected a fixed sample of biomedical abstracts. I will show the method, two short code walkthroughs, and the main results.”
The paper is Automated Knowledge Graph Construction using Large Language Models and Sentence Complexity Modelling, Anuyah et al., EMNLP 2025. Describe this work as an adapted partial reproduction. Do not imply that the full benchmark study or original model selection has been repeated.
Transition: “First, what does the pipeline actually do?”
Sources: https://aclanthology.org/2025.emnlp-main.783/; https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025

## Slide 2: The CoDe-KG method

**Time: 0:25–1:15**

Explain the stages in plain language (50 seconds).
“First, the model replaces references and abbreviations with explicit names. The code then splits the text into sentences. BERT classifies each sentence into one of five grammatical categories. Simple sentences go directly to extraction. Compound, complex and compound-complex sentences go through a class-specific simplification prompt. The pipeline drops incomplete sentences. Finally, the LLM extracts subject–relation–object triples.”
BERT is an encoder classifier, not the text-generating model. It supplies the routing decision. The same generative model currently handles all LLM roles, but the prompts differ by stage and sentence class.
Use the Anna example only as an intuition. A sentence can produce several triples. A syntactically simple sentence does not necessarily describe only one fact.
Transition: “I kept this stage structure, but adapted the execution.”
Sources: paper Sections 4.2–4.5; CoDe-KG/run_pipeline.py, lines 135–220; https://aclanthology.org/2025.emnlp-main.783.pdf

## Slide 3: Reproduction scope and adaptations

**Time: 1:15–2:10**

Explain scope (55 seconds).
“The public implementation uses Mixtral for coreference and relation extraction, Llama 3.1 8B for compound sentences and Llama 3.3 70B for complex sentences. My adapter maps every generative role to the available Llama 3.1 8B endpoint. That preserves the stage structure but changes the model configuration.”
“My M1 Pro has 16 GB of unified memory. BERT-Large training and inference run locally on MPS. Generative inference runs through the university API.”
Also acknowledge: API chat-message serialization differs from local generation, top_k and repetition_penalty are not transmitted, API batches execute sequentially, and relation extraction now allows 2,048 output tokens instead of 512. The first two imported document results lack complete historical configuration metadata. Those differences limit numerical comparability.
We did not repeat the paper’s prompt/model search or its relation-extraction benchmark evaluation. Pipeline completion and output correctness are separate questions.
Transition: “Here is where those decisions appear in the code.”
Sources: public run_pipeline.py; supplied llm_runtime.py and provenance.json; https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025

## Slide 4: Code walkthrough 1: routing and inference

**Time: 2:10–4:05**

CODE SWITCH 1 (1 minute 55 seconds, including navigation).
Have the editor open before starting. Paths below are relative to the repository root. Line numbers refer to the supplied snapshot; use the search anchors if they move.
1. Open CoDe-KG/run_pipeline.py at line 135. Show tokens_with_indices, build_coref_prompt, parse_coref_output and apply_coref_switch, through line 151. Explain: the LLM proposes token-span replacements and Python applies them.
2. Jump to line 172, bert.predict_batch. Show the class lists at lines 181–184, routing at 195–201 and final_simple_sentences at 207. Simple sentences bypass simplification. Incomplete sentences enter none of these four lists and are excluded.
3. Show extract_relationships at line 220. It receives the original simple sentences plus generated simple sentences.
4. Open CoDe-KG/llm_runtime.py at lines 53–62. Point to self.model_name = os.environ['LLM_MODEL']. Names such as mixtral_llm are role labels; they do not mean Mixtral actually runs.
5. Briefly show the request and cache lookup at lines 75–87, then atomic_json at line 122. Successful API calls persist individually. This supports resuming interrupted runs; it does not make a fresh sampled API response deterministic.
Do not read retry code line by line. Mention bounded retries and error logging in one sentence. The engineering supports the experiment, while classification and decomposition implement the paper’s method.
Fallback if time is tight: show run_pipeline.py only and explain the model override verbally.
RETURN TO SLIDE 5 by 4:05.
Sources: supplied run_pipeline.py and llm_runtime.py.

## Slide 5: BERT-Large evaluation

**Time: 4:05–5:30**

Present the quantitative result (1 minute 25 seconds).
“I fine-tuned BERT-Large for five sentence types. Starting from 2,000 training rows, I excluded both occurrences of one conflicting sentence and removed four further duplicate rows. That left 1,994 examples: 1,594 for training and 400 for validation.”
“Training used three epochs, a learning rate of 2e-5, a maximum length of 256 tokens, batch size one and eight-step gradient accumulation. The local seed was 4000.”
“For the final test, I removed 17 rows overlapping the used training or validation sentence text and 29 further duplicate rows. This left 5,223 test sentences, with no matching PMIDs. Accuracy was 88.72 percent and macro-F1 was 87.60 percent.”
Explain macro-F1: compute F1 for each class and give all five classes equal weight. It matters because the test classes are uneven. No test sentence exceeded the 256-token limit. Validation accuracy was 90.25 percent. These figures measure classification only, not final triple quality.
Do not claim we outperform the paper: training split and preprocessing differ. The paper’s BERT-Large accuracy is 87.68 percent under its reported setup, which is not a controlled comparison with ours.
Optional code anchor if asked: CoDe-KG/scripts/train_bert.py line 87 for conflict removal, line 139 for TrainingArguments. Do not switch here during the timed talk.
Sources: outputs/bert_training/experiment.json and validation_results.json; outputs/bert_test/metrics.json and per_class_metrics.csv; scripts/train_bert.py and check_bert_test_overlap.py. Paper Table 3 for the paper number, if discussed.

## Slide 6: Exploratory end-to-end results

**Time: 5:30–6:35**

Explain the small experiment (1 minute 5 seconds).
“I fixed five abstracts before inspecting model outputs, sampled with seed 4000 from the 190-document coreference dataset. Across the retained attempts, four documents completed. Document 394 failed because its coreference response reached the output-token limit.”
“The completed documents contain 184 triple rows. Exact within-document deduplication leaves 150 different triples, so 34 rows repeat an existing triple. These counts measure output volume and repetition, not correctness.”
“Some documents required retries and configuration changes. Documents 168 and 313 are imported legacy results, with incomplete historical token-limit metadata. Therefore four out of five is eventual coverage across attempts, not the success probability of one fixed run.”
The available gold file annotates coreference, not a complete set of gold triples. We have inspected errors but have not completed a human annotation of every triple. Do not report an end-to-end precision, recall or F1 value.
Transition: “One trace shows why technical completion is not sufficient.”
Sources: outputs/pipeline_sample_5_consolidated/triples.csv, triples.trace.jsonl, triples.failures.jsonl and provenance.json; data/evaluation/pipeline_sample_5_gold.json.

## Slide 7: A trace exposes error propagation

**Time: 6:35–8:35**

CODE SWITCH 2 (2 minutes, including slide explanation).
Start on this slide for about 30 seconds. The original sentence in document 313 says: “During this period, the ASIR of lung cancer in females was always lower than that in males.” The generated simplification instead says: “Men in China had a lower ASIR of lung cancer than females in China.” The comparison reverses.
Open the prepared walkthrough_evidence.json file from the companion folder. It contains short, exact excerpts from the uploaded traces, so there is no need to search a long single-line JSONL record live. Show doc_313.original_comparison, doc_313.resolved_sentence, doc_313.simplified_sentence and doc_313.output_triples. The resolved sentence already contains a misplaced “men in China” and a merged sentence boundary. Simplification then produces the reversed comparison, and extraction preserves the wrong claim.
Next show doc_327.span_mismatch: the model says Expression='it', StartToken=40, EndToken=40, while the actual token is 'show'.
Open CoDe-KG/coref_stage.py at line 51, apply_coref_switch. Point to the replacements loop at line 62. It uses the returned start/end indices without checking that the source span matches Expression. Incorrect model indices therefore overwrite unrelated words. This behavior exists in the supplied public implementation, rather than being introduced by the cache.
Say: “This example suggests two contributors: unreliable model span predictions and an unchecked replacement step. It does not prove which model change caused the full quality gap.”
If time is tight, show only document 313 and explain the guard verbally. Return to slide 8 by 8:35.
Sources: supplied document 313 and 327 traces; CoDe-KG/coref_stage.py lines 51–88.

## Slide 8: What this reproduction establishes

**Time: 8:35–9:40**

Close with a bounded conclusion (1 minute 5 seconds).
“I reproduced the four-stage processing architecture in an adapted setting. I also trained and independently tested the BERT classifier. The saved traces make the model behavior inspectable.”
“I did not reproduce the paper’s selected model combination, its prompt/model search or its relation-extraction benchmark scores. The five-document study is exploratory and combines retained outputs across attempts.”
“The strongest observed weakness is coreference: wrong replacements corrupt the text before later stages see it. The paper itself identifies coreference as a limitation and reports missing, invented and mismatched relations.”
“The next controlled experiment would validate predicted spans before replacement and compare the same fixed sample with and without that validation. I would keep the baseline outputs and label this as an extension. I have not run that experiment.”
Stop around 9:40 to leave a short transition or question buffer. If the ten minutes include discussion, shorten the first code switch by 20 seconds and this conclusion by 20 seconds.
Sources: paper Section 6.1 and Limitations; uploaded code and saved results; https://aclanthology.org/2025.emnlp-main.783.pdf

## Slide 9: Backup: classifier details

**Time: Q&A only**

Use only for questions about training or class differences. Hidden during the main slide show.
Per-class F1 and support are copied from outputs/bert_test/per_class_metrics.csv. Complex sentences have precision 70.03 percent and recall 92.47 percent, so the classifier predicts this class too broadly. The confusion matrix shows 129 actual compound-complex examples predicted as complex and 185 as compound.
Training: BERT-Large uncased, 3 epochs, learning rate 2e-5, max length 256, microbatch 1, gradient accumulation 8, eager attention, gradient checkpointing, seed 4000. Use MPS locally.
The classification head starts randomly initialized and is learned during fine-tuning. The warning about newly initialized classifier weights during model loading was expected before training.
Do not retrain during the seminar.
Sources: per_class_metrics.csv, confusion_matrix.csv, training_arguments.json.

## Slide 10: Backup: coreference evaluation

**Time: Q&A only**

This is a strict diagnostic, not the paper’s coreference metric. Hidden during the main slide show.
On the four completed documents, compare unique (StartToken, EndToken, normalized RefersTo) tuples. Normalization case-folds and collapses whitespace. Keep token indices exact. There are 46 unique predicted tuples and 28 gold tuples. Three match exactly, all in document 315. Exact matches divided by gold annotations give 3/28, or 10.71 percent. Dividing by unique predictions gives 3/46, or 6.52 percent. This measure can reject semantically equivalent referring expressions and should not be presented as a standard CoNLL score.
The gold file for failed document 394 also contains two reversed spans and an exact duplicate annotation. This was outside the four-document comparison. Errors in that gold file are an additional data-quality finding, not a reason to ignore our observed output errors.
The paper uses MUC, B3, CEAF and CoNLL scores and discusses cosine-similarity matching. Table 2 reports CoNLL 57.88 for Mixtral and 55.43 for Llama 3.1 8B. These scores are not directly comparable to our exact annotation count. The modest reported gap also means we cannot attribute all our failures to using a smaller model.
Sources: data/evaluation/pipeline_sample_5_gold.json; retained traces; paper Table 2 and Section 5.1; https://aclanthology.org/2025.emnlp-main.783.pdf

## Slide 11: Backup: evidence and repository map

**Time: Q&A only**

Sources and fallback navigation. Hidden during the main slide show.
Paper: Automated Knowledge Graph Construction using Large Language Models and Sentence Complexity Modelling, Anuyah et al., EMNLP 2025. https://aclanthology.org/2025.emnlp-main.783/
Public implementation: https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025
Local source of truth: uploaded CoDe-KG.zip, outputs.zip, data.zip and tests.zip, supplied 11 November 2026.
Main entry point: CoDe-KG/run_pipeline.py. Sentence classifier: bert_sentence_classifier.py. Prompt definitions: prompts.py. Per-stage parsing: coref_stage.py, simplify_stage.py and relation_extract_stage.py. API transport/cache: llm_runtime.py. Resume and CSV reconstruction: pipeline_resume.py. Training and evaluation: CoDe-KG/scripts/.
Do not open .env, private credentials or the API cache while sharing the screen. Use the prepared evidence file for the demo. Keep terminal/API execution out of the timed presentation.

## Likely questions and short answers

**Did you reproduce the paper?**
I reproduced its pipeline architecture with an adapted inference setup and evaluated the sentence classifier. I did not reproduce the full original experiment or establish the published end-to-end scores.

**Why use the API?**
My laptop can fine-tune BERT-Large with small batches, but I used the gateway for the larger generative models. The available endpoint also enabled a low-cost exploratory run.

**Why not just extract triples directly?**
The paper hypothesizes that resolving references and decomposing complex sentences improves extraction. We reproduced that structure. We have not yet run a direct-extraction ablation on the same sample, so we cannot establish that benefit for our configuration.

**Does BERT make embeddings or classify sentences?**
Both are compatible. BERT encodes the input into contextual representations. A trained classification head maps that representation to five sentence classes. The pipeline uses the predicted class to choose the next step.

**Are the 184 triples correct?**
No. That is a raw output count. There are exact duplicates and documented semantic errors. A complete human review is still required before reporting extraction precision.

**Why did the model overwrite the wrong words?**
It returned incorrect token positions. The replacement function trusts those positions without verifying the Expression against the source span. We traced one example where token 40 is “show” but the model calls it “it”.

**Would a larger model fix this?**
It might help, but we have not measured that. The paper itself reports coreference weaknesses. A stronger model also does not replace validation of structured output.

**Does setting seed 4000 make API output reproducible?**
No. It controls local randomness and sample selection. The adapter does not send a generation seed. Cached successful answers support exact replay of retained calls, which is different from reproducing an independent sampled run.

**What did caching change scientifically?**
It changes execution and replay. It does not repair wrong answers. Cached requests include the actual model, gateway, messages and transmitted sampling parameters in their identity.

**What remains to be done?**
A complete manual triple review, a controlled same-configuration run, an ablation for span validation or coreference, and the original benchmark evaluation if resources permit.

## Rehearsal plan

1. First pass: follow the notes, open every named function and locate each exact trace excerpt. Allow about 45 minutes.
2. Second pass: rehearse with a ten-minute timer. Practise the two switches without scrolling through unrelated code.
3. Final pass: speak from slide headings. Keep the appendix closed unless someone asks. Stop the main talk by 9:40.
