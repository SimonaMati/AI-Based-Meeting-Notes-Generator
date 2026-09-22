# AI-Based-Meeting-Notes-Generator
The AI-based meeting notes generator turns meeting video, audio or text into structured notes - a summary as well as speaker-attributed action items, deadlines, key decisions and a domain label.

Course project for a graduate AI course at Northeastern University (2026) by [Siddhi Kakani](https://github.com/siddhi1703) and [Simona Matiukaite](https://github.com/SimonaMati). Written up as a research paper.

## Results
Two abstractive summarization models were fine-tuned and compared on the 28-sample AMI test split. ROUGE computed with 'rouge_score' using stemming.
| Model | Params | ROUGE-1 R | ROUGE-2 R | ROUGE-L R | ROUGE-1 F1 | Max input |
|---|---|---|---|---|---|---|
| **BART-large-CNN-samsum** | 406.3M | **79.05%** | **45.64%** | **44.44%** | **37.78%** | 1024 |
| Flan-T5-base | 247.6M | 71.22% | 40.06% | 43.32% | 31.99% | 512 |

BART-samsum beat the Flan-T5 baseline on every metric — **+7.83 points on ROUGE-1 recall**. Two likely reasons: it was pretrained on the SAMSum dialogue corpus, which matches the turn-based multi-speaker structure of meetings, and its 1024-token context window needs less aggressive chunking.

Per-sample ROUGE-1 recall ranged from 45.9% to 99.1% across the 28 test meetings, tracking differences in meeting length, speaker count, and discussion complexity.

**Information extraction**, run over the 40-meeting domain collection: 880 speaker-attributed action items, 15 key decisions, 4 explicit deadline references. The keyword-based domain classifier placed all 40 meetings in the correct domain.

**Caveat on scale:** 28 test samples. The gap between models is wide enough to be informative, but this is a course-scale evaluation, not a benchmark result.

## Pipeline
```
video ─► FFmpeg ─┐
                 ├─► resample 16kHz ─► 30s chunks ─► Whisper small ─► transcript ─┐
audio ───────────┘                                                                │
                                                                                   │
text ──────────────────────────────────────────────────────────────────────────────┤
                                                                                   ▼
                                                                             cleaning
                                                                                   │
                     ┌─────────────────────────────┬───────────────────────────────┤
                     ▼                             ▼                               ▼
              summarization                  extraction                     classification
         abstractive (BART / T5)          action items                    term frequency
                    +                      deadlines                   business/edu/health
         extractive (TF-IDF centroid)      decisions
                     │                             │                               │
                     └─────────────────────────────┴───────────────────────────────┘
                                            │
                                            ▼
                                    Gradio web interface
```
**Hybrid summarization** is the core design choice. Transcripts are chunked and summarized abstractively, and in parallel each source sentence is scored by TF-IDF cosine similarity to the document centroid; top-ranked sentences are combined with the generated output. This prioritizes content coverage - hence high recall relative to precision.

**Structure extraction** is rule-based: regex over modal and imperative constructions (`we will`, `please`, `make sure`, `let's`). Action items are linked to speakers using the speaker tags present in the transcript.

## Data
| Source | Use |
|---|---|
| [`edinburghcstr/ami`](https://huggingface.co/datasets/edinburghcstr/ami) (`ihm`) | AMI Meeting Corpus audio and utterances |
| [`knkarthick/AMI`](https://huggingface.co/datasets/knkarthick/AMI) | Human reference summaries |
| [`medalpaca/medical_meadow_medqa`](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) | Source text for healthcare-domain meetings |

The AMI Meeting Corpus is roughly 100 hours of English meeting recordings with aligned transcripts and summary annotations. Records missing a transcript or summary were dropped, leaving **279 samples: 209 train / 42 validation / 28 test**, as input-target pairs.

A separate **40-meeting domain collection** (15 business, 10 education, 15 healthcare) was assembled to test the extraction and classification modules across registers. The healthcare portion is text transcripts rather than recorded meetings — real clinical audio was not available for privacy reasons.

Note the AMI `ihm` split is ~15 GB and takes several minutes to download.

## Models

## Running it

## Evaluation

## Repository contents

## Limitations

## License
