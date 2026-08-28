# Customer Sentiment & Review Response Automation

A GenAI case study that turns 3,000 raw product reviews into sentiment scores, customer segments, and a working support-response assistant — built end-to-end in Google Colab as a 3-milestone project.

**Business question:** *What are customers actually feeling, which of them need urgent attention, and can GenAI help a support team respond faster without losing accuracy or consistency?*

## What's in this repo

| Path | What it is |
|---|---|
| `notebooks/pwc_Project.ipynb` | The full, commented Colab notebook — all 3 milestones, code + write-ups, in one file |
| `reports/Review_Insights.pdf` | Milestone 1 report — sentiment analysis, keyword/theme extraction, GenAI insights |
| `reports/Review_Comparison.pdf` | Milestone 2 report — Positive/Neutral/Negative segment comparison and prioritization |
| `reports/Review_App_Snapshot.pdf` | Milestone 3 report — the packaged pipeline, the Gradio app, and real test runs |
| `reports/Reflection.txt` | Written reflection on what the project showed and what it would take to move it to production |

The raw dataset (a SQLite database of 3,000 reviews) is **not** included in this repo — see [Dataset](#dataset) below for how to get it.

## Project overview

The project runs in three milestones, each building on the last:

**Milestone 1 — Review Understanding & Insight Generation.** Loads the reviews from SQLite into pandas, analyzes rating distribution and review length, runs VADER sentiment scoring on the review text, extracts complaint themes with keyword matching, and asks Gemini to turn the raw statistics into a plain-language business summary.

**Milestone 2 — Customer Segment Comparison & Prioritization.** Splits the 3,000 reviews into three equal segments by star rating (Positive / Neutral / Negative), measures each segment numerically (sentiment score, urgency score, word count, distinctive vocabulary via TF-IDF), and uses Gemini to explain *why* the segments differ and what each one needs from support.

**Milestone 3 — GenAI-Powered Review Assistant (Deployment).** Packages the Milestone 1–2 logic into a single reusable `ReviewAnalyzer` class, saves it with pickle, and wires it into a Gradio app that takes a review and returns predicted sentiment, urgency score, detected themes, and a Gemini-drafted response for a support agent to review before sending — a Human-in-the-Loop design, not a fully automated responder.

### Key findings

- Star rating alone is a weak signal: 45% of reviews show a mismatch between their star rating and their actual text sentiment, and 78% of "neutral" 3-star reviews are actually positive in tone.
- The three rating-based segments are genuinely distinct customer populations (cosine similarity of 0.23–0.27 between all pairs), each needing a different response strategy — fast escalation for Negative, proactive outreach for Neutral, advocacy requests for Positive.
- Keyword-based theme detection has a real, documented limitation: a generic word ("money") can trigger a false-positive "Refund Request" tag on a review that's simply commenting on value for price — a concrete argument for keeping a human in the loop rather than auto-routing on AI output.
- This dataset shows signs of being templated/synthetically generated (a narrow review-length distribution, repeated phrasing). The techniques here work well, but findings and keyword lists should be re-validated against organic, real-world review data before being trusted in production.

## Tech stack

- **Data:** SQLite (`sqlite3`), pandas
- **NLP:** NLTK (stopwords, VADER sentiment), custom rule-based urgency scoring and theme extraction
- **ML:** scikit-learn (`TfidfVectorizer`, cosine similarity)
- **GenAI:** Google Gemini via the `google-genai` SDK (`gemini-2.5-flash`)
- **Visualization:** Matplotlib
- **Deployment:** Gradio, pickle
- **Reporting:** ReportLab (PDF generation)

## Dataset

The notebook expects a SQLite database (`reviews.db`) with a `customer_reviews` table (`rating`, `review_title`, `review_text` — 3,000 rows). It is not committed to this repo. To run the notebook:

1. Obtain `reviews.db` from your own project source (this was distributed as part of a GenAI case study dataset).
2. In Colab, run the first cell and use the file-upload prompt to upload `reviews.db` into the session.

## Running it

The notebook is written for Google Colab:

1. Open `notebooks/pwc_Project.ipynb` in Colab.
2. Upload `reviews.db` when prompted (Milestone 1, first cell).
3. Add your Gemini API key to Colab Secrets under the name `gemini_API_key` (used for the GenAI insight-generation cells in each milestone).
4. Run cells top to bottom. Milestone 3's Gradio cell launches an in-notebook app you can interact with directly.

To run locally instead of in Colab, install the dependencies and swap the `google.colab.files.upload()` / `userdata.get()` calls for a local file path and an environment variable:

```bash
pip install -r requirements.txt
```

## Known limitations

- The keyword-based theme extractor can false-positive on generic words that also appear in unrelated contexts (documented in `reports/Review_App_Snapshot.pdf`).
- VADER is a general-purpose, rule-based sentiment tool — it isn't tuned for this domain and can misread sarcasm or product-specific language.
- Segmentation is rating-based, not sentiment-based, so it inherits the same rating/sentiment mismatch noted above.
- The dataset itself appears templated; conclusions should be re-validated on organic review data before any production use.

## License

Released under the [MIT License](LICENSE).

## Author

Dev
