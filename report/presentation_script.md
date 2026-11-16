# Presentation script — FinSight-AI (about 9 min, limit 10)

Draft by Claude at J's request, from the final report; shortened to about 9 minutes. J and H: rewrite in your own words and rehearse with a timer.
The same text is in the speaker notes of `FinSight-AI_slides.pptx`.

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Design and Evaluation of a RAG Chatbot | J | 0:15 |
| 2 | A RAG answer is only as good as its retrieval | J | 0:40 |
| 3 | FiQA: real forum questions, incomplete labels | J | 0:40 |
| 4 | One pipeline, two final configurations | J | 0:40 |
| 5 | How we evaluate | H | 0:40 |
| 6 | E1: an approximate index buys speed with accuracy | H | 1:05 |
| 7 | E2: 128 dimensions is the knee of the PCA curve | J | 0:50 |
| 8 | E3: a similarity threshold can refuse unanswerable questions | J | 0:50 |
| 9 | Test set (run once): the dev conclusions hold | H | 0:40 |
| 10 | Answers fail when retrieval fails | H | 0:50 |
| 11 | Demo: the chatbot with config A | J | 0:45 |
| 12 | Limitations and our AI teammate | H | 0:45 |
| 13 | Retrieval is where accuracy is won or lost | J | 0:20 |

## 1. Design and Evaluation of a RAG Chatbot (J, 0:15)

Hi, we are Jing Cannon and Haiyan Zhu. Our CS 541 project is a retrieval-augmented chatbot for financial questions. We study its retrieval layer: how each design choice trades accuracy against speed and memory.

## 2. A RAG answer is only as good as its retrieval (J, 0:40)

Language models answer fluently, but in finance they can make things up. Retrieval-augmented generation first retrieves forum posts that match the question, then asks the model to answer only from them, with citations. If the right post is missing, the answer is built on the wrong evidence. We ask three questions. One: what does approximate search cost in quality, and what does it buy in speed? Two: how far can PCA shrink the embeddings? Three: how many posts should the model see, and can a similarity threshold make it refuse what it cannot answer?

## 3. FiQA: real forum questions, incomplete labels (J, 0:40)

We use FiQA from the BEIR benchmark: 57,638 finance forum posts and real user questions. We tuned everything on 500 dev questions and used the 648 test questions only once, at the end. Two facts matter. First, 19 percent of the posts exceed the encoder's 256-token limit and are cut off. Second, the labels are incomplete: we removed the labelled posts of 50 questions to make them unanswerable, checked each one by hand, and only 10 really were.

## 4. One pipeline, two final configurations (J, 0:40)

The pipeline has four steps. MiniLM embeds the question into 384 normalised dimensions, so cosine similarity is an inner product. We retrieve the top k posts. If even the best cosine is below a threshold tau, the system refuses without calling the LLM; otherwise gpt-oss-120b answers from the posts with citations. From dev we fixed two configurations: A, exact search in 384 dimensions, and B, PCA to 128 dimensions plus an IVF index. Haiyan will explain how we evaluate them.

## 5. How we evaluate (H, 0:40)

We measure three things. Retrieval: Recall at 10, plus nDCG and MRR, which reward ranking relevant posts high, since the LLM sees only the top five. Speed: latency per query and index memory. Refusal: the answer rate on answerable questions and the correct-refusal rate on unanswerable ones, combined as balanced accuracy, because with 450 versus 10 questions, always answering already scores 98 percent accuracy. Answers get a one-to-three score for correctness and for groundedness. Everything was tuned on dev; test was run once.

## 6. E1: an approximate index buys speed with accuracy (H, 1:05)

First, search. Our PyTorch KNN and FAISS Flat return identical scores, which validates our implementation, and their speed is a tie. IVF splits the corpus into 240 k-means cells and searches only the closest few. One cell is 47 times faster but loses 18 points of recall; at 64 cells it is exact again. We chose 16: minus 2.8 points for 6.7 times lower latency. The gain grows with the corpus: on one thread, 3.4 times on 10 thousand posts and 7.2 times on the full corpus, with the same small loss. Our own PyTorch version of the index matches FAISS on accuracy but is 17 times slower, because it loops in Python.

## 7. E2: 128 dimensions is the knee of the PCA curve (J, 0:50)

Second, PCA, with exact search, so any change comes from PCA alone. At 128 dimensions we keep 80 percent of the variance and lose only 2.5 points of recall, for three times less memory and 2.6 times faster search. Below that, quality drops fast: minus 10 points at 64 dimensions, minus 20 at 32. Unlike the IVF loss, this one cannot be bought back later. Config B combines both and is 30 times faster than A on dev, for minus 4.9 points.

## 8. E3: a similarity threshold can refuse unanswerable questions (J, 0:50)

Third, reliability. The top-1 cosine separates answerable from unanswerable questions with an AUC of 0.82; before the manual check it was only 0.56. With tau at 0.60, config A answers 76 percent of answerable questions and rejects 8 of the 10 unanswerable ones. Config B needs its own tau, 0.66, because PCA raises all cosines. With only ten questions this is uncertain. For k, more posts raised correctness but lowered groundedness, so we chose five.

## 9. Test set (run once): the dev conclusions hold (H, 0:40)

We then ran the test set once, with nothing changed. Both configurations score slightly lower than on dev, as expected after tuning, and the gap stays almost the same: B is 5.4 points below A on test, 4.9 on dev. Background load slowed A during the test run, so for speed we report the dev same-session result: B is 30 times faster.

## 10. Answers fail when retrieval fails (H, 0:50)

For generation we sampled 100 test questions. The threshold refused 16, the model refused 14 more, and 70 got an answer, with mean correctness 2.3 out of 3. The key result is on the right: with a relevant post among the five, correctness is 2.36; without one, it drops to 1.87, but groundedness stays the same. The model stays faithful to the posts it gets, even the wrong ones. So the errors come mainly from retrieval.

## 11. Demo: the chatbot with config A (J, 0:45)

Here is our Streamlit demo with config A: k equals five and tau 0.60. [Play the screen recording:] For a finance question, it shows the five retrieved posts with their scores and a cited answer. For an off-topic question, the best score is far below tau, so it refuses instead of guessing.

## 12. Limitations and our AI teammate (H, 0:45)

Our main limitations: one in five posts is truncated; tau rests on only ten unanswerable questions; and each answer has one joint score, so we have no inter-rater agreement. We disclose all AI use in AI_USAGE.md. Claude wrote most of the code, except the search back ends, which Haiyan wrote; it also ran experiments, made the figures and drafted the report and slides, which we revised. ChatGPT drafted the scores and verdicts; we decided every item together. The AI was sometimes confidently wrong: Claude claimed a merge was pushed when it was not, and a ChatGPT draft gave refusals full groundedness. Tests and saved run records let us catch this.

## 13. Retrieval is where accuracy is won or lost (J, 0:20)

To conclude: IVF with 16 probes keeps most of the accuracy at a fraction of the cost, 128 dimensions is the knee of the PCA curve, and a threshold refuses most unanswerable questions. Above all, answers fail when retrieval fails. Thank you.
