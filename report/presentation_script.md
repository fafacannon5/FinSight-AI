# Presentation script — FinSight-AI (10 min)

Draft by Claude at J's request, from the final report. J and H: rewrite in your own words and rehearse with a timer.
The same text is in the speaker notes of `FinSight-AI_slides.pptx`.

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Design and Evaluation of a RAG Chatbot | J | 0:20 |
| 2 | A RAG answer is only as good as its retrieval | J | 0:50 |
| 3 | FiQA: real forum questions, incomplete labels | J | 0:50 |
| 4 | One pipeline, two final configurations | J | 0:50 |
| 5 | How we evaluate | H | 0:50 |
| 6 | E1: an approximate index buys speed with accuracy | H | 1:15 |
| 7 | E2: 128 dimensions is the knee of the PCA curve | J | 1:00 |
| 8 | E3: a similarity threshold can refuse unanswerable questions | J | 1:10 |
| 9 | Test set (run once): the dev conclusions hold | H | 0:50 |
| 10 | Answers fail when retrieval fails | H | 1:00 |
| 11 | Demo: the chatbot with config A | J | 0:30 |
| 12 | Limitations and our AI teammate | H | 0:50 |
| 13 | Retrieval is where accuracy is won or lost | J | 0:25 |

## 1. Design and Evaluation of a RAG Chatbot (J, 0:20)

Hi, we are Jing Cannon and Haiyan Zhu. Our CS 541 project is a retrieval-augmented chatbot for financial questions. We do not focus on the user interface; we study the retrieval layer underneath it, and how its design choices trade accuracy against speed and memory.

## 2. A RAG answer is only as good as its retrieval (J, 0:50)

Large language models answer fluently, but in finance they can make things up, and the user cannot check. Retrieval-augmented generation first retrieves forum posts that match the question, then asks the model to answer only from those posts, with citations. If the right post is not retrieved, the model answers from the wrong evidence. So we asked three questions. One: exact or approximate nearest-neighbour search — how much quality do we lose for how much speed? Two: how far can PCA shrink the embeddings? Three: how many documents should the model see, and can a similarity threshold make the system refuse questions it cannot answer?

## 3. FiQA: real forum questions, incomplete labels (J, 0:50)

We use FiQA from the BEIR benchmark: 57,638 posts from finance forums, with real user questions. We tune everything on the 500 dev questions and used the 648 test questions only once, at the very end. Two things in the data matter for our results. First, 19 percent of the posts are longer than our encoder's 256-token limit, so their end is cut off. Second, the labels are incomplete. To test refusal, we removed the labelled posts of 50 questions and checked each one by hand — ChatGPT drafted a verdict and we decided together. Only 10 of the 50 became really unanswerable; for the other 40, another post still answered the question.

## 4. One pipeline, two final configurations (J, 0:50)

Our pipeline has four steps. We embed the question with MiniLM into 384 dimensions and normalise it, so cosine similarity is just an inner product. We retrieve the top k posts. If even the best post has a cosine below a threshold tau, the system refuses without calling the LLM. Otherwise gpt-oss-120b answers from the posts and cites them. After tuning on dev we fixed two configurations: A is exact search in full 384 dimensions, the accurate one; B is PCA to 128 dimensions plus an IVF index, the efficient one. Haiyan will now explain how we evaluate them.

## 5. How we evaluate (H, 0:50)

We measure three things. Retrieval: Recall at 10 — the share of relevant posts in the top 10 — plus nDCG and MRR, which reward ranking them high, because the LLM only sees the top five and refusal only looks at the top one. We also measure latency per single query and index memory. Refusal: the answer rate on answerable questions and the correct-refusal rate on unanswerable ones; we pick tau by balanced accuracy, because with 450 versus 10 questions, 'always answer' already gets 98 percent plain accuracy. Answers: a one-to-three score for correctness and for groundedness. The protocol is strict: everything is tuned on dev, and test was run once.

## 6. E1: an approximate index buys speed with accuracy (H, 1:15)

First, search. Our PyTorch KNN and FAISS Flat give exactly the same scores, which validates our implementation, and their speed is a tie. IVF splits the corpus into 240 k-means cells and only searches the closest few. Probing one cell is 47 times faster but loses 18 points of recall. Each extra cell buys accuracy back, at a growing cost in speed; at 64 cells it is exact again. We chose 16 cells: minus 2.8 points for 6.7 times lower latency. As a bonus we implemented the same index ourselves in PyTorch. It matches FAISS on accuracy but is 17 times slower, because it loops in Python while FAISS uses optimised C++.

## 7. E2: 128 dimensions is the knee of the PCA curve (J, 1:00)

Second, PCA. Here we kept exact search, so any change is caused by PCA alone. Memory falls exactly with the dimension. At 128 dimensions we keep 80 percent of the variance and lose only 2.5 points of recall, for three times less memory and 2.6 times faster search. Below that it falls quickly: 64 dimensions lose almost 10 points, 32 lose 20. One difference from IVF: the IVF loss can be bought back with more probes, but what PCA removes is gone. Config B combines both and is 30 times faster than A on dev, for minus 4.9 points.

## 8. E3: a similarity threshold can refuse unanswerable questions (J, 1:10)

Third, reliability. The top-1 cosine score separates answerable from unanswerable questions with an AUC of 0.82. Before the manual check it was only 0.56, because most 'unanswerable' candidates were in fact answerable. With tau at 0.60, config A answers 76 percent of answerable questions and refuses 8 of the 10 unanswerable ones. Config B needs its own tau, 0.66, because PCA raises all cosine values. With only ten questions this is uncertain. For k, giving the model more posts raised correctness but lowered groundedness, so we chose k equals five as the compromise.

## 9. Test set (run once): the dev conclusions hold (H, 0:50)

Then we ran the test set once, with nothing changed. Both configurations score a little lower than on dev, which is expected after tuning on dev, and the gap between them stays almost the same: B is 5.4 points below A on test, 4.9 on dev, with three times less memory. One honest caveat: the machine was under heavy background load during the test run, which slowed A's multithreaded search, so we report the dev same-session speed-up of 30 times, not the test one.

## 10. Answers fail when retrieval fails (H, 1:00)

For generation, we sampled 100 test questions. The threshold refused 16, the model refused 14 more, and 70 got a real answer, with mean correctness 2.3 out of 3. The most useful result is on the right. When a relevant post is among the five the model sees, correctness is 2.36. When none is, correctness drops to 1.87 — but groundedness stays the same. The model stays faithful to the posts it gets, even when they are the wrong posts. So in our system the errors come mainly from retrieval, and better retrieval would help more than a better prompt.

## 11. Demo: the chatbot with config A (J, 0:30)

Here is our Streamlit demo, which uses config A with k equals five and tau equals 0.60. [Play the screen recording:] I type a question, the app shows the five retrieved posts with their similarity scores, and then the cited answer. For a question the corpus cannot answer, it refuses instead of guessing.

## 12. Limitations and our AI teammate (H, 0:50)

Our main limitations: one in five posts is truncated by the encoder; tau rests on only ten unanswerable questions; and each answer has one joint score, so we have no inter-rater agreement. We also used AI tools, and we disclose all of it in AI_USAGE.md. Claude helped write most of the code, experiments and this report, and ChatGPT drafted the scores, which we then decided together. It made us much faster, but it also made convincing mistakes — for example it once claimed a merge was pushed when it was not. We kept control with tests, a locked test set, and by checking every number against the result files.

## 13. Retrieval is where accuracy is won or lost (J, 0:25)

To conclude: IVF with 16 probes keeps most of the accuracy at a fraction of the cost; 128 dimensions is the knee of the PCA curve; and a similarity threshold refuses most unanswerable questions. Above all, answers fail when retrieval fails, so that is where to improve next. Thank you.
