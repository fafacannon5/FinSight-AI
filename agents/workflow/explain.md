# Workflow: Explain

Parent: [AGENTS.md](../../AGENTS.md)

**Trigger:** "explain" / "teach me" / "讲解" / "我不懂" / "quiz me" / "考考我" — and after writing full code.

Goal: the member can present and defend the code in the 10-minute video and answer the professor's questions.

1. **Big picture first (3–5 sentences):** what problem this piece solves and where it sits in the pipeline
   (encode → index → (PCA) → retrieve → decide → generate).
2. **Link to the course:** name the CS 541 topic (KNN = Week 10, PCA = Week 5, K-means = Week 4,
   efficiency = CLO2) and the textbook idea it uses.
3. **Walk the code** block by block: inputs/outputs with shapes (e.g. `Q: (n_q, 384)`), complexity
   (e.g. brute-force KNN = O(n_q · N · d)).
4. **One tiny worked example** with numbers the member can check by hand.
5. **Quiz (if asked or after full code):** 3 questions, one at a time, wait for the answer, then give feedback.
   Example: "Why must query vectors use the PCA mean of the corpus, not their own mean?"

Answer in the member's language (中文 if they write in Chinese). Write a wiki page only if the explanation
is reusable (e.g. `wiki/literature/ivf_explained.md`).
