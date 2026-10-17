---
geometry: "left=1in,right=1in,top=1in,bottom=1in"
output: pdf_document
---

# Machine Learning (LT2326)

## Project Proposal

### Andreas Bartsiokas - Classify code samples as human-origin vs LLM-origin

#### Core Idea

I would like to investigate the topic of distinguishing between human written & machine written code. There is quite a lot of research on the general matter out there. Some examples: [AICD Bench](https://aclanthology.org/2026.eacl-long.325.pdf), [MultiAIGCD](https://link.springer.com/article/10.1007/s00521-026-12336-0).

#### Why it matters

Asside from this topic being genuinely interesting to me, I also think that it has been in the spotlight for the last couple of years for various reasons. In academia, proffesors are concerned about their students using AI for submissions, in industry many companies adopt AI, giving it a lot of control, while others enforce strict no-AI policies, and so on and so forth. Aside of model 'native' quirks, like e.g. halucinations, what makes llm generated code different from human code, and how well can we (a machine) tell those apart?

#### Proposed approach and resources

For now, my current approach is this (might variate depending on obstacles that might be encountered):

- Pick a ready to use dataset, most probable suspect: [AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet) (might change).
- supervised binary classification (human or llm?) OR supervised multiclass classification (human or what llm of those used)
- implementation of a simple baseline like e.g. tf-idf to Logic Regression
- implementaion of a transformer like e.g. CodeBERT
- evaluation

#### Evaluation and Analyses

- Metrics like Accuracy, Precision, Recall, F1. Also a confusion matrix.
- Comparison between the baseline and the transformer.
- If AIGCodeSet is used or a similar one with a published study and results, comparison of my results, with those of that specific study.
- IF there is enough time towards the end:
  - Manually create a small dataset. (Possibly using something like: [mbpp](https://huggingface.co/datasets/Muennighoff/mbpp) then manually prompting llms with the problem descriptions resulting with an equivalent dataset which instead includes machine-generated solutions. Then shuffling the original dataset within the generated one.)
  - This new dataset could be used in various ways, for example:
    1. Test my trained model with this new dataset. Compare the metrics produced between the original test split, with the results of this new 'synthetic' one.
    2. Train using this new tiny dataset and analyse in what ways the results deteriorate compared to the original huge dataset.
  - I know that this isn't an easy task, for various reasons. Obstacles, time or complications. It is to be investigated, only if suficient time is available after the main study and objective have been concluded.
