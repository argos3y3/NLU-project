
### First Part: Subjectivity & Polarity (2 points)
Create a pipeline model for Subjectivity and Polarity detection tasks. The pipeline has to be composed of two different models:

    1. The first model predicts if a sentence is subjective or objective;
    2. The second model performs the polarity detection of a document after removing the objective sentences predicted by the first model;


You have to report the results of the first and the second models. For the second model, you have to report the resutls achieved with and without the removal of the objective sentences to see if the pipeline can actually improve the performance. 

**The type of model**: You have to choose a Neutral Network in PyTorch (e.g. MLP or RNN o adattare quelli che abbiamo visto in lab)  or a pre-trained language model (e.g. BERT or T5).

**Datasets**:

    - NLTK: subjectivity (Subjectivity task)
    - NLTK: movie reviews (Polarity task)
**Evaluation**:

Use a K-fold evaluation for both tasks where with K = 10 PER COMPARARE CON SOTA


```
import nltk
nltk.download("subjectivity")
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity
print(len(subjectivity.sents()))
```

SOSTANZIALMENTE SI TRATTA DI MIGLIORARE PIPELINE ULTIMO ESERCIZIO

NOTA CHE SI POSSONO USARE MODELLI DIVERSI PER I DUE TASK

### Second Part: Aspect Based Sentiment Analysis (4 points) for a mark $\ge$ 30
Implement a joint model based on a Neural Network or a Pre-trained Language model for the Aspect Based Sentiment Analysis task:

    - Task 1: Extract the Aspect terms;
    - Task 2: Detect the polarity of these terms;
 
**Dataset**: The dataset that you have to use is the Laptop partition (LA TERZA) of SemEval2014 task 4, you can downloaded it from [here](https://github.com/lixin4ever/E2E-TBSA/tree/master/data).

**Evaluation**:  For the evaluation you can refert to this [script](https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py) or the official script provided by [SemEval](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) QUESST ULTIMO VA ADATTATO (Baseline, Evaluation and Evaluation link). Report F1, Precision and Recall for Task 1 alone, where you just consider the span ids, and jointly with Task2. You can jointly evaluate your model on these two tasks by considering both the span ids and polarity label in one single triple such as (id_start, id_end, polarity).


