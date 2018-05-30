# Paper report
### 8. LSDSem 2017 Shared Task: The Story Cloze Test: W17-0906.pdf

###### abstract
survey paper for SCT task and presents 8 methods from each teams.


### Systems

+ Linear Classifier (SVM or something?)

Feature: P(entire story) given by a language model.
feature: sentence length, character n-gram and word n-gram from only ending sentences

+ Linear classifier 

Feature:
P(entire story w.r.t sentiment)
P(entire story w.r.t frame-based) (??)
topical consistency score

+ fc + LSTM

Feature: pretrained word embedding

+ Bi LSTM + traditional feature

Feature: sentiment, negation, pronominalization, n gram overlap

+ RNN

feature: skip gram for sentences + Data augmentation

+ pointwise mutual information scores

P(coherence for the rest of the sentences | sentences)

+ ?

Feature: sentimente and sentiment overlap between ending and the rest of the story

+ DSSM

project ending and 4 sentences into 1 common space 


Tips: using pretrained embedding might be important.

simple logistic regression for word2vec embedding can achieve high score

78% of validation data contians sentiment words.


### 9. Sentiment Analysis and Lexical Cohesion for the Story Cloze Task
###### abstract

two systems: 

1. classification based on sentiment
2. on coherence.

##### systems

1. classification :
they adapted VADER dictionary (Hutto and Gilbert, 2014), which contains positive and negative words to calculate the snetence sentiment score by summing up all the semantic score for each words.
if negation found, they flip the score by multiplying -1.
In the end, ending sentence with higher sentimental similarity to the preceding sentences will be picked.


2, estimate the coherence probability of ending with the rest of the stories by referencing to the count from all the word co-occurence for each story. Whichever gets higher score will be regarded as the true ending.


##### trivials: 

- they remove stop words(the, a, ...)
- 78$ of the sentences have sentiment words.

##### drawbacks:

- Estimating coherence by counting all the cooccurrence is too simple and the result is miserably low. (0.536) Use can we use n-gram instead?

- predicting by focusing only on sentiment words (this is what system 1 does) seem to make sense, but the result is 0.60, we need to check the sentence to see why this does not work.

### 10. An RNN-based Binary Classifier for the Story Cloze Test

###### Abstract
Dummy sentence + BookCorpus ending gave 67% result.
###### Systems
1. Embedding part: word embedding or sentence embeddings(Kiros, 2015)
2. Learning method: supervised or unsupervised. 

2.1 unsupervised

2.1.1. count word co-occurence from large dataset to estimate how one sentence is likely to followed by next sentence.

2.2 supervised

2.2.1 RNN from created dummy data.

###### how to create dummy data?
1. choose ending sentence randomly from another stories.
2. choose ending from previous sentences in the story
3. find N most similar ending from similar stories by using interactive storytelling system(Swanson and Gordon, 2012) 
4. language model built from ROC Corpus. 


