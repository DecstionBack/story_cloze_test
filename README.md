# NLUProject2

## currently implemented

### SCNN (acc: 58%)
- https://dl.acm.org/citation.cfm?id=3184558.3186958
- self-weight layer and CNN omitted
- only use word2vec feature as input feature
- only use last output of LSTM instead of all outputs along timestep (if we use all outputs, how do we ignore the output corresponding to <pad> inputs?)

### Nico model (name not specified) (acc: 60%)
- https://www.researchgate.net/publication/318740853_Resource-Lean_Modeling_of_Coherence_in_Commonsense_Stories
- output shape is changed to 1 instead of 2
- use [4 story sentences + 1 candidate sentence] as input and scalar (0 or 1) as Ground Truth, while the author gives [4 sentences + 2 candidate sentences] and 2 dim one hot vector as GT

