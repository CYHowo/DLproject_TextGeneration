### Deep Learning Project - Shakespeare Text Generation

##### **Objective**

Implement a transformer-based, character-level language model (GPT-like) and train it on the Shakespeare dataset. By the end of this project, you should be able to generate Shakespearean-like text given a seed string.

#### **Environment**

```
pip install -r /path/to/requirements.txt

```

#### **Dataset**:

The Shakespeare dataset contains the complete works of William Shakespeare, including his plays, poems, and sonnets.

[**Download link**](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

### Model Implementation Approaches

We implemented the model using three different approaches:

1. **Manual Implementation** (`shakesperean_generation.py`)
2. **Custom TransformerBlock** (`modelito_final.ipynb`)
3. **Using GPT2LMHead Inheritance** (`shakespearean_generation_gpt.py`)


Through implementing these different approaches, we observed that each method varied significantly in terms of training time, complexity, and results. The `manual implementation` provided a deeper understanding of the model architecture but was time-consuming. The custom `TransformerBlock` approach struck a balance between flexibility and efficiency, while leveraging the `GPT2LMHead` allowed us to achieve faster results with minimal effort by utilizing pre-existing frameworks.

This exploration highlights the importance of choosing the right approach based on the project's goals, resources, and time constraints. Each method offers unique insights, and combining this knowledge has greatly enriched our understanding of transformer-based language models.
