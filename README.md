# ML_Journey
Objective: Repo to capture ML links and learning in a mindmap using markdown.

# Evaluation

## Benchmark

- [StandEval](https://scandeval.github.io/)
  - QA
    - Dataset: ScandiQA (based on MKQA)
    - Metric: Cosine similarity
  - Sentiment Analysis:
    - Dataset: Danish:AngryTweets, Norwegian: NoReC, Swedish: SweReC
    - Metric: Matthew’s Corre- lation Coefficient and macro-average F1.
  - Linguistic Acceptability: 
    - Dataset: ScaLA based on CoLA (Corpus of Linguistic Acceptability) dataset
    - Metric: 
- [SuperLIM](https://github.com/spraakbanken/SuperLim-2)
  - QA: 
    - Dataset:.swefaq
    - Metric: Pseudoalpha = (Accuracy - 109/2049) / (1940/2049)
  - Sentiment Analysis:
    - Dataset: absabank-imm
    - Metric: Alpha
- [NorBench](https://github.com/ltgoslo/norbench)
  - QA: 
    - Dataset: NorQuAD.
    - Metric: token-level F1
  - Sentiment Analysis:
    - Dataset: The Norwegian Review Corpus (NoReC; 2nd release) (Velldal et al., 2018). 
    - Metric: macro F1
  - Linguistic Acceptability: 
    - Dataset: NoCoLA Norwegian corpus of linguistic acceptance (NoCoLA; Jentoft and Samuel, 2023). 
    - Metric: Matthews correlation coefficient (MCC; Matthews, 1975).
  - Translation: 
    - Dataset: Bokma ̊l–Nynorsk bitexts. 
    - Metric: SacreBLEU (Lin and Och, 2004; Post, 2018)

## Dataset

- [ScadiQA](https://scandeval.github.io/)

  - [MKQA (Multilingual Knowledge Questions and Answers)](https://aclanthology.org/2021.tacl-1.82/)
  - [Natural Questions (NQ)](https://aclanthology.org/Q19-1026/)

- [NorBench](https://github.com/ltgoslo/norbench)

- [SuperLim](https://huggingface.co/datasets/sbx/superlim-2)

  - GLUE/SuperGLUE

    

# Models

## Deep Learning Models

### Transformers

#### Generative AI

##### Models
- Model: [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- Model: [ColossalAI](https://github.com/hpcaitech/ColossalAI)
- Model: [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf?utm_source=substack&utm_medium=email)
- Model: [Falcon](https://huggingface.co/tiiuae/falcon-40b)
- Model: [Claude 2: Model Card and Evaluations for Claude Models](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)
- Model: [FalconLite](https://huggingface.co/amazon/FalconLite)
- Model: [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)

##### Data
- [🆕 Appreciating the complexity of large language models data pipelines](https://blog.christianperone.com/2023/06/appreciating-llms-data-pipelines/)

##### Fine Tuning and RLHF
- 🆕 [Interactively fine-tune Falcon-40B and other LLMs on Amazon SageMaker Studio notebooks using QLoRA](https://aws.amazon.com/blogs/machine-learning/interactively-fine-tune-falcon-40b-and-other-llms-on-amazon-sagemaker-studio-notebooks-using-qlora/)
- 🆕 [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
- [Fine-tuning notebooks](https://platform.openai.com/docs/guides/fine-tuning/example-notebooks)

##### Performance

- Optimization: [Optimizing Memory Usage for Training LLMs and Vision Transformers in PyTorch](https://lightning.ai/pages/community/tutorial/pytorch-memory-vit-llm/)
- Evaluation: [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/latest/?groups=1)
- Bias: [Evaluating Language Model Bias with 🤗 Evaluate](https://huggingface.co/blog/evaluating-llm-bias)
- Leaderboard: [Open LLM Leaderboard by 🤗](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

##### Compliance
- [Do Foundation Model Providers Comply with the EU AI Act?](https://crfm.stanford.edu/2023/06/15/eu-ai-act.html)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

##### Security
- [Are aligned neural networks adversarially aligned?](https://arxiv.org/pdf/2306.15447.pdf), [video](https://www.youtube.com/watch?v=uqOfC3KSZFc&t=1s)

##### Readings

- Paper: [Transformer Modes: An Introduction and Catalog](https://arxiv.org/pdf/2302.07730.pdf)
- Paper: [Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/pdf/2212.10559.pdf)
- Paper:[Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)
- Paper: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- Paper: [Prompting Is Programming: A Query Language For Large Language Models](https://arxiv.org/pdf/2212.06094.pdf?utm_source=substack&utm_medium=email)
- Paper: [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)
- Paper: [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327)
- Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Paper: [Elucidating the Design Space of Diffusion-Based Generative Models -neurips2022](https://openreview.net/pdf?id=k7FuTOWMOc7)
- Paper: [An empirical analysis of compute-optimal large language model training -neurips2022](https://openreview.net/pdf?id=iBBcRUlOAPR)
- Paper: [Training Compute-Optimal Large Language Models by J. Hoffmann et al.](https://arxiv.org/abs/2203.15556)
- Paper: [LongNet: Scaling Transformers to 1,000,000,000 Tokens][https://arxiv.org/abs/2307.02486]
- Paper: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- Paper: [Self-Alignment with Instruction Backtranslation](https://arxiv.org/pdf/2308.06259.pdf)
- Book: [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

#### Evaluation
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [h2o-LLM-eval](https://github.com/h2oai/h2o-LLM-eval)
- [open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [helm](https://crfm.stanford.edu/helm/latest/)
- [alpaca_eval](https://tatsu-lab.github.io/alpaca_eval/)
- [chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

#### Demos:

- [Stable-Diffusion-ControlNet](https://huggingface.co/spaces/ArtGAN/Stable-Diffusion-ControlNet-WebUI)

#### Courses:
- [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/)
- [LangChain AI Handbook by Pinecone](https://www.pinecone.io/learn/langchain/)
- [State of GPT by Andrej Karpathy](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- [LLM Application Development by Deeplearning.AI](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [Generative AI by Google ](https://www.cloudskillsboost.google/paths/118)
- [Generative AI with Large Language Models by DeepLearning.AI and AWS](https://www.coursera.org/learn/generative-ai-with-llms)

#### Articles:
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [Low-Rank Adaptation of Large Language Models (LoRA)](https://huggingface.co/docs/diffusers/main/en/training/lora)
- [LLM Economics: ChatGPT vs Open-Source](https://towardsdatascience.com/llm-economics-chatgpt-vs-open-source-dfc29f69fec1)
- [Building LLM applications for production by Chip Huyen ](https://huyenchip.com/2023/04/11/llm-engineering.html)
- [Emerging Architectures for LLM Applications](https://a16z.com/2023/06/20/emerging-architectures-for-llm-applications/?utm_source=tldrai)

#### Videos:
- [State of GPT by Andrej Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A)

#### Internal:

- [Overview of LLMs and ChatGPT: QUIP](https://quip-amazon.com/JdVgAZaYzFV4#CPU9AAGYhtf)
- [Aligning AI with Human Values: QUIP ](https://quip-amazon.com/k2mQAFupUYuS/Aligning-AI-with-Human-Values)

#### Debug:
-[How 🤗 Accelerate runs very large models thanks to PyTorch ](https://huggingface.co/blog/accelerate-large-models)

# Tools


## Sagemaker
Section to provide sagemaker links.

### LLM
- [Hugging Face LLM Container](https://huggingface.co/blog/sagemaker-huggingface-llm)
- [Deep learning containers for large model inference](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-dlc.html)
- [Deploy BLOOM-176B and OPT-30B on Amazon SageMaker with large model inference Deep Learning Containers and DeepSpeed](https://aws.amazon.com/blogs/machine-learning/deploy-bloom-176b-and-opt-30b-on-amazon-sagemaker-with-large-model-inference-deep-learning-containers-and-deepspeed/)
- [Deploy large models at high performance using FasterTransformer on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/deploy-large-models-at-high-performance-using-fastertransformer-on-amazon-sagemaker/)

### JumpStart
- [🆕 Introduction to SageMaker JumpStart - Text Generation with Falcon models](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-falcon.ipynb)

### Training

- [Sagemaker Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)

### Evaluation

- [TensorBoard in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html)

### Deployment

- [SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit)



### Serving

- [Blog: Generative AI inference using AWS Inferentia2 and AWS Trainium](https://aws.amazon.com/blogs/machine-learning/achieve-high-performance-with-lowest-cost-for-generative-ai-inference-using-aws-inferentia2-and-aws-trainium-on-amazon-sagemaker/?sc_channel=sm&sc_campaign=Machine_Learning&sc_publisher=LINKEDIN&sc_geo=GLOBAL&sc_outcome=awareness&sc_content=ml_infrastructure&trk=machine_learning&linkId=213679883)

### Monitoring

### Maintenance

### Others

#### Development

- [SageMaker SSH Help](https://github.com/aws-samples/sagemaker-ssh-helper)

## AWS ML Chips

### SDKs
- [ 🤗 Optimum Neuron ](https://huggingface.co/docs/optimum-neuron/index)
- [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)

