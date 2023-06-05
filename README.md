# ML_Journey
Objective: Repo to capture ML links and learning in a mindmap using markdown.

# Models

## Deep Learning Models

### Transformers

#### Generative AI

##### Models
- Model: [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- Model: [ColossalAI](https://github.com/hpcaitech/ColossalAI)
- Model: [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf?utm_source=substack&utm_medium=email)
- Model: [Falcon](https://huggingface.co/tiiuae/falcon-40b)

##### Evaluation

- Evaluation: [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/latest/?groups=1)
- [Open LLM Leaderboard by 🤗](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

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

#### Demos:

- [Stable-Diffusion-ControlNet](https://huggingface.co/spaces/ArtGAN/Stable-Diffusion-ControlNet-WebUI)

#### Courses:
- [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/)
- [LangChain AI Handbook by Pinecone](https://www.pinecone.io/learn/langchain/)
- [State of GPT by Andrej Karpathy](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- [LLM Application Development by Deeplearning.AI](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)

#### Articles:
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [Low-Rank Adaptation of Large Language Models (LoRA)](https://huggingface.co/docs/diffusers/main/en/training/lora)

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

### Training

- [Sagemaker Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)

Your own docker container

### Evaluation

- [TensorBoard in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html)

### Deployment

- [SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit)

Implements model server stack and can be added to any Docker container, making it deployable to SageMaker. See example [Amazon SageMaker Multi-Model Endpoints using your own algorithm container](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/multi_model_endpoint_bring_your_own.ipynb) using pre-trained ResNet 18 and ResNet 152 models, both trained on the ImageNet datset.

### Serving

- [Blog: Generative AI inference using AWS Inferentia2 and AWS Trainium](https://aws.amazon.com/blogs/machine-learning/achieve-high-performance-with-lowest-cost-for-generative-ai-inference-using-aws-inferentia2-and-aws-trainium-on-amazon-sagemaker/?sc_channel=sm&sc_campaign=Machine_Learning&sc_publisher=LINKEDIN&sc_geo=GLOBAL&sc_outcome=awareness&sc_content=ml_infrastructure&trk=machine_learning&linkId=213679883)

### Monitoring

### Maintenance

### Others

#### Development

- [SageMaker SSH Help](https://github.com/aws-samples/sagemaker-ssh-helper)

Allows you to connect your IDE, such as VSCode, to Amazon SageMaker's training jobs, processing jobs, realtime inference endpoints, and SageMaker Studio notebook containers for fast interactive experimentation, remote debugging, and advanced troubleshooting.

