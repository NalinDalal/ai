# COMPLETE AI/ML MASTERY ROADMAP - Everything Included

## Phase 0: Programming Foundations & Mathematics (2-3 weeks)

### Week 1-2: Python + Math Essentials

**What to Learn:**

- Linear Algebra: Matrix operations, eigenvalues, eigenvectors
- Calculus: Differentiation and integration, partial derivatives
- Probability & Statistics: Bayes theorem, distributions, statistical significance
- Programming: Python is a must, with libraries like NumPy, Pandas, and Matplotlib
- Algorithms: Basics of data structures and algorithms (search, sort, graph traversal)

- **Practice platforms:** Kaggle for Python

**Projects:**

- [Implement matrix operations from scratch](./week-1-2/23.matrixOpr.py)
- [Build basic statistical analysis & data visualization dashboard](./week-1-2/33.dashboard.py)

**Time Investment:** 20-30 hours

---

## Phase 1: Classical Machine Learning Foundations (3-4 weeks)

### Week 3-6: Core & Advanced Machine Learning

**What to Learn:**

- Supervised Learning: Linear regression, logistic regression, decision trees
  - [Linear Regression: Galton (1886)](https://galton.org/essays/1880-1889/galton-1886-family-likeness-stature.pdf)
  - [Decision Trees: Quinlan (1986)](https://hunch.net/~coms-4771/quinlan.pdf)

- Unsupervised Learning: Clustering (K-means, DBSCAN), PCA
- Model Evaluation: Metrics like accuracy, precision, recall, F1-score
- Feature Engineering: Data preprocessing, handling missing data
- Frameworks: Scikit-learn for ML models

- **Hands-on:**
  - Work on datasets like Titanic (classification) or Boston Housing (regression) on Kaggle

5. **Projects:**

- Predictive model for stock price forecasting
- Customer segmentation analysis
- Complete Kaggle Titanic competition

**Time Investment:** 60-70 hours

https://www.cse.iitd.ac.in/~mausam/courses/col333/autumn2025/

---

## Phase 2: [Deep Learning](https://d2l.ai/chapter_preface/index.html) Foundations + Core Papers (4-5 weeks)

### Week 7: Neural Network Fundamentals + Stage A Papers

[Paper reading](https://jalexine.github.io/fix-your-paper-reading-game.html)

**Stage A — Roots & Learning Fundamentals (EXACT ORDER):**

1. [**McCulloch & Pitts (1943)** — neurons as logic gates](https://pabloinsente.github.io/the-mcculloch-pitts-artificial-neuron-model)
   - **Mini-project:** [Implement a logical gate network](https://ijettjournal.org/2017/volume-45/number-2/IJETT-V45P212.pdf)
   - **Why it matters:** Foundation of neural computation

2. [**Rosenblatt (1958)** — Perceptron](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)
   - **Mini-project:** Perceptron on AND/OR
   - **Why it matters:** First trainable neural classifier
   - [simulation](https://playbackpress.com/books/cppbook/chapter/12/3?comment=2)

3. [**Rumelhart, Hinton & Williams (1986)** — Backpropagation](https://github.com/georgezoto/Convolutional-Neural-Networks/blob/master/Papers/1986%20Backpro%20Learning%20representations%20by%20back-propagating%20errors%20-%20Rumelhart,%20Hinton,%20Williams.pdf)
   - **Mini-project:** Single-hidden-layer NN in NumPy, implement gradient check
   - **Why it matters:** Enables training of deep networks

**How to Learn:**

- "Deep Learning Specialization" by Andrew Ng (Coursera)
- "Fast.ai Deep Learning for Coders"

**Time Investment:** 25-30 hours

### Week 8: Convolutional Networks + Stage B Papers

**Stage B — Conv nets & Vision Basics:**

4. [**LeCun et al. (1989)** - "Backpropagation Applied to Handwritten Zip Code Recognition"; Early CNNs](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
   - **Mini-project:** Small CNN on MNIST (NumPy → PyTorch)

5. [ImageNet Classification with Deep Convolutional Neural Networks **Krizhevsky, Sutskever & Hinton (2012)** — AlexNet (ImageNet breakthrough)](http://papers.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
   - **Mini-project:** Train small CNN on CIFAR-10; use data augmentation

6. [**He et al. (2016)** — Deep Residual Learning for Image Recognition ; ResNet (skip connections)](https://arxiv.org/pdf/1512.03385)

[Medium Article](https://medium.com/@shadmansobhan114/understanding-residual-network-resnet-skip-connection-c444a1accfe9)

- **Mini-project:** Implement ResNet-18 in PyTorch; compare with plain net

**Projects:**

- Build a digit recognizer (MNIST dataset)
- Create an image classifier using CNNs
- Build CNN-based object detector

**Additional Resources:**

- [Vision Transformer](https://paperswithcode.com/method/vision-transformer)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556) (2014)

**Time Investment:** 30-35 hours

### Week 9-10: Sequence Models + Stage C Papers

**Stage C — Sequences → Attention:**

7. [**Hochreiter & Schmidhuber (1997)** - LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
   - **Mini-project:** Char-level LSTM text generator (Shakespeare)
   - **Resource:** ["Understanding LSTM Networks" by Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   - [**Implementation**](https://github.com/wojzaremba/lstm)

8. [**Word2Veq** Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781) (2013)

9. [**Distributed Representations of Words and Phrases and their Compositionality**](https://arxiv.org/pdf/1310.4546) (2013), [implementation](https://github.com/SkalskiP/vlms-zero-to-hero/blob/master/01_natural_language_processing_fundamentals/01_01_word2vec_with_sub_sampling_and_negative_sampling_in_pytorch.ipynb)

10. [**Sutskever, Vinyals & Le (2014)** — Seq2Seq](https://arxiv.org/pdf/1409.3215)

- **Mini-project:** Toy English→French translator (seq2seq)

11. [**Bahdanau, Cho & Bengio (2014)** — Attention for NMT](https://arxiv.org/pdf/1409.0473)

- **Mini-project:** Add attention to your seq2seq

10. [**Bahdanau, Cho & Bengio (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/pdf/1409.0473)

**Additional Resources:**

- ["The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) ([Code](https://github.com/karpathy/char-rnn))
- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) ([Code](https://github.com/wojzaremba/lstm), [karpathy blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/))
- [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks)
- [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391)

**Harkirat Resources:**

- [NL/DL](https://youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [Karpathy DL Series](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [NLP Playlist](https://youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

**Time Investment:** 35-40 hours

### Week 11: Deep Learning Mastery

**Resources:**

- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- Complete advanced CNN architectures
- GPU acceleration course

**Time Investment:** 25-30 hours

---

## Phase 3: The Transformer Revolution (3-4 weeks)

### Week 12: Transformer Architecture + Stage D Papers

**Stage D — The Core Shift: Transformer Family:**

14. [**Vaswani et al. (2017)** — "Attention Is All You Need" (Transformer) **LANDMARK PAPER**](https://arxiv.org/abs/1706.03762)
    - **Mini-project:** Implement a mini-Transformer from scratch (character-level), reproduce a simple translation or next-token model
    - **Resource:** `The Annotated Transformer` [Blog](https://nlp.seas.harvard.edu/annotated-transformer/) ,[Code](https://github.com/harvardnlp/annotated-transformer/)
      - [Vision Transformer](https://paperswithcode.com/method/vision-transformer)
      - [**Switch Transformer** - "Switch Transformer: Scaling to Trillion Parameter Models"](https://arxiv.org/pdf/2101.03961)
      - [**Char-RNN**](https://github.com/karpathy/char-rnn)

- **Why it matters:** Attention-only, no RNNs - changed everything

**Time Investment:** 30-35 hours

### Week 13-14: Pre-trained Language Models + Stage E Papers

**Stage E — Large Pretraining & Language Modelling:**

15. [**Devlin et al. (2018)** — BERT (bidirectional pretraining) **LANDMARK PAPER**](https://arxiv.org/abs/1810.04805)
    - **Mini-project:** Finetune BERT on sentiment or QA
    - **Why it matters:** Changed how we pretrain language models

16. [**Radford et al. (2018)** — GPT-1 (autoregressive pretraining)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - **Mini-project:** Run a small autoregressive LM (GPT-style) on tiny dataset

17. **Radford et al. / OpenAI (2019)** — GPT-2 (scaling + sampling)
    - **Mini-project:** Fine-tune GPT-2 small for a domain (e.g., commit messages)

18. [**Brown et al. (2020)** — GPT-3 (few-shot behaviour at scale) **LANDMARK PAPER**](https://arxiv.org/abs/2005.14165)
    - **Mini-project:** Train a small [nanoGPT](https://github.com/karpathy/nanoGPT) and experiment with in-context examples
    - **Why it matters:** Few-shot emergent behavior at huge scale
    - **Resource:** [GPT-3](https://paperswithcode.com/method/gpt-3)

**Time Investment:** 35-40 hours

12. **Radford et al. (2018)** - "Improving Language Understanding by Generative Pre-Training" (GPT-1)
13. **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners" (GPT-2)

### Week 15: Scaling Laws + Stage F Papers

**Stage F — Why Scale, and How Far It Goes:**

19. [**Kaplan et al. (2020)** — "Scaling Laws for Neural Language Models" **LANDMARK PAPER**](https://arxiv.org/abs/2001.08361)
    - **Mini-project:** Run ablations on small models (change depth/width/data) and plot loss vs params
    - **Why it matters:** Predictable gains with scale; guides resource allocation

**Time Investment:** 20-25 hours

[GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
[LLM from scratch](https://github.com/rasbt/LLMs-from-scratch)

---

## Phase 4: Large Language Models & Applications (3-4 weeks)

### Week 16-17: LLM Fundamentals & Fine-tuning

**What to Learn:**

- Transformers: Self-attention, positional embeddings
- Tokenization: Byte Pair Encoding (BPE), SentencePiece
- Fine-tuning LLMs: Using pre-trained models like GPT, BERT, etc.
- Frameworks: Hugging Face Transformers library

CLIP: [Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/pdf/2103.00020) (2021)

**How to Learn:**

- Tutorials: Hugging Face blog and documentation
- **Projects:**
  - Create a chatbot with OpenAI API or Hugging Face
  - Fine-tune a pre-trained BERT model on a custom dataset

**NVIDIA Courses:**

- [**Building A Brain in 10 Minutes**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+T-FX-01+V1)
  - Delve into the intricate process of how neural networks glean insights from data
  - Gain insight into the mathematical underpinnings of neuron functionality

**AI/ML Resources:**

- https://github.com/armankhondker/awesome-ai-ml-resources
- [Illya 30 paper](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- https://github.com/InterviewReady/ai-engineering-resources
- https://github.com/ashishps1/learn-ai-engineering
- [llm-zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)

**Time Investment:** 30-40 hours

- [llm from scratch](https://www.freecodecamp.org/news/code-an-llm-from-scratch-theory-to-rlhf/)

### Week 18-19: RAG Systems & Document AI

**Building RAG Agents with LLMs:**

- [**NVIDIA Course**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-15+V1)
  - Discover scalable deployment strategies and vector databases
  - Master modern LangChain paradigms for dialog management and document retrieval
  - Implement advanced models and steps for production with ease

**Augment your LLM with Retrieval Augmented Generation:**

- [**NVIDIA Course**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:NVIDIA+S-FX-16+v1)
  - Master the basics of Retrieval Augmented Generation (RAG)
  - Dive into the RAG retrieval process for enhanced learning

**Projects:**

- [**Build RAG LLM App**](https://youtu.be/f-AXdiCyiT8?si=NLjADorzxHy-intt)
- [**LLM Based App**](https://youtu.be/bcCFTk_uwUw?si=V869EVQIHB_dKkEY)
- [**Projects docs**](https://dswharshit.medium.com/start-building-these-projects-to-become-an-llm-engineer-0064e9e68d9d)
- [**Web Search Engine with Neural Embedding**](https://blog.wilsonl.in/search-engine/)

**Gen AI Resources:**

- https://github.com/bhav09/Generative-AI-Resources
- https://github.com/DataTalksClub/llm-zoomcamp

[**RAG Architecture Article**](https://www.analyticsvidhya.com/blog/2025/01/agentic-rag-system-architectures/)

**Time Investment:** 35-40 hours

- [checkout once](https://aiengineering.academy/)

- [Multi-Agent AI RAG](https://youtu.be/4MTtfTZnH5Y?si=QBqeGkmxuXo7Tgb6)
- [AI Engineer Summit 2025](https://www.youtube.com/watch?v=D7BzTxVVMuw)

---

## Phase 5: Generative AI & Advanced Models (3-4 weeks)

### Week 20: Generative Models + Stage G Papers

**Stage G — Generative Models (Images & Denoising):**

20. [**Kingma & Welling (2013)** — VAE](https://arxiv.org/pdf/1312.6114)
    - **Mini-project:** Implement simple VAE on MNIST
    - [code](https://github.com/NoviceStone/VAE?tab=readme-ov-file)

21. [**Goodfellow et al. (2014)** — GANs](https://arxiv.org/pdf/1406.2661)
    - **Mini-project:** Vanilla GAN on MNIST

22. [**Ho et al. (2020)** — DDPM / Diffusion models](https://arxiv.org/pdf/2006.11239)
    - **Mini-project:** Use Diffusers to fine-tune small diffusion model on a tiny dataset

- [**NVIDIA's Course Generative AI Explained**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-07+V1)
  - Grasp the concept and mechanics behind Generative AI
  - Explore its diverse applications across industries
  - Understand the inherent challenges and potential of Generative AI

**Advanced/Supplementary Papers for Further Exploration:**

- [**Multi-Scale Context Aggregation by Dilated Convolutions**](https://arxiv.org/abs/1511.07122)
- [**Neural Message Passing for Quantum Chemistry**](https://arxiv.org/abs/1704.01212)
- [**Identity Mappings in Deep Residual Networks**](https://arxiv.org/abs/1603.05027)
- [**A Simple Neural Network Module for Relational Reasoning**](https://arxiv.org/abs/1706.01427)
- [**Variational Lossy Autoencoder**](https://arxiv.org/abs/1611.02731)
- [**Relational Recurrent Neural Networks**](https://arxiv.org/abs/1806.01822)
- [**Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton**](https://arxiv.org/abs/2304.01258)
- [**Neural Turing Machines**](https://arxiv.org/abs/1410.5401)
- [**Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**](https://arxiv.org/abs/1512.02595)
- [**A Tutorial Introduction to the Minimum Description Length Principle**](https://arxiv.org/abs/math/0307138)
- [**Machine Super Intelligence by Shane Legg**](https://www.doc.ic.ac.uk/~shm/theresearchassistant/resources/legg-agi.pdf)
- [**Kolmogorov Complexity and Algorithmic Randomness**](https://arxiv.org/abs/1808.06305)

**Time Investment:** 25-30 hours

---

### Week 21-22: GenAI Roadmap Implementation

**Phase 0: Foundations of GenAI**

1. **Intro to GenAI & LLMs**
   - What is Generative AI? LLMs? RAG?
   - Overview of OpenAI, Hugging Face, and GPT
   - Tools: Jupyter, VSCode, Python setup
   - [resource](https://github.com/anirudhuuu/GenAI/)

2. **Project 1:** Your First Chatbot with OpenAI API
   - Use OpenAI chat-completion API
   - Simple CLI chatbot
   - Intro to prompt engineering

**Phase 1: Prompt Engineering & Token Management**

1. **Prompt Engineering Deep Dive**
   - Zero-shot, Few-shot, Chain-of-thought
   - Temperature, top_p, tokens, max_length

2. **Project 2:** Smart Email Generator
   - Take a subject and generate email copy
   - Use prompt templates and roles

**Phase 2: LangChain Essentials**

1. [**LangChain Basics**](https://docs.langchain.com/)
   - Components: Chains, Tools, Agents, Memory, PromptTemplates

2. **Project 3:** AI-Powered PDF Q&A Bot
   - Upload PDF → Chunk it → Embed → Query using OpenAI
   - Tools: LangChain, FAISS, PyPDF, OpenAIEmbeddings

**Phase 3: RAG (Retrieval-Augmented Generation)**

1. **Intro to Embeddings & Vector Stores**
   - ChromaDB, Pinecone
   - Cosine similarity, chunking, indexing

2. **Project 4:** Resume Analyzer Bot
   - Upload resume, analyze it, and suggest job matches
   - RAG pipeline using Chroma + LangChain

3. **Project 5:** YouTube Video Q&A Bot
   - Use yt-dlp to extract transcripts
   - Create embeddings, and answer questions based on video

**Time Investment:** 35-40 hours

---

### Week 23: Advanced GenAI Projects

**Phase 4: Agents & Tools**

1. **LangChain Agents Explained**
   - ReAct, MRKL, Tool usage

2. **Project 6:** Multi-Tool Research Assistant
   - Toolset: SerpAPI, Calculator, WebSearch, DocsReader

3. **Project 7:** AI Travel Planner
   - Input: Dates + preferences → Output: Itinerary
   - Uses tools like Maps, Flights, Weather, Budget planner

**Phase 5: LangGraph & Multi-Agent Systems**

1. **LangGraph Intro**
   - Graph-based reasoning
   - Building agent workflows

2. **Project 8:** Autonomous Startup Ideation Bot
   - One agent ideates, one critiques, one validates market fit

**Time Investment:** 30-35 hours

**Reference:**

- [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford Course Notes)](http://cs231n.stanford.edu/)

---

## Phase 6: AI Alignment & Safety (2-3 weeks)

### Week 24-25: Alignment + Stage H Papers

**Stage H — Alignment, Instruction Tuning & Safety:**

23. [**Christiano et al. (2017)** — RL from Human Preferences (foundational RLHF ideas)](https://arxiv.org/pdf/1706.03741)
    - **Mini-project:** Toy reward model + PPO fine-tune on GPT-2 small (toy env)

24. [**Ouyang et al. (2022)** — InstructGPT / Training LMs with human feedback **PRACTICAL ALIGNMENT**](https://arxiv.org/pdf/2203.02155)
    - **Mini-project:** Collect human rankings for 50 prompts, train a small reward model and rerank outputs

25. **Anthropic (2023)** — Constitutional AI (alternative alignment technique)
    - **Mini-project:** Implement a simple rule-based "constitution" reranker

**Time Investment:** 25-30 hours

---

## Phase 7: AI Agents & Multi-Agent Systems (3-4 weeks)

### Week 26: Basic Agents + Stage I Papers

**What to Learn:**

- Reinforcement Learning (RL): Q-learning, policy gradients
- Autonomous Agents: Concepts like goals, actions, state transitions
- Multi-agent Systems: Collaboration, competition among agents
- Frameworks: OpenAI Gym, Ray RLlib

**Stage I — Agents, Tool Use & Systems:**

26. [**Yao et al. (2022)** — ReAct: Reasoning + Acting integrated **PRACTICAL AGENT PATTERN**](https://arxiv.org/abs/2210.03629)
    - **Mini-project:** Build a ReAct prompt agent that queries Wikipedia and runs a calculator API

27. **AutoGPT / BabyAGI / LangChain patterns** (repos & blogs)
    - **Mini-project:** Build a two-agent system: planner + worker, via [LangChain](https://docs.langchain.com/)

28. [**Model Context Protocol (MCP, 2023-2024 spec)** — Protocol standard for model-agent-tool context passing](https://modelcontextprotocol.io/introduction)
    - **Guide:** https://composio.dev/blog/mcp-server-step-by-step-guide-to-building-from-scrtch/
    - **Mini-project:** Wrap your ReAct agent in an MCP-style JSON schema; connect two MCP services

**How to Learn:**

- **Courses:**
  - "Reinforcement Learning Specialization" by the University of Alberta (Coursera)
  - Tutorials from OpenAI Gym
- **Projects:**
  - Train an RL agent to solve a game environment like CartPole

**Time Investment:** 30-35 hours

### Week 27-28: Multi-Agent Systems & Frameworks

**Frameworks:**

- **Best 5 frameworks:** https://medium.com/@amosgyamfi/best-5-frameworks-to-build-multi-agent-ai-applications-1f88530ef8d8
  - Phidata
  - Swarm
  - CrewAI
  - Autogen
  - LangGraph

**Agent Resources:**

- [**Chip Huyen's Agents**](https://huyenchip.com//2025/01/07/agents.html)
- [**AI Agents for Beginners**](https://github.com/microsoft/ai-agents-for-beginners)
- [**Multi-Agent AI RAG** ](https://youtu.be/4MTtfTZnH5Y?si=QBqeGkmxuXo7Tgb6)

**8 Free Sources about AI Agents:**

1. [**"Agents" Google's whitepaper** by Julia Wiesinger, Patrick Marlow and Vladimir Vuskovic](https://www.kaggle.com/whitepaper-agents)
   - Covers agents, their functions, tool use and how they differ from models

2. [**"Agents in the Long Game of AI"** book by Marjorie McShane, Sergei Nirenburg, and Jesse English](https://direct.mit.edu/books/oa-monograph/5833/Agents-in-the-Long-Game-of-AIComputational)
   - Explores building AI agents, using Hybrid AI, that combines ML with knowledge-based reasoning

3. [**"AI Engineer Summit 2025: Agent Engineering"** 8-hour video](https://www.youtube.com/watch?v=D7BzTxVVMuw)
   - Experts' talks that share insights on the freshest Agent Engineering advancements

4. [**AI Agents Course from Hugging Face**](https://huggingface.co/learn/agents-course/en/unit0/introduction)
   - Agents' theory and practice to learn how to build them using top libraries and tools

5. [**"Artificial Intelligence: Foundations of Computational Agents"** 3rd Edition](https://artint.info/3e/html/ArtInt3e.html)
   - Agents' architectures, how they learn, reason, plan and act with certainty and uncertainty

6. [**"Intelligent Agents: Theory and Practice"** book by Michael Wooldridge](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/ker95/ker95-html.html)
   - How agents were seen in 1995 and explore their theory, architectures and agent languages

7. [**The Turing Post articles "AI Agents and Agentic Workflows"** on Hugging Face](https://huggingface.co/Kseniase)
   - Explore agentic workflows in detail and agents' building blocks, such as memory and knowledge

8. [**Collection "8 Free Sources to Master Building AI Agents"**](https://www.turingpost.com/p/building-ai-agents-sources)

9. [Book- Building AI Agents with LLMs, RAG, and Knowledge Graphs]

   **Time Investment:** 40-45 hours

### Week 29: Advanced Agent Projects

**Phase 6: API Deployment + Web App Integration**

1. **Serving LLM Apps with FastAPI**
   - API routing, auth, JSON I/O

2. **Project 9:** AI Code Review API
   - Input: PR diff → Output: Review comment suggestions

3. **Frontend Integration** (Optional React/Firebase)
   - Connecting FastAPI backend with frontend
   - Deploy on Vercel/Render

**Phase 7: MCP Integration**

- Personalize LLM behavior per user, domain, or app context

**Projects:**

- Multi-agent collaboration for a game-playing task
- Autonomous system combining ML predictions with agentic decision-making

**Time Investment:** 30-35 hours

---

## Phase 8: Cutting-Edge & Production (2-3 weeks)

### Week 30-32: Advanced Topics + Stage J Papers

**Stage J — Cutting Edge / Alternatives:**

30. **Mixture-of-Experts / Switch Transformer papers** (Google / others)
31. **Structured State Space Models / SSM / Mamba style papers** (2023-2024)
32. **Latest model evals, alignment evaluations** (2023-2025)
    - **Mini-project:** Swap one Transformer block in your mini-GPT with an MoE/SSM block

**Production & Deployment:**

- **NVIDIA Course:** Accelerate data science workflows seamlessly
  - https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+T-DS-03+V1
  - Discover the advantages of unified CPU and GPU workflows
  - Speed up data processing and ML without altering any code

**Phase 8: Deployment & Production-Ready AI**

1. **Caching, Rate Limiting, and Logging**
   - Redis, Pinecone persistence
   - Tracing with LangSmith / OpenTelemetry

2. **Project 10:** Full-stack AI Feedback App
   - Input: Student project uploads
   - Output: Instant AI feedback, stored in database
   - Dashboard view with ranking/score

- [Switch Transformer: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

**Time Investment:** 30-35 hours

[voice agents1](https://youtu.be/vaCTaUEpqvE?si=mFWG6JGWVcLsY4BF)
[voice agents2](https://youtu.be/oU_rr-bOrK8?si=dRQ5GqN1YhR9qAcw)
[building ai voice agents](https://www.freecodecamp.org/news/how-to-build-advanced-ai-agents/)

#### Multi-Agent Frameworks

- [**Framework Comparison**](https://medium.com/@amosgyamfi/best-5-frameworks-to-build-multi-agent-ai-applications-1f88530ef8d8)
- [**Building A Brain in 10 Minutes**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+T-FX-01+V1)
- [**RAG Architecture**](https://www.analyticsvidhya.com/blog/2025/01/agentic-rag-system-architectures/)
- [**Building RAG Agents**](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-15+V1)

#### MCP (Model Context Protocol)

- [**Official Documentation** ](https://modelcontextprotocol.io/introduction)
- [**Building Guide** ](https://composio.dev/blog/mcp-server-step-by-step-guide-to-building-from-scrtch/)

#### AI Agents Deep Dive

- [**Google's Agents Whitepaper**](https://www.kaggle.com/whitepaper-agents)
- [**MIT Book on Agents:** ](https://direct.mit.edu/books/oa-monograph/5833/Agents-in-the-Long-Game-of-AIComputational)
- [**Hugging Face Agents Course**](https://huggingface.co/learn/agents-course/en/unit0/introduction)
- [**Computational Agents Book**](https://artint.info/3e/html/ArtInt3e.html)
- [**Intelligent Agents Theory**](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/ker95/ker95-html.html)
- [**Turing Post on Agents**](https://huggingface.co/Kseniase)
- [**Building Agents Collection**](https://www.turingpost.com/p/building-ai-agents-sources)

---

## Bonus Modules & Additional Papers

### Ilya Sutskever & John Carmack's 30 Essential Papers

**Additional Core Papers (integrate throughout phases):**

- **The First Law of Complexodynamics** by Scott Aaronson ([Blog](https://scottaaronson.blog/?p=762))
- **Keeping Neural Networks Simple by Minimizing the Description Length of the Weights** ([PDF](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf))
- **GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism** ([ArXiv](https://arxiv.org/abs/1811.06965))
- **Multi-Scale Context Aggregation by Dilated Convolutions**
- **Neural Message Passing for Quantum Chemistry**
- **Identity Mappings in Deep Residual Networks**
- **A simple neural network module for relational reasoning**
- **Variational Lossy Autoencoder**
- **Relational recurrent neural networks**
- **Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton**
- **Neural Turing Machines**
- **Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**
- **A Tutorial Introduction to the Minimum Description Length Principle**
- **Machine Super Intelligence** by Shane Legg
- **Kolmogorov Complexity and Algorithmic Randomness**
- **CS231n: Convolutional Neural Networks for Visual Recognition**

---

## How to Read Each Paper (Strict 6-Step Recipe)

**Do this for EVERY paper before moving on:**

1. **Skim (10-15m):** Title, abstract, intro, conclusion, figures. Note the single-sentence claim.
2. **Questions (5m):** Write 3 questions the paper answers.
3. **Deep read (30-60m):** Methods + key equations; rederive one equation by hand.
4. **Code hunt (15-30m):** Find official / PapersWithCode / GitHub implementation.
5. **Implement (4-12h depending on project):** Reproduce one core experiment or reimplement a core block (small dataset). **Always** implement from scratch at least once (NumPy or PyTorch).
6. **Summarize (20-30m):** 300-500 words: 1) problem, 2) method, 3) main result, 4) limitations, 5) a one-line idea for extending it.

**Mini-rule:** If you can't implement a minimal version in 2 days, you didn't read it properly.

---

#### Project Documentation

- Building a chatGPT / Midjourney - like bot for a niche persona / use-case and integrate it with WhatsApp or Slack or Discord or build an app using Streamlit /Gradio.
- Chrome extension to summarise / ideate / extract takeaways / research with web pages.
- Create a news aggregator for a targeted persona (PMs, Al Engg., )
- Multi-modal generation via Discord.

---

## Final Milestone: Capstone Project

### Ultimate Integration Project (Week 32-34)

**Build a Complete AI System That Combines Everything You've Learned**

1. Agent framework
2. RL Finetuning project + writing evals
3. Devin
4. Memory Framework

#### Project Requirements:

1. **Data Pipeline:** Automated data collection and preprocessing
2. **Classical ML:** Feature engineering and baseline models
3. **Deep Learning:** Custom neural network architecture
4. **LLM Integration:** Language understanding and generation
5. **RAG System:** Knowledge retrieval and augmentation
6. **AI Agents:** Multi-agent workflow orchestration
7. **Safety & Alignment:** Content filtering and ethical AI practices
8. **Production Deployment:** Scalable API with monitoring
9. **User Interface:** Web app or mobile interface
10. **Documentation:** Complete technical documentation

#### Example Capstone Ideas:

- **AI Research Assistant:** Multi-agent system that reads papers, synthesizes findings, and generates research proposals
- **Intelligent Code Review System:** Combines static analysis, ML models, and LLMs for comprehensive code review
- **Personal AI Tutor:** Adaptive learning system with multimodal understanding and personalized teaching strategies
- **Autonomous Trading System:** Combines market analysis, sentiment analysis, and risk management with AI agents
- **Creative AI Studio:** Multi-modal generative system for content creation across text, images, and code

---

[voice ai](https://youtu.be/oU_rr-bOrK8?si=krryAFS9_GfHt6IV)
[live ai voice](https://youtu.be/vaCTaUEpqvE?si=o0yM9FzIAE3EwgDS)

---

Let's build our own agent, like I give my schedule to an api, which sets it
Then every time I push anything, like say I have a slot for 10:30 to read for system design and interviews

Now I can across something like an article so I push it to api the link of it

Now at 10:30 it reminds me of my slot, also gives option to read that article

I have slot in morning 7 to 9 for dsa and cp, so I say to it I wanna practice good stuff in dp, it fetches me awesome questions on them and push me then at specified timemaybe also gives out good stuff to read and revise

How's the idea, basically does everything based upon the context
