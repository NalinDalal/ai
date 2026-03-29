# COMPLETE AI/ML MASTERY ROADMAP - Everything Included

## Phase 0: Programming Foundations & Mathematics (2-3 weeks)

### [Week 1-2: Python + Math Essentials](./week-1-2-python-basics/readme.md)

---

## Phase 1: Classical Machine Learning Foundations (3-4 weeks)

### [Week 3-6: Core & Advanced Machine Learning](./week-3-ml/readme.md)

---

## Phase 2: Deep Learning Foundations + Core Papers (4-5 weeks)

### Week 7: [Neural Network Fundamentals](./week-7-neural-networks/readme.md)

### Week 8: [Convolutional Networks](./week-8-cnn/readme.md)

### Week 9: [Deep Learning](./week-9-deep-learning/readme.md) + [Recurrent Neural networks](./week-9-rnn/readme.md) + [Karpaathy implementation](./week-9-rnn-karpathy/readme.md)

### Week 10: [LSTM+Word2Vec](./week-10-lstm-Word2Vec/readme.md)

### Week 11: [Sequence Models + Stage C Papers](./week-11-sequence/readme.md)

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
    - [yt-lec](https://youtu.be/XXtpJxZBa2c?si=oc6fOm5VPPCYFauY)

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



