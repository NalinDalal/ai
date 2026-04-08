# COMPLETE AI/ML MASTERY ROADMAP - Everything Included

## Phase 0: Programming Foundations & Mathematics

### [Week 1-2: Python + Math Essentials](./week-1-2-python-basics/readme.md)

---

## Phase 1: Classical Machine Learning Foundations

### [Week 3-6: Core & Advanced Machine Learning](./week-3-ml/readme.md)

---

## Phase 2: Deep Learning Foundations + Core Papers

### Week 7: [Neural Network Fundamentals](./week-7-neural-networks/readme.md)

### Week 8: [Convolutional Networks](./week-8-cnn/readme.md)

### Week 9: [Deep Learning](./week-9-deep-learning/readme.md) + [Recurrent Neural networks](./week-9-rnn/readme.md) + [Karpaathy implementation](./week-9-rnn-karpathy/readme.md)

### Week 10: [LSTM+Word2Vec](./week-10-lstm-Word2Vec/readme.md)

### Week 11: [Sequence Models + Stage C Papers](./week-11-sequence/readme.md)

---

## Phase 3: The Transformer Revolution

### Week 12: [Transformer Architecture + Stage D Papers](./week-12-transformers/readme.md)

### Week 13-14: Pre-trained Language Models + Stage E Papers

**Stage E — Large Pretraining & Language Modelling:**

15. [**Devlin et al. (2018)** — BERT (bidirectional pretraining) **LANDMARK PAPER**](https://arxiv.org/abs/1810.04805)
    - **Mini-project:** Finetune BERT on sentiment or QA
    - **Why it matters:** Changed how we pretrain language models
    - **Foundation:** [Building LLMs from the Ground Up](https://www.youtube.com/watch?v=quh7z1q7-uc) - understand the transformer backbone shared by BERT

16. [**Radford et al. (2018)** — Improving Language Understanding by Generative Pre-Training/GPT-1 (autoregressive pretraining)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - **Mini-project:** Run a small autoregressive LM (GPT-style) on tiny dataset
    - **Foundation:** [Building LLMs from the Ground Up](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)

17. **Radford et al. / OpenAI (2019)** — Language Models are Unsupervised Multitask Learners/GPT-2 (scaling + sampling)
    - **Mini-project:** Fine-tune GPT-2 small for a domain (e.g., commit messages)
    - **Foundation:** [Building LLMs from the Ground Up](https://www.youtube.com/watch?v=quh7z1q7-uc)

18. [**Brown et al. (2020)** — GPT-3 (few-shot behaviour at scale) **LANDMARK PAPER**](https://arxiv.org/abs/2005.14165)
    - **Mini-project:** Train a small [nanoGPT](https://github.com/karpathy/nanoGPT) and experiment with in-context examples
    - **Why it matters:** Few-shot emergent behavior at huge scale
    - **Resource:** [GPT-3](https://paperswithcode.com/method/gpt-3)
    - **Foundation:** [Building LLMs from the Ground Up](https://www.youtube.com/watch?v=quh7z1q7-uc)

**Time Investment:** 35-40 hours


**AI Engineering Resources - LLMs:**

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ← **MUST READ**
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models)
- [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)
- [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)
- [How Transformer LLMs Work - Deeplearning.ai](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)
- [Building GPT from scratch - Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) ← **WATCH THIS**
- [Building LLMs from the Ground Up - Sebastian Raschka](https://www.youtube.com/watch?v=quh7z1q7-uc) ← **WATCH THIS** (foundational for GPT, BERT, and all transformer-based LLMs)
- [Build an LLM from Scratch Series - Sebastian Raschka](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11) ← **WATCH THIS SERIES**
  - [1: Set up your code environment](https://www.youtube.com/watch?v=yAcWnfsZhzo)
  - [2: Working with text data](https://www.youtube.com/watch?v=341Rb8fJxY0)
  - [3: Coding attention mechanisms](https://www.youtube.com/watch?v=-Ll8DtpNtvk)
  - [4: Implementing a GPT model from Scratch](https://www.youtube.com/watch?v=YSAkgEarBGE)
  - [6: Finetuning for Classification](https://www.youtube.com/watch?v=5PFXJYme4ik)
- [Developing an LLM: Building, Training, Finetuning](https://www.youtube.com/watch?v=kPGTx4wcm_w)
- [LLMs: A Journey Through Time and Architecture](https://www.youtube.com/watch?v=itIab9ZTAqk)
- [Finetuning Open-Source LLMs](https://www.youtube.com/watch?v=gs-IDg-FoIQ)
- [LLM Building Blocks & Transformer Alternatives](https://www.youtube.com/watch?v=lONyteDR4XE)
- [Pretraining and Finetuning LLMs from the Ground Up | SciPy 2024](https://www.youtube.com/watch?v=40C6dqomM8U)
- [Developing and Training LLMs From Scratch](https://www.youtube.com/watch?v=qL4JY6Y5pmA)
- [LLM Course - GitHub](https://github.com/mlabonne/llm-course)
- [LLM Course - Hugging Face](https://huggingface.co/learn/llm-course/chapter1/1)

**LLM Frameworks:**

- [LangChain](https://www.langchain.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Ollama](https://ollama.com/)
- [Instructor](https://python.useinstructor.com/)
- [Outlines](https://github.com/dottxt-ai/outlines)

**LLM APIs:**

- [OpenAI](https://platform.openai.com/docs/overview)
- [Anthropic](https://docs.anthropic.com/en/docs/overview)
- [Gemini - Google](https://ai.google.dev/gemini-api/docs)
- [Groq - Inference](https://groq.com/)

### Week 15: Scaling Laws + Stage F Papers

**Stage F — Why Scale, and How Far It Goes:**

19. [**Kaplan et al. (2020)** — "Scaling Laws for Neural Language Models" **LANDMARK PAPER**](https://arxiv.org/abs/2001.08361)
    - **Mini-project:** Run ablations on small models (change depth/width/data) and plot loss vs params
    - **Why it matters:** Predictable gains with scale; guides resource allocation

**Time Investment:** 20-25 hours

[GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
[LLM from scratch](https://github.com/rasbt/LLMs-from-scratch)
- [Building LLMs from the Ground Up - Sebastian Raschka](https://www.youtube.com/watch?v=quh7z1q7-uc)

**AI Engineering - Infrastructure (parallelize your training):**

- [Ray](https://arxiv.org/abs/1712.05889) ← **Use Ray for distributed training**
- [TensorFlow](https://arxiv.org/pdf/1605.08695)
- [FAISS library](https://arxiv.org/pdf/2401.08281)
- [Billion Scale Similarity Search: FAISS](https://arxiv.org/pdf/1702.08734)

---

## Phase 4: Large Language Models & Applications (3-4 weeks)

### Week 16-17: LLM Fundamentals & Fine-tuning

zero-shot → few-shot → CoT → then Fine-tuning

**What to Learn:**

- Transformers: Self-attention, positional embeddings
- Tokenization: Byte Pair Encoding (BPE), SentencePiece

**AI Engineering - Chain of Thought (improve LLM reasoning):**

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903) ← **MUST READ**
- [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/pdf/2502.03373)


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

**AI Engineering - Vector Databases (deploy in production):**

- [Milvus DB](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD21_Milvus.pdf)

**AI Engineering - Context Engineering (optimize prompts):**

- [DSPy](https://arxiv.org/pdf/2310.03714)
- [Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering](https://arxiv.org/pdf/2404.17723v1)
- [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)

**RAG Resources:**

- [Introduction to RAG - Coursera](https://www.coursera.org/projects/introduction-to-rag)
- [RAG Techniques - Github](https://github.com/NirDiamant/RAG_Techniques)
- [Generative AI for Beginners - Microsoft](https://github.com/microsoft/generative-ai-for-beginners)
- [Generative AI for Everyone - Coursera](https://www.coursera.org/learn/generative-ai-for-everyone)
- [The Building Blocks of Generative AI](https://shriftman.substack.com/p/the-building-blocks-of-generative)

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

**AI Engineering - Image Transformers:**

- [Image is 16x16 word](https://arxiv.org/pdf/2010.11929) (ViT) ← **READ THIS FIRST**
- [CLIP](https://arxiv.org/pdf/2103.00020)
- [deepseek image generation](https://arxiv.org/pdf/2501.17811)

**AI Engineering - Video Transformers:**

- [ViViT: A Video Vision Transformer](https://arxiv.org/pdf/2103.15691)
- [Joint Embedding abstractions with self-supervised video masks](https://arxiv.org/pdf/2404.08471)
- [Facebook VideoJAM ai gen](https://arxiv.org/pdf/2502.02492)

**AI Engineering - Reasoning (Scale inference compute):**

- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/pdf/2407.21787) ← **KEY PAPER**
- [Scale model test times is better than scaling parameters](https://arxiv.org/pdf/2408.03314)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/pdf/2412.06769)


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

- [Fine-Tuning Language Models with RHLF](https://arxiv.org/pdf/1909.08593)

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

**MCP Resources:**

- [MCP - Anthropic Guide](https://modelcontextprotocol.io/introduction)
- [Building AI Apps using MCP - Deeplearning.ai](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/)
- [MCP Course - Hugging Face](https://huggingface.co/learn/mcp-course/unit0/introduction)
- [Awesome MCP Servers - Github](https://github.com/punkpeye/awesome-mcp-servers)

**How to Learn:**

- **Courses:**
  - "Reinforcement Learning Specialization" by the University of Alberta (Coursera)
  - Tutorials from OpenAI Gym
- **Projects:**
  - Train an RL agent to solve a game environment like CartPole

**Time Investment:** 30-35 hours

**AI Engineering - Case Studies (build real systems):**

- [Automated Unit Test Improvement using Large Language Models at Meta](https://arxiv.org/pdf/2402.09171)
- [OpenAI o1 System Card](https://arxiv.org/pdf/2412.16720)
- [LLM-powered bug catchers](https://arxiv.org/pdf/2501.12862)
- [Chain-of-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)
- [Swiggy Search](https://bytes.swiggy.com/improving-search-relevance-in-hyperlocal-food-delivery-using-small-language-models-ecda2acc24e6)
- [Swarm by OpenAI](https://github.com/openai/swarm)
- [Netflix Foundation Models](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)
- [uber queryGPT](https://www.uber.com/en-IN/blog/query-gpt/)

**AI Engineering Resources - AI Agents:**

- [A Visual Guide to LLM Agents](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents)
- [Agents - Chip Huyen](https://huyenchip.com/2025/01/07/agents.html)
- [AI Agents Course - Hugging Face](https://huggingface.co/learn/agents-course/)
- [Building AI Browser Agents - Deeplearning.ai](https://www.deeplearning.ai/short-courses/building-ai-browser-agents/)
- [GenAI Agents - Github](https://github.com/NirDiamant/GenAI_Agents)
- [AI Agents in Action, Second Edition - Book](https://www.manning.com/books/ai-agents-in-action-second-edition)

**LLM-based IDEs:**

- [Cursor](https://www.cursor.com/)
- [Windsurf](https://windsurf.com/editor)
- [GitHub Copilot](https://github.com/features/copilot)

**Agentic Coding Tools:**

- [Claude Code](https://code.claude.com/docs/en/overview)
- [Codex](https://openai.com/codex/)

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

- [FlashAttention](https://arxiv.org/pdf/2205.14135)
- [Multi Query Attention](https://arxiv.org/pdf/1911.02150) 
- [Grouped Query Attention](https://arxiv.org/pdf/2305.13245)

**AI Engineering - Vectorization:**

- [IMAGEBIND: One Embedding Space To Bind Them All](https://arxiv.org/pdf/2305.05665)
- [SONAR: Sentence-Level Multimodal and Language-Agnostic Representations](https://arxiv.org/pdf/2308.11466)
- [Facebook Large Concept Models](https://arxiv.org/pdf/2412.08821v2)

- [DeepSeek R1](https://arxiv.org/pdf/2501.12948v1)
- [A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods](https://arxiv.org/pdf/2502.01618)
- [Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/pdf/2502.05171)

**AI Engineering - VideoRoPE (position embeddings):**

- [VideoRoPE: Rotary Position Embedding](https://arxiv.org/pdf/2502.05173)


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

**AI Engineering - Mixture of Experts (scale to trillion params):**

- [Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)

**AI Engineering - SSMs (alternatives to Transformers):**

- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/pdf/2305.13048)
- [Mamba](https://arxiv.org/pdf/2312.00752) ← **MUST READ**
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/pdf/2405.21060)
- [Distilling Transformers to SSMs](https://arxiv.org/pdf/2408.10189)
- [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/pdf/2410.10254)
- [Think Slow, Fast](https://arxiv.org/pdf/2502.20339)

**AI Engineering - Optimizations (make it fast):**

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608)
- [ByteDance 1.58](https://arxiv.org/pdf/2412.18653v1)
- [Transformer Square](https://arxiv.org/pdf/2501.06252)
- [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/pdf/2501.09732)
- [1b outperforms 405b](https://arxiv.org/pdf/2502.06703)
- [Speculative Decoding](https://arxiv.org/pdf/2211.17192)

**AI Engineering - Distillation (compress models):**

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)
- [BYOL - Distilled Architecture](https://arxiv.org/pdf/2006.07733)
- [DINO](https://arxiv.org/pdf/2104.14294)

**AI Engineering - Advanced Reasoning (build smarter models):**

- [Transformer Reasoning Capabilities](https://arxiv.org/pdf/2405.18512)
- [DeepSeek R1](https://arxiv.org/pdf/2501.12948v1)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/pdf/2407.21787)

**AI Engineering - Competition Models (win benchmarks):**

- [Google Math Olympiad 2](https://arxiv.org/pdf/2502.03544)
- [Competitive Programming with Large Reasoning Models](https://arxiv.org/pdf/2502.06807)
- [Google Math Olympiad 1](https://www.nature.com/articles/s41586-023-06747-5)

**AI Engineering - Titans & Latest Architecture:**

- [Google Titans outperform Transformers](https://arxiv.org/pdf/2501.00663)

**AI Engineering - Hype (understand the limits):**

- [Can AI be made to think critically](https://arxiv.org/pdf/2501.04682)
- [Evolving Deeper LLM Thinking](https://arxiv.org/pdf/2501.09891)
- [LLMs Can Easily Learn to Reason from Demonstrations Structure](https://arxiv.org/pdf/2502.07374)
- [Separating communication from intelligence](https://arxiv.org/pdf/2301.06627)
- [Language is not intelligence](https://gwern.net/doc/psychology/linguistics/2024-fedorenko.pdf)

**AI Engineering - Infrastructure (cutting edge):**

- [Deepseek filesystem](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)

**AI Engineering Resources - MLOps & Deployment:**

- [ML in Production - Coursera](https://www.coursera.org/learn/introduction-to-machine-learning-in-production)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [ML System Design - Stanford](https://stanford-cs329s.github.io/syllabus.html)

**MLOps Tools:**

- [Streamlit](https://streamlit.io/)
- [MLflow](https://mlflow.org/docs/latest/index.html)

**AI Engineering Resources - Guides:**

- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Anthropic courses](https://github.com/anthropics/courses/tree/master)

**YouTube Channels:**

- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) ← **MUST WATCH**
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)

**Books:**

- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Deep Learning - Ian Goodfellow](https://www.deeplearningbook.org/)
- [Deep Learning with Python](https://www.amazon.in/Deep-Learning-Python-Francois-Chollet/dp/1617294438/)
- [Why Machines Learn](https://www.amazon.com/Why-Machines-Learn-Elegant-Behind/dp/0593185749)
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)
- [Build a LLM from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Prompt Engineering for LLMs](https://www.oreilly.com/library/view/prompt-engineering-for/9781098156145/)
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [Build a Multi-Agent System (from Scratch)](https://www.manning.com/books/build-a-multi-agent-system-from-scratch)
- [Build a Reasoning Model (From Scratch)](https://www.manning.com/books/build-a-reasoning-model-from-scratch)
- [Build an AI Agent (From Scratch)](https://www.manning.com/books/build-an-ai-agent-from-scratch)
- [Build an LLM Application (from Scratch)](https://www.manning.com/books/build-llm-applications-from-scratch)
- [LLMs in Production](https://www.manning.com/books/llms-in-production)

**Other Resources:**

- [Papers with Code](https://paperswithcode.com/)
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Awesome LLM Apps - GitHub](https://github.com/Shubhamsaboo/awesome-llm-apps)

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

### Must-Read AI Papers (Landmarks)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) ← **START HERE**
- [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661)
- [GPT: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)

---

## Quick Reference: AI Engineering Priority Papers

### MUST READ First (before anything else):
1. **Attention is All You Need** - Core Transformer
2. **Byte-pair Encoding** - Tokenization
3. **FlashAttention** - Speed optimization
4. **Chain-of-Thought Prompting** - Reasoning
5. **Mamba** - SSM alternative

### Build These Projects:
1. Mini-Transformer from scratch (Week 12)
2. BPE tokenizer (Week 12)
3. RAG pipeline with FAISS (Week 18-19)
4. ReAct agent (Week 26)
5. Fine-tuned model with RLHF (Week 24-25)

### Production Skills:
- Ray for distributed training
- FAISS/Milvus for vector search
- FastAPI for serving
- LangChain/LangGraph for agents
