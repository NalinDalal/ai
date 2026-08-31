# AI Curriculum

## Timeline : 8 months. September to April.

**Rule:** curriculum is guidance, not a checklist. We progress by building and breaking real systems, not by completing week numbers.

---

### Phase 0: Foundations (Weeks 1–2)

Python, NumPy, Pandas, matplotlib, linear algebra, calculus, statistics, probability.

→ [python-basics](https://github.com/NalinDalal/python-basics)

---

### Phase 1: Classical ML (Weeks 3–6)

Linear regression, logistic regression, trees, SVMs, clustering, PCA, autoencoders, RL basics.

→ [machine-learning](https://github.com/NalinDalal/machine-learning)

**Resources:**

- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Why Machines Learn](https://www.amazon.com/Why-Machines-Learn-Elegant-Behind/dp/0593185749/)
- [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)

---

### Phase 2: Deep Learning (Weeks 7–12)

**Week 7:** Perceptron, backpropagation, neural network fundamentals
→ [neural-networks](https://github.com/NalinDalal/neural-networks)

**Key papers:**

- McCulloch & Pitts (1943) — A Logical Calculus of the Ideas Immanent in Nervous Activity
- Rosenblatt (1958) — The Perceptron
- Rumelhart, Hinton & Williams (1986) — Learning representations by back-propagating errors

**Resources:**

- [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [Deep Learning - Ian Goodfellow](https://www.deeplearningbook.org/)
- [Deep Learning with Python - François Chollet](https://www.amazon.in/Deep-Learning-Python-Francois-Chollet/dp/1617294438/)

---

**Week 8:** CNNs, AlexNet, ResNet, object detection
→ [cnn-architectures](https://github.com/NalinDalal/cnn-architectures)

**Key papers:**

- Krizhevsky et al. (2012) — AlexNet
- Simonyan & Zisserman (2015) — VGG
- He et al. (2016) — ResNet
- Redmon et al. (2016) — YOLO

---

**Week 9:** Deep learning, initialization, regularization, optimization
→ [week-9-deep-learning](https://github.com/NalinDalal/week-9-deep-learning)

**Key papers:**

- Kingma & Ba (2015) — Adam
- Ioffe & Szegedy (2015) — BatchNorm
- He et al. (2015) — Delving Deep into Rectifiers
- Srivastava et al. (2014) — Dropout

---

**Week 10:** RNNs, LSTM from scratch, BPTT, gradient clipping
→ [rnn](https://github.com/NalinDalal/rnn)

**Key papers:**

- Hochreiter & Schmidhuber (1997) — LSTM

---

**Week 11:** Character-level LSTM, truncated BPTT, temperature sampling
→ [rnn-karpathy](https://github.com/NalinDalal/rnn-karpathy)

**Key papers:**

- Karpathy (2015) — The Unreasonable Effectiveness of Recurrent Neural Networks

---

**Week 12:** LSTM language modeling, Word2Vec, t-SNE visualizations
→ [lstm-word2Vec](https://github.com/NalinDalal/lstm-word2Vec)

---

### Phase 3: Sequence Models (Weeks 13–14)

Seq2Seq, NMT, attention, English→French translator.

→ [Sequence-Models](https://github.com/NalinDalal/Sequence-Models)

**Key papers:**

- Sutskever et al. (2014) — Seq2Seq
- Bahdanau et al. (2015) — NMT with attention

---

### Phase 4: Transformers (Weeks 15–16)

Transformer architecture, multi-head attention, ViT, BPE tokenization, Switch Transformer.

→ [transformers](https://github.com/NalinDalal/transformers)

**Key papers:**

- Vaswani et al. (2017) — Attention Is All You Need ← **START HERE**
- Devlin et al. (2018) — BERT
- Brown et al. (2020) — GPT-3
- Dosovitskiy et al. (2021) — Vision Transformer
- Fedus et al. (2022) — Switch Transformer
- BPE: Sennrich et al. (2016)
- Google (2025) — Titans outperform Transformers

**Resources:**

- [Building GPT from scratch - Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) ← **WATCH THIS**
- [Building LLMs from the Ground Up - Sebastian Raschka](https://www.youtube.com/watch?v=quh7z1q7-uc)
- [Build a LLM from Scratch (book)](https://www.manning.com/books/build-a-large-language-model-from-scratch)

---

### Phase 5: LLMs & Applications (Weeks 17–20)

Zero-shot → few-shot → CoT → fine-tuning → RAG → agents.

**Key topics:**

- Chain-of-Thought prompting
- Fine-tuning BERT/GPT
- RAG pipelines, vector databases
- LangChain, LlamaIndex
- Model Context Protocol (MCP)

**Key papers:**

- Brown et al. (2020) — GPT-3
- Devlin et al. (2018) — BERT
- Lewis et al. (2020) — RAG
- Yao et al. (2022) — ReAct

**Time Investment:** 35–40 hours

---

### Phase 6: Generative AI (Weeks 21–22)

VAEs, GANs, diffusion models, image/video transformers.

**Key papers:**

- Kingma & Welling (2013) — VAE
- Goodfellow et al. (2014) — GANs
- Ho et al. (2020) — DDPM
- Rombach et al. (2022) — LDM

**Time Investment:** 25–30 hours

---

### Phase 7: Alignment & Safety (Weeks 23–24)

RLHF, instruction tuning, constitutional AI.

**Key papers:**

- Christiano et al. (2017) — RL from Human Preferences
- Ouyang et al. (2022) — InstructGPT
- Anthropic (2023) — Constitutional AI

**Time Investment:** 25–30 hours

---

### Phase 8: Agents & Multi-Agent Systems (Weeks 25–28)

ReAct, tool use, planner-executor, multi-agent frameworks, MCP.

**Key topics:**

- OpenAI Swarm, LangGraph, CrewAI, Phidata
- Autonomous coding agents
- API deployment with FastAPI

**Frameworks & Tools:**

- [LangChain](https://www.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Ollama](https://ollama.com/)
- [OpenAI Swarm](https://github.com/openai/swarm)
- [CrewAI](https://www.crewai.io/)
- [Phidata](https://www.phidata.com/)

**Resources:**

- [AI Agents in Action, Second Edition (book)](https://www.manning.com/books/ai-agents-in-action-second-edition)
- [Build an AI Agent (From Scratch)](https://www.manning.com/books/build-an-ai-agent-from-scratch)
- [Building AI Agents with LLMs, RAG, and Knowledge Graphs](https://www.manning.com/books/build-a-multi-agent-system-from-scratch)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [MCP Course - Hugging Face](https://huggingface.co/learn/mcp-course/unit0/introduction)
- [AI Engineer Summit 2025](https://www.youtube.com/watch?v=D7BzTxVVMuw)
- [A Visual Guide to LLM Agents](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents)
- [Agents - Chip Huyen](https://huyenchip.com/2025/01/07/agents.html)
- [AI Agents for Beginners - GitHub](https://github.com/microsoft/ai-agents-for-beginners)
- [GenAI Agents - GitHub](https://github.com/NirDiamant/GenAI_Agents)

- [MCP - Official Documentation](https://modelcontextprotocol.io/introduction)
- [Awesome MCP Servers - GitHub](https://github.com/punkpeye/awesome-mcp-servers)

**Time Investment:** 40–45 hours

---

### Phase 9: Cutting-Edge & Production (Weeks 29–32)

Mixture of Experts, SSMs/Mamba, FlashAttention, speculative decoding, production deployment.

**Key papers:**

- Fedus et al. (2022) — Switch Transformer
- Gu & Dao (2023) — Mamba
- Dao et al. (2022) — FlashAttention-2
- Hu et al. (2024) — LoLCATs
- DeepSeek (2025) — DeepSeek R1

**Time Investment:** 30–35 hours

---

## Project Ideas

See [project.md](./project.md) for detailed project briefs and future plans.

## Paper Reading Tracker

See [paper-read.md](./paper-read.md) for current reading progress.
