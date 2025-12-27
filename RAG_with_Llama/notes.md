# What a “Large Language Model” Really Is

A model like **LLaMA / Mistral / Phi** is just a large collection of numbers (parameters).

Example: **LLaMA-2-7B**
- ~7 billion parameters
- Stored as tensors
- Each parameter occupies memory depending on precision

## Memory usage (approx)

| Precision | Memory |
|--------|--------|
| FP32 | ~28 GB |
| FP16 / BF16 | ~14 GB |
| INT8 | ~7 GB |
| INT4 | ~3.5 GB |

Extra memory is needed for:
- Attention KV cache
- Activations
- Optimizer states (training only)


# Inference vs Fine-Tuning

## Inference
- Read-only model
- Forward pass only
- Used for chat / RAG / generation

Memory needs:
- Model weights
- KV cache

## Fine-Tuning
- Forward + backward pass
- Updates parameters

Memory needs:
- Weights
- Gradients
- Optimizer states
- Activations

> Fine-tuning requires **4–8× more memory** than inference.


# Where the Model Lives

All usage methods fall into one of these:

1. **Remote APIs** (OpenAI, Gemini, Claude)
2. **Local inference** (Hugging Face, Ollama)
3. **Managed cloud models** (HF Endpoints, SageMaker)

Each option trades:
- Control
- Cost
- Performance
- Privacy


# Fully Hosted APIs

Examples:
- OpenAI
- Google Gemini
- Anthropic Claude
- Groq
- Hugging Face Inference API

## What you control
- Prompt
- Temperature
- Max tokens

## What you do NOT control
- Model weights
- Architecture
- Tokenizer
- Training data

## Fine-tuning
❌ No real weight updates  
⚠️ Sometimes instruction tuning only

Best for:
- Rapid prototyping
- Production apps


# Hugging Face Local Inference

## Non-quantized (FP16 / BF16)
- Best quality
- High VRAM usage
- Slow startup

## Quantized (INT8 / INT4)
- Lower memory
- Faster inference
- Small quality loss

### Popular formats
| Format | Used by |
|-----|------|
| INT8 | bitsandbytes |
| NF4 | QLoRA |
| GPTQ | ExLlama |
| AWQ | Fast kernels |
| GGUF | llama.cpp |


# Ollama & llama.cpp

Ollama uses **GGUF models** with custom inference engines.

## Why it works on laptops
- Heavy quantization
- Efficient KV cache
- CPU / GPU / Apple Metal support

## Capabilities
- Inference
- System prompts
- RAG

## Limitations
❌ No training  
❌ No LoRA fine-tuning  

Best for:
- Local experimentation
- Demos


# Full Fine-Tuning

Means updating **all parameters**.

## Hardware needs
| Model | GPUs |
|----|----|
| 7B | 4×A100 |
| 13B | 8×A100 |
| 70B | Cluster |

## Memory cost
Weights + Gradients + Optimizer ≈ **6–8× model size**

Used only by:
- Meta
- OpenAI
- Google


# Parameter-Efficient Fine-Tuning (PEFT)

Instead of training everything, train small components.

## LoRA
- Adds low-rank matrices
- Base model frozen
- Very memory efficient

## QLoRA (Industry Standard)
- Base model: 4-bit
- LoRA adapters: FP16

| Model | GPU |
|----|----|
| 7B | 16GB |
| 13B | 24GB |

Quality ≈ **98% of full fine-tuning**


# Other PEFT Methods

| Method | Idea |
|----|----|
| Prefix Tuning | Virtual tokens |
| Prompt Tuning | Soft prompts |
| IA³ | Activation scaling |
| Adapters | Insert trainable layers |

LoRA dominates because:
- Simple
- Mergeable
- Framework-agnostic


# Serving Your Own Model

## Popular inference engines
| Engine | Use |
|----|----|
| vLLM | High throughput |
| TGI | HF official |
| llama.cpp | CPU |
| Ollama | Local dev |
| ExLlama | Fast GPTQ |

## Architecture
Client → API → Inference Engine → GPU


# RAG vs Fine-Tuning

## Why RAG dominates production
- No retraining
- Cheap
- Always fresh data
- Debuggable

## RAG Flow
Query → Retrieve docs → Inject context → Generate answer

> 80–90% of real systems use **RAG**, not fine-tuning


# Hybrid & Advanced Approaches

## RAG + LoRA
- LoRA for style/behavior
- RAG for facts

## Other techniques
- Mixture of Experts (Mixtral)
- Speculative decoding (Groq)
- Distillation (large → small)


# Mental Model Summary
```
Prompt
  ↓
LangChain
  ↓
┌───────────┬───────────┬────────────┐
│ Ollama    │ HF Local  │ Hosted API │
│ (GGUF)    │ (LoRA)    │ (BlackBox) │
└───────────┴───────────┴────────────┘
```
RAG → default  
LoRA → behavior  
Full fine-tuning → rarely


1️⃣ Memory Diagrams (Inference vs Full FT vs QLoRA)
🔹 A) Inference (FP16, no training)
```
GPU VRAM
┌──────────────────────────────┐
│ Model Weights (FP16) ~14 GB  │
├──────────────────────────────┤
│ KV Cache (depends on tokens) │
├──────────────────────────────┤
│ Activations (minimal)        │
└──────────────────────────────┘
```

✔️ Fast
✔️ Cheap
❌ No learning

🔹 B) Full Fine-Tuning (Why it’s impossible for you)
```
GPU VRAM
┌──────────────────────────────┐
│ Model Weights (FP16)         │
├──────────────────────────────┤
│ Gradients (FP16)             │
├──────────────────────────────┤
│ Optimizer States (Adam x2)   │
├──────────────────────────────┤
│ Activations (HUGE)           │
└──────────────────────────────┘
```

💥 6–8× model size
💥 7B ≈ 80–120 GB VRAM

❌ Practically dead outside Big Tech

🔹 C) QLoRA (What everyone actually uses)
```
GPU VRAM
┌──────────────────────────────┐
│ Base Model (4-bit NF4) ~3GB  │  ← frozen
├──────────────────────────────┤
│ LoRA Adapters (FP16) ~200MB │  ← trainable
├──────────────────────────────┤
│ Activations (paged)          │
└──────────────────────────────┘
```

✔️ Fits on 16–24GB GPU
✔️ Near full-FT quality
✔️ Mergeable adapters

This is the industry standard

2️⃣ Exact QLoRA Training Code (REAL, MODERN)

This is not tutorial code — this is what people actually run.

🔹 Install
```python
pip install -U transformers accelerate peft bitsandbytes datasets trl
```
🔹 Load Quantized Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
```
🔹 Attach LoRA (THIS is the magic)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Output:

Trainable params: ~5M
Total params: ~7B

🔹 Train (TRL SFTTrainer)
```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    packing=True,
    args={
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "fp16": False,
        "bf16": True,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 500,
        "output_dir": "./qlora-out",
        "optim": "paged_adamw_8bit",
    },
)

trainer.train()
```

🔥 This runs on a single 24GB GPU
```
3️⃣ vLLM vs Ollama (Real Comparison)
Feature	vLLM	Ollama
Purpose	Production serving	Local dev
Throughput	⭐⭐⭐⭐⭐	⭐⭐
GPU utilization	Excellent	Moderate
Batching	Continuous	Limited
RAG-ready	Yes	Yes
Fine-tuning	No	No
Model formats	HF	GGUF
Used by startups	YES	Rare
```
🔹 Why vLLM is dominant
```
Requests
   ↓
Continuous Batching
   ↓
PagedAttention
   ↓
GPU stays 90–95% busy
```

Ollama:
```
Request → Generate → Idle → Next
```

✔️ Ollama = laptop tool
✔️ vLLM = revenue-generating infra

4️⃣ Why Fine-Tuning FAILS for RAG Use Cases
❌ Myth

“If I fine-tune the model on my documents, I don’t need RAG”

Reality:
## 🔹 1) Hallucinations increase

Model generalizes, not retrieves.

Document says:
"Policy updated on March 2024"

Model learns:
"Policies are usually updated annually"


❌ Wrong answer

## 🔹 2) Knowledge freezes instantly
```
Fine-tuned today
↓
Policy changes tomorrow
↓
Model is already wrong
```

RAG = real-time
Fine-tuning = static

## 🔹 3) Token inefficiency

Model stores facts in weights

Retrieval stores facts in documents

Weights ≠ database

## 🔹 When fine-tuning DOES help
```
Use case	Method
Tone	LoRA
Reasoning style	LoRA
Domain language	LoRA
Facts	❌ RAG
```
Correct architecture:

Base LLM
 + LoRA (style)
 + RAG (facts)

# 5️⃣ Real Infrastructure Used by Startups (No BS)
## 🔹 Typical Series A Stack
```
Client
  ↓
FastAPI / Node
  ↓
LangChain / Custom RAG
  ↓
Vector DB (Qdrant / Pinecone)
  ↓
vLLM Inference Server
  ↓
A10 / L40 / A100 GPU
```
## 🔹 Models in production
```
Use	Model
General chat	Llama 3 / Mistral
Coding	DeepSeek / CodeLlama
Cheap inference	Phi / Gemma
RAG	Any 7B–13B
```
## 🔹 Cost reality (monthly)
```
Setup	Cost
OpenAI API	$$$
Self-hosted A10	~$800
Self-hosted L40	~$2,000
A100	$$$$

Startups move off APIs once:

Latency matters

Token cost explodes

Data privacy required
```
# 6️⃣ Final Mental Model (Burn This In)
```
❌ Fine-tune to learn facts
✅ RAG to retrieve facts

❌ Ollama for production
✅ vLLM for production

❌ Full fine-tuning
✅ QLoRA

❌ Store docs in weights
✅ Store docs in vector DB
```


# 2️⃣ QLoRA Training — Exact Production-Grade Code

## 🔹 Install
pip install -U transformers accelerate peft bitsandbytes datasets trl

---

## 🔹 Load Quantized Base Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
```


# 2️⃣ QLoRA Training — Exact Production-Grade Code

## 🔹 Install
```
pip install -U transformers accelerate peft bitsandbytes datasets trl
```
---

## 🔹 Attach LoRA Adapters
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
---

## 🔹 Train (TRL SFTTrainer)

```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    packing=True,
    args={
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "bf16": True,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 500,
        "output_dir": "./qlora-out",
        "optim": "paged_adamw_8bit",
    },
)

trainer.train()
```

---


# 3️⃣ vLLM vs Ollama — Real-World Comparison

| Feature | vLLM | Ollama |
|------|------|-------|
| Purpose | Production serving | Local dev |
| Throughput | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| GPU utilization | Excellent | Moderate |
| Batching | Continuous | Limited |
| RAG-ready | Yes | Yes |
| Fine-tuning | No | No |
| Model formats | HF | GGUF |
| Used by startups | YES | Rare |

---

## 🔹 Why vLLM Dominates

Requests  
↓  
Continuous batching  
↓  
PagedAttention  
↓  
GPU stays ~90–95% busy  

---

## 🔹 Ollama Execution Model

Request → Generate → Idle → Next  

✔️ Ollama = laptop experimentation  
✔️ vLLM = production revenue engine  


# 4️⃣ Why Fine-Tuning Fails for RAG Use Cases

## ❌ Myth
"If I fine-tune on my documents, I don’t need RAG"

---

## 🔹 Reality

### 1) Hallucinations Increase
Documents:
"Policy updated March 2024"

Model learns:
"Policies usually update annually" ❌

---

### 2) Knowledge Freezes
Fine-tune today  
↓  
Policy changes tomorrow  
↓  
Model is instantly wrong  

---

### 3) Token Inefficiency
- Weights ≠ database  
- Retrieval ≠ memorization  

---

## 🔹 When Fine-Tuning Actually Helps

| Use Case | Method |
|-------|-------|
| Tone | LoRA |
| Reasoning style | LoRA |
| Domain phrasing | LoRA |
| Facts | ❌ RAG |

---

## ✅ Correct Architecture

Base LLM  
+ LoRA (behavior/style)  
+ RAG (facts & freshness)  

---

# 5️⃣ Real Infrastructure Used by Startups

## 🔹 Typical Series-A Stack

Client  
↓  
FastAPI / Node  
↓  
LangChain / Custom RAG  
↓  
Vector DB (Qdrant / Pinecone)  
↓  
vLLM Inference Server  
↓  
A10 / L40 / A100 GPU  

---

## 🔹 Models in Production

| Use | Model |
|----|------|
| Chat | Llama 3 / Mistral |
| Coding | DeepSeek / CodeLlama |
| Cheap inference | Phi / Gemma |
| RAG | Any 7B–13B |

---

## 🔹 Monthly Cost Reality

| Setup | Cost |
|----|-----|
| OpenAI API | $$$ |
| A10 (self-hosted) | ~$800 |
| L40 | ~$2,000 |
| A100 | $$$$ |

Teams move off APIs when:
- Latency matters  
- Token cost explodes  
- Privacy is required  


# 6️⃣ Final Mental Model (Non-Negotiable)

❌ Fine-tune to store facts  
✅ Use RAG to retrieve facts  

❌ Ollama for production  
✅ vLLM for production  

❌ Full fine-tuning  
✅ QLoRA  

❌ Knowledge in weights  
✅ Knowledge in vector DB  

**RAG is default.  
Fine-tuning is optional.  
Instruction models are the base.**


