
# Crime & Punishment — Pinecone RAG + Email (Ollama + Gmail)

Retrieve-augmented QA over *Crime and Punishment* using **Pinecone** for vector search, **Ollama** (local) for embeddings + generation, and **Gmail** to email the answer.

## What it does

1. **Ingest**: chunk the novel, embed chunks (768-d) with `nomic-embed-text`, and upsert into Pinecone.
2. **Answer**: embed the user’s question, retrieve top-k chunks, and have `qwen2:7b` answer **only** from that context.
3. **Email**: send the answer + short citations to your inbox.

## Stack

* Python, LangChain 0.2+
* `langchain-ollama` (Chat + Embeddings over Ollama)
* Pinecone serverless
* Gmail API (OAuth Desktop) — **each user must bring their own Google Cloud project**
* tqdm (progress bars), python-dotenv

---

## Repository layout

```
.
├─ data/
│  └─ crime_and_punishment.txt          # you provide this file (not committed)
├─ ask_and_email.py                     # ask a question + email the answer
├─ ingest.py                            # chunk -> embed -> upsert
├─ rag.py                               # retrieval + generation
├─ emailer.py                           # Gmail API (OAuth) sender
├─ pinecone_utils.py                    # create/index/query helpers
├─ requirements.txt
├─ .env.example
└─ .gitignore
```

---

## Requirements

* Python 3.10–3.12
* [Ollama](https://ollama.com/) running locally
* Pinecone account + API key
* **Gmail OAuth:** each user creates their **own** Google Cloud project + OAuth client (see below)

---

## Install

```bash
pip install -r requirements.txt

# models for Ollama
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

---

## Configure environment

Create your `.env` from the template:

```bash
cp .env.example .env
```

Fill:

```
PINECONE_API_KEY=pcn-xxxxxxxxxxxxxxxx
PINECONE_INDEX=candp
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Optional defaults for ask_and_email.py
RECIPIENT_EMAIL=you@example.com
SENDER_LABEL=Crime&Punishment RAG
```

> Never commit `.env`, `credentials.json`, or `token.json`. They’re ignored by `.gitignore`.

---

## Provide the data

Place a plain-text copy of the novel at:

```
data/crime_and_punishment.txt
```

(Project Gutenberg has public-domain text.)

---

## Ingest (chunk → embed → upsert)

```bash
python ingest.py
```

You’ll see progress bars for embedding and upserting, then Pinecone index stats like:

```
namespaces.c_and_p.vector_count: 2698
dimension: 768 (cosine)
```

---

## Ask & email

```bash
# first arg: the question; second arg: recipient (optional if set in .env)
python ask_and_email.py "Why did Raskolnikov visit Sonya?" you@example.com
```

The script prints the answer + brief citations and emails the same content.

---

## Email delivery (OAuth — **each user needs their own**)

Every developer uses **their own** Google Cloud project and OAuth client. Do **not** share credentials/tokens.

1. **Create/select a project** in Google Cloud Console.
2. **APIs & Services → Library** → enable **Gmail API**.
3. **APIs & Services → OAuth consent screen → Audience**

   * Publishing status: **Testing**
   * Under **Test users**, add the Gmail account you will use.
4. **APIs & Services → Credentials → Create Credentials → OAuth client ID**

   * Application type: **Desktop app**
   * Download and save as **`credentials.json`** in the repo root (do **not** commit).
5. First run opens a browser → allow the app → **`token.json`** is created locally (also **do not** commit).

Re-auth: delete `token.json` and run again.
**403 “Access blocked”**: make sure your Gmail is added as a **Test user** in the same project that produced `credentials.json`.

> **Headless servers**: in `emailer.py`, change `flow.run_local_server(...)` to `flow.run_console()`.

### Optional: SMTP fallback (no OAuth)

If you prefer, set a Gmail **App Password** (requires 2-Step Verification) and use SMTP. Add to `.env.example`:

```
GMAIL_ADDRESS=
GMAIL_APP_PASSWORD=
```

Then call the SMTP sender (if you include `smtp_emailer.py`) instead of the Gmail API.

---

## Tips & tuning

* **Retrieval**: change `k` (top-k) and context length in `rag.py`.
* **Hallucination guard**: add a score threshold (e.g., require top score ≥ 0.78; otherwise say “I don’t know”).
* **Models**: switch `qwen2:7b` to another local model by changing the name in `rag.py`.
* **Deprecation warnings**: this repo uses `langchain-ollama`; if you see community deprecation warnings, update imports to:

  ```python
  from langchain_ollama import OllamaEmbeddings, ChatOllama
  ```

---

## Troubleshooting

* **Ollama not running / model not found**
  Run `ollama serve` and `ollama pull <model>`.
* **Pinecone “dimension mismatch”**
  Index dimension must be **768** for `nomic-embed-text`. Recreate if you switch embedding models.
* **Gmail 403 in consent**
  Add your Gmail as a **Test user** (Consent screen → Audience). Delete `token.json` and retry.
* **Windows CRLF warnings**
  Harmless. Add `.gitattributes` with `* text=auto` if you want normalization.

---

## Security & repo hygiene

* `.gitignore` already excludes: `.env`, `credentials.json`, `token.json`, `.venv/`, `__pycache__/`, IDE folders.
* Never commit secrets or OAuth tokens. If you accidentally did, rotate keys and purge history if needed.

---

