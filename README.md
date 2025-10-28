# Crime & Punishment — Pinecone RAG + Email (Ollama + Gmail API)

> Retrieve-augmented question-answering over *Crime and Punishment* using Pinecone for search, Ollama for a local LLM + embeddings, and the Gmail API for sending the answer by email.

This project demonstrates a full RAG (Retrieval-Augmented Generation) pipeline:
1.  It **ingests** a large text (Dostoevsky's *Crime and Punishment*).
2.  It **chunks and embeds** the text using a local embeddings model (Ollama's `nomic-embed-text`).
3.  It **stores** these embeddings in a Pinecone vector database.
4.  When you **ask a question**, it embeds the query, searches Pinecone for relevant passages, and passes the context to a local LLM (Ollama's `qwen2:7b`).
5.  It **delivers** the generated answer to your console and sends it to an email address using the Gmail API.

---

## 🛠️ Tech Stack

* **LLM:** Ollama (`qwen2:7b`)
* **Embeddings:** Ollama (`nomic-embed-text`, 768-dim)
* **Vector Database:** Pinecone
* **Email:** Google Gmail API (via OAuth 2.0)
* **Orchestration:** Python

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10–3.12
* [Ollama](https://ollama.com/) running locally
* A [Pinecone](https://www.pinecone.io/) account and API key
* A [Google Cloud](https://console.cloud.google.com/) project (see Step 3 for details)

### 2. Installation & Model Setup

1.  Clone this repository and install the Python dependencies:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    pip install -r requirements.txt
    ```

2.  Pull the required Ollama models:
    ```bash
    ollama pull qwen2:7b
    ollama pull nomic-embed-text
    ```

### 3. Configuration

#### A. Environment Variables (`.env`)

1.  Create your local environment file from the template:
    ```bash
    cp .env.example .env
    ```

2.  Fill in the `.env` file with your credentials. The Pinecone index name can be anything, but the dimensions must match your embedding model (768 for `nomic-embed-text`).

    ```ini
    # Pinecone
    PINECONE_API_KEY=pcn-...
    PINECONE_INDEX=candp
    PINECONE_CLOUD=aws
    PINECONE_REGION=us-east-1

    # Email
    RECIPIENT_EMAIL=you@example.com
    SENDER_LABEL="Crime&Punishment RAG"
    ```

#### B. Book Data

Place the full text of *Crime and Punishment* in the `data/` directory.

* **Required file path:** `data/crime_and_punishment.txt`
* You can download a plain text version from sources like [Project Gutenberg](https://www.gutenberg.org/ebooks/2554).

#### C. Google (Gmail API) Authentication

> **Note:** Each developer must use their own Google Cloud project. Do not share `credentials.json` or `token.json` files.

1.  **Create/Select Project:** Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project or select an existing one.

2.  **Enable API:**
    * Navigate to *APIs & Services* → *Library*.
    * Search for and enable the **Gmail API**.

3.  **Configure Consent Screen:**
    * Go to *APIs & Services* → *OAuth consent screen*.
    * **Publishing status:** Set to **Testing**.
    * **Test users:** Click "Add users" and add the Gmail account you will be sending emails *from* (the account that will authenticate).

4.  **Create Credentials:**
    * Go to *APIs & Services* → *Credentials*.
    * Click *Create Credentials* → *OAuth client ID*.
    * **Application type:** Select **Desktop app**.
    * Give it a name (e.g., "RAG Client").

5.  **Download Credentials:**
    * After creation, click the "Download JSON" icon for the new client ID.
    * Rename the downloaded file to `credentials.json` and place it in the **root of this repository**.

---

## Usage

### 1. Ingest Data

Run the ingest script to chunk the book, generate embeddings, and upload them to your Pinecone index.

```bash
python ingest.py
