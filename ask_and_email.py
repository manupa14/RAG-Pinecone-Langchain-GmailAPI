import os, sys, textwrap
from dotenv import load_dotenv
from rag import answer
from emailer import send_email

load_dotenv()
DEFAULT_TO = os.getenv("RECIPIENT_EMAIL")
SENDER_LABEL = os.getenv("SENDER_LABEL", "C&P RAG")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask_and_email.py \"<your question>\" [to_email]")
        sys.exit(1)
    question = sys.argv[1]
    to_email = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TO
    if not to_email:
        print("Provide a recipient email as arg or set RECIPIENT_EMAIL in .env")
        sys.exit(2)

    res = answer(question)
    ans = res["answer"]
    cites = "\n".join(f"- {c}" for c in res["citations"])

    email_body = textwrap.dedent(f"""
    Q: {question}

    A:
    {ans}

    ---
    Citations:
    {cites if cites else "(none)"}
    """)

    # Print to console
    print(email_body)

    # Email it
    send_email(
        to_email,
        subject=f"[C&P RAG] Answer to: {question}",
        body=email_body,
        sender_label=SENDER_LABEL
    )
    print(f"\n✔ Sent to {to_email}")

if __name__ == "__main__":
    main()
