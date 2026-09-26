"""Optional OpenAI SDK example. Install `openai` separately and set OPENAI_API_KEY.

This is an example adapter, not a bundled dependency or an endorsement.
Only use it with sources you may send to a third-party service.
"""
from newsrag import NewsroomRAG


def make_rag(client):
    def embed(texts: list[str]) -> list[list[float]]:
        result = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]

    def generate(prompt: str) -> str:
        result = client.responses.create(model="gpt-4.1-mini", input=prompt)
        return result.output_text

    return NewsroomRAG(embed=embed, generate=generate)


if __name__ == "__main__":
    from openai import OpenAI

    rag = make_rag(OpenAI())
    print("Adapter ready. Add your authorized Source records before calling ask().")
