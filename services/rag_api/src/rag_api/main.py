from rag_core.config import settings
from rag_core.pipeline import build_pipeline


def main() -> None:
	pipeline = build_pipeline()
	
	result = pipeline.invoke("Which bidding strategies does ASSUME support?")

	print("\n── Answer ──────────────────────────────────────────")
	print(result["answer"])

	print("\n── Sources ─────────────────────────────────────────", len(result["context"]))
	unique_sources = set()
	for doc in result["context"]:
		source = doc.metadata.get("source", "unknown")
		if source not in unique_sources:
			unique_sources.add(source)
			print(f"[{len(unique_sources)}] {source}")
	


if __name__ == "__main__":
	main()