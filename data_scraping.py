import asyncio
from crawl4ai import *
from pathlib import Path

async def main():
    target_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    target_name = "RAG_Wiki"
    
    # 1. Create a data directory to store your raw markdown documents
    output_dir = Path("rag_knowledge_base")
    output_dir.mkdir(exist_ok=True)
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=target_url)
        
        if result.success:
            # 2. Generate a safe filename from the URL
            filename = target_name
            file_path = output_dir / filename
            
            # 3. Save the markdown content to the file
            file_path.write_text(result.markdown, encoding="utf-8")
            
            print(f"Successfully scraped and saved to: {file_path}")
        else:
            print(f"Failed to crawl the page. Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())


