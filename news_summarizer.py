"""
News Article Summarizer
------------------------
Core logic for fetching a news article and generating an AI summary
using either a hosted Gemini model or a local Ollama model.

This module contains no UI code; see interface.py for the
Streamlit front end built on top of this class.
"""

import os
from typing import Optional

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from newspaper import Article

from dotenv import load_dotenv

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SUMMARY_PROMPTS = {
    "detailed": {
        "map": """Write a detailed summary of the following text:
            "{text}"

            DETAILED SUMMARY:""",
        "combine": """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:""",
    },
    "concise": {
        "map": """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:""",
        "combine": """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
                        FINAL CONCISE SUMMARY:""",
    },
}


class NewsArticleSummarizer:
    """Fetches a news article and summarizes it with a chosen LLM backend."""

    def __init__(
        self,
        model_type: str = "gemini",
        model_name: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.llm = self._build_llm(model_type, model_name, api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    @staticmethod
    def _build_llm(model_type: str, model_name: str, api_key: Optional[str]):
        if model_type == "gemini":
            if not api_key:
                raise ValueError("API key is required for Gemini models")
            return ChatOpenAI(
                temperature=0,
                model_name=model_name,
                api_key=api_key,
                base_url=GEMINI_BASE_URL,
            )
        if model_type == "ollama":
            return ChatOllama(
                model=model_name,
                temperature=0,
                timeout=120,
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    def fetch_article(self, url: str) -> Optional[Article]:
        """Downloads and parses a news article with newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """Splits raw article text into LangChain Document chunks."""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def summarize(
        self,
        url: str,
        summary_type: str = "detailed",
        chain_type: str = "stuff",
    ) -> dict:

        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        docs = self.create_documents(article.text)
        prompts = SUMMARY_PROMPTS[summary_type]

        if chain_type == "map_reduce":
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=PromptTemplate(
                    template=prompts["map"], input_variables=["text"]
                ),
                combine_prompt=PromptTemplate(
                    template=prompts["combine"], input_variables=["text"]
                ),
            )
        else:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=PromptTemplate(
                    template=prompts["map"], input_variables=["text"]
                ),
                document_variable_name="text",
            )

        result = chain.invoke(docs)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": result["output_text"],
            "chunks": docs,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main():
    """Quick manual test without the Streamlit UI."""
    summarizer = NewsArticleSummarizer(model_type="ollama", model_name="llama3.2")
    url = "https://www.artificialintelligence-news.com/news/nvidia-circular-financing-ai-labs/"

    result = summarizer.summarize(url, summary_type="detailed", chain_type="stuff")

    if "error" in result:
        print(result["error"])
        return

    print(f"Title: {result['title']}")
    print(f"Authors: {', '.join(result['authors'])}")
    print(f"Published: {result['publish_date']}")
    print(f"\nSummary:\n{result['summary']}")


if __name__ == "__main__":
    main()