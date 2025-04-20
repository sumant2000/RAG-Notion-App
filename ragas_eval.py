import os
import re
import yaml
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the RAG app
from llm import RAGApp
from notion_client import Client

# Load environment variables
load_dotenv()

class RAGEvaluator:
    def __init__(self, config_path="config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize RAG app
        self.rag_app = RAGApp(config_path=config_path)
        
        # Initialize Notion client for retrieving page statuses
        notion_api_key = os.getenv("NOTION_API_KEY")
        if notion_api_key:
            self.notion = Client(auth=notion_api_key)
        else:
            self.notion = None
            print("Warning: Notion API key not set, some tests may fail")
    
    def get_page_status(self, page_id: str) -> str:
        """Get status property of a Notion page"""
        if not self.notion:
            return None
        
        try:
            page = self.notion.pages.retrieve(page_id)
            properties = page.get("properties", {})
            
            for prop_name, prop in properties.items():
                if prop_name.lower() == "status" and prop.get("type") == "select":
                    status_data = prop.get("select", {})
                    if status_data:
                        return status_data.get("name")
        except Exception as e:
            print(f"Error getting page status: {e}")
        
        return None
    
    def test_private_pages_excluded(self) -> Dict[str, Any]:
        """Test that chunks from Private pages never appear in responses"""
        print("\n== Testing Private Pages Exclusion ==")
        
        # Get all documents in vector store
        all_docs = self.rag_app.vectorstore.similarity_search("", k=1000)
        
        # Get unique page IDs
        page_ids = set()
        for doc in all_docs:
            page_id = doc.metadata.get("page_id")
            if page_id:
                page_ids.add(page_id)
        
        print(f"Found {len(page_ids)} unique pages in vector store")
        
        # Check status of each page
        results = {
            "total_pages": len(page_ids),
            "private_pages_found": 0,
            "passed": True
        }
        
        for page_id in tqdm(page_ids):
            status = self.get_page_status(page_id)
            if status == "Private":
                results["private_pages_found"] += 1
                results["passed"] = False
                print(f"ERROR: Found Private page in vector store: {page_id}")
        
        print(f"Private pages found: {results['private_pages_found']}")
        print(f"Test result: {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def test_source_citations(self, test_questions: List[str], max_workers: int = 2) -> Dict[str, Any]:
        """Test that answers always include at least one source marker"""
        print("\n== Testing Source Citations ==")
        
        results = {
            "total_questions": len(test_questions),
            "questions_with_sources": 0,
            "passed": True,
            "details": []
        }
        
        # Define test worker function
        def process_question(question):
            try:
                response = self.rag_app.get_response(question)
                
                # Check if response contains source markers
                url_pattern = r'\[\[(https://www\.notion\.so/[a-zA-Z0-9-]+#[a-zA-Z0-9-]+)\]\]'
                sources = re.findall(url_pattern, response)
                
                has_sources = len(sources) > 0
                
                return {
                    "question": question,
                    "has_sources": has_sources,
                    "num_sources": len(sources),
                    "sources": sources
                }
            except Exception as e:
                return {
                    "question": question,
                    "has_sources": False,
                    "num_sources": 0,
                    "sources": [],
                    "error": str(e)
                }
        
        # Process questions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_question, q) for q in test_questions]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                detail = future.result()
                results["details"].append(detail)
                
                if detail["has_sources"]:
                    results["questions_with_sources"] += 1
                else:
                    results["passed"] = False
                    print(f"ERROR: Question without sources: {detail['question']}")
                    if "error" in detail:
                        print(f"       Error: {detail['error']}")
        
        # Calculate percentage
        results["percent_with_sources"] = (results["questions_with_sources"] / results["total_questions"]) * 100
        
        print(f"Questions with sources: {results['questions_with_sources']} / {results['total_questions']} ({results['percent_with_sources']:.1f}%)")
        print(f"Test result: {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def run_all_tests(self, test_questions: List[str]) -> Dict[str, Any]:
        """Run all evaluation tests and return results"""
        all_results = {}
        
        # Test 1: Private pages excluded
        all_results["private_pages"] = self.test_private_pages_excluded()
        
        # Test 2: Source citations
        all_results["source_citations"] = self.test_source_citations(test_questions)
        
        # Overall result
        all_results["overall_passed"] = all_results["private_pages"]["passed"] and all_results["source_citations"]["passed"]
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "evaluation_results.json"):
        """Save evaluation results to a file"""
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


def main():
    # Sample test questions
    test_questions = [
        "What are the main features of the product?",
        "How do I use the application?",
        "What are the system requirements?",
        "Can you explain the pricing structure?",
        "What is the company's history?",
        "How do I contact customer support?",
        "What security measures are in place?",
        "Are there any recent updates?",
        "How does the technology work?",
        "What differentiates this from competitors?"
    ]
    
    # Create evaluator
    evaluator = RAGEvaluator()
    
    # Run all tests
    results = evaluator.run_all_tests(test_questions)
    
    # Save results
    evaluator.save_results(results)
    
    # Print overall result
    print("\n== Overall Evaluation Result ==")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")


if __name__ == "__main__":
    main() 