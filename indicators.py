from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List
import re
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
import streamlit as st
from langchain_openai import AzureChatOpenAI
from ragas.llms import LangchainLLMWrapper
from config import Config

def initialize_llm() -> AzureChatOpenAI:
    """Initialize Azure OpenAI chat model with configuration.
    
    Returns:
        AzureChatOpenAI: Configured language model instance
    """
    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY,
        azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_version=Config.AZURE_OPENAI_API_VERSION,
        temperature=0.1,
    )

def calculate_text_similarity(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate similarity between two texts using multiple methods

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dictionary with different similarity scores
    """
    # Clean texts
    text1_clean = re.sub(r'\s+', ' ', text1.strip())
    text2_clean = re.sub(r'\s+', ' ', text2.strip())

    # Sequence similarity (character-based)
    sequence_similarity = SequenceMatcher(
        None, text1_clean, text2_clean).ratio()

    # TF-IDF Cosine similarity (word-based)
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        cosine_sim = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0

    # Word overlap similarity
    words1 = set(text1_clean.lower().split())
    words2 = set(text2_clean.lower().split())
    word_overlap = len(words1.intersection(words2)) / \
        max(len(words1.union(words2)), 1)

    return {
        "sequence_similarity": sequence_similarity,
        "cosine_similarity": cosine_sim,
        "word_overlap": word_overlap,
        "average_similarity": (sequence_similarity + cosine_sim + word_overlap) / 3
    }


def calculate_length_comparison(text1: str, text2: str) -> Dict[str, Any]:
    """
    Compare lengths of two texts

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dictionary with length comparison metrics
    """
    len1 = len(text1)
    len2 = len(text2)

    length_ratio = min(len1, len2) / \
        max(len1, len2) if max(len1, len2) > 0 else 1.0
    length_diff = abs(len1 - len2)
    length_diff_percentage = (
        length_diff / max(len1, len2)) * 100 if max(len1, len2) > 0 else 0

    # Word counts
    words1 = len(text1.split())
    words2 = len(text2.split())
    word_ratio = min(words1, words2) / max(words1,
                                           words2) if max(words1, words2) > 0 else 1.0

    return {
        "text1_length": len1,
        "text2_length": len2,
        "length_ratio": length_ratio,
        "length_difference": length_diff,
        "length_diff_percentage": length_diff_percentage,
        "text1_words": words1,
        "text2_words": words2,
        "word_ratio": word_ratio,
        "length_similarity_score": length_ratio  # Score based on length similarity
    }


def extract_text_samples(text: str, sample_size: int = 200) -> List[str]:
    """
    Extract arbitrary samples from text for comparison

    Args:
        text: Input text
        sample_size: Size of each sample

    Returns:
        List of text samples
    """
    if len(text) <= sample_size:
        return [text]

    samples = []
    # Beginning sample
    samples.append(text[:sample_size])

    # Middle sample
    mid_start = len(text) // 2 - sample_size // 2
    samples.append(text[mid_start:mid_start + sample_size])

    # End sample
    samples.append(text[-sample_size:])

    return samples


def ragas_evaluate(questions: List[str], answers: List[str], expected_answers: List[str], sources: List[str]) -> Dict[str, Any]:
    """
    Evaluate the answer using RAGAS
    """
    dataset = []

    for question, answer, expected_answer in zip(questions, answers, expected_answers):
        dataset.append(
            {
                "user_input": question,
                "retrieved_contexts": sources,
                "response": answer,
                "reference": expected_answer
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    evaluator_llm = LangchainLLMWrapper(initialize_llm())

    evaluation = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )
    return evaluation
