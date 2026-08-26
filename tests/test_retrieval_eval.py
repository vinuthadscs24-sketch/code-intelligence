import sys
import time
import argparse
from typing import List, Dict, Any, Set
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import VectorStore
from src.knowledge_graph import KnowledgeGraph as GraphDB
from src.hybrid_retriever import HybridRetriever
from src.llm_engine import CodeIntelligenceEngine