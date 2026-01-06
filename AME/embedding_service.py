import logging
from sentence_transformers import SentenceTransformer
from typing import List
import config

# Set the log level for sentence-transformers to ERROR to suppress internal progress bars
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing EmbeddingService...")
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            try:
                cls._instance.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                logger.info(f"Successfully loaded the embedding model: {config.EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to load the embedding model: {e}", exc_info=True)
                cls._instance = None
                raise
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text input."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}...: {e}", exc_info=True)
            return [0.0] * config.EMBEDDING_DIMENSION

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text inputs."""
        try:
            # Disable internal progress bar (show_progress_bar=False)
            embeddings = self.model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch of texts: {e}", exc_info=True)
            return [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)

    def get_dimension(self) -> int:
        """Retrieve the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()

# Singleton instance to share across the application
embedding_service_instance = EmbeddingService()