import logging
from sentence_transformers import SentenceTransformer
from typing import List
import config

# Set the log level for sentence-transformers to ERROR to suppress its internal progress bars
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("初始化 EmbeddingService...")
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            try:
                cls._instance.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                logger.info(f"成功加载本地嵌入模型: {config.EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"加载嵌入模型失败: {e}", exc_info=True)
                cls._instance = None
                raise
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """为单个文本生成嵌入"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"文本嵌入失败 for text: {text[:50]}...: {e}", exc_info=True)
            return [0.0] * config.EMBEDDING_DIMENSION

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """为一批文本生成嵌入"""
        try:
            # 移除内置的进度条 (show_progress_bar=False)
            embeddings = self.model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"批量文本嵌入失败: {e}", exc_info=True)
            return [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)

    def get_dimension(self) -> int:
        """获取模型的嵌入维度"""
        return self.model.get_sentence_embedding_dimension()

# 单例模式，以便在应用中共享
embedding_service_instance = EmbeddingService()