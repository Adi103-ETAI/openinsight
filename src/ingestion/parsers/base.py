from abc import ABC, abstractmethod

from src.ingestion.document_db import DocumentRecord


class BaseParser(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str:
        pass

    @abstractmethod
    def parse(self) -> list[DocumentRecord]:
        pass
