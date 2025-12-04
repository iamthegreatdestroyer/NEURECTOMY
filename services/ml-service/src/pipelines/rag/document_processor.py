"""
Document Processor for RAG Pipeline.

@LINGUA @SCRIBE - Processes various document formats for indexing.

Supports:
- Text files (.txt, .md)
- PDF documents
- HTML/Web content
- Code files
- JSON/YAML configuration
"""

import asyncio
import hashlib
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CODE = "code"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""
    doc_id: str
    filename: str
    doc_type: DocumentType
    source: str = ""
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    size_bytes: int = 0
    word_count: int = 0
    language: str = "en"
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    """A processed document ready for chunking."""
    metadata: DocumentMetadata
    content: str
    sections: list[dict] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    code_blocks: list[dict] = Field(default_factory=list)


class DocumentProcessor:
    """
    Document Processor for RAG Pipeline.
    
    @LINGUA - Processes various document formats into a unified structure
    suitable for chunking and embedding.
    """
    
    # File extension to document type mapping
    TYPE_MAPPING = {
        ".txt": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".pdf": DocumentType.PDF,
        ".py": DocumentType.CODE,
        ".js": DocumentType.CODE,
        ".ts": DocumentType.CODE,
        ".tsx": DocumentType.CODE,
        ".jsx": DocumentType.CODE,
        ".rs": DocumentType.CODE,
        ".go": DocumentType.CODE,
        ".java": DocumentType.CODE,
        ".cpp": DocumentType.CODE,
        ".c": DocumentType.CODE,
        ".h": DocumentType.CODE,
        ".json": DocumentType.JSON,
        ".yaml": DocumentType.YAML,
        ".yml": DocumentType.YAML,
        ".csv": DocumentType.CSV,
    }
    
    # Code language detection from extension
    CODE_LANGUAGES = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".sql": "sql",
        ".sh": "bash",
        ".rb": "ruby",
        ".php": "php",
    }
    
    def __init__(self):
        self._extractors: dict[DocumentType, callable] = {
            DocumentType.TEXT: self._extract_text,
            DocumentType.MARKDOWN: self._extract_markdown,
            DocumentType.HTML: self._extract_html,
            DocumentType.CODE: self._extract_code,
            DocumentType.JSON: self._extract_json,
            DocumentType.YAML: self._extract_yaml,
        }
    
    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate a unique document ID from content hash."""
        hash_input = f"{source}:{content[:1000]}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _detect_type(self, path: Path) -> DocumentType:
        """Detect document type from file extension."""
        suffix = path.suffix.lower()
        return self.TYPE_MAPPING.get(suffix, DocumentType.UNKNOWN)
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    async def process_file(
        self,
        file_path: str | Path,
        metadata_overrides: Optional[dict] = None,
    ) -> ProcessedDocument:
        """
        Process a file into a structured document.
        
        @LINGUA - Main entry point for file processing.
        
        Args:
            file_path: Path to the file
            metadata_overrides: Optional metadata to override extracted values
            
        Returns:
            ProcessedDocument ready for chunking
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc_type = self._detect_type(path)
        
        # Read file content
        content = await self._read_file(path)
        
        # Extract based on type
        extractor = self._extractors.get(doc_type, self._extract_text)
        extracted = await extractor(content, path)
        
        # Build metadata
        stats = path.stat()
        metadata = DocumentMetadata(
            doc_id=self._generate_doc_id(content, str(path)),
            filename=path.name,
            doc_type=doc_type,
            source=str(path.absolute()),
            size_bytes=stats.st_size,
            word_count=self._count_words(extracted.get("content", content)),
            created_at=datetime.fromtimestamp(stats.st_ctime),
            modified_at=datetime.fromtimestamp(stats.st_mtime),
            **(metadata_overrides or {}),
        )
        
        # Update metadata from extraction
        if extracted.get("title"):
            metadata.title = extracted["title"]
        if extracted.get("tags"):
            metadata.tags = extracted["tags"]
        if extracted.get("language"):
            metadata.language = extracted["language"]
        
        return ProcessedDocument(
            metadata=metadata,
            content=extracted.get("content", content),
            sections=extracted.get("sections", []),
            links=extracted.get("links", []),
            images=extracted.get("images", []),
            code_blocks=extracted.get("code_blocks", []),
        )
    
    async def process_text(
        self,
        content: str,
        source: str = "inline",
        doc_type: DocumentType = DocumentType.TEXT,
        metadata: Optional[dict] = None,
    ) -> ProcessedDocument:
        """
        Process raw text content.
        
        Args:
            content: Raw text content
            source: Source identifier
            doc_type: Document type hint
            metadata: Additional metadata
            
        Returns:
            ProcessedDocument
        """
        doc_id = self._generate_doc_id(content, source)
        
        # Extract based on type
        extractor = self._extractors.get(doc_type, self._extract_text)
        extracted = await extractor(content, None)
        
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=f"{doc_id}.txt",
            doc_type=doc_type,
            source=source,
            size_bytes=len(content.encode()),
            word_count=self._count_words(content),
            **(metadata or {}),
        )
        
        return ProcessedDocument(
            metadata=doc_metadata,
            content=extracted.get("content", content),
            sections=extracted.get("sections", []),
            links=extracted.get("links", []),
            code_blocks=extracted.get("code_blocks", []),
        )
    
    async def process_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
    ) -> AsyncGenerator[ProcessedDocument, None]:
        """
        Process all documents in a directory.
        
        Args:
            dir_path: Directory path
            recursive: Whether to process subdirectories
            extensions: File extensions to include (None = all supported)
            
        Yields:
            ProcessedDocument for each file
        """
        path = Path(dir_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        # Determine which extensions to process
        valid_extensions = extensions or list(self.TYPE_MAPPING.keys())
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in valid_extensions:
                continue
            
            try:
                doc = await self.process_file(file_path)
                yield doc
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
    
    # =========================================================================
    # File Reading
    # =========================================================================
    
    async def _read_file(self, path: Path) -> str:
        """Read file content asynchronously."""
        import aiofiles
        
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='replace') as f:
            return await f.read()
    
    # =========================================================================
    # Type-Specific Extractors
    # =========================================================================
    
    async def _extract_text(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from plain text."""
        # Simple text - just clean it up
        content = content.strip()
        
        # Try to detect title from first line
        lines = content.split('\n')
        title = lines[0].strip()[:100] if lines else None
        
        return {
            "content": content,
            "title": title,
        }
    
    async def _extract_markdown(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from Markdown content."""
        sections = []
        links = []
        images = []
        code_blocks = []
        
        # Extract title from first H1
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = h1_match.group(1).strip() if h1_match else None
        
        # Extract all headers as sections
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            heading = match.group(2).strip()
            sections.append({
                "level": level,
                "title": heading,
                "position": match.start(),
            })
        
        # Extract links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            links.append(match.group(2))
        
        # Extract images
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(image_pattern, content):
            images.append(match.group(2))
        
        # Extract code blocks
        code_pattern = r'```(\w*)\n(.*?)```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            code_blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2).strip(),
                "position": match.start(),
            })
        
        # Extract tags from frontmatter if present
        tags = []
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            fm = frontmatter_match.group(1)
            tags_match = re.search(r'tags:\s*\[([^\]]+)\]', fm)
            if tags_match:
                tags = [t.strip().strip('"\'') for t in tags_match.group(1).split(',')]
        
        return {
            "content": content,
            "title": title,
            "sections": sections,
            "links": links,
            "images": images,
            "code_blocks": code_blocks,
            "tags": tags,
        }
    
    async def _extract_html(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from HTML content."""
        # Simple HTML text extraction
        # In production, use BeautifulSoup or lxml
        
        # Remove scripts and styles
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        
        # Extract links
        links = re.findall(r'href=["\']([^"\']+)["\']', content)
        
        # Extract images
        images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content)
        
        # Strip HTML tags for text content
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "content": text,
            "title": title,
            "links": links,
            "images": images,
        }
    
    async def _extract_code(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from source code files."""
        language = "unknown"
        if path:
            language = self.CODE_LANGUAGES.get(path.suffix.lower(), "unknown")
        
        # Extract docstrings/comments as title
        title = None
        
        # Python docstring
        if language == "python":
            docstring_match = re.match(r'^["\'][\'"]{2}(.*?)["\'][\'"]{2}', content, re.DOTALL)
            if docstring_match:
                title = docstring_match.group(1).strip().split('\n')[0]
        
        # JavaScript/TypeScript JSDoc
        elif language in ["javascript", "typescript"]:
            jsdoc_match = re.match(r'^/\*\*(.*?)\*/', content, re.DOTALL)
            if jsdoc_match:
                title = jsdoc_match.group(1).strip().split('\n')[0].strip('* ')
        
        # Extract function/class definitions
        sections = []
        
        # Python functions and classes
        if language == "python":
            for match in re.finditer(r'^(class|def)\s+(\w+)', content, re.MULTILINE):
                sections.append({
                    "type": match.group(1),
                    "name": match.group(2),
                    "position": match.start(),
                })
        
        # JavaScript/TypeScript
        elif language in ["javascript", "typescript"]:
            # Functions
            for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)\s*[=:]?\s*(?:function|\(|async)', content):
                sections.append({
                    "type": "function",
                    "name": match.group(1),
                    "position": match.start(),
                })
            # Classes
            for match in re.finditer(r'class\s+(\w+)', content):
                sections.append({
                    "type": "class",
                    "name": match.group(1),
                    "position": match.start(),
                })
        
        return {
            "content": content,
            "title": title,
            "language": language,
            "sections": sections,
            "code_blocks": [{
                "language": language,
                "code": content,
                "position": 0,
            }],
        }
    
    async def _extract_json(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from JSON files."""
        import json
        
        try:
            data = json.loads(content)
            # Try to get a title from common fields
            title = (
                data.get("name") or
                data.get("title") or
                data.get("id") or
                (path.stem if path else None)
            )
            
            # Convert back to formatted JSON
            formatted = json.dumps(data, indent=2)
            
            return {
                "content": formatted,
                "title": title,
                "language": "json",
            }
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _extract_yaml(
        self,
        content: str,
        path: Optional[Path],
    ) -> dict[str, Any]:
        """Extract from YAML files."""
        # Simple extraction - just return content
        # In production, use PyYAML
        
        # Try to get title from name/title field
        title = None
        name_match = re.search(r'^(?:name|title):\s*(.+)$', content, re.MULTILINE)
        if name_match:
            title = name_match.group(1).strip().strip('"\'')
        
        return {
            "content": content,
            "title": title,
            "language": "yaml",
        }
