"""
Neurectomy Python SDK
Production-ready SDK for the Neurectomy API
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class NeurectomyError(Exception):
    """Base exception for Neurectomy SDK"""
    pass


class APIError(NeurectomyError):
    """API error response"""
    def __init__(self, code: str, message: str, status_code: int = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"API error {code}: {message}")


class ConfigError(NeurectomyError):
    """Configuration error"""
    pass


@dataclass
class TokenUsage:
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """Text completion response"""
    text: str
    tokens_generated: int
    finish_reason: str
    usage: Optional[TokenUsage] = None


@dataclass
class CompressionResponse:
    """Text compression response"""
    compressed_data: str
    compression_ratio: float
    original_size: int
    compressed_size: int
    algorithm: str


@dataclass
class StorageResponse:
    """File storage response"""
    object_id: str
    path: str
    size: int
    timestamp: str


@dataclass
class RetrievedFile:
    """Retrieved file data"""
    data: str
    path: str
    size: int
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StatusResponse:
    """API status response"""
    status: str
    version: str


class NeurectomyClient:
    """
    Neurectomy API Client
    
    Type-safe, production-ready Python SDK for Neurectomy API.
    
    Example:
        >>> client = NeurectomyClient(api_key="your-api-key")
        >>> response = client.complete("Hello, world!")
        >>> print(response.text)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.neurectomy.ai",
        timeout: int = 30,
        retry_on_failure: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize Neurectomy client
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API (default: https://api.neurectomy.ai)
            timeout: Request timeout in seconds (default: 30)
            retry_on_failure: Enable automatic retry (default: True)
            max_retries: Maximum retry attempts (default: 3)
            
        Raises:
            ConfigError: If API key is empty or invalid
        """
        if not api_key:
            raise ConfigError("API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Create session with retry strategy
        self.session = requests.Session()
        
        if retry_on_failure:
            retry_strategy = Retry(
                total=max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
                backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

        # Set headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"neurectomy-python-sdk/{__version__}",
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            data: Request payload
            
        Returns:
            Response data
            
        Raises:
            APIError: If API returns error
            NeurectomyError: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=self.timeout)
            else:  # POST
                response = self.session.post(url, json=data, timeout=self.timeout)

            if not response.ok:
                try:
                    error_data = response.json()
                    raise APIError(
                        error_data.get("code", "unknown"),
                        error_data.get("message", "Unknown error"),
                        response.status_code,
                    )
                except ValueError:
                    raise APIError(
                        "http_error",
                        f"HTTP {response.status_code}",
                        response.status_code,
                    )

            return response.json()

        except requests.exceptions.Timeout:
            raise NeurectomyError("Request timeout")
        except requests.exceptions.RequestException as e:
            raise NeurectomyError(f"Request failed: {e}")

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> CompletionResponse:
        """
        Generate text completion
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate (default: 100)
            temperature: Sampling temperature (0-2, default: 0.7)
            model: Model name (default: ryot-bitnet-7b)
            top_p: Nucleus sampling parameter (default: 1.0)
            frequency_penalty: Frequency penalty (default: 0)
            presence_penalty: Presence penalty (default: 0)
            
        Returns:
            CompletionResponse object
            
        Raises:
            APIError: If API returns error
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens or 100,
            "temperature": temperature if temperature is not None else 0.7,
            "model": model or "ryot-bitnet-7b",
            "top_p": top_p if top_p is not None else 1.0,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else 0,
            "presence_penalty": presence_penalty if presence_penalty is not None else 0,
        }

        data = self._make_request("POST", "/v1/completions", payload)

        return CompletionResponse(
            text=data["text"],
            tokens_generated=data["tokens_generated"],
            finish_reason=data["finish_reason"],
            usage=TokenUsage(**data["usage"]) if "usage" in data else None,
        )

    def compress(
        self,
        text: str,
        target_ratio: Optional[float] = None,
        compression_level: Optional[int] = None,
        algorithm: Optional[str] = None,
    ) -> CompressionResponse:
        """
        Compress text
        
        Args:
            text: Text to compress
            target_ratio: Target compression ratio (0-1, default: 0.1)
            compression_level: Compression level (1-9, default: 2)
            algorithm: Algorithm to use (default: lz4)
            
        Returns:
            CompressionResponse object
            
        Raises:
            APIError: If API returns error
        """
        payload = {
            "text": text,
            "target_ratio": target_ratio or 0.1,
            "compression_level": compression_level or 2,
            "algorithm": algorithm or "lz4",
        }

        data = self._make_request("POST", "/v1/compress", payload)

        return CompressionResponse(
            compressed_data=data["compressed_data"],
            compression_ratio=data["compression_ratio"],
            original_size=data["original_size"],
            compressed_size=data["compressed_size"],
            algorithm=data["algorithm"],
        )

    def store_file(
        self,
        path: str,
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StorageResponse:
        """
        Store file in ΣVAULT
        
        Args:
            path: File path
            data: File data (base64 encoded)
            metadata: Optional metadata
            
        Returns:
            StorageResponse object
            
        Raises:
            APIError: If API returns error
        """
        payload = {
            "path": path,
            "data": data,
        }
        if metadata:
            payload["metadata"] = metadata

        data = self._make_request("POST", "/v1/storage/store", payload)

        return StorageResponse(
            object_id=data["object_id"],
            path=data["path"],
            size=data["size"],
            timestamp=data["timestamp"],
        )

    def retrieve_file(self, object_id: str) -> RetrievedFile:
        """
        Retrieve file from ΣVAULT
        
        Args:
            object_id: Object ID
            
        Returns:
            RetrievedFile object
            
        Raises:
            APIError: If API returns error
        """
        data = self._make_request("GET", f"/v1/storage/{object_id}")

        return RetrievedFile(
            data=data["data"],
            path=data["path"],
            size=data["size"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata"),
        )

    def delete_file(self, object_id: str) -> bool:
        """
        Delete file from ΣVAULT
        
        Args:
            object_id: Object ID
            
        Returns:
            True if successful
            
        Raises:
            APIError: If API returns error
        """
        self._make_request("DELETE", f"/v1/storage/{object_id}")
        return True

    def get_status(self) -> StatusResponse:
        """
        Get API status
        
        Returns:
            StatusResponse object
            
        Raises:
            APIError: If API returns error
        """
        data = self._make_request("GET", "/v1/status")

        return StatusResponse(
            status=data["status"],
            version=data["version"],
        )
