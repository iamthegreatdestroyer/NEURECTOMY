// Package neurectomy provides a Go SDK for the Neurectomy API.
package neurectomy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	// DefaultBaseURL is the default API base URL
	DefaultBaseURL = "https://api.neurectomy.ai"
	// DefaultTimeout is the default request timeout
	DefaultTimeout = 30 * time.Second
)

// Error types
var (
	ErrMissingAPIKey = errors.New("API key is required")
	ErrAPIError      = errors.New("API error")
)

// Config holds the client configuration
type Config struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// ClientOptions allows customization of the client
type ClientOptions struct {
	BaseURL string
	Timeout time.Duration
}

// Client is the Neurectomy API client
type Client struct {
	apiKey  string
	baseURL string
	timeout time.Duration
	client  *http.Client
}

// CompletionRequest is a request for text completion
type CompletionRequest struct {
	Prompt           string  `json:"prompt"`
	MaxTokens        int     `json:"max_tokens,omitempty"`
	Temperature      float32 `json:"temperature,omitempty"`
	Model            string  `json:"model,omitempty"`
	TopP             float32 `json:"top_p,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32 `json:"presence_penalty,omitempty"`
}

// TokenUsage holds token usage information
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse is the response from text completion
type CompletionResponse struct {
	Text            string      `json:"text"`
	TokensGenerated int         `json:"tokens_generated"`
	FinishReason    string      `json:"finish_reason"`
	Usage           *TokenUsage `json:"usage,omitempty"`
}

// CompressionRequest is a request for text compression
type CompressionRequest struct {
	Text             string  `json:"text"`
	TargetRatio      float32 `json:"target_ratio,omitempty"`
	CompressionLevel int     `json:"compression_level,omitempty"`
	Algorithm        string  `json:"algorithm,omitempty"`
}

// CompressionResponse is the response from text compression
type CompressionResponse struct {
	CompressedData   string  `json:"compressed_data"`
	CompressionRatio float32 `json:"compression_ratio"`
	OriginalSize     int64   `json:"original_size"`
	CompressedSize   int64   `json:"compressed_size"`
	Algorithm        string  `json:"algorithm"`
}

// StorageRequest is a request to store a file
type StorageRequest struct {
	Path     string                 `json:"path"`
	Data     string                 `json:"data"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// StorageResponse is the response from file storage
type StorageResponse struct {
	ObjectID  string `json:"object_id"`
	Path      string `json:"path"`
	Size      int64  `json:"size"`
	Timestamp string `json:"timestamp"`
}

// RetrievedFile is the response from file retrieval
type RetrievedFile struct {
	Data      string                 `json:"data"`
	Path      string                 `json:"path"`
	Size      int64                  `json:"size"`
	Timestamp string                 `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// StatusResponse is the response from the status endpoint
type StatusResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
}

// ErrorResponse is an error response from the API
type ErrorResponse struct {
	Code    string      `json:"code"`
	Message string      `json:"message"`
	Details interface{} `json:"details,omitempty"`
}

// NewClient creates a new Neurectomy API client
func NewClient(apiKey string) (*Client, error) {
	return NewClientWithOptions(apiKey, &ClientOptions{})
}

// NewClientWithOptions creates a new client with custom options
func NewClientWithOptions(apiKey string, opts *ClientOptions) (*Client, error) {
	if apiKey == "" {
		return nil, ErrMissingAPIKey
	}

	if opts == nil {
		opts = &ClientOptions{}
	}

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = DefaultTimeout
	}

	return &Client{
		apiKey:  apiKey,
		baseURL: baseURL,
		timeout: timeout,
		client: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Complete generates text completion
func (c *Client) Complete(prompt string, maxTokens int, temperature float32) (*CompletionResponse, error) {
	req := &CompletionRequest{
		Prompt:           prompt,
		MaxTokens:        maxTokens,
		Temperature:      temperature,
		Model:            "ryot-bitnet-7b",
		TopP:             1.0,
		FrequencyPenalty: 0,
		PresencePenalty:  0,
	}

	if maxTokens == 0 {
		req.MaxTokens = 100
	}
	if temperature == 0 {
		req.Temperature = 0.7
	}

	var resp CompletionResponse
	if err := c.post("/v1/completions", req, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// Compress compresses text
func (c *Client) Compress(text string, targetRatio float32, compressionLevel int) (*CompressionResponse, error) {
	req := &CompressionRequest{
		Text:             text,
		TargetRatio:      targetRatio,
		CompressionLevel: compressionLevel,
		Algorithm:        "lz4",
	}

	if targetRatio == 0 {
		req.TargetRatio = 0.1
	}
	if compressionLevel == 0 {
		req.CompressionLevel = 2
	}

	var resp CompressionResponse
	if err := c.post("/v1/compress", req, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// StoreFile stores a file in ΣVAULT
func (c *Client) StoreFile(path, data string) (*StorageResponse, error) {
	req := &StorageRequest{
		Path: path,
		Data: data,
	}

	var resp StorageResponse
	if err := c.post("/v1/storage/store", req, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// RetrieveFile retrieves a file from ΣVAULT
func (c *Client) RetrieveFile(objectID string) (*RetrievedFile, error) {
	var resp RetrievedFile
	if err := c.get(fmt.Sprintf("/v1/storage/%s", objectID), &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// DeleteFile deletes a file from ΣVAULT
func (c *Client) DeleteFile(objectID string) error {
	return c.delete(fmt.Sprintf("/v1/storage/%s", objectID))
}

// GetStatus gets the API status
func (c *Client) GetStatus() (*StatusResponse, error) {
	var resp StatusResponse
	if err := c.get("/v1/status", &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}

// post makes a POST request to the API
func (c *Client) post(endpoint string, req, resp interface{}) error {
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequest("POST", fmt.Sprintf("%s%s", c.baseURL, endpoint), bytes.NewReader(body))
	if err != nil {
		return err
	}

	return c.do(httpReq, resp)
}

// get makes a GET request to the API
func (c *Client) get(endpoint string, resp interface{}) error {
	httpReq, err := http.NewRequest("GET", fmt.Sprintf("%s%s", c.baseURL, endpoint), nil)
	if err != nil {
		return err
	}

	return c.do(httpReq, resp)
}

// delete makes a DELETE request to the API
func (c *Client) delete(endpoint string) error {
	httpReq, err := http.NewRequest("DELETE", fmt.Sprintf("%s%s", c.baseURL, endpoint), nil)
	if err != nil {
		return err
	}

	return c.do(httpReq, nil)
}

// do executes an HTTP request and parses the response
func (c *Client) do(req *http.Request, resp interface{}) error {
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "neurectomy-go-sdk/1.0.0")

	httpResp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return err
	}

	if httpResp.StatusCode >= 400 {
		var errResp ErrorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return fmt.Errorf("HTTP %d: %s", httpResp.StatusCode, string(body))
		}
		return fmt.Errorf("%w %s: %s", ErrAPIError, errResp.Code, errResp.Message)
	}

	if resp != nil {
		if err := json.Unmarshal(body, resp); err != nil {
			return err
		}
	}

	return nil
}
