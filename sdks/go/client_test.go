package neurectomy

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	t.Run("valid API key", func(t *testing.T) {
		client, err := NewClient("test-key")
		require.NoError(t, err)
		assert.NotNil(t, client)
		assert.Equal(t, "test-key", client.apiKey)
		assert.Equal(t, DefaultBaseURL, client.baseURL)
		assert.Equal(t, DefaultTimeout, client.timeout)
	})

	t.Run("empty API key", func(t *testing.T) {
		client, err := NewClient("")
		assert.Error(t, err)
		assert.Nil(t, client)
		assert.Equal(t, ErrMissingAPIKey, err)
	})
}

func TestNewClientWithOptions(t *testing.T) {
	t.Run("custom options", func(t *testing.T) {
		opts := &ClientOptions{
			BaseURL: "https://custom.api.com",
		}
		client, err := NewClientWithOptions("test-key", opts)
		require.NoError(t, err)
		assert.Equal(t, "https://custom.api.com", client.baseURL)
	})

	t.Run("nil options", func(t *testing.T) {
		client, err := NewClientWithOptions("test-key", nil)
		require.NoError(t, err)
		assert.Equal(t, DefaultBaseURL, client.baseURL)
	})
}

func TestCompletionRequest(t *testing.T) {
	req := &CompletionRequest{
		Prompt:      "test",
		MaxTokens:   100,
		Temperature: 0.7,
	}

	assert.Equal(t, "test", req.Prompt)
	assert.Equal(t, 100, req.MaxTokens)
	assert.Equal(t, float32(0.7), req.Temperature)
}

func TestCompressionRequest(t *testing.T) {
	req := &CompressionRequest{
		Text:             "test text",
		TargetRatio:      0.1,
		CompressionLevel: 5,
	}

	assert.Equal(t, "test text", req.Text)
	assert.Equal(t, float32(0.1), req.TargetRatio)
	assert.Equal(t, 5, req.CompressionLevel)
}

func TestStorageRequest(t *testing.T) {
	req := &StorageRequest{
		Path: "test/file.txt",
		Data: "base64-data",
	}

	assert.Equal(t, "test/file.txt", req.Path)
	assert.Equal(t, "base64-data", req.Data)
}

func TestClientConfiguration(t *testing.T) {
	client, err := NewClient("test-key")
	require.NoError(t, err)

	assert.NotNil(t, client.client)
	assert.Equal(t, DefaultTimeout, client.timeout)
}

func TestConfig(t *testing.T) {
	config := Config{
		APIKey:  "test-key",
		BaseURL: "https://api.test.com",
		Timeout: DefaultTimeout,
	}

	assert.Equal(t, "test-key", config.APIKey)
	assert.Equal(t, "https://api.test.com", config.BaseURL)
}
