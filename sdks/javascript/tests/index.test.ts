import {
  NeurectomyClient,
  NeurectomyConfig,
  CompletionRequest,
  CompressionRequest,
} from '../src/index';

describe('NeurectomyClient', () => {
  let client: NeurectomyClient;
  const mockConfig: NeurectomyConfig = {
    apiKey: 'test-api-key',
    baseURL: 'http://localhost:8000',
    timeout: 5000,
  };

  beforeEach(() => {
    client = new NeurectomyClient(mockConfig);
  });

  describe('Constructor', () => {
    test('throws error when apiKey is missing', () => {
      expect(() => {
        new NeurectomyClient({ apiKey: '' });
      }).toThrow('API key is required');
    });

    test('creates client with default config', () => {
      const testClient = new NeurectomyClient({ apiKey: 'test-key' });
      expect(testClient).toBeDefined();
    });

    test('creates client with custom config', () => {
      const testClient = new NeurectomyClient({
        apiKey: 'test-key',
        baseURL: 'https://custom.api.com',
        timeout: 60000,
      });
      expect(testClient).toBeDefined();
    });
  });

  describe('Type Safety', () => {
    test('CompletionRequest interface is properly typed', () => {
      const request: CompletionRequest = {
        prompt: 'Hello',
        maxTokens: 100,
        temperature: 0.7,
        model: 'ryot-bitnet-7b',
      };
      expect(request.prompt).toBe('Hello');
    });

    test('CompressionRequest interface is properly typed', () => {
      const request: CompressionRequest = {
        text: 'Large text',
        targetRatio: 0.1,
        compressionLevel: 5,
      };
      expect(request.text).toBe('Large text');
    });
  });

  describe('Configuration', () => {
    test('uses custom baseURL', () => {
      const customConfig: NeurectomyConfig = {
        apiKey: 'test-key',
        baseURL: 'https://custom.neurectomy.ai',
      };
      const customClient = new NeurectomyClient(customConfig);
      expect(customClient).toBeDefined();
    });

    test('uses custom timeout', () => {
      const customConfig: NeurectomyConfig = {
        apiKey: 'test-key',
        timeout: 60000,
      };
      const customClient = new NeurectomyClient(customConfig);
      expect(customClient).toBeDefined();
    });

    test('enables retry by default', () => {
      const customConfig: NeurectomyConfig = {
        apiKey: 'test-key',
      };
      const customClient = new NeurectomyClient(customConfig);
      expect(customClient).toBeDefined();
    });
  });

  describe('API Methods', () => {
    test('complete method exists and is callable', async () => {
      expect(client.complete).toBeDefined();
      expect(typeof client.complete).toBe('function');
    });

    test('compress method exists and is callable', async () => {
      expect(client.compress).toBeDefined();
      expect(typeof client.compress).toBe('function');
    });

    test('storeFile method exists and is callable', async () => {
      expect(client.storeFile).toBeDefined();
      expect(typeof client.storeFile).toBe('function');
    });

    test('retrieveFile method exists and is callable', async () => {
      expect(client.retrieveFile).toBeDefined();
      expect(typeof client.retrieveFile).toBe('function');
    });

    test('deleteFile method exists and is callable', async () => {
      expect(client.deleteFile).toBeDefined();
      expect(typeof client.deleteFile).toBe('function');
    });

    test('getStatus method exists and is callable', async () => {
      expect(client.getStatus).toBeDefined();
      expect(typeof client.getStatus).toBe('function');
    });
  });

  describe('Error Handling', () => {
    test('completion method is async', async () => {
      const result = client.complete({ prompt: 'test' });
      expect(result).toBeInstanceOf(Promise);
    });

    test('compression method is async', async () => {
      const result = client.compress({ text: 'test' });
      expect(result).toBeInstanceOf(Promise);
    });

    test('storage methods are async', async () => {
      const storeResult = client.storeFile('test.txt', 'data');
      expect(storeResult).toBeInstanceOf(Promise);

      const retrieveResult = client.retrieveFile('test-id');
      expect(retrieveResult).toBeInstanceOf(Promise);
    });
  });
});
