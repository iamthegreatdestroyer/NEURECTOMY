/**
 * @neurectomy/test-config - Mock Index
 *
 * Re-export all mocks for convenient access.
 */

// API Mocks
export {
  mswServer,
  startMockServer,
  stopMockServer,
  resetMockHandlers,
  mockRestEndpoint,
  mockGraphQLQuery,
  mockErrorResponse,
  mockNetworkError,
  createFetchMock,
  createWebSocketMock,
  type MockResponseOptions,
  type MockGraphQLResponse,
  type FetchMock,
  type WebSocketMock,
} from "./api";

// Storage Mocks
export {
  createMockStore,
  createMockIndexedDB,
  createMockFileSystem,
  createMockCache,
  type MockStore,
  type MockIndexedDB,
  type MockFileSystem,
  type MockCache,
} from "./storage";

// Event Mocks
export {
  createMockEventEmitter,
  createMockMessageQueue,
  createMockPubSub,
  createMockJetStream,
  dispatchCustomEvent,
  createKeyboardEvent,
  createMouseEvent,
  createTouchEvent,
  type EventHandler,
  type MockEventEmitter,
  type MockMessageQueue,
  type MockPubSub,
  type MockJetStream,
  type MockJetStreamMessage,
  type MockJetStreamConsumer,
} from "./events";
