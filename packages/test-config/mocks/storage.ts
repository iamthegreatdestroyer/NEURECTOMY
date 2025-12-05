/**
 * @neurectomy/test-config - Storage Mocks
 *
 * Mock implementations for various storage backends.
 */

import { vi } from "vitest";

// ============================================================================
// In-Memory Key-Value Store
// ============================================================================

export interface MockStore<T = unknown> {
  get: (key: string) => T | undefined;
  set: (key: string, value: T) => void;
  delete: (key: string) => boolean;
  has: (key: string) => boolean;
  clear: () => void;
  keys: () => string[];
  values: () => T[];
  entries: () => [string, T][];
  size: () => number;
}

/**
 * Create an in-memory key-value store mock
 */
export function createMockStore<T = unknown>(): MockStore<T> {
  const store = new Map<string, T>();

  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: T) => {
      store.set(key, value);
    }),
    delete: vi.fn((key: string) => store.delete(key)),
    has: vi.fn((key: string) => store.has(key)),
    clear: vi.fn(() => store.clear()),
    keys: vi.fn(() => Array.from(store.keys())),
    values: vi.fn(() => Array.from(store.values())),
    entries: vi.fn(() => Array.from(store.entries())),
    size: vi.fn(() => store.size),
  };
}

// ============================================================================
// IndexedDB Mock
// ============================================================================

export interface MockIndexedDB {
  databases: Map<string, MockIDBDatabase>;
  open: (name: string, version?: number) => MockIDBOpenRequest;
  deleteDatabase: (name: string) => MockIDBOpenRequest;
  reset: () => void;
}

export interface MockIDBDatabase {
  name: string;
  version: number;
  objectStoreNames: string[];
  stores: Map<string, MockIDBObjectStore>;
  createObjectStore: (
    name: string,
    options?: IDBObjectStoreParameters
  ) => MockIDBObjectStore;
  deleteObjectStore: (name: string) => void;
  transaction: (
    storeNames: string | string[],
    mode?: IDBTransactionMode
  ) => MockIDBTransaction;
  close: () => void;
}

export interface MockIDBObjectStore {
  name: string;
  keyPath: string | string[] | null;
  autoIncrement: boolean;
  data: Map<IDBValidKey, unknown>;
  indexes: Map<string, MockIDBIndex>;
  add: (value: unknown, key?: IDBValidKey) => MockIDBRequest;
  put: (value: unknown, key?: IDBValidKey) => MockIDBRequest;
  get: (key: IDBValidKey) => MockIDBRequest;
  getAll: () => MockIDBRequest;
  delete: (key: IDBValidKey) => MockIDBRequest;
  clear: () => MockIDBRequest;
  createIndex: (
    name: string,
    keyPath: string | string[],
    options?: IDBIndexParameters
  ) => MockIDBIndex;
  deleteIndex: (name: string) => void;
  index: (name: string) => MockIDBIndex;
}

export interface MockIDBIndex {
  name: string;
  keyPath: string | string[];
  unique: boolean;
  multiEntry: boolean;
  get: (key: IDBValidKey) => MockIDBRequest;
  getAll: () => MockIDBRequest;
}

export interface MockIDBTransaction {
  db: MockIDBDatabase;
  mode: IDBTransactionMode;
  objectStore: (name: string) => MockIDBObjectStore;
  oncomplete: (() => void) | null;
  onerror: ((event: Event) => void) | null;
  onabort: (() => void) | null;
  commit: () => void;
  abort: () => void;
}

export interface MockIDBRequest<T = unknown> {
  result: T | null;
  error: Error | null;
  source: unknown;
  transaction: MockIDBTransaction | null;
  readyState: "pending" | "done";
  onsuccess: ((event: Event) => void) | null;
  onerror: ((event: Event) => void) | null;
}

export interface MockIDBOpenRequest extends MockIDBRequest<MockIDBDatabase> {
  onupgradeneeded: ((event: IDBVersionChangeEvent) => void) | null;
  onblocked: ((event: Event) => void) | null;
}

/**
 * Create an IndexedDB mock for testing offline storage
 */
export function createMockIndexedDB(): MockIndexedDB {
  const databases = new Map<string, MockIDBDatabase>();
  let autoIncrementCounter = 0;

  function createObjectStore(
    name: string,
    options?: IDBObjectStoreParameters
  ): MockIDBObjectStore {
    const store: MockIDBObjectStore = {
      name,
      keyPath: options?.keyPath ?? null,
      autoIncrement: options?.autoIncrement ?? false,
      data: new Map(),
      indexes: new Map(),

      add(value: unknown, key?: IDBValidKey) {
        const request = createRequest<IDBValidKey>();
        const actualKey =
          key ?? (store.autoIncrement ? ++autoIncrementCounter : undefined);
        if (actualKey === undefined) {
          request.error = new Error("Key required");
          setTimeout(() => request.onerror?.(new Event("error")), 0);
        } else if (store.data.has(actualKey)) {
          request.error = new Error("Key already exists");
          setTimeout(() => request.onerror?.(new Event("error")), 0);
        } else {
          store.data.set(actualKey, value);
          request.result = actualKey;
          setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        }
        return request;
      },

      put(value: unknown, key?: IDBValidKey) {
        const request = createRequest<IDBValidKey>();
        const actualKey =
          key ?? (store.autoIncrement ? ++autoIncrementCounter : undefined);
        if (actualKey === undefined) {
          request.error = new Error("Key required");
          setTimeout(() => request.onerror?.(new Event("error")), 0);
        } else {
          store.data.set(actualKey, value);
          request.result = actualKey;
          setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        }
        return request;
      },

      get(key: IDBValidKey) {
        const request = createRequest();
        request.result = store.data.get(key) ?? null;
        setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        return request;
      },

      getAll() {
        const request = createRequest<unknown[]>();
        request.result = Array.from(store.data.values());
        setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        return request;
      },

      delete(key: IDBValidKey) {
        const request = createRequest();
        store.data.delete(key);
        setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        return request;
      },

      clear() {
        const request = createRequest();
        store.data.clear();
        setTimeout(() => request.onsuccess?.(new Event("success")), 0);
        return request;
      },

      createIndex(
        name: string,
        keyPath: string | string[],
        options?: IDBIndexParameters
      ) {
        const index: MockIDBIndex = {
          name,
          keyPath,
          unique: options?.unique ?? false,
          multiEntry: options?.multiEntry ?? false,
          get(key: IDBValidKey) {
            const request = createRequest();
            for (const [, value] of store.data) {
              const indexKey =
                typeof keyPath === "string"
                  ? (value as Record<string, unknown>)[keyPath]
                  : keyPath.map((k) => (value as Record<string, unknown>)[k]);
              if (indexKey === key) {
                request.result = value;
                break;
              }
            }
            setTimeout(() => request.onsuccess?.(new Event("success")), 0);
            return request;
          },
          getAll() {
            const request = createRequest<unknown[]>();
            request.result = Array.from(store.data.values());
            setTimeout(() => request.onsuccess?.(new Event("success")), 0);
            return request;
          },
        };
        store.indexes.set(name, index);
        return index;
      },

      deleteIndex(name: string) {
        store.indexes.delete(name);
      },

      index(name: string) {
        const index = store.indexes.get(name);
        if (!index) throw new Error(`Index "${name}" not found`);
        return index;
      },
    };

    return store;
  }

  function createRequest<T = unknown>(): MockIDBRequest<T> {
    return {
      result: null,
      error: null,
      source: null,
      transaction: null,
      readyState: "pending",
      onsuccess: null,
      onerror: null,
    };
  }

  function createOpenRequest(): MockIDBOpenRequest {
    return {
      ...createRequest<MockIDBDatabase>(),
      onupgradeneeded: null,
      onblocked: null,
    };
  }

  const mockIndexedDB: MockIndexedDB = {
    databases,

    open(name: string, version = 1) {
      const request = createOpenRequest();
      const existingDb = databases.get(name);

      setTimeout(() => {
        if (!existingDb || existingDb.version < version) {
          const db: MockIDBDatabase = {
            name,
            version,
            objectStoreNames: [],
            stores: new Map(),
            createObjectStore(
              storeName: string,
              options?: IDBObjectStoreParameters
            ) {
              const store = createObjectStore(storeName, options);
              db.stores.set(storeName, store);
              db.objectStoreNames.push(storeName);
              return store;
            },
            deleteObjectStore(storeName: string) {
              db.stores.delete(storeName);
              const index = db.objectStoreNames.indexOf(storeName);
              if (index > -1) db.objectStoreNames.splice(index, 1);
            },
            transaction(
              storeNames: string | string[],
              mode: IDBTransactionMode = "readonly"
            ) {
              const names = Array.isArray(storeNames)
                ? storeNames
                : [storeNames];
              const transaction: MockIDBTransaction = {
                db,
                mode,
                objectStore(storeName: string) {
                  const store = db.stores.get(storeName);
                  if (!store)
                    throw new Error(`Object store "${storeName}" not found`);
                  return store;
                },
                oncomplete: null,
                onerror: null,
                onabort: null,
                commit() {
                  setTimeout(() => this.oncomplete?.(), 0);
                },
                abort() {
                  setTimeout(() => this.onabort?.(), 0);
                },
              };
              return transaction;
            },
            close() {
              // No-op for mock
            },
          };

          databases.set(name, db);

          if (!existingDb) {
            request.onupgradeneeded?.({
              oldVersion: 0,
              newVersion: version,
              target: { result: db },
            } as unknown as IDBVersionChangeEvent);
          } else if (existingDb.version < version) {
            request.onupgradeneeded?.({
              oldVersion: existingDb.version,
              newVersion: version,
              target: { result: db },
            } as unknown as IDBVersionChangeEvent);
          }

          request.result = db;
          request.readyState = "done";
          request.onsuccess?.(new Event("success"));
        } else {
          request.result = existingDb;
          request.readyState = "done";
          request.onsuccess?.(new Event("success"));
        }
      }, 0);

      return request;
    },

    deleteDatabase(name: string) {
      const request = createOpenRequest();
      setTimeout(() => {
        databases.delete(name);
        request.readyState = "done";
        request.onsuccess?.(new Event("success"));
      }, 0);
      return request;
    },

    reset() {
      databases.clear();
      autoIncrementCounter = 0;
    },
  };

  // Install mock
  Object.defineProperty(globalThis, "indexedDB", {
    value: mockIndexedDB,
    configurable: true,
  });

  return mockIndexedDB;
}

// ============================================================================
// File System Mock (for Node.js tests)
// ============================================================================

export interface MockFileSystem {
  files: Map<string, string | Buffer>;
  readFile: (path: string) => Promise<string | Buffer>;
  writeFile: (path: string, data: string | Buffer) => Promise<void>;
  exists: (path: string) => Promise<boolean>;
  unlink: (path: string) => Promise<void>;
  mkdir: (path: string) => Promise<void>;
  readdir: (path: string) => Promise<string[]>;
  reset: () => void;
}

/**
 * Create a mock file system for testing file operations
 */
export function createMockFileSystem(): MockFileSystem {
  const files = new Map<string, string | Buffer>();
  const directories = new Set<string>();

  return {
    files,

    readFile: vi.fn(async (path: string) => {
      const content = files.get(path);
      if (content === undefined) {
        throw new Error(`ENOENT: no such file or directory, open '${path}'`);
      }
      return content;
    }),

    writeFile: vi.fn(async (path: string, data: string | Buffer) => {
      files.set(path, data);
    }),

    exists: vi.fn(async (path: string) => {
      return files.has(path) || directories.has(path);
    }),

    unlink: vi.fn(async (path: string) => {
      if (!files.has(path)) {
        throw new Error(`ENOENT: no such file or directory, unlink '${path}'`);
      }
      files.delete(path);
    }),

    mkdir: vi.fn(async (path: string) => {
      directories.add(path);
    }),

    readdir: vi.fn(async (path: string) => {
      const entries: string[] = [];
      const prefix = path.endsWith("/") ? path : `${path}/`;

      for (const filePath of files.keys()) {
        if (filePath.startsWith(prefix)) {
          const relativePath = filePath.slice(prefix.length);
          const firstSegment = relativePath.split("/")[0];
          if (!entries.includes(firstSegment)) {
            entries.push(firstSegment);
          }
        }
      }

      return entries;
    }),

    reset() {
      files.clear();
      directories.clear();
    },
  };
}

// ============================================================================
// Cache Mock
// ============================================================================

export interface MockCache {
  store: Map<string, { value: unknown; expires?: number }>;
  get: <T>(key: string) => Promise<T | undefined>;
  set: <T>(key: string, value: T, ttl?: number) => Promise<void>;
  delete: (key: string) => Promise<boolean>;
  has: (key: string) => Promise<boolean>;
  clear: () => Promise<void>;
  reset: () => void;
}

/**
 * Create a cache mock with TTL support
 */
export function createMockCache(): MockCache {
  const store = new Map<string, { value: unknown; expires?: number }>();

  return {
    store,

    get: vi.fn(async <T>(key: string): Promise<T | undefined> => {
      const entry = store.get(key);
      if (!entry) return undefined;
      if (entry.expires && Date.now() > entry.expires) {
        store.delete(key);
        return undefined;
      }
      return entry.value as T;
    }),

    set: vi.fn(async <T>(key: string, value: T, ttl?: number) => {
      store.set(key, {
        value,
        expires: ttl ? Date.now() + ttl * 1000 : undefined,
      });
    }),

    delete: vi.fn(async (key: string) => {
      return store.delete(key);
    }),

    has: vi.fn(async (key: string) => {
      const entry = store.get(key);
      if (!entry) return false;
      if (entry.expires && Date.now() > entry.expires) {
        store.delete(key);
        return false;
      }
      return true;
    }),

    clear: vi.fn(async () => {
      store.clear();
    }),

    reset() {
      store.clear();
    },
  };
}
