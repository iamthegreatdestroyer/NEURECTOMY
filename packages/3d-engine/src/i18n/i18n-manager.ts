/**
 * i18n Service Implementation
 *
 * Core internationalization service with language detection,
 * translation resolution, and formatting utilities.
 *
 * @module @neurectomy/3d-engine/i18n
 * @agents @LINGUA @APEX
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  LanguageCode,
  LanguageInfo,
  TranslationNamespace,
  TranslationKey,
  TranslateOptions,
  InterpolationValues,
  PluralTranslation,
  I18nConfig,
  LanguageChangeEvent,
  LanguageChangeListener,
  LoadingState,
  MissingKeyHandler,
  DateFormatOptions,
  NumberFormatOptions,
  RelativeTimeFormatOptions,
  I18nService,
  TranslationObject,
} from "./types";
import {
  resources,
  languages,
  getLanguageInfo,
  getSupportedLanguages,
  isRTL,
} from "./locales";

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Default i18n configuration
 */
export const DEFAULT_I18N_CONFIG: I18nConfig = {
  defaultLanguage: "en",
  fallbackLanguage: "en",
  supportedLanguages: ["en", "es", "fr", "de", "ja", "zh", "ar", "he"],
  defaultNamespace: "common",
  debug: false,
  cache: true,
  lazyLoad: false,
  detectBrowserLanguage: true,
  persistLanguage: true,
  storageKey: "neurectomy-language",
};

// ============================================================================
// I18n Manager Class
// ============================================================================

/**
 * I18n Manager - Core internationalization service
 */
export class I18nManager implements I18nService {
  private config: I18nConfig;
  private currentLanguage: LanguageCode;
  private listeners: Set<LanguageChangeListener>;
  private loadingState: LoadingState;
  private translationCache: Map<string, string>;
  private missingKeyHandler: MissingKeyHandler | null;

  // Formatters (cached per language)
  private dateFormatters: Map<string, Intl.DateTimeFormat>;
  private numberFormatters: Map<string, Intl.NumberFormat>;
  private relativeTimeFormatters: Map<string, Intl.RelativeTimeFormat>;

  constructor(config: Partial<I18nConfig> = {}) {
    this.config = { ...DEFAULT_I18N_CONFIG, ...config };
    this.listeners = new Set();
    this.translationCache = new Map();
    this.missingKeyHandler = null;

    // Initialize formatters
    this.dateFormatters = new Map();
    this.numberFormatters = new Map();
    this.relativeTimeFormatters = new Map();

    // Initialize loading state
    this.loadingState = {
      isLoading: false,
      loadedNamespaces: new Set(["common"]),
      loadedLanguages: new Set(["en"]),
      errors: new Map(),
    };

    // Detect initial language
    this.currentLanguage = this.detectLanguage();
  }

  // ==========================================================================
  // Language Detection
  // ==========================================================================

  /**
   * Detect the best language to use
   */
  private detectLanguage(): LanguageCode {
    // Check persisted preference
    if (this.config.persistLanguage && typeof localStorage !== "undefined") {
      const stored = localStorage.getItem(this.config.storageKey);
      if (stored && this.isValidLanguage(stored)) {
        return stored as LanguageCode;
      }
    }

    // Detect from browser
    if (this.config.detectBrowserLanguage && typeof navigator !== "undefined") {
      const browserLang = navigator.language.split("-")[0];
      if (browserLang && this.isValidLanguage(browserLang)) {
        return browserLang as LanguageCode;
      }
    }

    // Fall back to default
    return this.config.defaultLanguage;
  }

  /**
   * Check if a language code is valid
   */
  private isValidLanguage(code: string): boolean {
    return this.config.supportedLanguages.includes(code as LanguageCode);
  }

  // ==========================================================================
  // Core Translation
  // ==========================================================================

  /**
   * Get current language
   */
  getLanguage(): LanguageCode {
    return this.currentLanguage;
  }

  /**
   * Set the current language
   */
  async setLanguage(language: LanguageCode): Promise<void> {
    if (!this.isValidLanguage(language)) {
      throw new Error(`Unsupported language: ${language}`);
    }

    const previousLanguage = this.currentLanguage;
    this.currentLanguage = language;

    // Persist preference
    if (this.config.persistLanguage && typeof localStorage !== "undefined") {
      localStorage.setItem(this.config.storageKey, language);
    }

    // Clear cached formatters (they're language-specific)
    this.dateFormatters.clear();
    this.numberFormatters.clear();
    this.relativeTimeFormatters.clear();

    // Clear translation cache if caching is enabled
    if (this.config.cache) {
      this.translationCache.clear();
    }

    // Notify listeners
    const event: LanguageChangeEvent = {
      previousLanguage,
      newLanguage: language,
      timestamp: Date.now(),
    };

    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error("Language change listener error:", error);
      }
    }
  }

  /**
   * Translate a key
   */
  t(key: TranslationKey, options: TranslateOptions = {}): string {
    const ns = options.ns || this.config.defaultNamespace;
    const cacheKey = `${this.currentLanguage}:${ns}:${key}:${JSON.stringify(options)}`;

    // Check cache
    if (this.config.cache && this.translationCache.has(cacheKey)) {
      return this.translationCache.get(cacheKey)!;
    }

    // Resolve translation
    let result = this.resolveKey(key, ns, options);

    // Handle missing key
    if (result === undefined) {
      if (this.missingKeyHandler) {
        const handled = this.missingKeyHandler(
          this.currentLanguage,
          ns,
          key,
          options.defaultValue
        );
        if (handled !== undefined) {
          result = handled;
        }
      }

      if (result === undefined) {
        if (options.returnKeyOnMissing !== false && !options.defaultValue) {
          result = key;
        } else {
          result = options.defaultValue || "";
        }

        if (this.config.debug) {
          console.warn(
            `Missing translation: ${this.currentLanguage}:${ns}:${key}`
          );
        }
      }
    }

    // Apply interpolation
    if (options.values) {
      result = this.interpolate(result, options.values);
    }

    // Apply pluralization
    if (typeof options.count === "number") {
      result = this.pluralize(result, options.count);
    }

    // Cache result
    if (this.config.cache) {
      this.translationCache.set(cacheKey, result);
    }

    return result;
  }

  /**
   * Resolve a translation key
   */
  private resolveKey(
    key: string,
    namespace: TranslationNamespace,
    options: TranslateOptions
  ): string | undefined {
    // Get translation resources
    const langResources = resources[this.currentLanguage];
    if (!langResources) {
      return this.resolveFallback(key, namespace, options);
    }

    const nsTranslations = langResources.translation[namespace] as
      | TranslationObject
      | undefined;
    if (!nsTranslations) {
      return this.resolveFallback(key, namespace, options);
    }

    // Navigate to the key
    const value = this.getNestedValue(nsTranslations, key);

    if (value === undefined) {
      return this.resolveFallback(key, namespace, options);
    }

    // Handle plural form objects
    if (typeof value === "object" && !Array.isArray(value)) {
      const pluralObj = value as unknown as PluralTranslation;
      if ("one" in pluralObj || "other" in pluralObj) {
        // Return the base form, pluralization will be applied later
        return pluralObj.one || pluralObj.other;
      }
    }

    return typeof value === "string" ? value : undefined;
  }

  /**
   * Try to resolve from fallback language
   */
  private resolveFallback(
    key: string,
    namespace: TranslationNamespace,
    _options: TranslateOptions
  ): string | undefined {
    if (this.currentLanguage === this.config.fallbackLanguage) {
      return undefined;
    }

    const fallbackResources = resources[this.config.fallbackLanguage];
    if (!fallbackResources) {
      return undefined;
    }

    const nsTranslations = fallbackResources.translation[namespace] as
      | TranslationObject
      | undefined;
    if (!nsTranslations) {
      return undefined;
    }

    const value = this.getNestedValue(nsTranslations, key);
    return typeof value === "string" ? value : undefined;
  }

  /**
   * Get a nested value from an object using dot notation
   */
  private getNestedValue(obj: TranslationObject, key: string): unknown {
    const parts = key.split(".");
    let current: unknown = obj;

    for (const part of parts) {
      if (current === null || current === undefined) {
        return undefined;
      }
      if (typeof current !== "object") {
        return undefined;
      }
      current = (current as Record<string, unknown>)[part];
    }

    return current;
  }

  /**
   * Interpolate values into a string
   */
  private interpolate(text: string, values: InterpolationValues): string {
    return text.replace(/\{\{(\w+)\}\}/g, (_, key: string) => {
      const value = values[key];
      if (value === undefined) {
        return `{{${key}}}`;
      }
      if (value instanceof Date) {
        return this.formatDate(value);
      }
      return String(value);
    });
  }

  /**
   * Apply pluralization rules
   */
  private pluralize(text: string, count: number): string {
    // If text contains plural markers, try to resolve them
    const langInfo = getLanguageInfo(this.currentLanguage);
    const pluralCategory = langInfo.pluralRules.select(count);

    // Check if the original translation object had plural forms
    // This is a simplified implementation - full i18n libraries handle this better
    return text.replace(/\{\{count\}\}/g, String(count));
  }

  /**
   * Check if a key exists
   */
  exists(
    key: TranslationKey,
    options?: { ns?: TranslationNamespace }
  ): boolean {
    const ns = options?.ns || this.config.defaultNamespace;
    const langResources = resources[this.currentLanguage];
    if (!langResources) return false;

    const nsTranslations = langResources.translation[ns] as
      | TranslationObject
      | undefined;
    if (!nsTranslations) return false;

    return this.getNestedValue(nsTranslations, key) !== undefined;
  }

  // ==========================================================================
  // Formatting
  // ==========================================================================

  /**
   * Format a date
   */
  formatDate(date: Date, options: DateFormatOptions = {}): string {
    const langInfo = getLanguageInfo(this.currentLanguage);
    const cacheKey = `date:${langInfo.dateLocale}:${JSON.stringify(options)}`;

    let formatter = this.dateFormatters.get(cacheKey);
    if (!formatter) {
      formatter = new Intl.DateTimeFormat(langInfo.dateLocale, options);
      this.dateFormatters.set(cacheKey, formatter);
    }

    return formatter.format(date);
  }

  /**
   * Format a number
   */
  formatNumber(value: number, options: NumberFormatOptions = {}): string {
    const langInfo = getLanguageInfo(this.currentLanguage);
    const cacheKey = `number:${langInfo.numberLocale}:${JSON.stringify(options)}`;

    let formatter = this.numberFormatters.get(cacheKey);
    if (!formatter) {
      formatter = new Intl.NumberFormat(langInfo.numberLocale, options);
      this.numberFormatters.set(cacheKey, formatter);
    }

    return formatter.format(value);
  }

  /**
   * Format relative time
   */
  formatRelativeTime(
    value: number,
    unit: Intl.RelativeTimeFormatUnit,
    options: RelativeTimeFormatOptions = {}
  ): string {
    const langInfo = getLanguageInfo(this.currentLanguage);
    const cacheKey = `reltime:${langInfo.dateLocale}:${JSON.stringify(options)}`;

    let formatter = this.relativeTimeFormatters.get(cacheKey);
    if (!formatter) {
      formatter = new Intl.RelativeTimeFormat(langInfo.dateLocale, options);
      this.relativeTimeFormatters.set(cacheKey, formatter);
    }

    return formatter.format(value, unit);
  }

  // ==========================================================================
  // Language Info
  // ==========================================================================

  /**
   * Get all supported languages
   */
  getSupportedLanguages(): LanguageInfo[] {
    return this.config.supportedLanguages.map((code) => languages[code]);
  }

  /**
   * Get text direction for current language
   */
  getDirection(): "ltr" | "rtl" {
    return isRTL(this.currentLanguage) ? "rtl" : "ltr";
  }

  /**
   * Get loading state
   */
  getLoadingState(): LoadingState {
    return { ...this.loadingState };
  }

  /**
   * Load a namespace (for lazy loading)
   */
  async loadNamespace(namespace: TranslationNamespace): Promise<void> {
    // In this implementation, all namespaces are bundled
    // This is a placeholder for dynamic loading
    this.loadingState.loadedNamespaces.add(namespace);
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  /**
   * Add a language change listener
   */
  onLanguageChange(listener: LanguageChangeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Set a custom missing key handler
   */
  setMissingKeyHandler(handler: MissingKeyHandler | null): void {
    this.missingKeyHandler = handler;
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  /**
   * Get current configuration
   */
  getConfig(): I18nConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<I18nConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Clear translation cache
   */
  clearCache(): void {
    this.translationCache.clear();
    this.dateFormatters.clear();
    this.numberFormatters.clear();
    this.relativeTimeFormatters.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: I18nManager | null = null;

/**
 * Get the singleton i18n manager instance
 */
export function getI18nManager(config?: Partial<I18nConfig>): I18nManager {
  if (!instance) {
    instance = new I18nManager(config);
  }
  return instance;
}

/**
 * Reset the singleton instance (for testing)
 */
export function resetI18nManager(): void {
  instance = null;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Translate a key using the singleton instance
 */
export function t(key: TranslationKey, options?: TranslateOptions): string {
  return getI18nManager().t(key, options);
}

/**
 * Get current language
 */
export function getCurrentLanguage(): LanguageCode {
  return getI18nManager().getLanguage();
}

/**
 * Set current language
 */
export async function setLanguage(language: LanguageCode): Promise<void> {
  return getI18nManager().setLanguage(language);
}

/**
 * Format a date
 */
export function formatDate(date: Date, options?: DateFormatOptions): string {
  return getI18nManager().formatDate(date, options);
}

/**
 * Format a number
 */
export function formatNumber(
  value: number,
  options?: NumberFormatOptions
): string {
  return getI18nManager().formatNumber(value, options);
}

/**
 * Get text direction
 */
export function getDirection(): "ltr" | "rtl" {
  return getI18nManager().getDirection();
}
