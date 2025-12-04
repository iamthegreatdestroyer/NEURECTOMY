/**
 * Internationalization (i18n) Types
 *
 * Type definitions for the multi-language support system.
 *
 * @module @neurectomy/3d-engine/i18n
 * @agents @LINGUA @APEX
 * @phase Phase 3 - Dimensional Forge
 */

/**
 * Supported languages with their metadata
 */
export type LanguageCode =
  | "en" // English (default)
  | "es" // Spanish
  | "fr" // French
  | "de" // German
  | "ja" // Japanese
  | "zh" // Chinese (Simplified)
  | "ar" // Arabic
  | "he"; // Hebrew

/**
 * Language metadata
 */
export interface LanguageInfo {
  /** ISO 639-1 code */
  code: LanguageCode;
  /** Native name */
  nativeName: string;
  /** English name */
  englishName: string;
  /** Writing direction */
  direction: "ltr" | "rtl";
  /** Date format locale */
  dateLocale: string;
  /** Number format locale */
  numberLocale: string;
  /** Plural rules */
  pluralRules: Intl.PluralRules;
}

/**
 * Namespace for organizing translations
 */
export type TranslationNamespace =
  | "common" // Common UI elements
  | "graph" // Graph-related terms
  | "agent" // Agent-related terms
  | "visualization" // 3D visualization terms
  | "accessibility" // Accessibility-related
  | "errors" // Error messages
  | "tooltips" // Tooltip text
  | "notifications" // Notification messages
  | "settings" // Settings UI
  | "help"; // Help text

/**
 * Translation key structure (nested object paths)
 */
export type TranslationKey = string;

/**
 * Interpolation values for translation
 */
export interface InterpolationValues {
  [key: string]: string | number | boolean | Date | undefined;
}

/**
 * Plural forms for a translation
 */
export interface PluralTranslation {
  zero?: string;
  one: string;
  two?: string;
  few?: string;
  many?: string;
  other: string;
}

/**
 * Translation value - can be string or plural forms
 */
export type TranslationValue = string | PluralTranslation;

/**
 * Nested translation structure
 */
export interface TranslationObject {
  [key: string]: TranslationValue | TranslationObject;
}

/**
 * Complete translation file for a namespace
 */
export interface NamespaceTranslations {
  [namespace: string]: TranslationObject;
}

/**
 * Translation resource for a language
 */
export interface LanguageResource {
  translation: NamespaceTranslations;
}

/**
 * All translation resources
 */
export interface TranslationResources {
  [language: string]: LanguageResource;
}

/**
 * i18n configuration
 */
export interface I18nConfig {
  /** Default language */
  defaultLanguage: LanguageCode;
  /** Fallback language if translation missing */
  fallbackLanguage: LanguageCode;
  /** Supported languages */
  supportedLanguages: LanguageCode[];
  /** Default namespace */
  defaultNamespace: TranslationNamespace;
  /** Debug mode */
  debug: boolean;
  /** Cache translations */
  cache: boolean;
  /** Lazy load namespaces */
  lazyLoad: boolean;
  /** Auto-detect browser language */
  detectBrowserLanguage: boolean;
  /** Store preference in localStorage */
  persistLanguage: boolean;
  /** localStorage key */
  storageKey: string;
}

/**
 * Translation function options
 */
export interface TranslateOptions {
  /** Namespace to look in */
  ns?: TranslationNamespace;
  /** Default value if key not found */
  defaultValue?: string;
  /** Interpolation values */
  values?: InterpolationValues;
  /** Count for pluralization */
  count?: number;
  /** Context for contextual translations */
  context?: string;
  /** Return key if missing (vs empty string) */
  returnKeyOnMissing?: boolean;
}

/**
 * Language change event
 */
export interface LanguageChangeEvent {
  previousLanguage: LanguageCode;
  newLanguage: LanguageCode;
  timestamp: number;
}

/**
 * Language change listener
 */
export type LanguageChangeListener = (event: LanguageChangeEvent) => void;

/**
 * Translation loading state
 */
export interface LoadingState {
  isLoading: boolean;
  loadedNamespaces: Set<TranslationNamespace>;
  loadedLanguages: Set<LanguageCode>;
  errors: Map<string, Error>;
}

/**
 * Missing key handler
 */
export type MissingKeyHandler = (
  language: LanguageCode,
  namespace: TranslationNamespace,
  key: string,
  fallbackValue?: string
) => string | undefined;

/**
 * Translation context
 */
export interface TranslationContext {
  /** Current language */
  language: LanguageCode;
  /** Current namespace */
  namespace: TranslationNamespace;
  /** Direction for current language */
  direction: "ltr" | "rtl";
  /** Is RTL language */
  isRTL: boolean;
}

/**
 * Format options for dates
 */
export interface DateFormatOptions {
  style?: "short" | "medium" | "long" | "full";
  dateStyle?: "short" | "medium" | "long" | "full";
  timeStyle?: "short" | "medium" | "long" | "full";
  weekday?: "narrow" | "short" | "long";
  year?: "numeric" | "2-digit";
  month?: "numeric" | "2-digit" | "narrow" | "short" | "long";
  day?: "numeric" | "2-digit";
  hour?: "numeric" | "2-digit";
  minute?: "numeric" | "2-digit";
  second?: "numeric" | "2-digit";
  timeZone?: string;
}

/**
 * Format options for numbers
 */
export interface NumberFormatOptions {
  style?: "decimal" | "currency" | "percent" | "unit";
  currency?: string;
  unit?: string;
  notation?: "standard" | "scientific" | "engineering" | "compact";
  minimumFractionDigits?: number;
  maximumFractionDigits?: number;
  minimumSignificantDigits?: number;
  maximumSignificantDigits?: number;
}

/**
 * Relative time format options
 */
export interface RelativeTimeFormatOptions {
  style?: "narrow" | "short" | "long";
  numeric?: "always" | "auto";
}

/**
 * i18n service interface
 */
export interface I18nService {
  /** Get current language */
  getLanguage(): LanguageCode;
  /** Set language */
  setLanguage(language: LanguageCode): Promise<void>;
  /** Translate a key */
  t(key: TranslationKey, options?: TranslateOptions): string;
  /** Check if key exists */
  exists(key: TranslationKey, options?: { ns?: TranslationNamespace }): boolean;
  /** Get all supported languages */
  getSupportedLanguages(): LanguageInfo[];
  /** Add language change listener */
  onLanguageChange(listener: LanguageChangeListener): () => void;
  /** Format date */
  formatDate(date: Date, options?: DateFormatOptions): string;
  /** Format number */
  formatNumber(value: number, options?: NumberFormatOptions): string;
  /** Format relative time */
  formatRelativeTime(
    value: number,
    unit: Intl.RelativeTimeFormatUnit,
    options?: RelativeTimeFormatOptions
  ): string;
  /** Get direction for current language */
  getDirection(): "ltr" | "rtl";
  /** Load namespace */
  loadNamespace(namespace: TranslationNamespace): Promise<void>;
  /** Get loading state */
  getLoadingState(): LoadingState;
}
