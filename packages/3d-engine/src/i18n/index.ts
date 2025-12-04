/**
 * i18n Module Index
 *
 * Complete internationalization system for the 3D engine.
 * Supports 8 languages including RTL languages (Arabic, Hebrew).
 *
 * @module @neurectomy/3d-engine/i18n
 * @agents @LINGUA @APEX
 * @phase Phase 3 - Dimensional Forge
 */

// Types
export type {
  LanguageCode,
  LanguageInfo,
  TranslationNamespace,
  TranslationKey,
  InterpolationValues,
  PluralTranslation,
  TranslationValue,
  TranslationObject,
  NamespaceTranslations,
  LanguageResource,
  TranslationResources,
  I18nConfig,
  TranslateOptions,
  LanguageChangeEvent,
  LanguageChangeListener,
  LoadingState,
  MissingKeyHandler,
  TranslationContext,
  DateFormatOptions,
  NumberFormatOptions,
  RelativeTimeFormatOptions,
  I18nService,
} from "./types";

// i18n Manager
export {
  I18nManager,
  DEFAULT_I18N_CONFIG,
  getI18nManager,
  resetI18nManager,
  // Convenience functions
  t,
  getCurrentLanguage,
  setLanguage,
  formatDate,
  formatNumber,
  getDirection,
} from "./i18n-manager";

// Locales
export {
  resources,
  languages,
  getLanguageInfo,
  getSupportedLanguages,
  getRTLLanguages,
  isRTL,
} from "./locales";

// ============================================================================
// React Integration (for when used with React)
// ============================================================================

import type { LanguageCode, TranslationContext } from "./types";
import { getI18nManager, getCurrentLanguage } from "./i18n-manager";
import { isRTL, getLanguageInfo } from "./locales";

/**
 * Create a translation context for React
 */
export function createTranslationContext(
  namespace: string = "common"
): TranslationContext {
  const language = getCurrentLanguage();
  const direction = isRTL(language) ? "rtl" : "ltr";

  return {
    language,
    namespace: namespace as import("./types").TranslationNamespace,
    direction,
    isRTL: direction === "rtl",
  };
}

/**
 * Initialize i18n with custom configuration
 */
export function initializeI18n(
  config?: Partial<import("./types").I18nConfig>
): {
  manager: import("./i18n-manager").I18nManager;
  t: typeof import("./i18n-manager").t;
  context: TranslationContext;
} {
  const manager = getI18nManager(config);
  const { t } = require("./i18n-manager");

  return {
    manager,
    t,
    context: createTranslationContext(),
  };
}

/**
 * Detect the best language based on browser settings
 */
export function detectBrowserLanguage(): LanguageCode | null {
  if (typeof navigator === "undefined") return null;

  const browserLang = navigator.language.split("-")[0];
  const supported = ["en", "es", "fr", "de", "ja", "zh", "ar", "he"];

  if (supported.includes(browserLang)) {
    return browserLang as LanguageCode;
  }

  return null;
}

/**
 * Get language display name in its native script
 */
export function getNativeLanguageName(code: LanguageCode): string {
  return getLanguageInfo(code).nativeName;
}

/**
 * Get language display name in English
 */
export function getEnglishLanguageName(code: LanguageCode): string {
  return getLanguageInfo(code).englishName;
}

/**
 * Apply RTL styles to a container
 */
export function applyRTLStyles(
  container: HTMLElement,
  language: LanguageCode
): void {
  const direction = isRTL(language) ? "rtl" : "ltr";
  container.setAttribute("dir", direction);
  container.style.direction = direction;

  if (direction === "rtl") {
    container.style.textAlign = "right";
  } else {
    container.style.textAlign = "left";
  }
}

/**
 * Language selector options
 */
export function getLanguageSelectorOptions(): Array<{
  code: LanguageCode;
  nativeName: string;
  englishName: string;
  direction: "ltr" | "rtl";
}> {
  const supported: LanguageCode[] = [
    "en",
    "es",
    "fr",
    "de",
    "ja",
    "zh",
    "ar",
    "he",
  ];

  return supported.map((code) => {
    const info = getLanguageInfo(code);
    return {
      code,
      nativeName: info.nativeName,
      englishName: info.englishName,
      direction: info.direction,
    };
  });
}
