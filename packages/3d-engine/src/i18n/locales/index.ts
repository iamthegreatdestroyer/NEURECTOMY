/**
 * Locales Index
 *
 * Export all translation files for supported languages.
 *
 * @module @neurectomy/3d-engine/i18n/locales
 * @agents @LINGUA
 * @phase Phase 3 - Dimensional Forge
 */

import { en } from "./en";
import { es } from "./es";
import { ja } from "./ja";
import type {
  LanguageCode,
  LanguageInfo,
  TranslationResources,
} from "../types";

/**
 * All translation resources
 */
export const resources: TranslationResources = {
  en: { translation: en },
  es: { translation: es },
  ja: { translation: ja },
  // Placeholder for additional languages - use English as fallback
  fr: { translation: en }, // TODO: Add French translations
  de: { translation: en }, // TODO: Add German translations
  zh: { translation: en }, // TODO: Add Chinese translations
  ar: { translation: en }, // TODO: Add Arabic translations
  he: { translation: en }, // TODO: Add Hebrew translations
};

/**
 * Language metadata for all supported languages
 */
export const languages: Record<LanguageCode, LanguageInfo> = {
  en: {
    code: "en",
    nativeName: "English",
    englishName: "English",
    direction: "ltr",
    dateLocale: "en-US",
    numberLocale: "en-US",
    pluralRules: new Intl.PluralRules("en"),
  },
  es: {
    code: "es",
    nativeName: "Español",
    englishName: "Spanish",
    direction: "ltr",
    dateLocale: "es-ES",
    numberLocale: "es-ES",
    pluralRules: new Intl.PluralRules("es"),
  },
  fr: {
    code: "fr",
    nativeName: "Français",
    englishName: "French",
    direction: "ltr",
    dateLocale: "fr-FR",
    numberLocale: "fr-FR",
    pluralRules: new Intl.PluralRules("fr"),
  },
  de: {
    code: "de",
    nativeName: "Deutsch",
    englishName: "German",
    direction: "ltr",
    dateLocale: "de-DE",
    numberLocale: "de-DE",
    pluralRules: new Intl.PluralRules("de"),
  },
  ja: {
    code: "ja",
    nativeName: "日本語",
    englishName: "Japanese",
    direction: "ltr",
    dateLocale: "ja-JP",
    numberLocale: "ja-JP",
    pluralRules: new Intl.PluralRules("ja"),
  },
  zh: {
    code: "zh",
    nativeName: "中文",
    englishName: "Chinese (Simplified)",
    direction: "ltr",
    dateLocale: "zh-CN",
    numberLocale: "zh-CN",
    pluralRules: new Intl.PluralRules("zh"),
  },
  ar: {
    code: "ar",
    nativeName: "العربية",
    englishName: "Arabic",
    direction: "rtl",
    dateLocale: "ar-SA",
    numberLocale: "ar-SA",
    pluralRules: new Intl.PluralRules("ar"),
  },
  he: {
    code: "he",
    nativeName: "עברית",
    englishName: "Hebrew",
    direction: "rtl",
    dateLocale: "he-IL",
    numberLocale: "he-IL",
    pluralRules: new Intl.PluralRules("he"),
  },
};

/**
 * Get language info by code
 */
export function getLanguageInfo(code: LanguageCode): LanguageInfo {
  return languages[code];
}

/**
 * Get all supported language codes
 */
export function getSupportedLanguages(): LanguageCode[] {
  return Object.keys(languages) as LanguageCode[];
}

/**
 * Get RTL languages
 */
export function getRTLLanguages(): LanguageCode[] {
  return (Object.entries(languages) as [LanguageCode, LanguageInfo][])
    .filter(([, info]) => info.direction === "rtl")
    .map(([code]) => code);
}

/**
 * Check if a language is RTL
 */
export function isRTL(code: LanguageCode): boolean {
  return languages[code].direction === "rtl";
}

// Re-export individual translations
export { en } from "./en";
export { es } from "./es";
export { ja } from "./ja";
