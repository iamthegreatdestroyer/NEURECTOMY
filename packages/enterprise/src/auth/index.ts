/**
 * @fileoverview Enterprise Authentication Module
 * @module @neurectomy/enterprise/auth
 *
 * Agent Assignment: @CIPHER (Cryptography) + @SYNAPSE (Integration)
 *
 * Unified authentication module providing:
 * - SSO/SAML 2.0 authentication
 * - OAuth 2.0 / OpenID Connect
 * - Session management
 * - Token handling
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

// Core SSO Provider
export {
  SSOManager,
  DEFAULT_AUTH_CONFIG,
  DEFAULT_ATTRIBUTE_MAPPING,
  createDefaultIdentity,
  type SSOEventType,
  type SSOEvent,
  type LoginRequest,
  type LoginResponse,
  type TokenClaims,
  type IProviderHandler,
} from "./sso-provider.js";

// SAML Handler
export {
  SAMLHandler,
  type SAMLAssertion,
  type SAMLResponse,
} from "./saml-handler.js";

// OAuth/OIDC Handler
export {
  OAuthHandler,
  type TokenResponse,
  type OAuthError,
  type PKCEPair,
  type OIDCDiscovery,
} from "./oauth-handler.js";

// Re-export relevant types
export type {
  AuthProvider,
  AuthSession,
  UserIdentity,
  SSOProviderConfig,
  SAMLConfig,
  OAuth2Config,
  OIDCConfig,
  LDAPConfig,
  AttributeMapping,
  SessionState,
  MFAMethod,
  IdentityMetadata,
  AuthModuleConfig,
  CookieSettings,
  PasswordPolicy,
  OAuth2GrantType,
  SAMLBinding,
} from "../types.js";
