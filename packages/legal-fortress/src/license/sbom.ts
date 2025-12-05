/**
 * @fileoverview SBOM Generator - Software Bill of Materials
 * @module @neurectomy/legal-fortress/license/sbom
 *
 * @agents @AEGIS @FORGE - Compliance + Build Systems Specialists
 *
 * Generates comprehensive SBOMs in industry-standard formats:
 * - SPDX 2.3 (ISO/IEC 5962:2021)
 * - CycloneDX 1.5
 * - Custom NEURECTOMY format
 */

import { v4 as uuidv4 } from "uuid";
import {
  SBOMDocument,
  SBOMFormat,
  SBOMComponent,
  SBOMRelationship,
  DetectedLicense,
} from "../types";
import { LicenseDetectionEngine, LicenseDefinition } from "./detection";

// ============================================================================
// SBOM TYPES (@AEGIS)
// ============================================================================

/**
 * Package information for SBOM
 */
export interface PackageInfo {
  name: string;
  version: string;
  description?: string;
  homepage?: string;
  repository?: string;
  license?: string;
  author?: string;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
}

/**
 * SBOM generation options
 */
export interface SBOMOptions {
  format: SBOMFormat;
  includeDevDependencies: boolean;
  includePeerDependencies: boolean;
  includeTransitive: boolean;
  detectLicenses: boolean;
  generateHashes: boolean;
  namespace?: string;
}

/**
 * SPDX document structure
 */
export interface SPDXDocument {
  spdxVersion: string;
  dataLicense: string;
  SPDXID: string;
  name: string;
  documentNamespace: string;
  creationInfo: {
    created: string;
    creators: string[];
    licenseListVersion?: string;
  };
  packages: SPDXPackage[];
  relationships: SPDXRelationship[];
}

/**
 * SPDX package
 */
export interface SPDXPackage {
  SPDXID: string;
  name: string;
  versionInfo?: string;
  downloadLocation: string;
  filesAnalyzed: boolean;
  licenseConcluded?: string;
  licenseDeclared?: string;
  copyrightText?: string;
  supplier?: string;
  homepage?: string;
  description?: string;
  externalRefs?: Array<{
    referenceCategory: string;
    referenceType: string;
    referenceLocator: string;
  }>;
}

/**
 * SPDX relationship
 */
export interface SPDXRelationship {
  spdxElementId: string;
  relationshipType: string;
  relatedSpdxElement: string;
}

/**
 * CycloneDX document structure
 */
export interface CycloneDXDocument {
  $schema: string;
  bomFormat: string;
  specVersion: string;
  serialNumber: string;
  version: number;
  metadata: {
    timestamp: string;
    tools: Array<{ name: string; version: string }>;
    component?: CycloneDXComponent;
  };
  components: CycloneDXComponent[];
  dependencies?: Array<{
    ref: string;
    dependsOn: string[];
  }>;
}

/**
 * CycloneDX component
 */
export interface CycloneDXComponent {
  type: "library" | "application" | "framework" | "file";
  name: string;
  version?: string;
  description?: string;
  purl?: string;
  "bom-ref"?: string;
  licenses?: Array<{
    license: {
      id?: string;
      name?: string;
      url?: string;
    };
  }>;
  hashes?: Array<{
    alg: string;
    content: string;
  }>;
  externalReferences?: Array<{
    type: string;
    url: string;
  }>;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_SBOM_OPTIONS: SBOMOptions = {
  format: "spdx",
  includeDevDependencies: false,
  includePeerDependencies: true,
  includeTransitive: true,
  detectLicenses: true,
  generateHashes: false,
};

// ============================================================================
// SBOM GENERATOR (@AEGIS @FORGE)
// ============================================================================

/**
 * SBOM Generator - Creates Software Bill of Materials
 * @agent @AEGIS @FORGE - Complete SBOM implementation
 */
export class SBOMGenerator {
  private licenseEngine: LicenseDetectionEngine;
  private toolName = "@neurectomy/legal-fortress";
  private toolVersion = "1.0.0";

  constructor(licenseEngine?: LicenseDetectionEngine) {
    this.licenseEngine = licenseEngine ?? new LicenseDetectionEngine();
  }

  /**
   * Generate SBOM from package information
   */
  generate(
    rootPackage: PackageInfo,
    dependencies: PackageInfo[],
    options: Partial<SBOMOptions> = {}
  ): SBOMDocument {
    const opts = { ...DEFAULT_SBOM_OPTIONS, ...options };

    // Filter dependencies
    let filteredDeps = [...dependencies];
    if (!opts.includeDevDependencies) {
      filteredDeps = filteredDeps.filter(
        (d) => !this.isDevDependency(d, rootPackage)
      );
    }

    // Build components
    const components: SBOMComponent[] = [
      this.packageToComponent(rootPackage, opts.detectLicenses),
      ...filteredDeps.map((d) =>
        this.packageToComponent(d, opts.detectLicenses)
      ),
    ];

    // Build relationships
    const relationships = this.buildRelationships(rootPackage, filteredDeps);

    // Create SBOM document
    const document: SBOMDocument = {
      id: uuidv4(),
      format: opts.format,
      version: this.getFormatVersion(opts.format),
      createdAt: new Date(),
      rootComponent: rootPackage.name,
      components,
      relationships,
    };

    return document;
  }

  /**
   * Generate SBOM from package.json content
   * @agent @FORGE - Build system integration
   */
  generateFromPackageJson(
    packageJsonContent: string,
    lockFileContent?: string,
    options?: Partial<SBOMOptions>
  ): SBOMDocument {
    const pkg = JSON.parse(packageJsonContent) as PackageInfo;
    const dependencies = this.extractDependencies(pkg, lockFileContent);

    return this.generate(pkg, dependencies, options);
  }

  /**
   * Export SBOM to SPDX format
   */
  exportToSPDX(document: SBOMDocument, namespace?: string): SPDXDocument {
    const docNamespace =
      namespace ??
      `https://neurectomy.io/sbom/${document.rootComponent}/${uuidv4()}`;

    const spdx: SPDXDocument = {
      spdxVersion: "SPDX-2.3",
      dataLicense: "CC0-1.0",
      SPDXID: "SPDXRef-DOCUMENT",
      name: document.rootComponent,
      documentNamespace: docNamespace,
      creationInfo: {
        created: document.createdAt.toISOString(),
        creators: [
          `Tool: ${this.toolName}-${this.toolVersion}`,
          "Organization: NEURECTOMY",
        ],
        licenseListVersion: "3.21",
      },
      packages: document.components.map((c, i) =>
        this.componentToSPDXPackage(c, i)
      ),
      relationships: this.buildSPDXRelationships(document),
    };

    return spdx;
  }

  /**
   * Export SBOM to CycloneDX format
   */
  exportToCycloneDX(document: SBOMDocument): CycloneDXDocument {
    const rootComponent = document.components.find(
      (c) => c.name === document.rootComponent
    );

    const cdx: CycloneDXDocument = {
      $schema: "http://cyclonedx.org/schema/bom-1.5.schema.json",
      bomFormat: "CycloneDX",
      specVersion: "1.5",
      serialNumber: `urn:uuid:${uuidv4()}`,
      version: 1,
      metadata: {
        timestamp: document.createdAt.toISOString(),
        tools: [{ name: this.toolName, version: this.toolVersion }],
        component: rootComponent
          ? this.componentToCycloneDX(rootComponent)
          : undefined,
      },
      components: document.components
        .filter((c) => c.name !== document.rootComponent)
        .map((c) => this.componentToCycloneDX(c)),
      dependencies: this.buildCycloneDXDependencies(document),
    };

    return cdx;
  }

  /**
   * Export SBOM to JSON string
   */
  exportToJSON(document: SBOMDocument): string {
    switch (document.format) {
      case "spdx":
        return JSON.stringify(this.exportToSPDX(document), null, 2);
      case "cyclonedx":
        return JSON.stringify(this.exportToCycloneDX(document), null, 2);
      case "neurectomy":
      default:
        return JSON.stringify(document, null, 2);
    }
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Convert package to SBOM component
   */
  private packageToComponent(
    pkg: PackageInfo,
    detectLicense: boolean
  ): SBOMComponent {
    let licenses: DetectedLicense[] = [];

    if (detectLicense && pkg.license) {
      const detected = this.licenseEngine.detectFromSPDX(pkg.license);
      if (detected) {
        licenses = [detected];
      }
    }

    return {
      id: `pkg:npm/${pkg.name}@${pkg.version}`,
      name: pkg.name,
      version: pkg.version,
      type: "library",
      licenses,
      purl: `pkg:npm/${encodeURIComponent(pkg.name)}@${pkg.version}`,
      supplier: pkg.author,
      hashes: [],
    };
  }

  /**
   * Check if package is a dev dependency
   */
  private isDevDependency(pkg: PackageInfo, root: PackageInfo): boolean {
    return !!(root.devDependencies && pkg.name in root.devDependencies);
  }

  /**
   * Extract dependencies from package.json
   */
  private extractDependencies(
    pkg: PackageInfo,
    _lockFileContent?: string
  ): PackageInfo[] {
    const deps: PackageInfo[] = [];

    // Regular dependencies
    if (pkg.dependencies) {
      for (const [name, version] of Object.entries(pkg.dependencies)) {
        deps.push({
          name,
          version: this.cleanVersion(version),
        });
      }
    }

    // Dev dependencies
    if (pkg.devDependencies) {
      for (const [name, version] of Object.entries(pkg.devDependencies)) {
        deps.push({
          name,
          version: this.cleanVersion(version),
        });
      }
    }

    // Peer dependencies
    if (pkg.peerDependencies) {
      for (const [name, version] of Object.entries(pkg.peerDependencies)) {
        deps.push({
          name,
          version: this.cleanVersion(version),
        });
      }
    }

    return deps;
  }

  /**
   * Clean version string
   */
  private cleanVersion(version: string): string {
    return version.replace(/^[\^~><=]+/, "");
  }

  /**
   * Build relationships between components
   */
  private buildRelationships(
    root: PackageInfo,
    dependencies: PackageInfo[]
  ): SBOMRelationship[] {
    const relationships: SBOMRelationship[] = [];

    for (const dep of dependencies) {
      relationships.push({
        sourceId: `pkg:npm/${root.name}@${root.version}`,
        targetId: `pkg:npm/${dep.name}@${dep.version}`,
        type: "depends-on",
      });
    }

    return relationships;
  }

  /**
   * Get format version string
   */
  private getFormatVersion(format: SBOMFormat): string {
    switch (format) {
      case "spdx":
        return "SPDX-2.3";
      case "cyclonedx":
        return "1.5";
      case "neurectomy":
      default:
        return "1.0";
    }
  }

  /**
   * Convert component to SPDX package
   */
  private componentToSPDXPackage(
    component: SBOMComponent,
    index: number
  ): SPDXPackage {
    const spdxId =
      index === 0
        ? "SPDXRef-RootPackage"
        : `SPDXRef-Package-${component.name.replace(/[^a-zA-Z0-9]/g, "-")}`;

    return {
      SPDXID: spdxId,
      name: component.name,
      versionInfo: component.version,
      downloadLocation: component.purl ?? "NOASSERTION",
      filesAnalyzed: false,
      licenseConcluded: component.licenses[0]?.spdxId ?? "NOASSERTION",
      licenseDeclared: component.licenses[0]?.spdxId ?? "NOASSERTION",
      copyrightText: "NOASSERTION",
      supplier: component.supplier
        ? `Organization: ${component.supplier}`
        : undefined,
      externalRefs: component.purl
        ? [
            {
              referenceCategory: "PACKAGE_MANAGER",
              referenceType: "purl",
              referenceLocator: component.purl,
            },
          ]
        : undefined,
    };
  }

  /**
   * Build SPDX relationships
   */
  private buildSPDXRelationships(document: SBOMDocument): SPDXRelationship[] {
    const relationships: SPDXRelationship[] = [
      {
        spdxElementId: "SPDXRef-DOCUMENT",
        relationshipType: "DESCRIBES",
        relatedSpdxElement: "SPDXRef-RootPackage",
      },
    ];

    for (const rel of document.relationships) {
      const sourceComp = document.components.find((c) => c.id === rel.sourceId);
      const targetComp = document.components.find((c) => c.id === rel.targetId);

      if (sourceComp && targetComp) {
        const sourceId =
          sourceComp.name === document.rootComponent
            ? "SPDXRef-RootPackage"
            : `SPDXRef-Package-${sourceComp.name.replace(/[^a-zA-Z0-9]/g, "-")}`;
        const targetId = `SPDXRef-Package-${targetComp.name.replace(/[^a-zA-Z0-9]/g, "-")}`;

        relationships.push({
          spdxElementId: sourceId,
          relationshipType: "DEPENDS_ON",
          relatedSpdxElement: targetId,
        });
      }
    }

    return relationships;
  }

  /**
   * Convert component to CycloneDX format
   */
  private componentToCycloneDX(component: SBOMComponent): CycloneDXComponent {
    return {
      type: "library",
      name: component.name,
      version: component.version,
      purl: component.purl,
      "bom-ref": component.id,
      licenses: component.licenses.map((l) => ({
        license: {
          id: l.spdxId,
          name: l.name,
        },
      })),
      hashes: component.hashes?.map((h) => ({
        alg: h.algorithm.toUpperCase(),
        content: h.value,
      })),
    };
  }

  /**
   * Build CycloneDX dependencies
   */
  private buildCycloneDXDependencies(
    document: SBOMDocument
  ): Array<{ ref: string; dependsOn: string[] }> {
    const depMap = new Map<string, string[]>();

    for (const rel of document.relationships) {
      if (rel.type === "depends-on") {
        const deps = depMap.get(rel.sourceId) ?? [];
        deps.push(rel.targetId);
        depMap.set(rel.sourceId, deps);
      }
    }

    return [...depMap.entries()].map(([ref, dependsOn]) => ({
      ref,
      dependsOn,
    }));
  }
}

// ============================================================================
// SBOM VALIDATOR (@AEGIS)
// ============================================================================

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * SBOM Validator
 * @agent @AEGIS - Compliance validation
 */
export class SBOMValidator {
  /**
   * Validate SBOM document
   */
  validate(document: SBOMDocument): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Required fields
    if (!document.id) {
      errors.push("Document ID is required");
    }

    if (!document.rootComponent) {
      errors.push("Root component is required");
    }

    if (!document.components || document.components.length === 0) {
      errors.push("At least one component is required");
    }

    // Validate components
    for (const component of document.components) {
      if (!component.name) {
        errors.push(`Component missing name`);
      }

      if (!component.version) {
        warnings.push(`Component ${component.name} missing version`);
      }

      if (component.licenses.length === 0) {
        warnings.push(
          `Component ${component.name} missing license information`
        );
      }
    }

    // Validate relationships
    const componentIds = new Set(document.components.map((c) => c.id));

    for (const rel of document.relationships) {
      if (!componentIds.has(rel.sourceId)) {
        warnings.push(
          `Relationship source ${rel.sourceId} not found in components`
        );
      }

      if (!componentIds.has(rel.targetId)) {
        warnings.push(
          `Relationship target ${rel.targetId} not found in components`
        );
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Validate SPDX document
   */
  validateSPDX(spdx: SPDXDocument): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Required SPDX fields
    if (spdx.spdxVersion !== "SPDX-2.3") {
      warnings.push(
        `SPDX version ${spdx.spdxVersion} may not be fully supported`
      );
    }

    if (!spdx.documentNamespace) {
      errors.push("Document namespace is required for SPDX");
    }

    if (!spdx.creationInfo?.created) {
      errors.push("Creation timestamp is required");
    }

    // Validate packages
    for (const pkg of spdx.packages) {
      if (!pkg.SPDXID) {
        errors.push(`Package ${pkg.name} missing SPDXID`);
      }

      if (pkg.downloadLocation === "NOASSERTION" && !pkg.homepage) {
        warnings.push(`Package ${pkg.name} has no download location`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { SBOMGenerator, SBOMValidator, DEFAULT_SBOM_OPTIONS };

export type {
  PackageInfo,
  SBOMOptions,
  SPDXDocument,
  SPDXPackage,
  SPDXRelationship,
  CycloneDXDocument,
  CycloneDXComponent,
  ValidationResult,
};
