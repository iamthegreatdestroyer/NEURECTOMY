import { useState } from "react";
import {
  Shield,
  FileText,
  CheckCircle,
  AlertTriangle,
  Clock,
  Lock,
  Eye,
  Download,
  Search,
  Filter,
  Plus,
  ChevronRight,
  Scale,
  Gavel,
  FileCheck,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ComplianceItem {
  id: string;
  title: string;
  category: string;
  status: "compliant" | "warning" | "non-compliant" | "pending";
  lastAudit: string;
  nextAudit: string;
  description: string;
}

interface LegalDocument {
  id: string;
  title: string;
  type: string;
  version: string;
  lastUpdated: string;
  status: "active" | "draft" | "archived";
}

const mockComplianceItems: ComplianceItem[] = [
  {
    id: "1",
    title: "GDPR Data Protection",
    category: "Privacy",
    status: "compliant",
    lastAudit: "2025-11-15",
    nextAudit: "2026-02-15",
    description:
      "General Data Protection Regulation compliance for EU data subjects",
  },
  {
    id: "2",
    title: "SOC 2 Type II",
    category: "Security",
    status: "compliant",
    lastAudit: "2025-10-01",
    nextAudit: "2026-04-01",
    description:
      "Service Organization Control for security, availability, and confidentiality",
  },
  {
    id: "3",
    title: "HIPAA Compliance",
    category: "Healthcare",
    status: "warning",
    lastAudit: "2025-09-20",
    nextAudit: "2025-12-20",
    description:
      "Health Insurance Portability and Accountability Act requirements",
  },
  {
    id: "4",
    title: "PCI DSS",
    category: "Financial",
    status: "pending",
    lastAudit: "2025-08-10",
    nextAudit: "2025-12-10",
    description: "Payment Card Industry Data Security Standard",
  },
  {
    id: "5",
    title: "ISO 27001",
    category: "Security",
    status: "compliant",
    lastAudit: "2025-11-01",
    nextAudit: "2026-05-01",
    description: "Information security management system certification",
  },
];

const mockDocuments: LegalDocument[] = [
  {
    id: "1",
    title: "Privacy Policy",
    type: "Policy",
    version: "3.2.1",
    lastUpdated: "2025-11-20",
    status: "active",
  },
  {
    id: "2",
    title: "Terms of Service",
    type: "Agreement",
    version: "2.5.0",
    lastUpdated: "2025-11-15",
    status: "active",
  },
  {
    id: "3",
    title: "Data Processing Agreement",
    type: "Agreement",
    version: "1.8.3",
    lastUpdated: "2025-10-30",
    status: "active",
  },
  {
    id: "4",
    title: "Cookie Policy",
    type: "Policy",
    version: "2.0.0-draft",
    lastUpdated: "2025-12-01",
    status: "draft",
  },
  {
    id: "5",
    title: "Security Whitepaper",
    type: "Documentation",
    version: "4.1.0",
    lastUpdated: "2025-11-10",
    status: "active",
  },
];

export function LegalFortress() {
  const [activeTab, setActiveTab] = useState<
    "compliance" | "documents" | "audit"
  >("compliance");
  const [searchQuery, setSearchQuery] = useState("");

  const statusConfig = {
    compliant: {
      icon: CheckCircle,
      color: "text-green-500",
      bg: "bg-green-500/10",
      label: "Compliant",
    },
    warning: {
      icon: AlertTriangle,
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
      label: "Warning",
    },
    "non-compliant": {
      icon: AlertCircle,
      color: "text-red-500",
      bg: "bg-red-500/10",
      label: "Non-Compliant",
    },
    pending: {
      icon: Clock,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      label: "Pending",
    },
  };

  const complianceStats = {
    compliant: mockComplianceItems.filter((i) => i.status === "compliant")
      .length,
    warning: mockComplianceItems.filter((i) => i.status === "warning").length,
    nonCompliant: mockComplianceItems.filter(
      (i) => i.status === "non-compliant"
    ).length,
    pending: mockComplianceItems.filter((i) => i.status === "pending").length,
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex-none p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Shield className="h-7 w-7 text-primary" />
              Legal Fortress
            </h1>
            <p className="text-muted-foreground mt-1">
              Compliance management and legal documentation hub
            </p>
          </div>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            <Plus className="h-4 w-4" />
            New Audit
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-6 p-1 bg-muted rounded-lg w-fit">
          {[
            { id: "compliance", label: "Compliance", icon: Shield },
            { id: "documents", label: "Documents", icon: FileText },
            { id: "audit", label: "Audit Trail", icon: Eye },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stats Overview */}
      <div className="flex-none p-6 grid grid-cols-4 gap-4">
        <div className="p-4 bg-card rounded-xl border border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-green-500/10">
              <CheckCircle className="h-5 w-5 text-green-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{complianceStats.compliant}</p>
              <p className="text-sm text-muted-foreground">Compliant</p>
            </div>
          </div>
        </div>
        <div className="p-4 bg-card rounded-xl border border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-yellow-500/10">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{complianceStats.warning}</p>
              <p className="text-sm text-muted-foreground">Warnings</p>
            </div>
          </div>
        </div>
        <div className="p-4 bg-card rounded-xl border border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-red-500/10">
              <AlertCircle className="h-5 w-5 text-red-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {complianceStats.nonCompliant}
              </p>
              <p className="text-sm text-muted-foreground">Non-Compliant</p>
            </div>
          </div>
        </div>
        <div className="p-4 bg-card rounded-xl border border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-500/10">
              <Clock className="h-5 w-5 text-blue-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{complianceStats.pending}</p>
              <p className="text-sm text-muted-foreground">Pending</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6 pt-0">
        {/* Search and Filter */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search compliance items or documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-card border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          <button className="flex items-center gap-2 px-4 py-2 bg-card border border-border rounded-lg hover:bg-accent transition-colors">
            <Filter className="h-4 w-4" />
            Filter
          </button>
        </div>

        {activeTab === "compliance" && (
          <div className="space-y-4">
            {mockComplianceItems.map((item) => {
              const status = statusConfig[item.status];
              const StatusIcon = status.icon;

              return (
                <div
                  key={item.id}
                  className="p-4 bg-card rounded-xl border border-border hover:border-primary/50 transition-colors cursor-pointer"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4">
                      <div className={cn("p-2 rounded-lg", status.bg)}>
                        <StatusIcon className={cn("h-5 w-5", status.color)} />
                      </div>
                      <div>
                        <h3 className="font-semibold">{item.title}</h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          {item.description}
                        </p>
                        <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            Last audit: {item.lastAudit}
                          </span>
                          <span className="flex items-center gap-1">
                            <Scale className="h-3 w-3" />
                            Next: {item.nextAudit}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span
                        className={cn(
                          "px-2 py-1 rounded-full text-xs font-medium",
                          status.bg,
                          status.color
                        )}
                      >
                        {status.label}
                      </span>
                      <ChevronRight className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === "documents" && (
          <div className="grid grid-cols-2 gap-4">
            {mockDocuments.map((doc) => (
              <div
                key={doc.id}
                className="p-4 bg-card rounded-xl border border-border hover:border-primary/50 transition-colors cursor-pointer"
              >
                <div className="flex items-start gap-4">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold truncate">{doc.title}</h3>
                      {doc.status === "draft" && (
                        <span className="px-2 py-0.5 bg-yellow-500/10 text-yellow-500 text-xs rounded-full">
                          Draft
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {doc.type} • v{doc.version}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Updated {doc.lastUpdated}
                    </p>
                  </div>
                  <button
                    className="p-2 hover:bg-accent rounded-lg transition-colors"
                    aria-label="Download document"
                    title="Download document"
                  >
                    <Download className="h-4 w-4 text-muted-foreground" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === "audit" && (
          <div className="space-y-4">
            <div className="p-6 bg-card rounded-xl border border-border text-center">
              <Gavel className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-semibold text-lg">Audit Trail</h3>
              <p className="text-muted-foreground mt-2">
                Complete audit history and compliance tracking coming soon.
              </p>
              <button className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-lg">
                Generate Audit Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default LegalFortress;
