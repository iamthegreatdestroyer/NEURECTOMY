/**
 * Intelligence Foundry Feature
 * ML Model Training & Deployment Hub with MLflow Integration
 */

import { useState } from "react";
import { Brain, BarChart3, Sliders, Database, Zap } from "lucide-react";

// Import components
import {
  ModelTrainer,
  ExperimentDashboard,
  HyperparameterTuner,
} from "./components";

type Tab = "trainer" | "experiments" | "tuner" | "models";

export function IntelligenceFoundry() {
  const [activeTab, setActiveTab] = useState<Tab>("trainer");

  const tabs = [
    {
      id: "trainer" as Tab,
      label: "Model Trainer",
      icon: Brain,
      description: "Train ML models with MLflow",
    },
    {
      id: "experiments" as Tab,
      label: "Experiments",
      icon: BarChart3,
      description: "Track and compare experiments",
    },
    {
      id: "tuner" as Tab,
      label: "Hyperparameter Tuner",
      icon: Sliders,
      description: "Automated optimization with Optuna",
    },
    {
      id: "models" as Tab,
      label: "Model Registry",
      icon: Database,
      description: "Manage deployed models",
    },
  ];

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 flex items-center justify-center">
                <Zap className="w-5 h-5 text-violet-400" />
              </div>
              Intelligence Foundry
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Enterprise ML training, experiment tracking, and model management
              with MLflow & Optuna
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all
                  ${
                    isActive
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "bg-background text-muted-foreground hover:bg-muted hover:text-foreground"
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "trainer" && <ModelTrainer />}
        {activeTab === "experiments" && <ExperimentDashboard />}
        {activeTab === "tuner" && <HyperparameterTuner />}
        {activeTab === "models" && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Database className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">Model Registry</h2>
              <p className="text-muted-foreground">
                Coming soon: MLflow model registry integration
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default IntelligenceFoundry;
