/**
 * Discovery Engine Feature
 * Research & Knowledge Graph Navigation
 */

import { useState } from "react";
import {
  Search,
  BookOpen,
  FileText,
  Link2,
  Tag,
  Calendar,
  User,
  Star,
  Download,
  ExternalLink,
  Filter,
  Grid,
  List,
  Sparkles,
  TrendingUp,
  Clock,
  Bookmark,
  Share2,
} from "lucide-react";

// Research item type
interface ResearchItem {
  id: string;
  title: string;
  authors: string[];
  source: string;
  date: Date;
  abstract: string;
  tags: string[];
  citations: number;
  relevance: number;
  saved: boolean;
  url: string;
}

// Research card component
function ResearchCard({ item }: { item: ResearchItem }) {
  const [saved, setSaved] = useState(item.saved);

  return (
    <div className="bg-card border border-border rounded-xl p-5 hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <FileText className="w-4 h-4 text-primary" />
          </div>
          <span className="text-xs text-muted-foreground font-medium">
            {item.source}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 px-2 py-1 bg-green-500/10 text-green-500 rounded-full text-xs font-medium">
            <Sparkles className="w-3 h-3" />
            {item.relevance}% match
          </div>
          <button
            onClick={() => setSaved(!saved)}
            className={`p-1.5 rounded-lg transition-colors ${
              saved
                ? "bg-yellow-500/10 text-yellow-500"
                : "hover:bg-muted text-muted-foreground"
            }`}
          >
            <Bookmark className={`w-4 h-4 ${saved ? "fill-current" : ""}`} />
          </button>
        </div>
      </div>

      <h3 className="font-semibold text-foreground mb-2 line-clamp-2 hover:text-primary cursor-pointer transition-colors">
        {item.title}
      </h3>

      <p className="text-sm text-muted-foreground mb-3 line-clamp-3">
        {item.abstract}
      </p>

      {/* Authors */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
        <User className="w-3.5 h-3.5" />
        <span className="line-clamp-1">{item.authors.join(", ")}</span>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {item.tags.slice(0, 4).map((tag) => (
          <span
            key={tag}
            className="px-2 py-0.5 bg-muted text-muted-foreground text-xs rounded-full"
          >
            {tag}
          </span>
        ))}
        {item.tags.length > 4 && (
          <span className="px-2 py-0.5 text-muted-foreground text-xs">
            +{item.tags.length - 4}
          </span>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-border">
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5" />
            {item.date.toLocaleDateString()}
          </div>
          <div className="flex items-center gap-1">
            <Link2 className="w-3.5 h-3.5" />
            {item.citations} citations
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            className="p-2 hover:bg-muted rounded-lg transition-colors"
            aria-label="Share"
            title="Share"
          >
            <Share2 className="w-4 h-4 text-muted-foreground" />
          </button>
          <button
            className="p-2 hover:bg-muted rounded-lg transition-colors"
            aria-label="Download"
            title="Download"
          >
            <Download className="w-4 h-4 text-muted-foreground" />
          </button>
          <button
            className="p-2 hover:bg-muted rounded-lg transition-colors"
            aria-label="Open in new tab"
            title="Open in new tab"
          >
            <ExternalLink className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      </div>
    </div>
  );
}

// Trending topic card
function TrendingCard({
  topic,
}: {
  topic: { name: string; growth: number; papers: number };
}) {
  return (
    <div className="flex items-center justify-between p-3 bg-card border border-border rounded-lg hover:border-primary/30 cursor-pointer transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-lg flex items-center justify-center">
          <TrendingUp className="w-4 h-4 text-emerald-500" />
        </div>
        <div>
          <p className="font-medium text-sm">{topic.name}</p>
          <p className="text-xs text-muted-foreground">{topic.papers} papers</p>
        </div>
      </div>
      <span className="text-sm font-semibold text-emerald-500">
        +{topic.growth}%
      </span>
    </div>
  );
}

// Main Discovery Engine component
export function DiscoveryEngine() {
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [activeSource, setActiveSource] = useState<string | null>(null);

  // Mock research data
  const researchItems: ResearchItem[] = [
    {
      id: "1",
      title:
        "Attention Is All You Need: Transformer Architecture for Neural Machine Translation",
      authors: ["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J."],
      source: "arXiv",
      date: new Date("2017-06-12"),
      abstract:
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture based solely on attention mechanisms.",
      tags: ["transformers", "attention", "NLP", "deep learning"],
      citations: 89000,
      relevance: 98,
      saved: true,
      url: "https://arxiv.org/abs/1706.03762",
    },
    {
      id: "2",
      title:
        "Large Language Models as Tool Makers: Emergent Capabilities in Few-Shot Learning",
      authors: ["Chen, M.", "Tworek, J.", "Jun, H."],
      source: "NeurIPS 2023",
      date: new Date("2023-09-15"),
      abstract:
        "We demonstrate that large language models can synthesize executable tools from natural language specifications, enabling a new paradigm of AI-assisted programming and automation.",
      tags: ["LLM", "tool use", "few-shot", "agents"],
      citations: 156,
      relevance: 95,
      saved: false,
      url: "#",
    },
    {
      id: "3",
      title: "Constitutional AI: Harmlessness from AI Feedback",
      authors: ["Bai, Y.", "Kadavath, S.", "Kundu, S."],
      source: "Anthropic",
      date: new Date("2022-12-15"),
      abstract:
        "We describe a method for training AI systems to be harmless and helpful using AI feedback, without human labels for harmlessness. The key is to have AI evaluate and critique its own outputs.",
      tags: ["AI safety", "RLHF", "alignment", "constitutional AI"],
      citations: 892,
      relevance: 91,
      saved: false,
      url: "#",
    },
    {
      id: "4",
      title:
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
      authors: ["Wei, J.", "Wang, X.", "Schuurmans, D."],
      source: "Google Research",
      date: new Date("2022-01-28"),
      abstract:
        "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.",
      tags: ["prompting", "reasoning", "LLM", "chain-of-thought"],
      citations: 4500,
      relevance: 89,
      saved: true,
      url: "#",
    },
  ];

  // Trending topics
  const trendingTopics = [
    { name: "Multi-Modal LLMs", growth: 245, papers: 1243 },
    { name: "AI Agents", growth: 189, papers: 876 },
    { name: "RAG Systems", growth: 156, papers: 654 },
    { name: "Constitutional AI", growth: 134, papers: 432 },
  ];

  // Sources
  const sources = [
    { id: "arxiv", name: "arXiv", count: 12500 },
    { id: "semantic", name: "Semantic Scholar", count: 45000 },
    { id: "pubmed", name: "PubMed", count: 8900 },
    { id: "ieee", name: "IEEE Xplore", count: 5600 },
    { id: "acm", name: "ACM DL", count: 3400 },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <BookOpen className="w-7 h-7 text-emerald-500" />
            Discovery Engine
          </h1>
          <p className="text-muted-foreground mt-1">
            Research & Knowledge Graph Navigation
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode("grid")}
            className={`p-2 rounded-lg transition-colors ${
              viewMode === "grid"
                ? "bg-primary text-primary-foreground"
                : "bg-muted"
            }`}
          >
            <Grid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={`p-2 rounded-lg transition-colors ${
              viewMode === "list"
                ? "bg-primary text-primary-foreground"
                : "bg-muted"
            }`}
          >
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search papers, authors, topics..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-12 pr-4 py-4 bg-card border border-border rounded-xl text-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
          <button className="flex items-center gap-2 px-3 py-1.5 bg-muted rounded-lg text-sm">
            <Filter className="w-4 h-4" />
            Filters
          </button>
          <button className="flex items-center gap-2 px-4 py-1.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium">
            <Sparkles className="w-4 h-4" />
            AI Search
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Sidebar */}
        <div className="col-span-3 space-y-6">
          {/* Sources */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h3 className="font-semibold mb-4">Sources</h3>
            <div className="space-y-2">
              {sources.map((source) => (
                <button
                  key={source.id}
                  onClick={() =>
                    setActiveSource(
                      activeSource === source.id ? null : source.id
                    )
                  }
                  className={`w-full flex items-center justify-between p-2 rounded-lg text-sm transition-colors ${
                    activeSource === source.id
                      ? "bg-primary/10 text-primary"
                      : "hover:bg-muted"
                  }`}
                >
                  <span>{source.name}</span>
                  <span className="text-muted-foreground">
                    {source.count.toLocaleString()}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Trending */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-500" />
              Trending Topics
            </h3>
            <div className="space-y-2">
              {trendingTopics.map((topic) => (
                <TrendingCard key={topic.name} topic={topic} />
              ))}
            </div>
          </div>

          {/* Recent Searches */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Recent Searches
            </h3>
            <div className="space-y-2">
              {[
                "transformer architecture",
                "RLHF training",
                "agent frameworks",
              ].map((query) => (
                <button
                  key={query}
                  className="w-full text-left p-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-colors"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="col-span-9">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm text-muted-foreground">
              Found <span className="font-semibold text-foreground">2,847</span>{" "}
              results
            </p>
            <select className="px-3 py-1.5 bg-card border border-border rounded-lg text-sm">
              <option>Sort by Relevance</option>
              <option>Sort by Date</option>
              <option>Sort by Citations</option>
            </select>
          </div>

          <div
            className={
              viewMode === "grid" ? "grid grid-cols-2 gap-4" : "space-y-4"
            }
          >
            {researchItems.map((item) => (
              <ResearchCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DiscoveryEngine;
