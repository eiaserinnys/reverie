/**
 * Greeting Phase Component
 *
 * Phase 0: Welcome screen
 * - Reverie installation welcome message
 * - Project introduction
 * - Prerequisites (DB, API Key)
 */

import { useSetup } from '../useSetup';

export function GreetingPhase() {
  const { nextPhase } = useSetup();

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Welcome Message */}
      <div className="text-center space-y-4">
        <div className="text-6xl mb-6">🌙</div>
        <h1 className="text-3xl font-bold">
          Welcome to Reverie Setup
        </h1>
        <p className="text-lg text-muted-foreground">
          Reverie is a long-term memory system for AI agents.
          <br />
          Just a few simple steps to get started.
        </p>
      </div>

      {/* Project Introduction */}
      <div className="bg-muted/30 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-3">What is Reverie?</h2>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Reverie enables AI agents to remember conversations and leverage past context
          for more natural and consistent interactions.
          It uses vector embedding-based retrieval to quickly find relevant memories.
        </p>
        <p className="text-xs text-muted-foreground/80 mt-2 italic">
          Note: Memory injection into agents is planned after retrieval performance optimization.
        </p>
      </div>

      {/* Prerequisites Cards */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Before You Begin</h2>

        <div className="grid gap-4 sm:grid-cols-2">
          {/* Database Card */}
          <div className="border border-border rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🗄️</span>
              <h3 className="font-medium">Database</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              <strong>PostgreSQL</strong> or <strong>SQLite</strong>
            </p>
            <p className="text-xs text-muted-foreground">
              PostgreSQL requires the pgvector extension.
              SQLite works out of the box with no extra setup.
            </p>
          </div>

          {/* API Key Card */}
          <div className="border border-border rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🔑</span>
              <h3 className="font-medium">OpenAI API Key</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Used for generating text embeddings
            </p>
            <p className="text-xs text-muted-foreground">
              Get your key from the{' '}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                OpenAI Platform
              </a>
              .
            </p>
          </div>
        </div>
      </div>

      {/* Start Button */}
      <div className="flex justify-center pt-4">
        <button
          onClick={nextPhase}
          className="px-8 py-3 text-lg font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Get Started →
        </button>
      </div>
    </div>
  );
}
