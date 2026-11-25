/**
 * API Keys Phase Component
 *
 * Phase 3: API Key & Model Configuration
 * - OpenAI API Key 입력
 * - API Key 유효성 검증
 * - 모델 선택 (Embedding, Indexing NER, Query NER, Enrich Stage 1/2)
 */

import { useState, useCallback, useEffect } from 'react';
import { useSetup } from '../useSetup';
import { PasswordField, CheckItem, type CheckItemStatus } from '../shared';
import { saveApiKey } from '../../../lib/setup-api';

// Available OpenAI models for embeddings
const EMBEDDING_MODELS = [
  { id: 'text-embedding-3-small', name: 'text-embedding-3-small', description: 'Fast and cost-effective' },
  { id: 'text-embedding-3-large', name: 'text-embedding-3-large', description: 'Higher quality embeddings' },
  { id: 'text-embedding-ada-002', name: 'text-embedding-ada-002', description: 'Legacy model' },
];

// Available chat models (for NER and Enrich)
const CHAT_MODELS = [
  { id: 'gpt-4.1-mini', name: 'gpt-4.1-mini', description: 'Fast, cost-effective' },
  { id: 'gpt-4.1', name: 'gpt-4.1', description: 'Balanced performance' },
  { id: 'gpt-5-nano', name: 'gpt-5-nano', description: 'Ultra-fast, minimal cost' },
  { id: 'gpt-5-mini', name: 'gpt-5-mini', description: 'Good balance (Recommended)' },
  { id: 'gpt-5', name: 'gpt-5', description: 'Highest quality' },
  { id: 'gpt-5.2', name: 'gpt-5.2', description: 'Latest, best reasoning' },
];

// Tab definitions
const TABS = [
  { id: 'apikey', label: 'API Key', icon: '🔑' },
  { id: 'embedding', label: 'Embedding', icon: '📐' },
  { id: 'indexing', label: 'Indexing NER', icon: '📝' },
  { id: 'query', label: 'Query NER', icon: '🔍' },
  { id: 'preprocessor', label: 'Preprocessor', icon: '⚙️' },
  { id: 'enrich1', label: 'Enrich 1', icon: '✨' },
  { id: 'enrich2', label: 'Enrich 2', icon: '🎯' },
] as const;

type TabId = typeof TABS[number]['id'];

// Default model settings (production defaults)
const DEFAULT_MODELS = {
  embedding: 'text-embedding-3-small',
  indexing_ner: 'gpt-5-mini',
  query_ner: 'gpt-5-mini',
  preprocessor: 'gpt-5-mini',
  enrich_stage1: 'gpt-5-mini',
  enrich_stage2: 'gpt-5.2',
};

interface ModelSelectorProps {
  label: string;
  description: string;
  models: typeof EMBEDDING_MODELS | typeof CHAT_MODELS;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

function ModelSelector({ label, description, models, value, onChange, disabled }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <p className="text-xs text-muted-foreground">{description}</p>
      <div className="grid gap-2">
        {models.map((model) => (
          <label
            key={model.id}
            className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
              value === model.id
                ? 'border-primary bg-primary/5'
                : 'border-border hover:bg-muted/50'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <input
              type="radio"
              name={label}
              value={model.id}
              checked={value === model.id}
              onChange={(e) => onChange(e.target.value)}
              className="text-primary focus:ring-primary"
              disabled={disabled}
            />
            <div>
              <span className="text-sm font-medium">{model.name}</span>
              <span className="text-xs text-muted-foreground ml-2">
                {model.description}
              </span>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}

export function ApiKeysPhase() {
  const { status, updatePhaseStatus, nextPhase, prevPhase } = useSetup();

  const [activeTab, setActiveTab] = useState<TabId>('apikey');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState({
    embedding: DEFAULT_MODELS.embedding,
    indexing_ner: DEFAULT_MODELS.indexing_ner,
    query_ner: DEFAULT_MODELS.query_ner,
    preprocessor: DEFAULT_MODELS.preprocessor,
    enrich_stage1: DEFAULT_MODELS.enrich_stage1,
    enrich_stage2: DEFAULT_MODELS.enrich_stage2,
  });
  const [validationStatus, setValidationStatus] = useState<CheckItemStatus>('pending');
  const [validationMessage, setValidationMessage] = useState<string>();
  const [validationError, setValidationError] = useState<string>();
  const [isValidating, setIsValidating] = useState(false);

  // Masked placeholder for existing API key
  const MASKED_API_KEY = '********************************';

  // Check if already validated and pre-fill with masked key
  useEffect(() => {
    if (status?.phases.apikeys.validated) {
      setValidationStatus('passed');
      setValidationMessage('API key configured');
    }
    // Pre-fill with masked value if key exists in .env
    if (status?.phases.apikeys.has_api_key && !apiKey) {
      setApiKey(MASKED_API_KEY);
    }
  }, [status, apiKey]);

  const handleApiKeyChange = useCallback((value: string) => {
    setApiKey(value);
    // Reset validation when key changes
    if (validationStatus !== 'pending') {
      setValidationStatus('pending');
      setValidationMessage(undefined);
      setValidationError(undefined);
    }
  }, [validationStatus]);

  const handleModelChange = useCallback((key: keyof typeof models, value: string) => {
    setModels(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleValidateAndSave = useCallback(async () => {
    const trimmedKey = apiKey.trim();

    // If using masked key (existing key), just proceed with model settings
    const isUsingExistingKey = trimmedKey === MASKED_API_KEY && status?.phases.apikeys.has_api_key;

    if (!trimmedKey) {
      setValidationStatus('failed');
      setValidationError('API key is required');
      setActiveTab('apikey');
      return;
    }

    // Basic format validation (skip if using existing key)
    if (!isUsingExistingKey && !trimmedKey.startsWith('sk-')) {
      setValidationStatus('failed');
      setValidationError('API key should start with "sk-"');
      setActiveTab('apikey');
      return;
    }

    setIsValidating(true);
    setValidationStatus('checking');
    setValidationMessage(undefined);
    setValidationError(undefined);
    updatePhaseStatus('apikeys', { status: 'in_progress' });

    try {
      // Don't send masked key - send empty string to keep existing key
      const result = await saveApiKey({
        openai_api_key: isUsingExistingKey ? '' : trimmedKey,
        model: models.embedding,
        indexing_ner_model: models.indexing_ner,
        query_ner_model: models.query_ner,
        preprocessor_model: models.preprocessor,
        enrich_stage1_model: models.enrich_stage1,
        enrich_stage2_model: models.enrich_stage2,
      });

      if (result.success) {
        setValidationStatus('passed');
        setValidationMessage('API key validated and all models saved');
        updatePhaseStatus('apikeys', {
          status: 'complete',
          validated: true,
        });
        // Auto-advance after short delay
        setTimeout(() => nextPhase(), 500);
      } else {
        setValidationStatus('failed');
        setValidationError(result.error || 'Validation failed');
        updatePhaseStatus('apikeys', { status: 'failed' });
      }
    } catch (err) {
      setValidationStatus('failed');
      setValidationError(err instanceof Error ? err.message : 'Validation failed');
      updatePhaseStatus('apikeys', { status: 'failed' });
    } finally {
      setIsValidating(false);
    }
  }, [apiKey, models, status, updatePhaseStatus, nextPhase]);

  const isAlreadyConfigured = status?.phases.apikeys.validated === true && validationStatus === 'passed';

  const renderTabContent = () => {
    switch (activeTab) {
      case 'apikey':
        return (
          <PasswordField
            label="OpenAI API Key"
            value={apiKey}
            onChange={(e) => handleApiKeyChange(e.target.value)}
            placeholder="sk-..."
            description="Get your API key from https://platform.openai.com/api-keys"
            required
            disabled={isAlreadyConfigured}
          />
        );
      case 'embedding':
        return (
          <ModelSelector
            label="Embedding Model"
            description="Model for generating vector embeddings. Used to convert text into semantic vectors for similarity search."
            models={EMBEDDING_MODELS}
            value={models.embedding}
            onChange={(v) => handleModelChange('embedding', v)}
            disabled={isAlreadyConfigured}
          />
        );
      case 'indexing':
        return (
          <ModelSelector
            label="Indexing NER Model"
            description="Model for Named Entity Recognition during indexing. Extracts entities from conversation turns."
            models={CHAT_MODELS}
            value={models.indexing_ner}
            onChange={(v) => handleModelChange('indexing_ner', v)}
            disabled={isAlreadyConfigured}
          />
        );
      case 'query':
        return (
          <ModelSelector
            label="Query NER Model"
            description="Model for extracting entities from search queries. Faster model recommended for real-time use."
            models={CHAT_MODELS}
            value={models.query_ner}
            onChange={(v) => handleModelChange('query_ner', v)}
            disabled={isAlreadyConfigured}
          />
        );
      case 'preprocessor':
        return (
          <ModelSelector
            label="Preprocessor Model"
            description="Model for turn-level filtering before OpenIE. Filters out non-informative turns (greetings, simple tool calls)."
            models={CHAT_MODELS}
            value={models.preprocessor}
            onChange={(v) => handleModelChange('preprocessor', v)}
            disabled={isAlreadyConfigured}
          />
        );
      case 'enrich1':
        return (
          <ModelSelector
            label="Enrich Stage 1 Model"
            description="Model for parallel fact extraction from retrieved passages. Runs once per passage."
            models={CHAT_MODELS}
            value={models.enrich_stage1}
            onChange={(v) => handleModelChange('enrich_stage1', v)}
            disabled={isAlreadyConfigured}
          />
        );
      case 'enrich2':
        return (
          <ModelSelector
            label="Enrich Stage 2 Model"
            description="Model for re-ranking and synthesizing extracted facts. Runs once per query."
            models={CHAT_MODELS}
            value={models.enrich_stage2}
            onChange={(v) => handleModelChange('enrich_stage2', v)}
            disabled={isAlreadyConfigured}
          />
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Configure your OpenAI API key and select models for each pipeline stage.
          All settings will be stored in your .env file.
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-1 overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[200px]">
        {renderTabContent()}
      </div>

      {/* Validation Status */}
      {validationStatus !== 'pending' && (
        <CheckItem
          name="API Key Validation"
          description="Verifying API key with OpenAI..."
          status={validationStatus}
          message={validationMessage}
          error={validationError}
          onRetry={handleValidateAndSave}
        />
      )}

      {/* Already Configured Notice */}
      {isAlreadyConfigured && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <p className="text-sm text-green-600 dark:text-green-400">
            API key is already configured. You can proceed to the next step or
            enter a new key to replace it.
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={prevPhase}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>

        <button
          onClick={handleValidateAndSave}
          disabled={isValidating || !apiKey.trim()}
          className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isValidating
            ? 'Saving...'
            : apiKey === MASKED_API_KEY
              ? 'Save Model Settings'
              : 'Validate & Save All'}
        </button>

        {isAlreadyConfigured && (
          <button
            onClick={nextPhase}
            className="px-4 py-2 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors"
          >
            Continue →
          </button>
        )}
      </div>

      {/* Security Note */}
      <div className="text-xs text-muted-foreground">
        <p>
          <strong>Security:</strong> Your API key is stored locally in the .env file
          and is never sent anywhere except to OpenAI's API.
        </p>
      </div>
    </div>
  );
}
