/**
 * Profile Phase Component
 *
 * Phase 4: Profile Configuration
 * - 프로필 이름 설정
 * - Claude Code 프로젝트 다중 선택 (체크박스)
 * - 커스텀 경로 추가 지원
 */

import { useState, useCallback, useEffect } from 'react';
import { useSetup } from '../useSetup';
import { InputField } from '../shared';
import {
  saveProfile,
  listClaudeProjects,
  type ClaudeProject,
} from '../../../lib/setup-api';

interface FormData {
  profileName: string;
  selectedPaths: string[];  // Multi-folder selection
  customPath: string;       // Custom path input
}

interface FormErrors {
  profileName?: string;
  paths?: string;
}

export function ProfilePhase() {
  const { status, updatePhaseStatus, nextPhase, prevPhase } = useSetup();

  const [formData, setFormData] = useState<FormData>({
    profileName: 'default',
    selectedPaths: [],
    customPath: '',
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string>();
  const [showCustomInput, setShowCustomInput] = useState(false);

  // Claude projects state
  const [claudeProjects, setClaudeProjects] = useState<ClaudeProject[]>([]);
  const [claudeConfigPath, setClaudeConfigPath] = useState<string | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [claudeError, setClaudeError] = useState<string>();

  // Load Claude projects on mount
  useEffect(() => {
    async function loadProjects() {
      setLoadingProjects(true);
      try {
        const result = await listClaudeProjects();
        if (result.success) {
          setClaudeProjects(result.projects);
          setClaudeConfigPath(result.config_path);
          setClaudeError(undefined);
        } else {
          setClaudeError(result.error);
          setClaudeProjects([]);
        }
      } catch (err) {
        setClaudeError(err instanceof Error ? err.message : 'Failed to load projects');
        setClaudeProjects([]);
      } finally {
        setLoadingProjects(false);
      }
    }
    loadProjects();
  }, []);

  // Initialize from status
  useEffect(() => {
    if (status?.phases.profile) {
      const profile = status.phases.profile;
      setFormData(prev => ({
        ...prev,
        profileName: profile.name || prev.profileName,
        // Load existing paths from status
        selectedPaths: profile.session_paths || (profile.session_path ? [profile.session_path] : []),
      }));
    }
  }, [status]);

  const handleProfileNameChange = useCallback((value: string) => {
    setFormData(prev => ({ ...prev, profileName: value }));
    setErrors(prev => ({ ...prev, profileName: undefined }));
    setSaveError(undefined);
  }, []);

  const handlePathToggle = useCallback((path: string) => {
    setFormData(prev => {
      const isSelected = prev.selectedPaths.includes(path);
      const newPaths = isSelected
        ? prev.selectedPaths.filter(p => p !== path)
        : [...prev.selectedPaths, path];
      return { ...prev, selectedPaths: newPaths };
    });
    setErrors(prev => ({ ...prev, paths: undefined }));
    setSaveError(undefined);
  }, []);

  const handleAddCustomPath = useCallback(() => {
    const trimmedPath = formData.customPath.trim();
    if (!trimmedPath) return;

    // Validate path is absolute
    const isUnixPath = trimmedPath.startsWith('/') || trimmedPath.startsWith('~');
    const isWindowsPath = /^[A-Za-z]:[/\\]/.test(trimmedPath);
    if (!isUnixPath && !isWindowsPath) {
      setErrors(prev => ({ ...prev, paths: 'Path must be absolute (e.g., /path/to/sessions or D:/path)' }));
      return;
    }

    // Don't add duplicates
    if (formData.selectedPaths.includes(trimmedPath)) {
      setErrors(prev => ({ ...prev, paths: 'This path is already added' }));
      return;
    }

    setFormData(prev => ({
      ...prev,
      selectedPaths: [...prev.selectedPaths, trimmedPath],
      customPath: '',
    }));
    setShowCustomInput(false);
    setErrors(prev => ({ ...prev, paths: undefined }));
  }, [formData.customPath, formData.selectedPaths]);

  const handleRemovePath = useCallback((path: string) => {
    setFormData(prev => ({
      ...prev,
      selectedPaths: prev.selectedPaths.filter(p => p !== path),
    }));
  }, []);

  const validateForm = useCallback((): boolean => {
    const newErrors: FormErrors = {};

    const trimmedName = formData.profileName.trim();
    if (!trimmedName) {
      newErrors.profileName = 'Profile name is required';
    } else if (!/^[a-zA-Z0-9_-]+$/.test(trimmedName)) {
      newErrors.profileName = 'Only letters, numbers, hyphens, and underscores allowed';
    }

    // Paths are optional - user can configure later
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData.profileName]);

  const handleSaveAndContinue = useCallback(async () => {
    if (!validateForm()) return;

    setIsSaving(true);
    setSaveError(undefined);
    updatePhaseStatus('profile', { status: 'in_progress' });

    try {
      const result = await saveProfile({
        name: formData.profileName.trim(),
        session_paths: formData.selectedPaths,
      });

      if (result.success) {
        updatePhaseStatus('profile', {
          status: 'complete',
          name: formData.profileName.trim(),
          session_paths: formData.selectedPaths,
        });
        nextPhase();
      } else {
        setSaveError(result.error || 'Failed to save profile');
        updatePhaseStatus('profile', { status: 'failed' });
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save profile');
      updatePhaseStatus('profile', { status: 'failed' });
    } finally {
      setIsSaving(false);
    }
  }, [validateForm, formData, updatePhaseStatus, nextPhase]);

  const isAlreadyConfigured = status?.phases.profile.status === 'complete';

  // Find which projects are selected
  const selectedProjects = claudeProjects.filter(
    p => formData.selectedPaths.includes(p.session_path)
  );

  // Custom paths (not from Claude projects)
  const customPaths = formData.selectedPaths.filter(
    path => !claudeProjects.some(p => p.session_path === path)
  );

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Configure your Reverie profile. Select one or more Claude Code projects
          to index their session history for semantic search.
        </p>
      </div>

      {/* Form */}
      <div className="space-y-4">
        <InputField
          label="Profile Name"
          value={formData.profileName}
          onChange={(e) => handleProfileNameChange(e.target.value)}
          placeholder="default"
          description="A unique name for this Reverie instance"
          error={errors.profileName}
          required
        />

        {/* Claude Projects Multi-Select */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">
            Session Files Paths
          </label>
          <p className="text-xs text-muted-foreground">
            Select Claude Code projects to index (you can select multiple)
          </p>

          {loadingProjects ? (
            <div className="px-3 py-2 text-sm text-muted-foreground border border-border rounded-md">
              Loading Claude projects...
            </div>
          ) : claudeError ? (
            <div className="space-y-2">
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
                <p className="text-sm text-yellow-600 dark:text-yellow-400">
                  ⚠️ {claudeError}
                </p>
              </div>
            </div>
          ) : claudeProjects.length > 0 ? (
            <div className="space-y-2">
              {/* Checkbox list */}
              <div className="border border-border rounded-md divide-y divide-border max-h-64 overflow-y-auto">
                {claudeProjects.map((project) => {
                  const isSelected = formData.selectedPaths.includes(project.session_path);
                  return (
                    <label
                      key={project.folder_name}
                      className={`flex items-start gap-3 p-3 cursor-pointer hover:bg-muted/50 transition-colors ${
                        isSelected ? 'bg-primary/5' : ''
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handlePathToggle(project.session_path)}
                        className="mt-0.5 h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-foreground truncate">
                          {project.display_name}
                        </div>
                        <div className="text-xs text-muted-foreground truncate">
                          {project.original_path}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {project.session_count} sessions
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>

              {claudeConfigPath && (
                <p className="text-xs text-muted-foreground">
                  Claude config: <code className="bg-muted px-1 rounded">{claudeConfigPath}</code>
                </p>
              )}
            </div>
          ) : (
            <div className="bg-muted/30 rounded-md p-3 text-sm text-muted-foreground">
              No Claude projects found. Add a custom path below.
            </div>
          )}

          {/* Custom paths list */}
          {customPaths.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Custom paths:</p>
              <div className="space-y-1">
                {customPaths.map(path => (
                  <div
                    key={path}
                    className="flex items-center gap-2 px-3 py-2 bg-muted/30 rounded-md text-sm"
                  >
                    <code className="flex-1 truncate">{path}</code>
                    <button
                      onClick={() => handleRemovePath(path)}
                      className="text-red-500 hover:text-red-700 text-xs"
                      title="Remove path"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Add custom path */}
          {showCustomInput ? (
            <div className="flex gap-2">
              <input
                type="text"
                value={formData.customPath}
                onChange={(e) => setFormData(prev => ({ ...prev, customPath: e.target.value }))}
                placeholder="/path/to/claude/sessions"
                className="flex-1 px-3 py-2 text-sm rounded-md border border-input bg-background focus:outline-none focus:ring-2 focus:ring-ring"
                onKeyDown={(e) => e.key === 'Enter' && handleAddCustomPath()}
              />
              <button
                onClick={handleAddCustomPath}
                className="px-3 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Add
              </button>
              <button
                onClick={() => {
                  setShowCustomInput(false);
                  setFormData(prev => ({ ...prev, customPath: '' }));
                }}
                className="px-3 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowCustomInput(true)}
              className="text-sm text-primary hover:text-primary/80"
            >
              + Add custom path
            </button>
          )}

          {errors.paths && (
            <p className="text-xs text-red-500">{errors.paths}</p>
          )}

          {/* Selected summary */}
          {formData.selectedPaths.length > 0 && (
            <div className="bg-primary/5 border border-primary/20 rounded-md p-3">
              <p className="text-xs font-medium text-primary">
                {formData.selectedPaths.length} path{formData.selectedPaths.length !== 1 ? 's' : ''} selected
              </p>
              {selectedProjects.length > 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  Total: {selectedProjects.reduce((sum, p) => sum + p.session_count, 0)} sessions from Claude projects
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {saveError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <p className="text-sm text-red-600 dark:text-red-400">
            {saveError}
          </p>
        </div>
      )}

      {/* Already Configured Notice */}
      {isAlreadyConfigured && !saveError && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <p className="text-sm text-green-600 dark:text-green-400">
            Profile is already configured. You can proceed or update the settings.
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
          onClick={handleSaveAndContinue}
          disabled={isSaving || loadingProjects}
          className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSaving ? 'Saving...' : 'Save & Continue →'}
        </button>

        {isAlreadyConfigured && (
          <button
            onClick={nextPhase}
            className="px-4 py-2 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors"
          >
            Skip →
          </button>
        )}
      </div>

      {/* Help Text */}
      <div className="bg-muted/30 rounded-lg p-4">
        <h4 className="text-sm font-medium mb-2">About Multi-Folder Indexing</h4>
        <ul className="text-xs text-muted-foreground space-y-1">
          <li>• Select multiple Claude Code projects to index together</li>
          <li>• Each project's session history will be searchable</li>
          <li>• You can add custom paths for other session directories</li>
          <li>• Paths can be updated later in the settings</li>
        </ul>
      </div>
    </div>
  );
}
