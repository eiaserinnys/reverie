/**
 * useSetup hook
 *
 * Setup Wizard Context를 사용하기 위한 hook
 */

import { useContext } from 'react';
import { SetupContext } from './SetupContext';
import type { SetupContextType } from './SetupContext';

export function useSetup(): SetupContextType {
  const context = useContext(SetupContext);
  if (context === undefined) {
    throw new Error('useSetup must be used within a SetupProvider');
  }
  return context;
}
