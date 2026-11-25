import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QuerySettings, type QueryParams } from './QuerySettings';

describe('QuerySettings', () => {
  const defaultParams: QueryParams = {
    pprMinScore: 0.0001,
    pprMaxIter: 20,
    tau: 0.5,
    damping: 0.85,
    beta: 0.5,
    topK: 25,
    halfLifeDays: null,
  };

  const defaultProps = {
    params: defaultParams,
    onParamsChange: vi.fn(),
    visible: true,
    embedded: true, // Required for rendering content
  };

  it('renders nothing when not visible', () => {
    const { container } = render(<QuerySettings {...defaultProps} visible={false} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing in non-embedded mode (legacy behavior)', () => {
    const { container } = render(<QuerySettings {...defaultProps} embedded={false} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders header and content when visible and embedded', () => {
    render(<QuerySettings {...defaultProps} />);
    expect(screen.getByText('Query Settings')).toBeInTheDocument();
    expect(screen.getByText('PPR Min Score')).toBeInTheDocument();
  });

  it('displays all parameter labels', () => {
    render(<QuerySettings {...defaultProps} />);

    expect(screen.getByText('PPR Min Score')).toBeInTheDocument();
    expect(screen.getByText('PPR Max Iter')).toBeInTheDocument();
    expect(screen.getByText('Tau (Entity Activation)')).toBeInTheDocument();
    expect(screen.getByText('Damping (PPR)')).toBeInTheDocument();
    expect(screen.getByText('Beta (PPR Weight)')).toBeInTheDocument();
    expect(screen.getByText('Top K (Results)')).toBeInTheDocument();
    expect(screen.getByText('Time Decay (Half-Life)')).toBeInTheDocument();
  });

  it('displays current parameter values', () => {
    render(<QuerySettings {...defaultProps} />);

    // Check that values are displayed (formatted)
    expect(screen.getByText('0.0001000')).toBeInTheDocument(); // pprMinScore
    expect(screen.getByText('20')).toBeInTheDocument(); // pprMaxIter
    // tau and beta both have 0.500, so we use getAllByText
    expect(screen.getAllByText('0.500')).toHaveLength(2); // tau and beta
    expect(screen.getByText('0.850')).toBeInTheDocument(); // damping
    expect(screen.getByText('25')).toBeInTheDocument(); // topK
    expect(screen.getByText('Off')).toBeInTheDocument(); // halfLifeDays null
  });

  it('calls onParamsChange when slider is changed', () => {
    const onParamsChange = vi.fn();
    render(<QuerySettings {...defaultProps} onParamsChange={onParamsChange} />);

    // Find the topK slider (max="100")
    const sliders = screen.getAllByRole('slider');
    const topKSlider = sliders.find(s => s.getAttribute('max') === '100');

    if (topKSlider) {
      fireEvent.change(topKSlider, { target: { value: '75' } });

      expect(onParamsChange).toHaveBeenCalledWith({
        ...defaultParams,
        topK: 75,
      });
    }
  });

  it('shows parameter descriptions', () => {
    render(<QuerySettings {...defaultProps} />);

    expect(screen.getByText(/Min score for PPR nodes/i)).toBeInTheDocument();
    expect(screen.getByText(/Entity activation threshold/i)).toBeInTheDocument();
    expect(screen.getByText(/Number of passages to retrieve/i)).toBeInTheDocument();
  });

  it('enables time decay when checkbox is checked', () => {
    const onParamsChange = vi.fn();
    render(<QuerySettings {...defaultProps} onParamsChange={onParamsChange} />);

    const checkboxes = screen.getAllByRole('checkbox');
    const timeDecayCheckbox = checkboxes[0];

    fireEvent.click(timeDecayCheckbox);

    expect(onParamsChange).toHaveBeenCalledWith({
      ...defaultParams,
      halfLifeDays: 365,
    });
  });
});
