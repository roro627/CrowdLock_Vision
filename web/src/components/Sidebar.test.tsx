import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import Sidebar from './Sidebar';

describe('Sidebar', () => {
  it('calls onToggle for overlay buttons', async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();

    render(
      <Sidebar
        frame={null}
        stats={null}
        statsError={null}
        toggles={{ showBoxes: true, showHead: true, showBody: true, showDensity: true }}
        onToggle={onToggle}
        connection="open"
      />
    );

    await user.click(screen.getByRole('button', { name: /Bounding Boxes/i }));
    expect(onToggle).toHaveBeenCalledWith('showBoxes');

    await user.click(screen.getByRole('button', { name: /Head Markers/i }));
    expect(onToggle).toHaveBeenCalledWith('showHead');

    await user.click(screen.getByRole('button', { name: /Body Markers/i }));
    expect(onToggle).toHaveBeenCalledWith('showBody');

    await user.click(screen.getByRole('button', { name: /Density Heatmap/i }));
    expect(onToggle).toHaveBeenCalledWith('showDensity');
  });
});
