import { useMemo } from 'react';
import type { FramePayload, StatsPayload } from '../types';

interface Props {
  frame: FramePayload | null;
  stats: StatsPayload | null;
  statsError: string | null;
  toggles: {
    showBoxes: boolean;
    showHead: boolean;
    showBody: boolean;
    showDensity: boolean;
  };
  onToggle: (key: keyof Props['toggles']) => void;
  connection: string;
}

/** Render a single toggle button row. */
const ToggleRow = ({
  label,
  value,
  onClick,
}: {
  label: string;
  value: boolean;
  onClick: () => void;
}) => (
  <button
    className={`button-ghost flex items-center justify-between w-full ${value ? 'bg-slate-800 border-accent/60' : ''}`}
    onClick={onClick}
  >
    <span>{label}</span>
    <span className="text-accent font-semibold">{value ? 'ON' : 'OFF'}</span>
  </button>
);

/**
 * Right-side control panel.
 *
 * Shows connection/health info and exposes overlay toggles for the video canvas.
 */
export function Sidebar({ frame, stats, statsError, toggles, onToggle, connection }: Props) {
  const densest = useMemo(
    () => frame?.density?.max_cell ?? stats?.densest_cell ?? null,
    [frame, stats]
  );
  const fps = frame?.fps ?? stats?.fps ?? 0;
  const people = frame?.persons.length ?? stats?.total_persons ?? 0;
  return (
    <div className="card p-4 space-y-4 w-full md:w-80">
      <div>
        <p className="text-xs uppercase tracking-wide text-slate-400">Connection</p>
        <p className="text-lg font-semibold text-accent">{connection}</p>
        {statsError && <p className="text-xs text-red-400">Backend: {statsError}</p>}
      </div>
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-slate-400">Overlays</p>
        <ToggleRow
          label="Bounding Boxes"
          value={toggles.showBoxes}
          onClick={() => onToggle('showBoxes')}
        />
        <ToggleRow
          label="Head Markers"
          value={toggles.showHead}
          onClick={() => onToggle('showHead')}
        />
        <ToggleRow
          label="Body Markers"
          value={toggles.showBody}
          onClick={() => onToggle('showBody')}
        />
        <ToggleRow
          label="Density Heatmap"
          value={toggles.showDensity}
          onClick={() => onToggle('showDensity')}
        />
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="card p-3">
          <p className="text-slate-400 text-xs">People</p>
          <p className="text-2xl font-bold">{people}</p>
        </div>
        <div className="card p-3">
          <p className="text-slate-400 text-xs">FPS</p>
          <p className="text-2xl font-bold">{fps.toFixed(1)}</p>
        </div>
        <div className="card p-3 col-span-2">
          <p className="text-slate-400 text-xs">Densest Cell</p>
          <p className="text-lg font-semibold">{densest ? `${densest[0]}, ${densest[1]}` : '–'}</p>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
