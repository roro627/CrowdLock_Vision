import { useEffect, useState } from 'react';
import VideoOverlay from './components/VideoOverlay';
import Sidebar from './components/Sidebar';
import SourceSelector from './components/SourceSelector';
import { useMetadataStream } from './hooks/useWebsocket';
import { useStats } from './hooks/useStats';
import { api } from './api/client';
import './styles/index.css';

const API_BASE = api.base;

function App() {
  const wsUrl = API_BASE.replace('http', 'ws') + '/stream/metadata';
  const videoUrl = `${API_BASE}/stream/video`;
  const { latest, status, getById } = useMetadataStream(wsUrl);
  const { stats, error: statsError } = useStats(API_BASE);
  const [toggles, setToggles] = useState({
    showBoxes: true,
    showHead: true,
    showBody: true,
    showDensity: true,
  });
  const [configStatus, setConfigStatus] = useState<'idle' | 'saving' | 'error' | 'saved'>('idle');
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const handleToggle = (key: keyof typeof toggles) => {
    setToggles((t) => ({ ...t, [key]: !t[key] }));
  };

  return (
    <div className="min-h-screen text-slate-100">
      <header className="p-6 flex items-center justify-between">
        <div>
          <p className="uppercase tracking-[0.2em] text-xs text-slate-400">CrowdLock Vision</p>
          <h1 className="text-2xl font-bold text-accent">Dashboard</h1>
        </div>
        <div className="text-sm text-slate-400">{now.toLocaleString()}</div>
      </header>
      <main className="px-6 pb-10 grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-6 items-start">
        <VideoOverlay
          frame={latest}
          getFrameById={getById}
          showBoxes={toggles.showBoxes}
          showHead={toggles.showHead}
          showBody={toggles.showBody}
          showDensity={toggles.showDensity}
          videoUrl={videoUrl}
          connection={status}
          backendError={statsError}
        />
        <div className="space-y-4">
          <Sidebar
            frame={latest}
            stats={stats}
            statsError={statsError}
            toggles={toggles}
            onToggle={handleToggle}
            connection={status}
          />
          <SourceSelector onStatus={setConfigStatus} />
          {configStatus === 'saving' && <p className="text-sm text-slate-400">Updating backend…</p>}
          {configStatus === 'saved' && <p className="text-sm text-accent">Backend updated.</p>}
          {configStatus === 'error' && (
            <p className="text-sm text-red-400">Failed to update backend.</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
