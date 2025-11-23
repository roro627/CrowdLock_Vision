import { useMemo, useState } from 'react';
import VideoOverlay from './components/VideoOverlay';
import Sidebar from './components/Sidebar';
import SourceSelector from './components/SourceSelector';
import { useMetadataStream } from './hooks/useWebsocket';
import './styles/index.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const wsUrl = useMemo(() => API_BASE.replace('http', 'ws') + '/stream/metadata', []);
  const videoUrl = `${API_BASE}/stream/video`;
  const { latest, status } = useMetadataStream(wsUrl);
  const [toggles, setToggles] = useState({ showBoxes: true, showHead: true, showBody: true, showDensity: true });
  const [configStatus, setConfigStatus] = useState<'idle' | 'saving' | 'error' | 'saved'>('idle');

  const handleToggle = (key: keyof typeof toggles) => {
    setToggles((t) => ({ ...t, [key]: !t[key] }));
  };

  return (
    <div className="min-h-screen text-slate-100">
      <header className="p-6 flex items-center justify-between">
        <div>
          <p className="uppercase tracking-[0.2em] text-xs text-slate-400">CrowdLock Vision</p>
          <h1 className="text-2xl font-bold text-accent">Human Lock-On Dashboard</h1>
        </div>
        <div className="text-sm text-slate-400">{new Date().toLocaleString()}</div>
      </header>
      <main className="px-6 pb-10 grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-6 items-start">
        <VideoOverlay
          frame={latest}
          showBoxes={toggles.showBoxes}
          showHead={toggles.showHead}
          showBody={toggles.showBody}
          showDensity={toggles.showDensity}
          videoUrl={videoUrl}
        />
        <div className="space-y-4">
          <Sidebar frame={latest} toggles={toggles} onToggle={handleToggle} connection={status} />
          <SourceSelector onStatus={setConfigStatus} />
          {configStatus === 'saving' && <p className="text-sm text-slate-400">Updating backend…</p>}
          {configStatus === 'saved' && <p className="text-sm text-accent">Backend updated.</p>}
          {configStatus === 'error' && <p className="text-sm text-red-400">Failed to update backend.</p>}
        </div>
      </main>
    </div>
  );
}

export default App;
