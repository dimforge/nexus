import {useEffect, useState, useRef, type ReactNode} from 'react';
import Layout from '@theme/Layout';
import styles from './demos.module.css';

// Each demo is a full all_examples app with its own built-in demo picker UI.
const demos = [
  {
    name: '3d',
    demo: 'all_examples3',
    title: '3D Demos',
    description: 'Rigid-body dynamics demos in 3D',
    source: 'https://github.com/dimforge/nexus/tree/main/crates/examples3d',
  },
  {
    name: '2d',
    demo: 'all_examples2',
    title: '2D Demos',
    description: 'Rigid-body dynamics demos in 2D',
    source: 'https://github.com/dimforge/nexus/tree/main/crates/examples2d',
  },
];

export default function Demos(): ReactNode {
  const [selected, setSelected] = useState<string | null>(null);
  const [activeDemo, setActiveDemo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Nexus runs its physics as WebGPU compute shaders: no WebGPU, no demo.
  useEffect(() => {
    setWebgpuSupported(typeof navigator !== 'undefined' && !!(navigator as any).gpu);
  }, []);

  // Handle URL hash for deep linking
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash && demos.some((d) => d.name === hash)) {
      setSelected(hash);
    } else {
      setSelected('3d');
    }

    const handleHashChange = () => {
      const newHash = window.location.hash.slice(1);
      if (newHash && demos.some((d) => d.name === newHash)) setSelected(newHash);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Handle demo transitions - clear iframe first to release WebGPU context
  useEffect(() => {
    if (selected === activeDemo) return;

    setIsLoading(true);

    // Force iframe cleanup by setting src to blank first
    if (iframeRef.current) {
      iframeRef.current.src = 'about:blank';
    }
    setActiveDemo(null);

    // Wait for the iframe to be cleared and GPU context to be released
    const timer = setTimeout(() => {
      setActiveDemo(selected);
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, [selected]);

  const handleSelect = (name: string) => {
    setSelected(name);
    window.location.hash = name;
  };

  const current = demos.find((d) => d.name === selected);

  return (
    <Layout
      title="Demos"
      description="Interactive nexus GPU physics demos running in your browser"
      noFooter
    >
      <div className={styles.container}>
        <div className={styles.toolbar}>
          <div className={styles.tabs}>
            {demos.map((demo) => (
              <button
                key={demo.name}
                className={`${styles.tab} ${selected === demo.name ? styles.tabSelected : ''}`}
                onClick={() => handleSelect(demo.name)}
              >
                {demo.title}
              </button>
            ))}
          </div>
          <span className={styles.hint}>
            Pick individual demos from the panel inside the viewer. First load
            may take a while (the physics engine ships as a large WASM module).
          </span>
        </div>

        {!webgpuSupported && (
          <div className={styles.webgpuWarning}>
            <strong>WebGPU is not available in this browser.</strong> Nexus
            runs its physics as WebGPU compute shaders. On Firefox, enable{' '}
            <code>dom.webgpu.enabled</code> in <code>about:config</code>; on
            Chromium, enable <code>Unsafe WebGPU Support</code> in{' '}
            <code>chrome://flags</code>. Safari is currently not supported.
          </div>
        )}

        <div className={styles.viewer}>
          {activeDemo ? (
            <>
              <iframe
                ref={iframeRef}
                key={activeDemo}
                src={`/demos/${demos.find((d) => d.name === activeDemo)?.demo}/`}
                title={activeDemo}
                className={styles.viewerFrame}
              />
              <div className={styles.viewerControls}>
                <a
                  href={current?.source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.sourceLink}
                >
                  &lt;/&gt; Source
                </a>
              </div>
            </>
          ) : isLoading ? (
            <div className={styles.placeholder}>
              Loading...
            </div>
          ) : (
            <div className={styles.placeholder}>
              Select a demo
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
