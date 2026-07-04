import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img
          src="/img/nexus-logo.png"
          alt="nexus logo"
          className={styles.heroLogo}
        />
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/demos">
            View Demos
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="https://github.com/dimforge/nexus">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function Intro(): ReactNode {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2">A Multiphysics Engine on the GPU</Heading>
            <p>
              Nexus is a cross-platform GPU-accelerated multiphysics engine
              for Rust. The whole simulation pipeline runs as compute shaders
              written in Rust with{' '}
              <Link to="https://github.com/Rust-GPU/rust-gpu">Rust-GPU</Link>{' '}
              and compiled to SPIR-V, executed through WebGPU (on Windows,
              macOS, Linux, and the browser), or through Metal, Cuda, or CPU.
            </p>
            <p>
              Rigid-body dynamics is available today, with colliders, joints,
              articulated multibodies, and URDF/MJCF robots. Other solvers —
              MPM for fluids and deformables, DEM for granular media, etc. —
              are work in progress.
            </p>
            <div className={styles.codeLinks}>
              <Link
                className="button button--primary"
                to="/demos">
                Try the Demos
              </Link>
              <Link
                className="button button--outline button--primary"
                to="https://docs.rs/nexus3d">
                API Documentation
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Features(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureGrid}>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>⚡</span>
            <h3>GPU-Accelerated</h3>
            <p>The whole physics pipeline runs as compute shaders — thousands of bodies in real time.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🌐</span>
            <h3>Cross-Platform</h3>
            <p>WebGPU on Windows, macOS, Linux, and the browser. CUDA, Metal, and CPU backends too.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🦀</span>
            <h3>Shaders in Rust</h3>
            <p>Compute shaders written in Rust with Rust-GPU and compiled to SPIR-V.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🧊</span>
            <h3>Rigid-Body Dynamics</h3>
            <p>Boxes, balls, convex shapes, trimeshes, heightfields, joints, and multibodies.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🤖</span>
            <h3>Robotics-Ready</h3>
            <p>Load URDF and MJCF robot descriptions, including MuJoCo Menagerie models.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🐍</span>
            <h3>Python Bindings</h3>
            <p>The 3D engine and viewer are also available from Python as the nexus3d module.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Cross-platform GPU multiphysics simulation for Rust"
      description="Nexus is a cross-platform GPU-accelerated multiphysics engine for Rust: rigid-body dynamics running as compute shaders via WebGPU, with shaders written in Rust using Rust-GPU.">
      <HomepageHeader />
      <main>
        <Intro />
        <Features />
      </main>
    </Layout>
  );
}
