import React, {useEffect} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {useDoc, useDocsVersion} from '@docusaurus/plugin-content-docs/client';
import {parseGuideId, variants} from '@site/src/userGuideVariants';
import styles from './styles.module.css';

type ReadingPosition = {scrollY: number; sections: string[]; fraction: number};
let pendingScroll: {pathname: string; position: ReadingPosition} | undefined;

// Follow Rapier's selector: retain the fraction read in the current section,
// rather than a pixel offset, since the examples differ in length by language.
function readingLine() {
  const bar = document.querySelector<HTMLElement>(`.${styles.selector}`);
  return Math.max(0, bar?.getBoundingClientRect().bottom ?? 0);
}

function documentHeadings() {
  return Array.from(document.querySelectorAll<HTMLElement>(
    '.theme-doc-markdown :is(h1, h2, h3, h4, h5, h6)[id]',
  ))
    .filter(heading => heading.getClientRects().length > 0)
    .map(heading => ({id: heading.id, top: heading.getBoundingClientRect().top + window.scrollY}))
    .sort((a, b) => a.top - b.top);
}

function readingPosition(): ReadingPosition {
  const headings = documentHeadings();
  const reading = window.scrollY + readingLine();
  const next = headings.findIndex(heading => heading.top > reading + 1);
  const previous = (next === -1 ? headings.length : next) - 1;
  if (previous < 0) return {scrollY: window.scrollY, sections: [], fraction: 0};
  const end = next === -1 ? document.documentElement.scrollHeight : headings[next].top;
  return {
    scrollY: window.scrollY,
    sections: headings.slice(0, previous + 1).reverse().map(heading => heading.id),
    fraction: Math.min(1, Math.max(0, (reading - headings[previous].top) / Math.max(1, end - headings[previous].top))),
  };
}

function restoredScrollY(position: ReadingPosition) {
  const headings = documentHeadings();
  for (const [index, id] of position.sections.entries()) {
    const section = headings.findIndex(heading => heading.id === id);
    if (section !== -1) {
      const start = headings[section].top;
      const end = section + 1 < headings.length ? headings[section + 1].top : document.documentElement.scrollHeight;
      // A language-only section falls back to the nearest shared heading above it.
      return start + (index === 0 ? position.fraction * (end - start) : 0) - readingLine();
    }
  }
  return position.sections.length === 0 ? position.scrollY : 0;
}

function restoreScroll(position: ReadingPosition) {
  const content = document.querySelector('.theme-doc-markdown');
  const inputs = ['wheel', 'touchstart', 'keydown', 'mousedown'];
  const apply = () => window.scrollTo({top: restoredScrollY(position), behavior: 'instant'});
  const observer = typeof ResizeObserver !== 'undefined' && content ? new ResizeObserver(apply) : undefined;
  let timeout: ReturnType<typeof setTimeout>;
  const stop = () => {
    observer?.disconnect();
    clearTimeout(timeout);
    inputs.forEach(input => window.removeEventListener(input, stop, true));
  };
  apply();
  if (observer && content) {
    observer.observe(content);
    inputs.forEach(input => window.addEventListener(input, stop, true));
    timeout = setTimeout(stop, 3000);
  }
  return stop;
}

export default function GuideLanguageSelector() {
  const {metadata} = useDoc();
  const {docs} = useDocsVersion();
  const {pathname, search, hash} = useLocation();
  const current = parseGuideId(metadata.id);

  useEffect(() => {
    if (!pendingScroll || pendingScroll.pathname !== pathname) return undefined;
    const {position} = pendingScroll;
    pendingScroll = undefined;
    let dispose: (() => void) | undefined;
    let secondFrame: number | undefined;
    // Restore after the router has performed its own top/hash scrolling.
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => { dispose = restoreScroll(position); });
    });
    return () => {
      cancelAnimationFrame(firstFrame);
      if (secondFrame !== undefined) cancelAnimationFrame(secondFrame);
      dispose?.();
    };
  }, [pathname]);

  if (!current) return null;

  return (
    <nav className={styles.selector} aria-label="User-guide language">
      <div className={styles.languages}>
        {variants.map(variant => {
          const target = docs[`user_guides/${variant.id}/${current.page}`];
          if (!target) {
            return <span key={variant.id} className={clsx(styles.language, styles.missing)}
              aria-disabled="true" title={`${variant.description}: this page is unavailable`}>
              {variant.label}
            </span>;
          }
          const selected = variant.id === current.variant.id;
          const targetPath = metadata.permalink.replace(
            `/user_guides/${current.variant.id}/`, `/user_guides/${variant.id}/`,
          );
          return (
            <Link key={variant.id} to={`${targetPath}${search}${hash}`}
              className={clsx(styles.language, selected && styles.selected)}
              aria-current={selected ? 'page' : undefined} title={variant.description}
              onClick={event => {
                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                if (selected) {
                  event.preventDefault();
                  return;
                }
                pendingScroll = {pathname: targetPath, position: readingPosition()};
              }}>
              {variant.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
