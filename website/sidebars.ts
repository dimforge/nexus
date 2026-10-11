import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import groups from './guide-pages.json';
import {variants} from './src/userGuideVariants';

// Each language has its own sidebar and next/previous sequence, like Rapier.
const sidebars: SidebarsConfig = Object.fromEntries(variants.map(variant => [
  `user_guide_${variant.id}`,
  [{
    type: 'category' as const,
    label: `User Guide: ${variant.description}`,
    collapsible: false,
    items: groups.map(group => ({
      type: 'category' as const,
      label: group.label,
      items: group.pages.map(page => `user_guides/${variant.id}/${page}`),
    })),
  }],
]));
export default sidebars;
