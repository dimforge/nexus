import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'nexus',
  tagline: 'Cross-platform GPU multiphysics simulation for Rust',
  favicon: 'img/nexus-logo-small.png',

  future: {
    v4: true,
  },

  url: 'https://nexus.dimforge.com',
  baseUrl: '/',

  organizationName: 'dimforge',
  projectName: 'nexus',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          exclude: ['**/templates/**'],
          editUrl: ({docPath}) => {
            const templatePath = docPath.replace(/user_guides\/(rust|python|all)\//, 'user_guides/templates/');
            return `https://github.com/dimforge/nexus/edit/main/website/docs/${templatePath}`;
          },
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/nexus-logo.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'nexus',
      logo: {
        alt: 'nexus Logo',
        src: 'img/nexus-logo-small.png',
      },
      items: [
        {
          type: 'dropdown',
          label: 'User Guide',
          position: 'left',
          items: [
            {label: 'Rust (2D and 3D)', to: '/docs/user_guides/rust/getting_started'},
            {label: 'Python (3D)', to: '/docs/user_guides/python/getting_started'},
            {label: 'All languages', to: '/docs/user_guides/all/getting_started'},
          ],
        },
        {
          to: '/demos',
          label: 'Demos',
          position: 'left',
        },
        {
          href: 'https://docs.rs/nexus3d',
          label: 'API Docs',
          position: 'left',
        },
        {
          value: '<a class="header-button-donate" href="https://github.com/sponsors/dimforge" target="_blank" rel="noopener noreferrer">Donate ♥</a>',
          className: 'header-button-donate',
          position: 'right',
          type: 'html'
        },
        {
          href: 'https://dimforge.com',
          label: 'Dimforge',
          position: 'right',
        },
        {
          href: 'https://github.com/dimforge/nexus',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Resources',
          items: [
            {label: 'User Guide (Rust)', to: '/docs/user_guides/rust/getting_started'},
            {label: 'User Guide (Python)', to: '/docs/user_guides/python/getting_started'},
            {
              label: 'API Documentation (3D)',
              href: 'https://docs.rs/nexus3d',
            },
            {
              label: 'API Documentation (2D)',
              href: 'https://docs.rs/nexus2d',
            },
            {
              label: 'Examples',
              href: 'https://github.com/dimforge/nexus/tree/main/crates',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discord',
              href: 'https://discord.gg/vt9DJSW',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/dimforge/nexus/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/dimforge/nexus/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/dimforge/nexus',
            },
            {
              label: 'Crates.io',
              href: 'https://crates.io/crates/nexus3d',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Dimforge. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'python', 'toml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
