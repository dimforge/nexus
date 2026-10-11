// Shared by the page selector and the language-specific sidebars.
export const variants = [
  {id: 'rust', label: 'Rust', description: 'Rust (2D and 3D)'},
  {id: 'python', label: 'Python', description: 'Python (3D)'},
  {id: 'all', label: 'All', description: 'All languages'},
] as const;

export function parseGuideId(id: string) {
  const match = /^user_guides\/([^/]+)\/([^/]+)$/.exec(id);
  const variant = match && variants.find(candidate => candidate.id === match[1]);
  return variant ? {variant, page: match[2]} : undefined;
}
