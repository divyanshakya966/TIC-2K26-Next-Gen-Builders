import { NextResponse } from 'next/server';
import { getDashboardSession, type ThemeMode } from '@/lib/session';
import type { SocialLinks } from '@/lib/types';

const LINK_KEYS = ['github', 'linkedin', 'resume', 'twitter', 'portfolio', 'devto'] as const;

function sanitizeTheme(input: unknown): ThemeMode | undefined {
  if (input === 'light' || input === 'dark') {
    return input;
  }

  return undefined;
}

function sanitizeLinkValue(input: string): string | undefined {
  const trimmed = input.trim();
  if (!trimmed) {
    return undefined;
  }

  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return undefined;
    }

    parsed.hash = '';
    return parsed.toString();
  } catch {
    return undefined;
  }
}

function sanitizeLinks(input: unknown): Partial<SocialLinks> {
  if (!input || typeof input !== 'object') {
    return {};
  }

  const normalizedLinks: Partial<SocialLinks> = {};

  for (const key of LINK_KEYS) {
    const value = (input as Record<string, unknown>)[key];
    if (typeof value === 'string') {
      const sanitized = sanitizeLinkValue(value);
      if (sanitized) {
        normalizedLinks[key] = sanitized;
      }
    }
  }

  return normalizedLinks;
}

export async function GET() {
  const session = await getDashboardSession();

  return NextResponse.json({
    theme: session.preferences?.theme,
    links: session.preferences?.links ?? {},
  });
}

export async function POST(request: Request) {
  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }

  const payload = (body ?? {}) as Record<string, unknown>;
  const theme = sanitizeTheme(payload.theme);
  const links = sanitizeLinks(payload.links);

  const hasLinks = Object.keys(links).length > 0;
  const hasTheme = typeof theme !== 'undefined';

  if (!hasTheme && !hasLinks) {
    return NextResponse.json(
      { error: 'No valid preference values provided in payload.' },
      { status: 400 },
    );
  }

  const session = await getDashboardSession();
  const previousPreferences = session.preferences ?? {};

  session.preferences = {
    ...previousPreferences,
    ...(hasTheme ? { theme } : {}),
    ...(hasLinks ? { links: { ...previousPreferences.links, ...links } } : {}),
  };

  await session.save();

  return NextResponse.json({
    ok: true,
    preferences: session.preferences,
  });
}
