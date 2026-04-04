import { getIronSession, type SessionOptions } from 'iron-session';
import { cookies } from 'next/headers';
import type { SocialLinks, UserProfile } from '@/lib/types';

export type ThemeMode = 'light' | 'dark';

export interface DashboardPreferences {
  theme?: ThemeMode;
  links?: Partial<SocialLinks>;
}

export interface AnalysisSessionData {
  signature: string;
  profile: UserProfile;
}

interface DashboardSessionData {
  preferences?: DashboardPreferences;
  analysis?: AnalysisSessionData;
}

function resolveSessionPassword(): string {
  const configured = process.env.SESSION_SECRET;

  if (configured && configured.length >= 32) {
    return configured;
  }

  if (process.env.NODE_ENV === 'production') {
    throw new Error('SESSION_SECRET must be set and at least 32 characters long in production.');
  }

  return 'dev-only-change-this-session-secret-minimum-32-characters';
}

function getSessionOptions(): SessionOptions {
  return {
    password: resolveSessionPassword(),
    cookieName: 'session',
    cookieOptions: {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 30,
    },
  };
}

export async function getDashboardSession() {
  // Next.js 16 cookies() may be async; cast to any because iron-session expects CookieStore
  const cookieStore = (await cookies()) as unknown as any;
  return getIronSession<DashboardSessionData>(cookieStore, getSessionOptions());
}
