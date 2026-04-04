import { getIronSession, type SessionOptions } from 'iron-session';
import { cookies } from 'next/headers';
import type { SocialLinks } from '@/lib/types';

export type ThemeMode = 'light' | 'dark';

export interface DashboardPreferences {
  theme?: ThemeMode;
  links?: Partial<SocialLinks>;
  expandedJobs?: boolean;
}

interface DashboardSessionData {
  preferences?: DashboardPreferences;
}

const sessionPassword =
  process.env.SESSION_SECRET ?? 'dev-only-change-this-session-secret-minimum-32-characters';

const sessionOptions: SessionOptions = {
  password: sessionPassword,
  cookieName: 'tic2k26_session',
  cookieOptions: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 60 * 60 * 24 * 30,
  },
};

export async function getDashboardSession() {
  // Next.js 16 cookies() may be async; cast to any because iron-session expects CookieStore
  const cookieStore = (await cookies()) as unknown as any;
  return getIronSession<DashboardSessionData>(cookieStore, sessionOptions);
}
