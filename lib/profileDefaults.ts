import type { SocialLinks, UserProfile } from './types';

const DEFAULT_LINKS: SocialLinks = {
  github: '',
  linkedin: '',
  resume: '',
  twitter: '',
  portfolio: '',
  devto: '',
};

const DEFAULT_TECHNICAL_SKILLS = [
  { skill: 'Frontend', value: 0, fullMark: 100 },
  { skill: 'Backend', value: 0, fullMark: 100 },
  { skill: 'Cloud', value: 0, fullMark: 100 },
  { skill: 'Testing', value: 0, fullMark: 100 },
];

const DEFAULT_SOFT_SKILLS = [
  { skill: 'Communication', value: 0, fullMark: 100 },
  { skill: 'Problem Solving', value: 0, fullMark: 100 },
  { skill: 'Teamwork', value: 0, fullMark: 100 },
  { skill: 'Adaptability', value: 0, fullMark: 100 },
];

export function createEmptyUserProfile(links: Partial<SocialLinks> = {}): UserProfile {
  return {
    links: {
      ...DEFAULT_LINKS,
      ...links,
    },
    technicalSkills: DEFAULT_TECHNICAL_SKILLS,
    softSkills: DEFAULT_SOFT_SKILLS,
    jobRecommendations: [],
    aiSummary: {
      overview: 'Paste a public GitHub, LinkedIn, resume, or portfolio URL and run analysis to generate a live profile.',
      strengths: [],
      gaps: ['No live profile has been analyzed yet.'],
      industryRelevanceScore: 0,
      atsScore: 0,
      atsFeedback: ['Upload a resume and run analysis to receive ATS scoring feedback.'],
      industryInsights: 'Waiting for a public profile link to analyze.',
      topSkills: [],
    },
    learningPath: [],
  };
}

export const emptyUserProfile = createEmptyUserProfile();