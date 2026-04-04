export interface SocialLinks {
  github: string;
  linkedin: string;
  resume: string;
  twitter: string;
  portfolio: string;
  devto: string;
}

export interface SkillData {
  skill: string;
  value: number;
  fullMark: number;
}

export interface JobRecommendation {
  id: number;
  title: string;
  company: string;
  matchPercentage: number;
  salary: string;
  location: string;
  type: string;
  skills: string[];
  description: string;
  applyUrl: string;
}

export interface AISummary {
  overview: string;
  strengths: string[];
  gaps: string[];
  industryRelevanceScore: number;
  industryInsights: string;
  topSkills: string[];
}

export interface LearningResource {
  type: 'course' | 'book' | 'video' | 'documentation';
  title: string;
  provider?: string;
  url: string;
  free: boolean;
}

export interface LearningPathItem {
  id: number;
  topic: string;
  priority: 'high' | 'medium' | 'low';
  timeEstimate: string;
  explanation: string;
  resources: LearningResource[];
}

export interface UserProfile {
  links: SocialLinks;
  technicalSkills: SkillData[];
  softSkills: SkillData[];
  jobRecommendations: JobRecommendation[];
  aiSummary: AISummary;
  learningPath: LearningPathItem[];
}
