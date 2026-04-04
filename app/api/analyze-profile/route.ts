import { NextRequest, NextResponse } from 'next/server';
import { analyzeProfileText, type AIDetectedSkill } from '@/lib/ai';

interface SocialLinks {
  github?: string;
  linkedin?: string;
  resume?: string;
  portfolio?: string;
  twitter?: string;
  devto?: string;
}

interface Skill {
  name: string;
  confidence: number;
  source: string;
}

const BASE_SKILL_KEYWORDS = [
  'react',
  'javascript',
  'typescript',
  'node',
  'python',
  'vue',
  'angular',
  'tailwind',
  'nextjs',
  'express',
  'html',
  'css',
  'java',
  'graphql',
  'docker',
  'kubernetes',
  'sql',
  'aws',
];

function normalizeSkill(keyword: string) {
  return keyword
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((marker) => marker.charAt(0).toUpperCase() + marker.slice(1))
    .join(' ');
}

function extractSkillsFromText(text: string, source: string): Skill[] {
  const skills: Skill[] = [];
  const lowered = text.toLowerCase();
  const found = new Set<string>();

  BASE_SKILL_KEYWORDS.forEach((keyword) => {
    if (lowered.includes(keyword)) {
      found.add(keyword);
    }
  });

  found.forEach((keyword) => {
    skills.push({
      name: normalizeSkill(keyword),
      confidence: 0.6,
      source,
    });
  });

  return skills;
}

export async function POST(request: NextRequest) {
  try {
    const { links }: { links: SocialLinks } = await request.json();
    let extractedSkills: Skill[] = [];

    // Scrape GitHub
    if (links.github) {
      const githubSkills = await scrapeGitHub(links.github);
      extractedSkills.push(...githubSkills);
    }

    // Scrape LinkedIn
    if (links.linkedin) {
      const linkedinSkills = await scrapeLinkedIn(links.linkedin);
      extractedSkills.push(...linkedinSkills);
    }

    // Scrape Portfolio
    if (links.portfolio) {
      const portfolioSkills = await scrapePortfolio(links.portfolio);
      extractedSkills.push(...portfolioSkills);
    }

    // Scrape Dev.to
    if (links.devto) {
      const devtoSkills = await scrapeDevTo(links.devto);
      extractedSkills.push(...devtoSkills);
    }

    // Add fallback from resume and twitter URLs text
    if (links.resume) {
      extractedSkills.push(...extractSkillsFromText(links.resume, 'Resume URL'));
    }
    if (links.twitter) {
      extractedSkills.push(...extractSkillsFromText(links.twitter, 'Twitter URL'));
    }

    // Deduplicate skill results
    const uniqueSkillMap = new Map<string, Skill>();

    extractedSkills.forEach((skill) => {
      const key = `${skill.name.toLowerCase()}|${skill.source}`;
      if (!uniqueSkillMap.has(key)) {
        uniqueSkillMap.set(key, skill);
      }
    });

    extractedSkills = Array.from(uniqueSkillMap.values());

    // Combine with AI result
    const rawText = Object.values(links).filter(Boolean).join('\n\n') || 'No link text available';
    const aiAnalysis = await analyzeProfileText(rawText).catch((error) => {
      console.error('AI analysis failure: ', error);
      return {
        technicalSkills: [],
        softSkills: [],
        summary: '',
        gaps: [],
        recommendations: [],
        industryRelevanceScore: 0,
        industryInsights: '',
        topSkills: [],
        rawText,
      };
    });

    const aiSkills: Skill[] = [
      ...aiAnalysis.technicalSkills.map((s) => ({ name: s.name, confidence: s.confidence, source: s.source || 'AI-Technical' })),
      ...aiAnalysis.softSkills.map((s) => ({ name: s.name, confidence: s.confidence, source: s.source || 'AI-Soft' })),
    ];

    const combinedSkillsMap = new Map<string, Skill>();
    [...extractedSkills, ...aiSkills].forEach((skill) => {
      const key = `${skill.name.toLowerCase()}|${skill.source}`;
      if (!combinedSkillsMap.has(key)) {
        combinedSkillsMap.set(key, skill);
      }
    });

    const combinedSkills = Array.from(combinedSkillsMap.values());

    return NextResponse.json({
      skills: combinedSkills,
      aiSummary: aiAnalysis.summary,
      aiGaps: aiAnalysis.gaps,
      aiRecommendations: aiAnalysis.recommendations,
      aiTechnicalSkills: aiAnalysis.technicalSkills,
      aiSoftSkills: aiAnalysis.softSkills,
      aiIndustryRelevanceScore: aiAnalysis.industryRelevanceScore,
      aiIndustryInsights: aiAnalysis.industryInsights,
      aiTopSkills: aiAnalysis.topSkills,
    });
  } catch (error) {
    console.error('Error analyzing profile:', error);
    return NextResponse.json(
      { error: 'Failed to analyze profile' },
      { status: 500 }
    );
  }
}

async function scrapeGitHub(url: string): Promise<Skill[]> {
  try {
    const username = url.split('/').filter(Boolean).pop();
    if (!username) return [];

    const response = await fetch(`https://api.github.com/users/${username}/repos`);
    if (!response.ok) return [];

    const repos = await response.json();
    const languageSet = new Set<string>();

    for (const repo of repos.slice(0, 10)) {
      if (repo.language) {
        languageSet.add(repo.language.toLowerCase());
      }
      if (repo?.description) {
        extractSkillsFromText(repo.description, 'GitHub').forEach((skill) => {
          languageSet.add(skill.name.toLowerCase());
        });
      }
    }

    return Array.from(languageSet).map((lang): Skill => ({
      name: normalizeSkill(lang),
      confidence: 0.75,
      source: 'GitHub',
    }));
  } catch (error) {
    console.error('GitHub scrape failed:', error);
    return [];
  }
}

async function scrapeLinkedIn(url: string): Promise<Skill[]> {
  try {
    const candidateSkills = extractSkillsFromText(url, 'LinkedIn URL');
    if (candidateSkills.length > 0) return candidateSkills;

    const username = url.split('/').filter(Boolean).pop() || 'linkedin-user';
    return [
      { name: 'Leadership', confidence: 0.55, source: 'LinkedIn' },
      { name: normalizeSkill(username), confidence: 0.45, source: 'LinkedIn username' },
    ];
  } catch {
    return [{ name: 'Leadership', confidence: 0.5, source: 'LinkedIn' }];
  }
}

async function scrapePortfolio(url: string): Promise<Skill[]> {
  try {
    const response = await fetch(url);
    if (!response.ok) return [];

    const html = await response.text();
    const skills = extractSkillsFromText(html, 'Portfolio');
    return skills;
  } catch (error) {
    console.error('Portfolio scrape failed:', error);
    return [];
  }
}

async function scrapeDevTo(url: string): Promise<Skill[]> {
  try {
    const username = url.split('/').filter(Boolean).pop();
    if (!username) return [];

    const response = await fetch(`https://dev.to/api/articles?username=${username}`);
    if (!response.ok) return [];

    const articles = await response.json();
    let skillSet = new Set<string>();
    articles.forEach((article: { title: string; description?: string }) => {
      extractSkillsFromText(article.title, 'Dev.to').forEach((skill) => skillSet.add(skill.name.toLowerCase()));
      if (article.description) {
        extractSkillsFromText(article.description, 'Dev.to').forEach((skill) => skillSet.add(skill.name.toLowerCase()));
      }
    });

    return Array.from(skillSet).map((skill) => ({
      name: normalizeSkill(skill),
      confidence: 0.65,
      source: 'Dev.to',
    }));
  } catch (error) {
    console.error('Dev.to scrape failed:', error);
    return [];
  }
}