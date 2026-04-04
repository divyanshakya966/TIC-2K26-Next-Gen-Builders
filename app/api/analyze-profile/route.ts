import axios from 'axios';
import * as cheerio from 'cheerio';
import { NextRequest, NextResponse } from 'next/server';
import { analyzeProfileText } from '@/lib/ai';
import type { JobRecommendation, LearningPathItem, LearningResource } from '@/lib/types';

interface SocialLinks {
  github?: string;
  linkedin?: string;
  resume?: string;
  portfolio?: string;
  twitter?: string;
  devto?: string;
}

interface AnalyzeProfileRequestBody {
  links?: SocialLinks;
  resumeText?: string;
  resumeFileName?: string;
}

interface Skill {
  name: string;
  confidence: number;
  source: string;
}

interface ScrapedSource {
  source: string;
  url: string;
  title: string;
  description: string;
  text: string;
  skills: Skill[];
}

const SCRAPER_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
  Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
};

const SCRAPE_TIMEOUT_MS = 12000;

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
  'mongodb',
  'postgresql',
  'firebase',
  'git',
  'rest',
  'api',
  'testing',
  'jest',
  'webpack',
  'vite',
  'redux',
  'fastapi',
  'django',
  'flask',
  'rust',
  'go',
  'kotlin',
  'swift',
  'ci/cd',
  'devops',
  'cloud',
  'microservices',
];

const SOFT_SKILL_HINTS = ['communication', 'leadership', 'teamwork', 'problem solving', 'adaptability', 'creativity', 'ownership', 'presentation', 'mentoring'];

interface SkillFrequency {
  skill: string;
  count: number;
  sources: Set<string>;
}

function buildSkillFrequencyMap(allText: string): Map<string, SkillFrequency> {
  const frequencyMap = new Map<string, SkillFrequency>();
  const lowerText = allText.toLowerCase();

  for (const keyword of BASE_SKILL_KEYWORDS) {
    const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
    const matches = lowerText.match(regex);
    const count = matches ? matches.length : 0;

    if (count > 0) {
      const normalizedName = normalizeSkill(keyword);
      frequencyMap.set(normalizedName.toLowerCase(), {
        skill: normalizedName,
        count,
        sources: new Set(),
      });
    }
  }

  return frequencyMap;
}

function calculateFrequencyScore(frequency: SkillFrequency, maxCount: number): number {
  if (maxCount === 0) return 50;

  const relativeFrequency = frequency.count / maxCount;

  if (relativeFrequency >= 0.6) return 90 + Math.random() * 8;
  if (relativeFrequency >= 0.4) return 75 + Math.random() * 12;
  if (relativeFrequency >= 0.2) return 60 + Math.random() * 12;
  if (relativeFrequency >= 0.1) return 50 + Math.random() * 10;

  return Math.max(40, 45 + relativeFrequency * 5);
}

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
  const SOURCE_BASE_CONFIDENCE: Record<string, number> = {
    'GitHub': 0.75,
    'LinkedIn': 0.70,
    'Portfolio': 0.75,
    'Dev.to': 0.65,
    'Resume': 0.70,
    'Resume Upload': 0.84,
    'Twitter': 0.55,
  };

  const skills: Skill[] = [];
  const lowered = text.toLowerCase();
  const found = new Set<string>();

  BASE_SKILL_KEYWORDS.forEach((keyword) => {
    if (lowered.includes(keyword)) {
      found.add(keyword);
    }
  });

  const baseConfidence = SOURCE_BASE_CONFIDENCE[source] || 0.65;

  found.forEach((keyword) => {
    skills.push({
      name: normalizeSkill(keyword),
      confidence: baseConfidence,
      source,
    });
  });

  return skills;
}

function aggregateSkillsByNameWithBoost(skills: Skill[]): Skill[] {
  const SOURCE_CREDIBILITY_SCORE: Record<string, number> = {
    'GitHub': 0.95,
    'LinkedIn': 0.85,
    'Portfolio': 0.90,
    'Dev.to': 0.80,
    'Resume': 0.88,
    'Resume Upload': 0.95,
    'Twitter': 0.60,
  };

  const skillMap = new Map<string, { name: string; sources: Set<string>; confidences: number[] }>();

  for (const skill of skills) {
    const key = normalizeToken(skill.name);
    if (!skillMap.has(key)) {
      skillMap.set(key, {
        name: skill.name,
        sources: new Set(),
        confidences: [],
      });
    }

    const entry = skillMap.get(key)!;
    entry.sources.add(skill.source);
    entry.confidences.push(skill.confidence);
  }

  return Array.from(skillMap.values()).map((entry) => {
    const sourcesArray = Array.from(entry.sources);
    const sourceCredibilities = sourcesArray.map((s) => SOURCE_CREDIBILITY_SCORE[s] || 0.70);
    const avgSourceCredibility = sourceCredibilities.reduce((a, b) => a + b, 0) / sourceCredibilities.length;
    const maxConfidence = Math.max(...entry.confidences);
    const countBonus = Math.min(0.25, (entry.sources.size - 1) * 0.05);

    const finalConfidence = Math.min(0.99, maxConfidence + (avgSourceCredibility - 0.6) * 0.2 + countBonus);

    return {
      name: entry.name,
      confidence: finalConfidence,
      source: sourcesArray.join(' + '),
    };
  });
}

function cleanText(value: string | null | undefined): string {
  return value ? value.replace(/\s+/g, ' ').trim() : '';
}

function truncateText(value: string, maxLength = 3200): string {
  const normalized = cleanText(value);
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength)}…`;
}

function normalizeUrl(value: string): string {
  try {
    const parsed = new URL(value);
    parsed.hash = '';
    return parsed.toString();
  } catch {
    return value.trim();
  }
}

function buildUrlList(links: SocialLinks): string[] {
  return Object.values(links)
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .map((value) => normalizeUrl(value));
}

interface LinkVerificationResult {
  url: string;
  source: keyof SocialLinks;
  accessible: boolean;
  status?: number;
  error?: string;
}

async function verifyLinks(links: SocialLinks): Promise<{
  verified: LinkVerificationResult[];
  accessibleCount: number;
  inaccessibleLinks: LinkVerificationResult[];
}> {
  const results: LinkVerificationResult[] = [];

  const sourceNames = Object.entries(links)
    .filter(([, url]) => typeof url === 'string' && url.trim().length > 0)
    .map(([key, url]) => [key as keyof SocialLinks, normalizeUrl(url)] as const);

  const verificationPromises = sourceNames.map(async ([source, url]) => {
    try {
      const response = await axios.head(url, {
        headers: SCRAPER_HEADERS,
        timeout: 6000,
        validateStatus: () => true,
        maxRedirects: 5,
      });

      const accessible = response.status >= 200 && response.status < 400;
      const result: LinkVerificationResult = {
        url,
        source,
        accessible,
        status: response.status,
      };

      if (!accessible) {
        result.error = `HTTP ${response.status}: ${getStatusDescription(response.status)}`;
      }

      return result;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      return {
        url,
        source,
        accessible: false,
        error: `Connection failed: ${errorMsg.includes('timeout') ? 'Request timeout' : 'Unable to reach'}`,
      };
    }
  });

  const allResults = await Promise.all(verificationPromises);

  const accessibleCount = allResults.filter((r) => r.accessible).length;
  const inaccessibleLinks = allResults.filter((r) => !r.accessible);

  return {
    verified: allResults,
    accessibleCount,
    inaccessibleLinks,
  };
}

function getStatusDescription(status: number): string {
  const descriptions: Record<number, string> = {
    400: 'Bad Request',
    401: 'Unauthorized (Login Required)',
    403: 'Forbidden (Access Denied)',
    404: 'Not Found',
    410: 'Gone (Permanently Deleted)',
    429: 'Too Many Requests (Rate Limited)',
    500: 'Server Error',
    503: 'Service Unavailable',
  };

  return descriptions[status] || 'Inaccessible';
}

function extractTextBlock($: cheerio.CheerioAPI, selector: string, limit = 8): string {
  return $(selector)
    .toArray()
    .slice(0, limit)
    .map((element) => cleanText($(element).text()))
    .filter(Boolean)
    .join('\n');
}

function buildProfileText(links: SocialLinks, sources: ScrapedSource[], extractedSkills: Skill[]): string {
  const lines: string[] = [];

  for (const [key, value] of Object.entries(links)) {
    if (typeof value === 'string' && value.trim()) {
      lines.push(`${key}: ${value.trim()}`);
    }
  }

  for (const source of sources) {
    lines.push(`source: ${source.source}`);
    lines.push(`title: ${source.title}`);

    if (source.description) {
      lines.push(`description: ${source.description}`);
    }

    if (source.text) {
      lines.push(source.text);
    }
  }

  if (extractedSkills.length > 0) {
    lines.push(`detectedSkills: ${extractedSkills.map((skill) => `${skill.name} (${skill.source})`).join(', ')}`);
  }

  return lines.length > 0 ? truncateText(lines.join('\n\n'), 12000) : 'No profile links or detected skills available';
}

function buildUploadedResumeSource(resumeText: string, resumeFileName?: string): ScrapedSource {
  const cleanedResumeText = truncateText(resumeText, 6000);

  return {
    source: 'Resume Upload',
    url: resumeFileName ? `uploaded://${resumeFileName}` : 'uploaded://resume',
    title: resumeFileName || 'Uploaded Resume',
    description: 'Resume text imported from local upload.',
    text: cleanedResumeText,
    skills: extractSkillsFromText(cleanedResumeText, 'Resume Upload'),
  };
}

async function fetchHtml(url: string): Promise<{ html: string; finalUrl: string }> {
  const response = await axios.get(url, {
    headers: SCRAPER_HEADERS,
    timeout: SCRAPE_TIMEOUT_MS,
    responseType: 'text',
    validateStatus: (status) => status >= 200 && status < 400,
  });

  return {
    html: typeof response.data === 'string' ? response.data : String(response.data ?? ''),
    finalUrl: normalizeUrl(url),
  };
}

async function buildGenericSource(url: string, source: string): Promise<ScrapedSource> {
  const normalizedUrl = normalizeUrl(url);
  const { html, finalUrl } = await fetchHtml(normalizedUrl);
  const $ = cheerio.load(html);

  const title = cleanText(
    $('meta[property="og:title"]').attr('content')
      || $('meta[name="twitter:title"]').attr('content')
      || $('title').first().text()
      || source,
  );
  const description = cleanText(
    $('meta[property="og:description"]').attr('content')
      || $('meta[name="description"]').attr('content')
      || $('meta[name="twitter:description"]').attr('content')
      || '',
  );

  const text = truncateText(
    [
      title,
      description,
      extractTextBlock($, 'main'),
      extractTextBlock($, 'article'),
      extractTextBlock($, 'section'),
      extractTextBlock($, 'h1, h2, h3, p, li'),
    ]
      .filter(Boolean)
      .join('\n'),
  );

  return {
    source,
    url: finalUrl,
    title: title || source,
    description,
    text,
    skills: extractSkillsFromText(text, source),
  };
}

async function scrapeGitHub(url: string): Promise<ScrapedSource> {
  const normalizedUrl = normalizeUrl(url);

  try {
    const parsed = new URL(normalizedUrl);
    const username = parsed.pathname.split('/').filter(Boolean)[0];

    if (!username) {
      return await buildGenericSource(normalizedUrl, 'GitHub');
    }

    const profileUrl = `https://github.com/${username}`;
    const repositoriesUrl = `https://github.com/${username}?tab=repositories`;
    const [profilePage, repositoriesPage] = await Promise.all([fetchHtml(profileUrl), fetchHtml(repositoriesUrl)]);
    const profileDoc = cheerio.load(profilePage.html);
    const repositoriesDoc = cheerio.load(repositoriesPage.html);

    const title = cleanText(
      profileDoc('meta[property="og:title"]').attr('content')
        || profileDoc('title').first().text()
        || `${username} on GitHub`,
    );
    const description = cleanText(
      profileDoc('meta[property="og:description"]').attr('content')
        || profileDoc('meta[name="description"]').attr('content')
        || profileDoc('[data-testid="user-profile-bio"]').first().text()
        || '',
    );

    const repoHighlights = repositoriesDoc('article, .Box-row, li').toArray().slice(0, 10).map((element) => {
      const card = repositoriesDoc(element);
      const repoName = cleanText(card.find('a').first().text());
      const repoDescription = cleanText(card.find('p').first().text());
      const repoLanguage = cleanText(card.find('[itemprop="programmingLanguage"]').first().text());

      return [repoName, repoDescription, repoLanguage].filter(Boolean).join(' | ');
    }).filter(Boolean);

    const text = truncateText(
      [
        title,
        description,
        extractTextBlock(profileDoc, 'main'),
        extractTextBlock(profileDoc, 'h1, h2, h3, p, li'),
        extractTextBlock(repositoriesDoc, 'main'),
        extractTextBlock(repositoriesDoc, 'article, .Box-row, li'),
        repoHighlights.join('\n'),
      ]
        .filter(Boolean)
        .join('\n'),
    );

    return {
      source: 'GitHub',
      url: profilePage.finalUrl,
      title,
      description,
      text,
      skills: extractSkillsFromText(text, 'GitHub'),
    };
  } catch (error) {
    console.error('GitHub scrape failed:', error);
    return await buildGenericSource(normalizedUrl, 'GitHub');
  }
}

async function scrapeLinkedIn(url: string): Promise<ScrapedSource> {
  const normalizedUrl = normalizeUrl(url);

  try {
    const { html, finalUrl } = await fetchHtml(normalizedUrl);
    const $ = cheerio.load(html);

    const title = cleanText(
      $('meta[property="og:title"]').attr('content')
        || $('title').first().text()
        || 'LinkedIn profile',
    );
    const description = cleanText(
      $('meta[property="og:description"]').attr('content')
        || $('meta[name="description"]').attr('content')
        || $('meta[name="twitter:description"]').attr('content')
        || '',
    );

    const text = truncateText(
      [
        title,
        description,
        extractTextBlock($, 'main'),
        extractTextBlock($, 'section'),
        extractTextBlock($, 'article'),
        extractTextBlock($, 'h1, h2, h3, p, li'),
      ]
        .filter(Boolean)
        .join('\n'),
    );

    return {
      source: 'LinkedIn',
      url: finalUrl,
      title,
      description,
      text,
      skills: extractSkillsFromText(text, 'LinkedIn'),
    };
  } catch (error) {
    console.error('LinkedIn scrape failed:', error);
    return {
      source: 'LinkedIn',
      url: normalizedUrl,
      title: 'LinkedIn profile',
      description: 'Public LinkedIn content could not be fully scraped.',
      text: normalizedUrl,
      skills: [],
    };
  }
}

async function scrapePortfolio(url: string): Promise<ScrapedSource> {
  try {
    return await buildGenericSource(url, 'Portfolio');
  } catch (error) {
    console.error('Portfolio scrape failed:', error);
    return {
      source: 'Portfolio',
      url: normalizeUrl(url),
      title: 'Portfolio',
      description: 'Portfolio content could not be scraped.',
      text: normalizeUrl(url),
      skills: [],
    };
  }
}

async function scrapeResume(url: string): Promise<ScrapedSource> {
  try {
    return await buildGenericSource(url, 'Resume');
  } catch (error) {
    console.error('Resume scrape failed:', error);
    return {
      source: 'Resume',
      url: normalizeUrl(url),
      title: 'Resume',
      description: 'Resume content could not be scraped.',
      text: normalizeUrl(url),
      skills: [],
    };
  }
}

async function scrapeTwitter(url: string): Promise<ScrapedSource> {
  try {
    return await buildGenericSource(url, 'Twitter');
  } catch (error) {
    console.error('Twitter scrape failed:', error);
    return {
      source: 'Twitter',
      url: normalizeUrl(url),
      title: 'Twitter profile',
      description: 'Twitter content could not be scraped.',
      text: normalizeUrl(url),
      skills: [],
    };
  }
}

async function scrapeDevTo(url: string): Promise<ScrapedSource> {
  const normalizedUrl = normalizeUrl(url);

  try {
    const parsed = new URL(normalizedUrl);
    const username = parsed.pathname.split('/').filter(Boolean)[0];

    if (!username) {
      return await buildGenericSource(normalizedUrl, 'Dev.to');
    }

    const profileUrl = `https://dev.to/${username}`;
    const articlesUrl = `https://dev.to/api/articles?username=${username}`;
    const [profilePage, articlesResponse] = await Promise.all([fetchHtml(profileUrl), axios.get(articlesUrl, { headers: SCRAPER_HEADERS, timeout: SCRAPE_TIMEOUT_MS, validateStatus: (status) => status >= 200 && status < 400 })]);
    const doc = cheerio.load(profilePage.html);
    const articles = Array.isArray(articlesResponse.data) ? articlesResponse.data : [];

    const title = cleanText(
      doc('meta[property="og:title"]').attr('content')
        || doc('title').first().text()
        || `${username} on DEV`,
    );
    const description = cleanText(
      doc('meta[property="og:description"]').attr('content')
        || doc('meta[name="description"]').attr('content')
        || '',
    );

    const articleSummaries = articles.slice(0, 8).map((article: { title?: string; description?: string; tag_list?: string[]; reading_time_minutes?: number }) => {
      return [article.title, article.description, Array.isArray(article.tag_list) ? article.tag_list.join(', ') : '', typeof article.reading_time_minutes === 'number' ? `${article.reading_time_minutes} min read` : '']
        .filter(Boolean)
        .join(' | ');
    }).filter(Boolean);

    const text = truncateText(
      [
        title,
        description,
        extractTextBlock(doc, 'main'),
        extractTextBlock(doc, 'article'),
        extractTextBlock(doc, 'h1, h2, h3, p, li'),
        articleSummaries.join('\n'),
      ]
        .filter(Boolean)
        .join('\n'),
    );

    return {
      source: 'Dev.to',
      url: profilePage.finalUrl,
      title,
      description,
      text,
      skills: extractSkillsFromText(text, 'Dev.to'),
    };
  } catch (error) {
    console.error('Dev.to scrape failed:', error);
    return await buildGenericSource(normalizedUrl, 'Dev.to');
  }
}

async function collectScrapedSources(links: SocialLinks): Promise<ScrapedSource[]> {
  const tasks: Array<Promise<ScrapedSource>> = [];

  if (links.github) tasks.push(scrapeGitHub(links.github));
  if (links.linkedin) tasks.push(scrapeLinkedIn(links.linkedin));
  if (links.portfolio) tasks.push(scrapePortfolio(links.portfolio));
  if (links.resume) tasks.push(scrapeResume(links.resume));
  if (links.twitter) tasks.push(scrapeTwitter(links.twitter));
  if (links.devto) tasks.push(scrapeDevTo(links.devto));

  return await Promise.all(tasks);
}

function normalizeToken(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function createSkillLookup(values: string[]): Set<string> {
  return new Set(values.map(normalizeToken).filter(Boolean));
}

function getProfileSources(extractedSkills: Skill[]): string[] {
  const preferredOrder = ['GitHub', 'LinkedIn', 'Portfolio', 'Resume URL', 'Resume', 'Dev.to', 'Twitter URL', 'Twitter'];
  const sources = new Set<string>();

  for (const skill of extractedSkills) {
    skill.source.split(' + ').forEach((source) => {
      const cleaned = source.trim();
      if (cleaned) {
        sources.add(cleaned);
      }
    });
  }

  return preferredOrder.filter((source) => sources.has(source));
}

function formatList(items: string[]): string {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;
}

function detectFocusArea(skills: AnalysisResult['technicalSkills']): 'frontend' | 'fullstack' | 'platform' | 'general' {
  const techNames = skills.map((skill) => normalizeToken(skill.name));

  if (techNames.some((skill) => ['react', 'javascript', 'typescript', 'next.js', 'nextjs', 'frontend', 'ui'].includes(skill))) {
    if (techNames.some((skill) => ['node.js', 'node', 'sql', 'database', 'api', 'backend'].includes(skill))) {
      return 'fullstack';
    }

    return 'frontend';
  }

  if (techNames.some((skill) => ['docker', 'kubernetes', 'aws', 'cloud', 'devops', 'ci/cd', 'cicd'].includes(skill))) {
    return 'platform';
  }

  if (techNames.some((skill) => ['node.js', 'node', 'sql', 'database', 'api', 'backend', 'python'].includes(skill))) {
    return 'fullstack';
  }

  return 'general';
}

function buildPersonalizedSummary(
  analysis: AnalysisResult,
  sources: string[],
  focusArea: 'frontend' | 'fullstack' | 'platform' | 'general',
): string {
  const leadSkills = analysis.topSkills.slice(0, 3);
  const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
  const leadPhrase = leadSkills.length > 0 ? formatList(leadSkills) : 'core skills';

  if (focusArea === 'frontend') {
    return `${sourcePhrase} point to a frontend-leaning profile centered on ${leadPhrase}. GitHub and portfolio evidence suggest you build with React-oriented patterns, while your social/profile signals add communication and delivery strength.`;
  }

  if (focusArea === 'fullstack') {
    return `${sourcePhrase} show a balanced profile centered on ${leadPhrase}. Your signals suggest you can move between interface work and backend/problem-solving tasks, with enough collaboration evidence to operate across product teams.`;
  }

  if (focusArea === 'platform') {
    return `${sourcePhrase} indicate a profile leaning toward infrastructure and delivery around ${leadPhrase}. Your public/project footprint suggests you can build and ship, with room to deepen cloud and deployment ownership.`;
  }

  return `${sourcePhrase} highlight strengths in ${leadPhrase}. The strongest evidence comes from the skills and delivery patterns surfaced in your linked accounts, with collaboration signals showing up alongside technical work.`;
}

function buildPersonalizedGaps(
  analysis: AnalysisResult,
  technicalSkills: AnalysisResult['technicalSkills'],
  softSkills: AnalysisResult['softSkills'],
  sources: string[],
  focusArea: 'frontend' | 'fullstack' | 'platform' | 'general',
): string[] {
  const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
  const skillNames = createSkillLookup([...technicalSkills.map((skill) => skill.name), ...softSkills.map((skill) => skill.name)]);

  const gaps: string[] = [];

  if (focusArea === 'frontend') {
    if (!skillNames.has('node.js') && !skillNames.has('node') && !skillNames.has('backend')) {
      gaps.push(`Backend depth is not as visible across ${sourcePhrase} as your frontend work.`);
    }
    if (!skillNames.has('aws') && !skillNames.has('cloud') && !skillNames.has('docker')) {
      gaps.push(`Cloud and deployment signals are lighter than your React and UI evidence.`);
    }
    if (!skillNames.has('testing') && !skillNames.has('jest') && !skillNames.has('playwright')) {
      gaps.push('Testing and automation are not yet a dominant signal in the linked evidence.');
    }
  }

  if (focusArea === 'fullstack') {
    if (!skillNames.has('aws') && !skillNames.has('docker') && !skillNames.has('kubernetes')) {
      gaps.push(`Cloud delivery and infrastructure ownership are still less visible than your product work.`);
    }
    if (!skillNames.has('system design') && !skillNames.has('architecture')) {
      gaps.push('System design evidence is lighter than your implementation-level skills.');
    }
    if (!skillNames.has('testing') && !skillNames.has('ci/cd') && !skillNames.has('cicd')) {
      gaps.push('Testing depth and CI/CD automation could be stronger in the current profile signal.');
    }
  }

  if (focusArea === 'platform') {
    if (!skillNames.has('backend') && !skillNames.has('api') && !skillNames.has('sql')) {
      gaps.push('Backend and API ownership are less visible than your delivery-oriented signals.');
    }
    if (!skillNames.has('system design') && !skillNames.has('scalability')) {
      gaps.push('System design and scalability evidence is still thin across the linked profiles.');
    }
    if (!skillNames.has('testing') && !skillNames.has('observability')) {
      gaps.push('Reliability, testing, and observability signals are not yet prominent.');
    }
  }

  if (gaps.length === 0) {
    gaps.push(`Your linked profiles show solid technical direction, but the evidence across ${sourcePhrase} can be sharpened with more concrete project outcomes and role-specific depth.`);
    gaps.push('A few more project examples with measurable impact would make the profile easier to position for higher-fit roles.');
  }

  return gaps.slice(0, 4);
}

function buildPersonalizedStrengths(
  analysis: AnalysisResult,
  sources: string[],
  focusArea: 'frontend' | 'fullstack' | 'platform' | 'general',
): string[] {
  const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
  const strengths = new Set<string>();

  const topTech = analysis.technicalSkills.slice(0, 3);
  const topSoft = analysis.softSkills.slice(0, 2);

  if (topTech.length > 0) {
    strengths.add(`${sourcePhrase} show strength in ${formatList(topTech.map((skill) => skill.name))}.`);
  }

  if (topSoft.length > 0) {
    strengths.add(`Collaboration signals such as ${formatList(topSoft.map((skill) => skill.name))} are present across your profile evidence.`);
  }

  if (focusArea === 'frontend') {
    strengths.add('The profile is strongest where product-facing UI work, React patterns, and delivery quality intersect.');
  } else if (focusArea === 'fullstack') {
    strengths.add('Your evidence suggests you can connect interface work with backend problem solving when the role demands it.');
  } else if (focusArea === 'platform') {
    strengths.add('The strongest signal is execution: you appear comfortable shipping and iterating on engineering work.');
  } else {
    strengths.add('Your profile shows a mix of technical depth and communication that can be positioned toward multiple role types.');
  }

  return Array.from(strengths).slice(0, 5);
}

function aggregateSkillsByName(skills: Skill[]): Skill[] {
  const aggregated = new Map<string, { name: string; confidence: number; sources: string[] }>();

  for (const skill of skills) {
    const key = normalizeToken(skill.name);
    const existing = aggregated.get(key);

    if (!existing) {
      aggregated.set(key, {
        name: skill.name,
        confidence: skill.confidence,
        sources: [skill.source],
      });
      continue;
    }

    existing.confidence = Math.max(existing.confidence, skill.confidence);
    if (!existing.sources.includes(skill.source)) {
      existing.sources.push(skill.source);
    }
  }

  return Array.from(aggregated.values()).map((item) => ({
    name: item.name,
    confidence: item.confidence,
    source: item.sources.join(' + '),
  }));
}

type AnalysisResult = Awaited<ReturnType<typeof analyzeProfileText>>;

type SoftSignal = {
  name: string;
  score: number;
  confidence: number;
};

function buildSoftSkillSignals(extractedSkills: Skill[]): SoftSignal[] {
  const sourceText = extractedSkills.map((skill) => `${skill.name} ${skill.source}`).join(' ').toLowerCase();
  const sources = new Set(extractedSkills.map((skill) => normalizeToken(skill.source)));

  const signals: SoftSignal[] = [
    {
      name: 'Communication',
      score: 72 + (sources.has('linkedin') || sources.has('portfolio') ? 8 : 0),
      confidence: 0.82,
    },
    {
      name: 'Problem Solving',
      score: 70 + (sources.has('github') || sources.has('dev.to') ? 10 : 0),
      confidence: 0.8,
    },
    {
      name: 'Teamwork',
      score: 66 + (sources.has('github') || sources.has('linkedin') ? 8 : 0),
      confidence: 0.76,
    },
    {
      name: 'Leadership',
      score: 60 + (sources.has('linkedin') || sources.has('resume') ? 12 : 0),
      confidence: 0.7,
    },
    {
      name: 'Adaptability',
      score: 64 + (sources.has('portfolio') || sources.has('dev.to') ? 8 : 0),
      confidence: 0.74,
    },
    {
      name: 'Creativity',
      score: 62 + (sources.has('portfolio') || sourceText.includes('design') ? 10 : 0),
      confidence: 0.72,
    },
    {
      name: 'Ownership',
      score: 68 + (sources.has('github') || sources.has('portfolio') ? 8 : 0),
      confidence: 0.78,
    },
    {
      name: 'Presentation',
      score: 58 + (sources.has('portfolio') || sources.has('linkedin') ? 10 : 0),
      confidence: 0.68,
    },
    {
      name: 'Mentoring',
      score: 56 + (sources.has('dev.to') || sources.has('github') ? 8 : 0),
      confidence: 0.66,
    },
  ];

  return signals;
}

function deriveFallbackAISkills(extractedSkills: Skill[]): Pick<AnalysisResult, 'technicalSkills' | 'softSkills'> {
  const technical = new Map<string, { name: string; score: number; confidence: number }>();
  const soft = new Map<string, { name: string; score: number; confidence: number }>();

  for (const skill of extractedSkills) {
    const token = normalizeToken(skill.name);
    const score = Math.round(Math.min(96, Math.max(45, (skill.confidence || 0.6) * 100)));
    const normalized = normalizeSkill(skill.name);

    if (SOFT_SKILL_HINTS.some((hint) => token.includes(hint))) {
      soft.set(normalized, { name: normalized, score, confidence: clampConfidence(skill.confidence) });
    } else {
      technical.set(normalized, { name: normalized, score, confidence: clampConfidence(skill.confidence) });
    }
  }

  for (const signal of buildSoftSkillSignals(extractedSkills)) {
    if (!soft.has(signal.name)) {
      soft.set(signal.name, {
        name: signal.name,
        score: Math.min(96, Math.max(45, Math.round(signal.score))),
        confidence: clampConfidence(signal.confidence),
      });
    }
  }

  return {
    technicalSkills: Array.from(technical.values()).slice(0, 8).map((skill) => ({
      ...skill,
      type: 'technical' as const,
      source: 'Processed-Link-Data',
    })),
    softSkills: Array.from(soft.values()).slice(0, 6).map((skill) => ({
      ...skill,
      type: 'soft' as const,
      source: 'Processed-Link-Data',
    })),
  };
}

function clampConfidence(value: number): number {
  if (!Number.isFinite(value)) return 0.6;
  return Math.min(1, Math.max(0.35, value));
}

function buildIndustryInsights(
  industryRelevanceScore: number,
  topSkills: string[],
  gaps: string[],
  sources: string[],
  technicalSkills: AnalysisResult['technicalSkills'],
  softSkills: AnalysisResult['softSkills'],
): string {
  const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
  const strongest = topSkills.slice(0, 3);
  const strongestPhrase = strongest.length > 0 ? formatList(strongest) : 'core skills';
  const gapHeadline = gaps[0] ? gaps[0] : 'add clearer project outcomes and measurable impact to strengthen role fit';

  const technicalAvg = technicalSkills.length > 0
    ? Math.round(technicalSkills.reduce((sum, skill) => sum + skill.score, 0) / technicalSkills.length)
    : 0;
  const softAvg = softSkills.length > 0
    ? Math.round(softSkills.reduce((sum, skill) => sum + skill.score, 0) / softSkills.length)
    : 0;

  return `Relevance is ${industryRelevanceScore}% based on ${sourcePhrase}, with strongest evidence in ${strongestPhrase}. Technical signal averages ${technicalAvg}% and collaboration signal averages ${softAvg}%. Highest-impact next step: ${gapHeadline}`;
}

function enrichAnalysis(analysis: AnalysisResult, extractedSkills: Skill[]): AnalysisResult {
  const sources = getProfileSources(extractedSkills);
  const focusArea = detectFocusArea([...analysis.technicalSkills, ...analysis.softSkills]);
  const fallback = deriveFallbackAISkills(extractedSkills);

  const mergeSkills = (
    primary: AnalysisResult['technicalSkills'] | AnalysisResult['softSkills'],
    secondary: AnalysisResult['technicalSkills'] | AnalysisResult['softSkills'],
    type: 'technical' | 'soft',
  ) => {
    const combined = new Map<string, AnalysisResult['technicalSkills'][number]>();

    [...primary, ...secondary].forEach((skill) => {
      const key = normalizeToken(skill.name);
      const existing = combined.get(key);
      if (!existing || skill.score > existing.score) {
        combined.set(key, {
          ...skill,
          type,
          source: skill.source || 'Processed-Link-Data',
        });
      }
    });

    return Array.from(combined.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, type === 'technical' ? 8 : 6);
  };

  const technicalSkills = mergeSkills(
    analysis.technicalSkills.length > 0 ? analysis.technicalSkills : [],
    fallback.technicalSkills,
    'technical',
  );
  const softSkills = mergeSkills(
    analysis.softSkills.length > 0 ? analysis.softSkills : [],
    fallback.softSkills,
    'soft',
  );

  const mergedSkills = [...technicalSkills, ...softSkills];
  const topSkills = analysis.topSkills.length > 0
    ? analysis.topSkills
    : mergedSkills.sort((a, b) => b.score - a.score).slice(0, 6).map((skill) => skill.name);

  const summary = buildPersonalizedSummary({ ...analysis, technicalSkills, softSkills, topSkills }, sources, focusArea);

  const gaps = buildPersonalizedGaps({ ...analysis, technicalSkills, softSkills, topSkills }, technicalSkills, softSkills, sources, focusArea);

  const recommendations = analysis.recommendations.length > 0
    ? analysis.recommendations
    : [
        `Deepen evidence in ${topSkills[0] || 'your strongest area'} by attaching it to concrete portfolio or GitHub outcomes.`,
        focusArea === 'frontend'
          ? 'Add one backend or deployment project so your frontend strength looks more complete.'
          : focusArea === 'platform'
            ? 'Document one product-facing project to balance infrastructure strength with role breadth.'
            : 'Build one end-to-end project that shows product, implementation, and collaboration together.',
        `Use ${sources.length > 0 ? formatList(sources) : 'your linked profiles'} to show measurable impact, not just tooling keywords.`,
      ];

  const averageScore = mergedSkills.length > 0
    ? Math.round(mergedSkills.reduce((sum, skill) => sum + skill.score, 0) / mergedSkills.length)
    : 0;

  const industryRelevanceScore = analysis.industryRelevanceScore > 0 ? analysis.industryRelevanceScore : averageScore;
  const industryInsights = buildIndustryInsights(
    industryRelevanceScore,
    topSkills,
    gaps,
    sources,
    technicalSkills,
    softSkills,
  );

  const strengths = buildPersonalizedStrengths({ ...analysis, technicalSkills, softSkills, topSkills }, sources, focusArea);

  const hasResumeEvidence = extractedSkills.some((skill) => normalizeToken(skill.source).includes('resume'));
  const technicalAverage = technicalSkills.length > 0
    ? technicalSkills.reduce((sum, skill) => sum + skill.score, 0) / technicalSkills.length
    : 0;
  const softAverage = softSkills.length > 0
    ? softSkills.reduce((sum, skill) => sum + skill.score, 0) / softSkills.length
    : 0;

  const keywordCoverageComponent = Math.min(35, topSkills.length * 5 + technicalSkills.length * 2);
  const skillSignalComponent = (technicalAverage * 0.35) + (softAverage * 0.15);
  const resumeEvidenceComponent = hasResumeEvidence ? 16 : 6;
  const gapPenalty = Math.min(16, gaps.length * 3);

  const fallbackAtsScore = Math.max(
    38,
    Math.min(
      98,
      Math.round(keywordCoverageComponent + skillSignalComponent + resumeEvidenceComponent - gapPenalty),
    ),
  );

  const atsScore = analysis.atsScore > 0 ? analysis.atsScore : fallbackAtsScore;
  const atsFeedback = analysis.atsFeedback.length > 0
    ? analysis.atsFeedback
    : [
        'Mirror keywords from target job descriptions in your skills and project bullets.',
        'Add measurable impact lines (for example: reduced latency by 25%, built for 10k users).',
        'Keep section headings standard: Summary, Skills, Experience, Projects, Education.',
        'Use a clean one-column format to improve ATS parsing reliability.',
      ];

  return {
    ...analysis,
    technicalSkills,
    softSkills,
    summary,
    gaps,
    recommendations,
    industryRelevanceScore,
    atsScore,
    atsFeedback,
    industryInsights,
    topSkills,
    strengths,
  };
}

function buildAIStrengths(analysis: Awaited<ReturnType<typeof analyzeProfileText>>): string[] {
  const fromAnalysis = Array.isArray(analysis.strengths)
    ? analysis.strengths.map((item) => item.trim()).filter(Boolean)
    : [];

  if (fromAnalysis.length > 0) {
    const normalizedSeen = new Set<string>();
    const unique = fromAnalysis.filter((line) => {
      const normalized = normalizeToken(line);
      if (!normalized || normalizedSeen.has(normalized)) return false;
      normalizedSeen.add(normalized);
      return true;
    });

    return unique.slice(0, 5);
  }

  const fallback = new Set<string>();
  analysis.technicalSkills.slice(0, 3).forEach((skill) => fallback.add(`Technical depth shown in ${skill.name}.`));
  analysis.softSkills.slice(0, 2).forEach((skill) => fallback.add(`Collaboration signal present in ${skill.name}.`));

  return Array.from(fallback).slice(0, 5);
}

function buildAIJobRecommendations(analysis: Awaited<ReturnType<typeof analyzeProfileText>>): JobRecommendation[] {
  // If AI has generated job recommendations, use those
  if (analysis.jobRecommendations && analysis.jobRecommendations.length > 0) {
    const mapped = analysis.jobRecommendations
      .map((job, index) => ({
        id: index + 1,
        title: job.title,
        company: job.company,
        matchPercentage: job.matchPercentage,
        salary: job.salary || 'Competitive',
        location: job.location,
        type: job.type,
        skills: job.skills,
        description: job.description,
        applyUrl: '#',
        fitReason: job.fitReason,
      }))
      .sort((a, b) => b.matchPercentage - a.matchPercentage)
      .slice(0, 8);

    const selected: JobRecommendation[] = [];
    const used = new Set<string>();

    const addByType = (typeMatcher: (type: string) => boolean) => {
      const found = mapped.find((item) => typeMatcher(item.type) && !used.has(`${item.title}|${item.company}`));
      if (found) {
        used.add(`${found.title}|${found.company}`);
        selected.push(found);
      }
    };

    addByType((type) => type.toLowerCase() === 'full-time');
    addByType((type) => type.toLowerCase().includes('internship'));
    addByType((type) => type.toLowerCase() === 'contract');

    for (const item of mapped) {
      const key = `${item.title}|${item.company}`;
      if (!used.has(key)) {
        used.add(key);
        selected.push(item);
      }
      if (selected.length >= 5) break;
    }

    return selected.slice(0, 5).map((item, index) => ({ ...item, id: index + 1 }));
  }

  // Fallback to deterministic skill-based recommendations if AI didn't generate any
  const skillLookup = createSkillLookup([
    ...analysis.topSkills,
    ...analysis.technicalSkills.map((skill) => skill.name),
    ...analysis.softSkills.map((skill) => skill.name),
  ]);

  const topSkillsText = analysis.topSkills.slice(0, 3).join(', ') || 'your strongest signals';
  const gapTokens = createSkillLookup(analysis.gaps);

  const countHits = (items: string[]) => items.reduce((total, item) => total + (skillLookup.has(normalizeToken(item)) ? 1 : 0), 0);
  const countGapHits = (items: string[]) => items.reduce((total, item) => total + (gapTokens.has(normalizeToken(item)) ? 1 : 0), 0);
  const hasEarlyCareerSignals = analysis.technicalSkills.length < 7 || analysis.topSkills.length < 5;

  const roleCatalog = [
    {
      title: 'Frontend Engineer',
      company: 'Product Studio',
      salary: '$100K – $140K',
      location: 'Remote / Hybrid',
      type: 'Full-time',
      skills: ['React', 'TypeScript', 'Next.js', 'Accessibility'],
      signals: ['react', 'typescript', 'next.js', 'javascript', 'ui', 'css', 'frontend'],
      gapSignals: ['testing', 'accessibility'],
      description: 'Best for building polished, production-ready interfaces with strong component and UX quality.',
    },
    {
      title: 'Backend Engineer',
      company: 'Data Driven Products',
      salary: '$105K – $145K',
      location: 'Remote / Hybrid',
      type: 'Full-time',
      skills: ['Node.js', 'REST APIs', 'SQL', 'Testing'],
      signals: ['node', 'node.js', 'api', 'rest', 'sql', 'database', 'backend'],
      gapSignals: ['database', 'api', 'testing'],
      description: 'Fit for profiles with API and service-layer signals that can be expanded into ownership of backend systems.',
    },
    {
      title: 'Full Stack Engineer',
      company: 'ScaleUp Labs',
      salary: '$115K – $160K',
      location: 'Hybrid',
      type: 'Full-time',
      skills: ['React', 'Node.js', 'TypeScript', 'SQL'],
      signals: ['react', 'typescript', 'node', 'api', 'sql', 'fullstack', 'next.js'],
      gapSignals: ['system design', 'architecture'],
      description: 'Good fit when your profile shows both frontend delivery and backend implementation potential.',
    },
    {
      title: 'Cloud / DevOps Engineer',
      company: 'Cloud Native Systems',
      salary: '$115K – $165K',
      location: 'Remote',
      type: 'Full-time',
      skills: ['Docker', 'Kubernetes', 'CI/CD', 'AWS'],
      signals: ['docker', 'kubernetes', 'cloud', 'aws', 'devops', 'ci/cd', 'deployment'],
      gapSignals: ['cloud', 'devops', 'ci/cd'],
      description: 'Strong track when your signals include deployment, cloud tooling, or automation practices.',
    },
    {
      title: 'AI/ML Engineer Intern',
      company: 'Applied Intelligence Lab',
      salary: '$25 – $40 / hour',
      location: 'Remote',
      type: 'Internship (Paid)',
      skills: ['Python', 'Data Processing', 'APIs', 'Git'],
      signals: ['python', 'api', 'data', 'ml', 'ai', 'automation'],
      gapSignals: ['python', 'data'],
      description: 'Paid internship focused on practical model integration, data workflows, and engineering quality.',
    },
    {
      title: 'Frontend Developer Intern',
      company: 'Experience Design Collective',
      salary: '$18 – $28 / hour',
      location: 'Hybrid',
      type: 'Internship (Paid)',
      skills: ['React', 'JavaScript', 'CSS', 'Git'],
      signals: ['react', 'javascript', 'css', 'html', 'ui', 'frontend'],
      gapSignals: ['accessibility', 'testing'],
      description: 'Paid internship that turns strong UI foundations into measurable product-delivery outcomes.',
    },
    {
      title: 'Open Source Contributor Intern',
      company: 'Tech for Good Foundation',
      salary: 'Stipend / Unpaid',
      location: 'Remote',
      type: 'Internship (Unpaid)',
      skills: ['Git', 'Issue Triage', 'Documentation', 'Testing'],
      signals: ['git', 'github', 'testing', 'docs', 'communication'],
      gapSignals: ['communication', 'testing'],
      description: 'Unpaid internship designed for portfolio growth through shipped OSS contributions and maintainership practices.',
    },
    {
      title: 'Community Product Intern',
      company: 'Startup Incubator Network',
      salary: 'Unpaid (Certificate + Mentorship)',
      location: 'Remote / Hybrid',
      type: 'Internship (Unpaid)',
      skills: ['Communication', 'Research', 'Prototyping', 'Teamwork'],
      signals: ['communication', 'presentation', 'teamwork', 'ownership', 'product'],
      gapSignals: ['ownership', 'product'],
      description: 'Role emphasizes product storytelling, collaboration, and fast prototyping in startup environments.',
    },
    {
      title: 'Contract Frontend Developer',
      company: 'Freelance Product Team',
      salary: '$45 – $70 / hour',
      location: 'Remote',
      type: 'Contract',
      skills: ['React', 'Next.js', 'TypeScript', 'Performance'],
      signals: ['react', 'next.js', 'typescript', 'performance', 'ui'],
      gapSignals: ['testing', 'accessibility'],
      description: 'Contract role suited for shipping feature sprints and improving frontend performance in short cycles.',
    },
    {
      title: 'Contract API Developer',
      company: 'Integration Partners',
      salary: '$50 – $80 / hour',
      location: 'Remote',
      type: 'Contract',
      skills: ['Node.js', 'REST', 'Database', 'Cloud'],
      signals: ['node', 'api', 'rest', 'database', 'cloud'],
      gapSignals: ['api', 'database', 'cloud'],
      description: 'Contract role focused on integrations, API reliability, and production backend delivery.',
    },
  ];

  const ranked = roleCatalog
    .map((role) => {
      const signalHits = countHits(role.signals);
      const skillHits = countHits(role.skills);
      const gapHits = countGapHits(role.gapSignals);
      const topHits = analysis.topSkills.reduce((sum, skill) => sum + (role.signals.includes(normalizeToken(skill)) ? 1 : 0), 0);
      const typeBoost = hasEarlyCareerSignals && role.type.includes('Internship') ? 6 : role.type === 'Contract' ? 3 : 0;
      const weighted = 52 + signalHits * 7 + skillHits * 6 + topHits * 5 + gapHits * 2 + typeBoost;
      const matchPercentage = Math.max(55, Math.min(96, Math.round(weighted)));
      const fitReason = `Matched on ${Math.max(signalHits + skillHits, 1)} profile signals, with strongest overlap in ${topSkillsText}.`;

      return {
        ...role,
        matchPercentage,
        fitReason,
      };
    })
    .sort((a, b) => b.matchPercentage - a.matchPercentage);

  const selected: typeof ranked = [];
  const used = new Set<string>();

  const addByType = (typeMatcher: (type: string) => boolean) => {
    const found = ranked.find((item) => typeMatcher(item.type) && !used.has(`${item.title}|${item.company}`));
    if (found) {
      used.add(`${found.title}|${found.company}`);
      selected.push(found);
    }
  };

  // Force a varied recommendation set so the UI is not repetitive.
  addByType((type) => type.toLowerCase() === 'full-time');
  addByType((type) => type.toLowerCase().includes('paid'));
  addByType((type) => type.toLowerCase().includes('unpaid'));
  addByType((type) => type.toLowerCase() === 'contract');

  for (const role of ranked) {
    const key = `${role.title}|${role.company}`;
    if (!used.has(key)) {
      used.add(key);
      selected.push(role);
    }
    if (selected.length >= 5) break;
  }

  return selected.slice(0, 5).map((role, index) => ({
    id: index + 1,
    title: role.title,
    company: role.company,
    matchPercentage: role.matchPercentage,
    salary: role.salary,
    location: role.location,
    type: role.type,
    skills: role.skills,
    description: role.description,
    applyUrl: '#',
    fitReason: role.fitReason,
  }));
}

function buildAILearningPath(
  analysis: Awaited<ReturnType<typeof analyzeProfileText>>,
  sources: string[],
  focusArea: 'frontend' | 'fullstack' | 'platform' | 'general',
): LearningPathItem[] {
  const gapText = analysis.gaps.join(' ').toLowerCase();
  const topSkills = analysis.topSkills.slice(0, 3);
  const topSkillsText = topSkills.length > 0 ? formatList(topSkills) : 'your current strengths';
  const gapSet = createSkillLookup(analysis.gaps);
  const techSet = createSkillLookup(analysis.technicalSkills.map((skill) => skill.name));
  const softSet = createSkillLookup(analysis.softSkills.map((skill) => skill.name));

  const hasGap = (...terms: string[]) => terms.some((term) => gapSet.has(normalizeToken(term)) || gapText.includes(normalizeToken(term)));
  const hasTech = (...terms: string[]) => terms.some((term) => techSet.has(normalizeToken(term)));
  const hasSoft = (...terms: string[]) => terms.some((term) => softSet.has(normalizeToken(term)));

  const candidateResources: Record<string, LearningResource[]> = {
    'Python Backend & Automation': [
      {
        type: 'documentation',
        title: 'Python Documentation',
        provider: 'Python Software Foundation',
        url: 'https://docs.python.org/3/',
        free: true,
      },
      {
        type: 'documentation',
        title: 'FastAPI Docs',
        provider: 'FastAPI',
        url: 'https://fastapi.tiangolo.com/',
        free: true,
      },
      {
        type: 'documentation',
        title: 'pytest Docs',
        provider: 'pytest',
        url: 'https://docs.pytest.org/en/stable/',
        free: true,
      },
    ],
    'TypeScript & Architecture': [
      {
        type: 'documentation',
        title: 'TypeScript Handbook',
        provider: 'Official TypeScript Docs',
        url: 'https://www.typescriptlang.org/docs/',
        free: true,
      },
      {
        type: 'book',
        title: 'Programming TypeScript',
        provider: 'O\'Reilly',
        url: 'https://www.oreilly.com/library/view/programming-typescript/9781492037651/',
        free: false,
      },
      {
        type: 'course',
        title: 'TypeScript Deep Dive',
        provider: 'Udemy',
        url: 'https://www.udemy.com/',
        free: false,
      },
    ],
    'Frontend Delivery': [
      {
        type: 'documentation',
        title: 'React Docs',
        provider: 'React',
        url: 'https://react.dev/',
        free: true,
      },
      {
        type: 'documentation',
        title: 'Next.js Docs',
        provider: 'Vercel',
        url: 'https://nextjs.org/docs',
        free: true,
      },
      {
        type: 'course',
        title: 'Frontend Masters React Path',
        provider: 'Frontend Masters',
        url: 'https://frontendmasters.com/',
        free: false,
      },
    ],
    'Backend APIs & Data': [
      {
        type: 'documentation',
        title: 'Node.js Docs',
        provider: 'Node.js',
        url: 'https://nodejs.org/en/docs',
        free: true,
      },
      {
        type: 'documentation',
        title: 'PostgreSQL Docs',
        provider: 'PostgreSQL',
        url: 'https://www.postgresql.org/docs/',
        free: true,
      },
      {
        type: 'course',
        title: 'REST API Design',
        provider: 'Udemy',
        url: 'https://www.udemy.com/',
        free: false,
      },
    ],
    'Cloud, Deployment & CI/CD': [
      {
        type: 'documentation',
        title: 'Docker Docs',
        provider: 'Docker',
        url: 'https://docs.docker.com/',
        free: true,
      },
      {
        type: 'documentation',
        title: 'GitHub Actions Docs',
        provider: 'GitHub',
        url: 'https://docs.github.com/en/actions',
        free: true,
      },
      {
        type: 'course',
        title: 'AWS Skill Builder',
        provider: 'Amazon Web Services',
        url: 'https://skillbuilder.aws/',
        free: true,
      },
    ],
    'Testing & Quality': [
      {
        type: 'documentation',
        title: 'Playwright Docs',
        provider: 'Microsoft',
        url: 'https://playwright.dev/',
        free: true,
      },
      {
        type: 'documentation',
        title: 'Jest Docs',
        provider: 'Jest',
        url: 'https://jestjs.io/docs/getting-started',
        free: true,
      },
      {
        type: 'course',
        title: 'Testing JavaScript Applications',
        provider: 'Frontend Masters',
        url: 'https://frontendmasters.com/',
        free: false,
      },
    ],
    'System Design & Storytelling': [
      {
        type: 'book',
        title: 'Designing Data-Intensive Applications',
        provider: 'O\'Reilly',
        url: 'https://dataintensive.net/',
        free: false,
      },
      {
        type: 'video',
        title: 'ByteByteGo System Design Videos',
        provider: 'ByteByteGo',
        url: 'https://www.youtube.com/@ByteByteGo',
        free: true,
      },
      {
        type: 'documentation',
        title: 'System Design Primer',
        provider: 'GitHub',
        url: 'https://github.com/donnemartin/system-design-primer',
        free: true,
      },
    ],
    'Communication & Product Delivery': [
      {
        type: 'documentation',
        title: 'Product Discovery Reading List',
        provider: 'Mind the Product',
        url: 'https://www.mindtheproduct.com/',
        free: true,
      },
      {
        type: 'course',
        title: 'Technical Communication for Engineers',
        provider: 'Coursera',
        url: 'https://www.coursera.org/',
        free: false,
      },
      {
        type: 'book',
        title: 'Made to Stick',
        provider: 'Random House',
        url: 'https://heathbrothers.com/books/made-to-stick/',
        free: false,
      },
    ],
  };

  const candidates: Array<{
    id: number;
    topic: string;
    priority: LearningPathItem['priority'];
    timeEstimate: string;
    explanation: string;
    resources: LearningResource[];
    score: number;
  }> = [
    {
      id: 1,
      topic: 'Python Backend & Automation',
      priority: hasTech('python') ? 'high' : 'medium',
      timeEstimate: '3–5 weeks',
      explanation: `Your profile shows Python as a real signal, so this path turns that strength into more visible backend, API, and automation evidence.`,
      resources: candidateResources['Python Backend & Automation'],
      score: (hasTech('python') ? 4 : 0) + (hasGap('backend', 'api', 'testing') ? 2 : 0) + (focusArea === 'general' ? 1 : 0),
    },
    {
      id: 2,
      topic: 'TypeScript & Architecture',
      priority: hasTech('typescript', 'react') ? 'high' : 'medium',
      timeEstimate: '3–4 weeks',
      explanation: `Your profile around ${topSkillsText} suggests this is the fastest way to make your existing work more reusable and production-ready.`,
      resources: candidateResources['TypeScript & Architecture'],
      score: (hasTech('typescript') ? 3 : 0) + (hasTech('react', 'next.js') ? 2 : 0) + (hasGap('architecture', 'system design') ? 1 : 0),
    },
    {
      id: 3,
      topic: 'Frontend Delivery',
      priority: focusArea === 'frontend' || hasTech('react', 'next.js', 'ui') ? 'high' : 'medium',
      timeEstimate: '2–4 weeks',
      explanation: `This path sharpens the frontend work already visible in your linked profiles and makes the delivery side of your UI work easier to show.`,
      resources: candidateResources['Frontend Delivery'],
      score: (hasTech('react', 'next.js') ? 3 : 0) + (hasSoft('communication', 'ownership') ? 2 : 0) + (hasGap('testing') ? 1 : 0),
    },
    {
      id: 4,
      topic: 'Backend APIs & Data',
      priority: hasGap('backend', 'api', 'database', 'sql') || hasTech('node.js', 'graphql') ? 'high' : 'medium',
      timeEstimate: '4–5 weeks',
      explanation: `Your profile signals enough product and implementation depth to benefit from stronger backend and data-system evidence.`,
      resources: candidateResources['Backend APIs & Data'],
      score: (hasGap('backend', 'api', 'database') ? 3 : 0) + (hasTech('node.js', 'graphql', 'sql') ? 2 : 0) + (focusArea === 'fullstack' ? 1 : 0),
    },
    {
      id: 5,
      topic: 'Cloud, Deployment & CI/CD',
      priority: hasGap('cloud', 'devops', 'deployment', 'ci/cd', 'cicd') || hasTech('docker', 'aws', 'kubernetes') ? 'high' : 'medium',
      timeEstimate: '4–6 weeks',
      explanation: `The current profile can be lifted by adding more visible delivery, deployment, and infrastructure ownership.`,
      resources: candidateResources['Cloud, Deployment & CI/CD'],
      score: (hasGap('cloud', 'devops', 'ci/cd') ? 3 : 0) + (hasTech('docker', 'aws', 'kubernetes') ? 2 : 0),
    },
    {
      id: 6,
      topic: 'Testing & Quality',
      priority: hasGap('testing', 'automation', 'coverage') ? 'high' : 'medium',
      timeEstimate: '2–3 weeks',
      explanation: `More testing evidence will make your current skills feel more trustworthy and easier to position for stronger roles.`,
      resources: candidateResources['Testing & Quality'],
      score: (hasGap('testing', 'automation', 'coverage') ? 3 : 0) + (hasTech('jest', 'playwright') ? 2 : 0),
    },
    {
      id: 7,
      topic: 'System Design & Storytelling',
      priority: hasGap('system design', 'architecture', 'scalability') ? 'high' : 'medium',
      timeEstimate: '4–6 weeks',
      explanation: `This helps convert the skills surfaced in your linked profiles into a stronger story for interviews and role transitions.`,
      resources: candidateResources['System Design & Storytelling'],
      score: (hasGap('system design', 'architecture') ? 3 : 0) + (hasSoft('communication', 'presentation') ? 2 : 0),
    },
    {
      id: 8,
      topic: 'Communication & Product Delivery',
      priority: hasSoft('communication', 'presentation', 'mentoring') ? 'high' : 'medium',
      timeEstimate: '2–4 weeks',
      explanation: `Your profile sources show collaboration and delivery signals that can be translated into clearer product narratives and stronger cross-functional impact.`,
      resources: candidateResources['Communication & Product Delivery'],
      score: (hasSoft('communication', 'presentation') ? 3 : 0) + (sources.length > 1 ? 1 : 0),
    },
  ];

  const ranked = candidates
    .filter((candidate) => candidate.score > 0 || candidate.priority === 'high')
    .sort((a, b) => b.score - a.score || (a.priority === b.priority ? a.id - b.id : a.priority === 'high' ? -1 : 1));

  const selected = ranked.slice(0, 4);

  const buildTopicContext = (topic: string): string => {
    const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
    const skillPhrase = topSkills.length > 0 ? formatList(topSkills) : 'your core skills';

    if (topic.includes('Frontend')) {
      return `This extends ${skillPhrase} into visible UI delivery outcomes from ${sourcePhrase}, including measurable quality signals like testing and accessibility.`;
    }

    if (topic.includes('Backend') || topic.includes('API')) {
      return `This uses ${skillPhrase} as a base and adds stronger backend ownership signals from ${sourcePhrase}, especially around APIs, data, and reliability.`;
    }

    if (topic.includes('Cloud') || topic.includes('CI/CD')) {
      return `This turns current delivery signals from ${sourcePhrase} into clearer cloud and deployment evidence tied to ${skillPhrase}.`;
    }

    if (topic.includes('Testing')) {
      return `This strengthens trust in your profile by converting ${skillPhrase} into repeatable quality signals across ${sourcePhrase}.`;
    }

    if (topic.includes('Communication') || topic.includes('Product')) {
      return `This improves how ${sourcePhrase} communicates impact, helping ${skillPhrase} map to product and cross-functional outcomes.`;
    }

    if (topic.includes('System Design')) {
      return `This bridges implementation strength in ${skillPhrase} with architecture-level reasoning that is currently underrepresented in ${sourcePhrase}.`;
    }

    return `This path connects ${sourcePhrase} to ${skillPhrase} while targeting the most relevant growth opportunities.`;
  };

  return selected.map((item, index) => {
    const gapPhrase = analysis.gaps.length > 0 ? formatList(analysis.gaps.slice(0, 2)) : 'the current skill gaps';
    const contextSentence = buildTopicContext(item.topic);

    return {
      id: item.id,
      topic: item.topic,
      priority: item.priority,
      timeEstimate: item.timeEstimate,
      explanation: `${item.explanation} ${contextSentence} It directly addresses ${gapPhrase}.`,
      resources: item.resources,
    };
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as AnalyzeProfileRequestBody;
    const links = body?.links ?? {};
    const uploadedResumeText = typeof body?.resumeText === 'string' ? body.resumeText.trim() : '';
    const uploadedResumeFileName = typeof body?.resumeFileName === 'string' ? body.resumeFileName.trim() : '';
    const hasResumeUpload = uploadedResumeText.length > 0;
    const hasAnyProfileLinks = buildUrlList(links).length > 0;

    if (!hasAnyProfileLinks && !hasResumeUpload) {
      return NextResponse.json({ error: 'Please provide at least one public profile link or upload a resume.' }, { status: 400 });
    }

    if (hasAnyProfileLinks) {
      // Verify link accessibility before scraping
      const linkVerification = await verifyLinks(links);

      if (linkVerification.accessibleCount === 0 && !hasResumeUpload) {
        return NextResponse.json(
          {
            error: 'None of the provided links are accessible.',
            details: linkVerification.inaccessibleLinks.map((link) => ({
              source: link.source,
              url: link.url,
              reason: link.error,
            })),
          },
          { status: 400 },
        );
      }

      if (linkVerification.inaccessibleLinks.length > 0) {
        console.warn('Some links are inaccessible:', linkVerification.inaccessibleLinks);
      }
    }

    const scrapedSources = hasAnyProfileLinks ? await collectScrapedSources(links) : [];
    if (hasResumeUpload) {
      scrapedSources.push(buildUploadedResumeSource(uploadedResumeText, uploadedResumeFileName || undefined));
    }
    
    // Build raw text for AI analysis
    const allScrapedText = scrapedSources.map((s) => s.text).join(' ');
    
    // Calculate frequency map for data-driven scores
    const frequencyMap = buildSkillFrequencyMap(allScrapedText);
    const maxFrequency = Array.from(frequencyMap.values()).reduce((max, freq) => Math.max(max, freq.count), 1);
    
    // Convert frequency map to skills with authentic scores
    const frequencyBasedSkills: Skill[] = Array.from(frequencyMap.values()).map((freq) => ({
      name: freq.skill,
      confidence: calculateFrequencyScore(freq, maxFrequency) / 100,
      source: 'Frequency-Analysis',
    }));

    // Extract text-based skills as fallback
    const extractedSkills = aggregateSkillsByNameWithBoost(
      scrapedSources.flatMap((source) => source.skills.length > 0 ? source.skills : extractSkillsFromText(source.text, source.source)),
    );

    // Merge frequency-based with extracted skills, preferring frequency-based
    const mergedSkills = new Map<string, Skill>();
    
    for (const skill of frequencyBasedSkills) {
      mergedSkills.set(normalizeToken(skill.name), skill);
    }
    
    for (const skill of extractedSkills) {
      const key = normalizeToken(skill.name);
      if (!mergedSkills.has(key)) {
        mergedSkills.set(key, skill);
      }
    }
    
    const finalExtractedSkills = Array.from(mergedSkills.values());

    const rawText = buildProfileText(links, scrapedSources, finalExtractedSkills);
    const aiAnalysis = await analyzeProfileText(rawText).catch((error) => {
      console.error('AI analysis failure: ', error);
      return {
        technicalSkills: [],
        softSkills: [],
        summary: '',
        gaps: [],
        recommendations: [],
        topSkills: [],
        strengths: [],
        industryRelevanceScore: 0,
        atsScore: 0,
        atsFeedback: [],
        industryInsights: '',
        jobRecommendations: [],
        learningPath: [],
        rawText,
      };
    });

    const processedAnalysis = enrichAnalysis(aiAnalysis, finalExtractedSkills);
    const sources = getProfileSources(finalExtractedSkills);
    const focusArea = detectFocusArea(processedAnalysis.technicalSkills);
    const aiStrengths = buildAIStrengths(processedAnalysis);
    const aiJobRecommendations = buildAIJobRecommendations(processedAnalysis);
    const aiLearningPath = buildAILearningPath(processedAnalysis, sources, focusArea);

    const aiSkills: Skill[] = [
      ...processedAnalysis.technicalSkills.map((s) => ({ name: s.name, confidence: s.confidence, source: s.source || 'AI-Technical' })),
      ...processedAnalysis.softSkills.map((s) => ({ name: s.name, confidence: s.confidence, source: s.source || 'AI-Soft' })),
    ];

    const combinedSkillsMap = new Map<string, Skill>();
    [...finalExtractedSkills, ...aiSkills].forEach((skill) => {
      const key = `${skill.name.toLowerCase()}|${skill.source}`;
      if (!combinedSkillsMap.has(key)) {
        combinedSkillsMap.set(key, skill);
      }
    });

    const combinedSkills = aggregateSkillsByName(Array.from(combinedSkillsMap.values()));

    return NextResponse.json({
      skills: combinedSkills,
      aiSummary: processedAnalysis.summary,
      aiStrengths,
      aiGaps: processedAnalysis.gaps,
      aiRecommendations: processedAnalysis.recommendations,
      aiTechnicalSkills: processedAnalysis.technicalSkills,
      aiSoftSkills: processedAnalysis.softSkills,
      aiIndustryRelevanceScore: processedAnalysis.industryRelevanceScore,
      aiAtsScore: processedAnalysis.atsScore,
      aiAtsFeedback: processedAnalysis.atsFeedback,
      aiIndustryInsights: processedAnalysis.industryInsights,
      aiTopSkills: processedAnalysis.topSkills,
      aiJobRecommendations,
      aiLearningPath,
    });
  } catch (error) {
    console.error('Error analyzing profile:', error);
    return NextResponse.json(
      { error: 'Failed to analyze profile' },
      { status: 500 }
    );
  }
}