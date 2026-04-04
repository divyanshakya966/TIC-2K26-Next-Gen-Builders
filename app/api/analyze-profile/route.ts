import axios from 'axios';
import * as cheerio from 'cheerio';
import dns from 'node:dns/promises';
import net from 'node:net';
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

function isPrivateIpv4(ip: string): boolean {
  const segments = ip.split('.').map((part) => Number(part));
  if (segments.length !== 4 || segments.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return true;
  }

  if (segments[0] === 10) return true;
  if (segments[0] === 127) return true;
  if (segments[0] === 0) return true;
  if (segments[0] === 169 && segments[1] === 254) return true;
  if (segments[0] === 172 && segments[1] >= 16 && segments[1] <= 31) return true;
  if (segments[0] === 192 && segments[1] === 168) return true;

  return false;
}

function isPrivateIpv6(ip: string): boolean {
  const lowered = ip.toLowerCase();

  return lowered === '::1'
    || lowered === '::'
    || lowered.startsWith('fc')
    || lowered.startsWith('fd')
    || lowered.startsWith('fe80');
}

function isPrivateIpAddress(ip: string): boolean {
  const ipVersion = net.isIP(ip);
  if (ipVersion === 4) return isPrivateIpv4(ip);
  if (ipVersion === 6) return isPrivateIpv6(ip);
  return false;
}

async function resolvesToPublicIp(hostname: string): Promise<boolean> {
  try {
    const records = await dns.lookup(hostname, { all: true, verbatim: true });
    if (!records.length) return false;
    return records.every((record) => net.isIP(record.address) !== 0 && !isPrivateIpAddress(record.address));
  } catch {
    return false;
  }
}

type LinkValidationWarning = { source: keyof SocialLinks; url: string; reason: string };

async function sanitizeAndValidateLinks(inputLinks: SocialLinks): Promise<{
  links: SocialLinks;
  warnings: LinkValidationWarning[];
}> {
  const normalizedLinks: SocialLinks = {};
  const warnings: LinkValidationWarning[] = [];

  const linkEntries = Object.entries(inputLinks) as Array<[keyof SocialLinks, string | undefined]>;

  for (const [source, value] of linkEntries) {
    if (typeof value !== 'string' || !value.trim()) {
      continue;
    }

    let parsed: URL;
    try {
      parsed = new URL(value.trim());
    } catch {
      warnings.push({ source, url: value.trim(), reason: 'Invalid URL format.' });
      continue;
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      warnings.push({ source, url: parsed.toString(), reason: 'Only http/https links are allowed.' });
      continue;
    }

    const hostname = parsed.hostname.toLowerCase();
    if (hostname === 'localhost') {
      warnings.push({ source, url: parsed.toString(), reason: 'Localhost URLs are not allowed.' });
      continue;
    }

    if (isPrivateIpAddress(hostname)) {
      warnings.push({ source, url: parsed.toString(), reason: 'Private or loopback IP addresses are not allowed.' });
      continue;
    }

    const isPublicHost = await resolvesToPublicIp(hostname);
    if (!isPublicHost) {
      warnings.push({ source, url: parsed.toString(), reason: 'Host did not resolve to a public IP address.' });
      continue;
    }

    parsed.hash = '';
    normalizedLinks[source] = parsed.toString();
  }

  return { links: normalizedLinks, warnings };
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

function resolveResponseUrl(response: any, originalUrl: string): string {
  const fromNode = response?.request?.res?.responseUrl;
  const fromBrowserLike = response?.request?.responseURL;
  const candidate = typeof fromNode === 'string' && fromNode.trim()
    ? fromNode
    : typeof fromBrowserLike === 'string' && fromBrowserLike.trim()
      ? fromBrowserLike
      : originalUrl;

  return normalizeUrl(candidate);
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
      const headResponse = await axios.head(url, {
        headers: SCRAPER_HEADERS,
        timeout: 6000,
        validateStatus: () => true,
        maxRedirects: 5,
      });

      if (headResponse.status >= 200 && headResponse.status < 400) {
        return {
          url: resolveResponseUrl(headResponse, url),
          source,
          accessible: true,
          status: headResponse.status,
        } as LinkVerificationResult;
      }

      // Some hosts reject HEAD while allowing GET for public pages.
      if ([401, 403, 405, 406, 429, 999].includes(headResponse.status)) {
        const getResponse = await axios.get(url, {
          headers: SCRAPER_HEADERS,
          timeout: 7000,
          validateStatus: () => true,
          maxRedirects: 5,
          responseType: 'text',
          maxContentLength: 256 * 1024,
        });

        const getAccessible = getResponse.status >= 200 && getResponse.status < 400;
        if (getAccessible) {
          return {
            url: resolveResponseUrl(getResponse, url),
            source,
            accessible: true,
            status: getResponse.status,
          } as LinkVerificationResult;
        }

        return {
          url: resolveResponseUrl(getResponse, url),
          source,
          accessible: false,
          status: getResponse.status,
          error: `HTTP ${getResponse.status}: ${getStatusDescription(getResponse.status)}`,
        } as LinkVerificationResult;
      }

      const result: LinkVerificationResult = {
        url: resolveResponseUrl(headResponse, url),
        source,
        accessible: false,
        status: headResponse.status,
      };

      result.error = `HTTP ${headResponse.status}: ${getStatusDescription(headResponse.status)}`;

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
    999: 'Blocked by anti-bot protection',
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
    finalUrl: resolveResponseUrl(response, url),
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

// Method 1: Job Search APIs (Most Reliable & Legal)
async function fetchInternshipsFromJSearch(skills: string[]): Promise<JobRecommendation[]> {
  try {
    const topSkills = skills.slice(0, 3).join(' ').toLowerCase();
    const query = `internship ${topSkills} developer engineer`;

    // JSearch API (RapidAPI) - Professional job search API
    const options = {
      method: 'GET',
      url: 'https://jsearch.p.rapidapi.com/search',
      params: {
        query: `${query} internship OR intern`,
        page: '1',
        num_pages: '1',
        country: 'US',
        date_posted: 'all', // Include more internship opportunities
        employment_types: 'INTERN,PARTTIME' // Include both intern and part-time roles
      },
      headers: {
        'X-RapidAPI-Key': process.env.RAPIDAPI_KEY || '',
        'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
      }
    };

    const response = await axios.request(options);
    const jobs = response.data.data || [];

    return jobs.slice(0, 5).map((job: any, index: number) => ({
      id: 2000 + index,
      title: job.job_title || 'Internship Position',
      company: job.employer_name || 'Company',
      matchPercentage: Math.floor(70 + Math.random() * 25), // 70-95%
      salary: job.job_min_salary ? `$${job.job_min_salary} - $${job.job_max_salary}` : 'Competitive',
      location: job.job_city && job.job_state ? `${job.job_city}, ${job.job_state}` : job.job_country || 'Remote',
      type: job.job_employment_type?.toLowerCase().includes('intern') ? 'Internship (Paid)' : 'Internship',
      skills: job.job_required_skills || skills.slice(0, 3),
      description: job.job_description?.slice(0, 200) + '...' || 'Great internship opportunity',
      applyUrl: job.job_apply_link || job.job_google_link || '#',
      fitReason: `Found via job search API matching ${topSkills} skills`
    }));

  } catch (error) {
    console.error('JSearch API failed:', error);
    return [];
  }
}

// Method 2: Web Scraping (Direct from Job Sites)
async function scrapeInternshipsFromIndeed(skills: string[]): Promise<JobRecommendation[]> {
  try {
    const query = skills.slice(0, 3).join(' ').toLowerCase();
    const searchQuery = `${query} internship OR intern`;
    const url = `https://www.indeed.com/jobs?q=${encodeURIComponent(searchQuery)}&l=United%20States&fromage=any&limit=50`;

    const { html } = await fetchHtml(url);
    const $ = cheerio.load(html);
    const internships: JobRecommendation[] = [];

    $('.job_seen_beacon').each((index, element) => {
      if (index >= 6) return false; // Limit to 6 results for more variety

      const jobCard = $(element);
      const title = jobCard.find('.jobTitle').text().trim();
      const company = jobCard.find('.companyName').text().trim() || jobCard.find('[data-testid="company-name"]').text().trim() || 'Local Company';
      const location = jobCard.find('.companyLocation').text().trim() || 'Various Locations';
      const salary = jobCard.find('.salary-snippet').text().trim() || jobCard.find('.estimated-salary').text().trim() || 'Competitive';
      const description = jobCard.find('.job-snippet').text().trim() || 'Great internship opportunity';
      const jobLink = jobCard.find('a').attr('href');

      if (title && (title.toLowerCase().includes('intern') || title.toLowerCase().includes('internship'))) {
        internships.push({
          id: 3000 + index,
          title: title,
          company: company,
          matchPercentage: Math.floor(65 + Math.random() * 25),
          salary: salary,
          location: location || 'Remote',
          type: salary.includes('$') || salary.includes('hour') ? 'Internship (Paid)' : 'Internship (Unpaid)',
          skills: ['JavaScript', 'Python', 'React'], // Generic skills
          description: description || 'Exciting internship opportunity',
          applyUrl: jobLink ? `https://www.indeed.com${jobLink}` : '#',
          fitReason: 'Found via Indeed internship search'
        });
      }
    });

    return internships;

  } catch (error) {
    console.error('Indeed scraping failed:', error);
    return [];
  }
}

// Method 3: Company Career Pages (Targeted Scraping)
async function scrapeCompanyInternships(): Promise<JobRecommendation[]> {
  const companyConfigs = [
    {
      name: 'Google',
      careerUrl: 'https://careers.google.com/students/',
      selector: '.job-title, .opening-title',
      location: 'Mountain View, CA',
      salary: '$8,000 - $9,000 / month'
    },
    {
      name: 'Meta',
      careerUrl: 'https://www.metacareers.com/students/',
      selector: '.job-title, [data-testid*="job"]',
      location: 'Menlo Park, CA',
      salary: '$7,500 - $8,500 / month'
    },
    {
      name: 'Microsoft',
      careerUrl: 'https://careers.microsoft.com/students/us/en',
      selector: '.job-title, .card-title',
      location: 'Redmond, WA',
      salary: '$7,000 - $8,000 / month'
    }
  ];

  const internships: JobRecommendation[] = [];

  for (const config of companyConfigs) {
    try {
      const { html } = await fetchHtml(config.careerUrl);
      const $ = cheerio.load(html);

      // Check if internship opportunities exist
      const hasInternships = html.toLowerCase().includes('intern') ||
                           html.toLowerCase().includes('internship') ||
                           html.toLowerCase().includes('student');

      if (hasInternships) {
        internships.push({
          id: 4000 + internships.length,
          title: `Software Engineering Intern`,
          company: config.name,
          matchPercentage: 90 + Math.floor(Math.random() * 8),
          salary: config.salary,
          location: config.location,
          type: 'Internship (Paid)',
          skills: ['JavaScript', 'Python', 'React', 'TypeScript'],
          description: `Work on cutting-edge technologies at ${config.name} with full-time engineers and mentors.`,
          applyUrl: config.careerUrl,
          fitReason: `Direct from ${config.name} career page`
        });
      }
    } catch (error) {
      console.error(`Failed to scrape ${config.name}:`, error);
    }
  }

  return internships;
}

// Method 4: RSS Feeds (If available)
async function fetchInternshipsFromRSS(): Promise<JobRecommendation[]> {
  try {
    // Some job sites provide RSS feeds
    const rssSources = [
      'https://stackoverflow.com/jobs/feed?tags=internship',
      'https://weworkremotely.com/categories/remote-internships.rss'
    ];

    const internships: JobRecommendation[] = [];

    for (const rssUrl of rssSources) {
      try {
        const response = await axios.get(rssUrl, {
          headers: SCRAPER_HEADERS,
          timeout: 10000
        });

        // Parse RSS (simplified - in production use a proper RSS parser)
        const $ = cheerio.load(response.data, { xmlMode: true });

        $('item').each((index, element) => {
          if (index >= 2) return false; // Limit per source

          const item = $(element);
          const title = item.find('title').text().trim();
          const description = item.find('description').text().trim();
          const link = item.find('link').text().trim();

          if (title.toLowerCase().includes('intern')) {
            internships.push({
              id: 5000 + internships.length,
              title: title,
              company: 'Various Companies',
              matchPercentage: Math.floor(60 + Math.random() * 25),
              salary: 'Varies',
              location: 'Remote/Various',
              type: description.toLowerCase().includes('paid') || description.toLowerCase().includes('$') ? 'Internship (Paid)' : 'Internship (Unpaid)',
              skills: ['Various'],
              description: description.slice(0, 150) + '...',
              applyUrl: link,
              fitReason: 'Found via RSS feed'
            });
          }
        });
      } catch (error) {
        console.error(`RSS fetch failed for ${rssUrl}:`, error);
      }
    }

    return internships;

  } catch (error) {
    console.error('RSS fetching failed:', error);
    return [];
  }
}

// Main function that combines multiple methods
async function fetchLiveInternships(skills: string[]): Promise<JobRecommendation[]> {
  const allInternships: JobRecommendation[] = [];

  try {
    // Try multiple methods in parallel for better coverage
    const [jsearchResults, indeedResults, companyResults, rssResults] = await Promise.allSettled([
      fetchInternshipsFromJSearch(skills),
      scrapeInternshipsFromIndeed(skills),
      scrapeCompanyInternships(),
      fetchInternshipsFromRSS()
    ]);

    // Collect successful results
    if (jsearchResults.status === 'fulfilled') {
      allInternships.push(...jsearchResults.value);
    }
    if (indeedResults.status === 'fulfilled') {
      allInternships.push(...indeedResults.value);
    }
    if (companyResults.status === 'fulfilled') {
      allInternships.push(...companyResults.value);
    }
    if (rssResults.status === 'fulfilled') {
      allInternships.push(...rssResults.value);
    }

    // Remove duplicates and sort by match percentage
    const uniqueInternships = allInternships.filter((internship, index, self) =>
      index === self.findIndex(i => i.title === internship.title && i.company === internship.company)
    );

    // Return top 5 most relevant internships
    return uniqueInternships
      .sort((a, b) => b.matchPercentage - a.matchPercentage)
      .slice(0, 8);

  } catch (error) {
    console.error('Error fetching live internships:', error);

    // Fallback to basic company internships if all methods fail
    return [
      {
        id: 1001,
        title: 'Software Engineering Intern',
        company: 'Google',
        matchPercentage: 92,
        salary: '$8,000 - $9,000 / month',
        location: 'Mountain View, CA',
        type: 'Internship (Paid)',
        skills: skills.slice(0, 3),
        description: 'Work on cutting-edge AI and web technologies with full-time engineers.',
        applyUrl: 'https://careers.google.com/students/',
        fitReason: 'Fallback: Direct company career page'
      },
      {
        id: 1002,
        title: 'Frontend Developer Intern',
        company: 'Meta',
        matchPercentage: 88,
        salary: '$7,500 - $8,500 / month',
        location: 'Menlo Park, CA',
        type: 'Internship (Paid)',
        skills: ['React', 'JavaScript', 'TypeScript'],
        description: 'Build user-facing features for billions of users using modern web technologies.',
        applyUrl: 'https://www.metacareers.com/students/',
        fitReason: 'Fallback: Direct company career page'
      }
    ];
  }
}

// Fetch regular jobs (full-time, contract, etc.) from live sources
async function fetchLiveJobs(skills: string[]): Promise<JobRecommendation[]> {
  const allJobs: JobRecommendation[] = [];

  try {
    const topSkills = skills.slice(0, 3).join(' ').toLowerCase();
    const jobQuery = `${topSkills} developer engineer`;

    // JSearch API for regular jobs
    const jsearchOptions = {
      method: 'GET',
      url: 'https://jsearch.p.rapidapi.com/search',
      params: {
        query: `${jobQuery} full-time OR contract OR permanent OR entry-level`,
        page: '1',
        num_pages: '1',
        country: 'US',
        date_posted: 'all', // Include older jobs for more variety
        employment_types: 'FULLTIME,CONTRACTOR,PARTTIME,INTERN' // Include more types
      },
      headers: {
        'X-RapidAPI-Key': process.env.RAPIDAPI_KEY || '',
        'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
      }
    };

    const jsearchResponse = await axios.request(jsearchOptions);
    const jobs = jsearchResponse.data.data || [];

    const jsearchJobs = jobs.slice(0, 5).map((job: any, index: number) => ({
      id: 1000 + index, // Different ID range from internships
      title: job.job_title || 'Software Developer',
      company: job.employer_name || 'Tech Company',
      matchPercentage: Math.floor(75 + Math.random() * 20), // 75-95%
      salary: job.job_min_salary ? `$${job.job_min_salary} - $${job.job_max_salary}` : '$80K – $120K',
      location: job.job_city && job.job_state ? `${job.job_city}, ${job.job_state}` : job.job_country || 'Remote',
      type: job.job_employment_type === 'CONTRACTOR' ? 'Contract' : 'Full-time',
      skills: job.job_required_skills || skills.slice(0, 4),
      description: job.job_description?.slice(0, 200) + '...' || 'Exciting full-time opportunity',
      applyUrl: job.job_apply_link || job.job_google_link || '#',
      fitReason: `Live job opportunity matching ${topSkills} skills`
    }));

    allJobs.push(...jsearchJobs);

    // Scrape Indeed for regular jobs
    try {
      const indeedResponse = await axios.get('https://www.indeed.com/jobs', {
        params: {
          q: `${jobQuery} full time OR part time OR contract`,
          l: 'United States',
          sort: 'relevance', // Sort by relevance instead of date for more variety
          fromage: 'any', // Include jobs from any time period
          limit: '50' // Get more results
        },
        headers: SCRAPER_HEADERS,
        timeout: SCRAPE_TIMEOUT_MS
      });

      const $ = cheerio.load(indeedResponse.data);
      const indeedJobs: JobRecommendation[] = [];

      $('.job_seen_beacon').each((index, element) => {
        if (indeedJobs.length >= 5) return false; // Limit to 5 jobs

        const title = $(element).find('.jobTitle').text().trim() || 'Software Developer';
        const company = $(element).find('.companyName').text().trim() || 'Local Company';
        const location = $(element).find('.companyLocation').text().trim() || 'Various Locations';
        const link = $(element).find('a').attr('href');
        const jobLink = link ? `https://www.indeed.com${link}` : '#';

        // Include both jobs and internships, but skip obvious duplicates
        if (title.toLowerCase().includes('duplicate') || title.toLowerCase().includes('test')) {
          return;
        }

        indeedJobs.push({
          id: 1100 + index,
          title,
          company,
          matchPercentage: Math.floor(65 + Math.random() * 25),
          salary: '$50K – $90K', // More realistic local salaries
          location,
          type: title.toLowerCase().includes('intern') ? 'Internship' : 'Full-time',
          skills: skills.slice(0, 3),
          description: 'Local job opportunity with competitive compensation',
          applyUrl: jobLink,
          fitReason: 'Found via Indeed local job search'
        });
      });

      allJobs.push(...indeedJobs);
    } catch (error) {
      console.error('Indeed jobs scraping failed:', error);
    }

    // Remove duplicates and sort by match percentage
    const uniqueJobs = allJobs.filter((job, index, self) =>
      index === self.findIndex(j => j.title === job.title && j.company === job.company)
    );

    // Return top 7 most relevant jobs
    return uniqueJobs
      .sort((a, b) => b.matchPercentage - a.matchPercentage)
      .slice(0, 7);

  } catch (error) {
    console.error('Live jobs fetch failed:', error);
    return [];
  }
}

function normalizeToken(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function createSkillLookup(values: string[]): Set<string> {
  return new Set(values.map(normalizeToken).filter(Boolean));
}

function getProfileSources(extractedSkills: Skill[]): string[] {
  const preferredOrder = ['GitHub', 'LinkedIn', 'Portfolio', 'Resume Upload', 'Resume', 'Dev.to', 'Twitter'];
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
  const topSkillsPhrase = analysis.topSkills.length > 0 ? formatList(analysis.topSkills.slice(0, 3)) : 'your strongest skills';
  const weakestTechnical = [...technicalSkills]
    .sort((a, b) => a.score - b.score)
    .slice(0, 2)
    .map((skill) => skill.name);
  const weakestSoft = [...softSkills]
    .sort((a, b) => a.score - b.score)
    .slice(0, 2)
    .map((skill) => skill.name);
  const weakestTechnicalPhrase = weakestTechnical.length > 0 ? formatList(weakestTechnical) : 'backend and testing depth';
  const weakestSoftPhrase = weakestSoft.length > 0 ? formatList(weakestSoft) : 'communication and stakeholder clarity';

  const aiGaps = Array.isArray(analysis.gaps)
    ? analysis.gaps.map((gap) => gap.trim()).filter(Boolean).slice(0, 3)
    : [];

  const gaps: string[] = [...aiGaps];

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

  const dynamicGaps = [
    `Add quantified outcomes around ${topSkillsPhrase} so recruiters can see measurable impact, not only tool usage.`,
    `Improve weaker signals in ${weakestTechnicalPhrase} with one focused project that includes architecture, tests, and deployment notes.`,
    `Strengthen non-technical positioning in ${weakestSoftPhrase} with clearer project narratives, ownership statements, and collaboration examples.`,
    focusArea === 'frontend'
      ? 'Add one API or data-intensive project to balance strong UI skills with backend execution evidence.'
      : focusArea === 'platform'
        ? 'Add one customer-facing product project so infrastructure strengths connect more clearly to product outcomes.'
        : 'Publish one end-to-end case study that links discovery, implementation, testing, and production impact.',
    `Increase proof depth across ${sourcePhrase} with public artifacts like polished READMEs, demos, and technical write-ups.`,
  ];

  if (technicalSkills.length > 0) {
    const lowTechnical = technicalSkills.filter((skill) => skill.score < 62).slice(0, 2).map((skill) => skill.name);
    if (lowTechnical.length > 0) {
      dynamicGaps.unshift(
        `Your lowest technical signals are ${formatList(lowTechnical)}; improving these will quickly raise role match quality.`,
      );
    }
  }

  if (softSkills.length > 0) {
    const lowSoft = softSkills.filter((skill) => skill.score < 62).slice(0, 2).map((skill) => skill.name);
    if (lowSoft.length > 0) {
      dynamicGaps.push(
        `Soft-skill evidence is weakest in ${formatList(lowSoft)}; add clearer collaboration outcomes in project documentation.`,
      );
    }
  }

  const normalized = new Set(gaps.map((item) => normalizeToken(item)));
  for (const item of dynamicGaps) {
    const key = normalizeToken(item);
    if (!normalized.has(key)) {
      gaps.push(item);
      normalized.add(key);
    }
    if (gaps.length >= 5) break;
  }

  return gaps.slice(0, 5);
}

function buildPersonalizedStrengths(
  analysis: AnalysisResult,
  sources: string[],
  focusArea: 'frontend' | 'fullstack' | 'platform' | 'general',
): string[] {
  const sourcePhrase = sources.length > 0 ? formatList(sources) : 'your linked profiles';
  const strengths = new Set<string>();
  const topSkillsPhrase = analysis.topSkills.length > 0 ? formatList(analysis.topSkills.slice(0, 3)) : 'your core stack';
  const avgTechnicalScore = analysis.technicalSkills.length > 0
    ? Math.round(analysis.technicalSkills.reduce((sum, skill) => sum + skill.score, 0) / analysis.technicalSkills.length)
    : 0;
  const avgSoftScore = analysis.softSkills.length > 0
    ? Math.round(analysis.softSkills.reduce((sum, skill) => sum + skill.score, 0) / analysis.softSkills.length)
    : 0;

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

  const baseStrengths = Array.from(strengths);
  const fallbackStrengths = [
    `Your profile evidence across ${sourcePhrase} consistently reinforces strengths in ${topSkillsPhrase}.`,
    `Technical signal is currently around ${avgTechnicalScore}%, giving you a practical base for role-focused upskilling.`,
    `Collaboration signal is around ${avgSoftScore}%, which supports cross-functional fit and team readiness.`,
    analysis.topSkills.length >= 4
      ? `Breadth across ${analysis.topSkills.slice(0, 4).join(', ')} improves flexibility for varied role requirements.`
      : 'Your visible skill mix is coherent and can be positioned effectively for internship and early-career roles.',
    'The profile shows clear learning momentum, with reusable project skills that map to current market demand.',
  ];

  const normalized = new Set(baseStrengths.map((item) => normalizeToken(item)));
  for (const fallback of fallbackStrengths) {
    const key = normalizeToken(fallback);
    if (!normalized.has(key)) {
      baseStrengths.push(fallback);
      normalized.add(key);
    }
    if (baseStrengths.length >= 5) break;
  }

  return baseStrengths.slice(0, 5);
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
  const topSkillsPhrase = analysis.topSkills.length > 0 ? formatList(analysis.topSkills.slice(0, 3)) : 'your strongest areas';
  const topTechnical = analysis.technicalSkills.slice(0, 2).map((skill) => skill.name);
  const topSoft = analysis.softSkills.slice(0, 2).map((skill) => skill.name);
  const technicalPhrase = topTechnical.length > 0 ? formatList(topTechnical) : 'technical execution';
  const softPhrase = topSoft.length > 0 ? formatList(topSoft) : 'collaboration and communication';

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

    const fallback = [
      `Consistent technical foundation is visible in ${technicalPhrase}.`,
      `Collaboration strength is reinforced by signals in ${softPhrase}.`,
      `Your strongest evidence currently centers on ${topSkillsPhrase}.`,
      'Public profile artifacts indicate delivery focus and strong learning adaptability.',
      'Balanced implementation and communication signals improve cross-functional team fit.',
    ];

    for (const line of fallback) {
      const normalized = normalizeToken(line);
      if (!normalizedSeen.has(normalized)) {
        unique.push(line);
        normalizedSeen.add(normalized);
      }
      if (unique.length >= 5) break;
    }

    return unique.slice(0, 5);
  }

  const fallback = new Set<string>();
  analysis.technicalSkills.slice(0, 3).forEach((skill) => fallback.add(`Technical depth shown in ${skill.name}.`));
  analysis.softSkills.slice(0, 2).forEach((skill) => fallback.add(`Collaboration signal present in ${skill.name}.`));
  fallback.add(`The strongest profile evidence currently centers on ${topSkillsPhrase}.`);
  fallback.add(`Combined strengths in ${technicalPhrase} and ${softPhrase} improve role readiness.`);

  return Array.from(fallback).slice(0, 5);
}

function buildAIJobRecommendations(analysis: Awaited<ReturnType<typeof analyzeProfileText>>, liveJobs: JobRecommendation[] = [], liveInternships: JobRecommendation[] = []): JobRecommendation[] {
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

    // Add live jobs and internships to the recommendations (they already have proper URLs)
    const combinedRecommendations = [...selected, ...liveJobs, ...liveInternships];

    // Remove duplicates and sort by match percentage
    const uniqueRecommendations = combinedRecommendations.filter((item, index, self) =>
      index === self.findIndex(r => r.title === item.title && r.company === item.company)
    );

    return uniqueRecommendations
      .sort((a, b) => b.matchPercentage - a.matchPercentage)
      .slice(0, 15)
      .map((item, index) => ({ ...item, id: index + 1 }));
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
      applyUrl: 'https://github.com/topics/good-first-issue'
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
      applyUrl: 'https://angel.co/jobs'
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

  // Add live jobs and internships to the fallback recommendations
  const combinedRecommendations = [...selected, ...liveJobs, ...liveInternships];

  // Remove duplicates and sort by match percentage
  const uniqueRecommendations = combinedRecommendations.filter((item, index, self) =>
    index === self.findIndex(r => r.title === item.title && r.company === item.company)
  );

  return uniqueRecommendations
    .sort((a, b) => b.matchPercentage - a.matchPercentage)
    .slice(0, 15)
    .map((role, index) => ({
      id: index + 1,
      title: role.title,
      company: role.company,
      matchPercentage: role.matchPercentage,
      salary: role.salary,
      location: role.location,
      type: role.type,
      skills: role.skills,
      description: role.description,
      applyUrl: role.applyUrl || '#',
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
    const submittedLinks = body?.links ?? {};
    const uploadedResumeText = typeof body?.resumeText === 'string' ? body.resumeText.trim() : '';
    const uploadedResumeFileName = typeof body?.resumeFileName === 'string' ? body.resumeFileName.trim() : '';
    const hasResumeUpload = uploadedResumeText.length > 0;
    const validatedLinksResult = await sanitizeAndValidateLinks(submittedLinks);
    const links = validatedLinksResult.links;
    const hasAnyProfileLinks = buildUrlList(links).length > 0;
    let linkWarnings: Array<{ source: keyof SocialLinks; url: string; reason?: string }> = [...validatedLinksResult.warnings];

    if (!hasAnyProfileLinks && !hasResumeUpload) {
      return NextResponse.json(
        {
          error: 'Please provide at least one valid public profile link or upload a resume.',
          details: validatedLinksResult.warnings,
        },
        { status: 400 },
      );
    }

    if (hasAnyProfileLinks) {
      // Verify link accessibility before scraping
      const linkVerification = await verifyLinks(links);

      const accessibilityWarnings = linkVerification.inaccessibleLinks.map((link) => ({
        source: link.source,
        url: link.url,
        reason: link.error,
      }));
      linkWarnings = [...linkWarnings, ...accessibilityWarnings];

      if (accessibilityWarnings.length > 0) {
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

    // Fetch live jobs and internships based on user's skills
    const userSkills = [...processedAnalysis.topSkills, ...processedAnalysis.technicalSkills.map(s => s.name)];
    const [liveJobs, liveInternships] = await Promise.all([
      fetchLiveJobs(userSkills),
      fetchLiveInternships(userSkills)
    ]);

    const aiJobRecommendations = buildAIJobRecommendations(processedAnalysis, liveJobs, liveInternships);
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
      aiAtsScore: hasResumeUpload ? processedAnalysis.atsScore : null,
      aiAtsFeedback: hasResumeUpload ? processedAnalysis.atsFeedback : [],
      aiAtsAvailable: hasResumeUpload,
      aiIndustryInsights: processedAnalysis.industryInsights,
      aiTopSkills: processedAnalysis.topSkills,
      aiJobRecommendations,
      aiLearningPath,
      linkWarnings,
    });
  } catch (error) {
    console.error('Error analyzing profile:', error);
    return NextResponse.json(
      { error: 'Failed to analyze profile' },
      { status: 500 }
    );
  }
}