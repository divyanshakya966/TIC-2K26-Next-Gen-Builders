'use client';

import { useEffect, useMemo, useState } from 'react';
import axios from 'axios'
import { AnimatePresence, motion } from 'framer-motion';
import {
  BookOpen,
  Briefcase,
  ExternalLink,
  Github,
  Lightbulb,
  Linkedin,
  Moon,
  Sparkles,
  Sun,
  UserCircle2,
} from 'lucide-react';
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { mockUserProfile } from '@/lib/mockData';
import { cn } from '@/lib/utils';
import type { LearningResource, SocialLinks, SkillData } from '@/lib/types';

const animationVariant = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0 },
};

const resourceLabel: Record<LearningResource['type'], string> = {
  book: 'Book',
  course: 'Course',
  documentation: 'Documentation',
  video: 'Video',
};

const priorityClass = {
  high: 'bg-rose-100 text-rose-800 border-rose-200',
  medium: 'bg-amber-100 text-amber-800 border-amber-200',
  low: 'bg-emerald-100 text-emerald-800 border-emerald-200',
};

type ThemeMode = 'light' | 'dark';
type ThemeTransition = 'to-light' | 'to-dark' | 'dark-moon' | null;

const navSections = [
  { id: 'overview', label: 'Overview' },
  { id: 'links', label: 'Links' },
  { id: 'skills', label: 'Skill Graphs' },
  { id: 'jobs', label: 'Recommendations' },
  { id: 'summary', label: 'AI Summary' },
  { id: 'learning', label: 'Learning Path' },
];

export default function HomePage() {
  const [links, setLinks] = useState<SocialLinks>({
    github: '',
    linkedin: '',
    resume: '',
    portfolio: '',
    twitter: '',
    devto: '',
  });
  const [linkErrors, setLinkErrors] = useState<Partial<Record<keyof SocialLinks, string>>>({});
  const [expandedJobs, setExpandedJobs] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>('light');
  const [themeTransition, setThemeTransition] = useState<ThemeTransition>(null);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [extractedSkills, setExtractedSkills] = useState<Skill[]>([]);
  const [aiSummary, setAiSummary] = useState<string>('');
  const [aiGaps, setAiGaps] = useState<string[]>([]);
  const [aiRecommendations, setAiRecommendations] = useState<string[]>([]);
  const [aiTechnicalSkills, setAiTechnicalSkills] = useState<SkillData[]>([]);
  const [aiSoftSkills, setAiSoftSkills] = useState<SkillData[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysisMessage, setAnalysisMessage] = useState<string | null>(null);

  const [jobs, setJobs] = useState(mockUserProfile.jobRecommendations);
  const [learningPath, setLearningPath] = useState(mockUserProfile.learningPath);
  const [industryRelevanceScore, setIndustryRelevanceScore] = useState(mockUserProfile.aiSummary.industryRelevanceScore);
  const [industryInsights, setIndustryInsights] = useState(mockUserProfile.aiSummary.industryInsights);
  const [topSkills, setTopSkills] = useState(mockUserProfile.aiSummary.topSkills);

  const jobsToShow = useMemo(
    () => jobs.slice(0, expandedJobs ? 5 : 3),
    [expandedJobs, jobs],
  );

  const technicalSkillsData = useMemo(
    () => (aiTechnicalSkills.length > 0 ? aiTechnicalSkills : mockUserProfile.technicalSkills),
    [aiTechnicalSkills],
  );

  const softSkillsData = useMemo(
    () => (aiSoftSkills.length > 0 ? aiSoftSkills : mockUserProfile.softSkills),
    [aiSoftSkills],
  );

  const averageTechnical = useMemo(() => {
    const total = technicalSkillsData.reduce((sum, item) => sum + item.value, 0);
    return technicalSkillsData.length ? Math.round(total / technicalSkillsData.length) : 0;
  }, [technicalSkillsData]);

  const averageSoft = useMemo(() => {
    const total = softSkillsData.reduce((sum, item) => sum + item.value, 0);
    return softSkillsData.length ? Math.round(total / softSkillsData.length) : 0;
  }, [softSkillsData]);

  const isUrlValid = (value: string) => {
    if (!value) return true;
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  };

  const updateLink = (field: keyof SocialLinks, value: string) => {
    setLinks((prev) => ({ ...prev, [field]: value }));
    if (!isUrlValid(value)) {
      setLinkErrors((prev) => ({ ...prev, [field]: 'Please enter a valid URL (https://...).'}));
    } else {
      setLinkErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }
  };
  const handleAnalyzeProfile = async () => {
    if (!Object.values(links).some((link) => link.trim())) {
      setAnalysisError('Please provide at least one link.');
      setAnalysisMessage(null);
      return;
    }

    const invalidField = Object.entries(links).find(([, value]) => value && !isUrlValid(value));
    if (invalidField) {
      setAnalysisError(`Please enter a valid URL for ${invalidField[0]}.`);
      setAnalysisMessage(null);
      return;
    }

    setAnalyzing(true);
    setAnalysisError(null);
    setAnalysisMessage('Fetching and parsing skills from provided links...');
    setExtractedSkills([]);

    try {
      const response = await axios.post('/api/analyze-profile', { links });
      const skills = response.data.skills || [];
      const aiTech = (response.data.aiTechnicalSkills || []).map((s: any) => ({
        skill: s.name,
        value: Number(s.score || s.confidence * 100 || 0),
        fullMark: 100,
      }));
      const aiSoft = (response.data.aiSoftSkills || []).map((s: any) => ({
        skill: s.name,
        value: Number(s.score || s.confidence * 100 || 0),
        fullMark: 100,
      }));

      setExtractedSkills(skills);
      setAiSummary(response.data.aiSummary || '');
      setAiGaps(response.data.aiGaps || []);
      setAiRecommendations(response.data.aiRecommendations || []);
      setAiTechnicalSkills(aiTech);
      setAiSoftSkills(aiSoft);

      setIndustryRelevanceScore(response.data.aiIndustryRelevanceScore ?? Math.round((averageTechnical + averageSoft) / 2));
      setIndustryInsights(response.data.aiIndustryInsights || mockUserProfile.aiSummary.industryInsights);
      setTopSkills(response.data.aiTopSkills?.length ? response.data.aiTopSkills : mockUserProfile.aiSummary.topSkills);

      const detectedSkillNames = new Set<string>([...aiTech, ...aiSoft].map((item) => item.skill.toLowerCase()));

      if (detectedSkillNames.size > 0) {
        const updatedJobs = mockUserProfile.jobRecommendations
          .map((job) => {
            const matchCount = job.skills.filter((skill) => detectedSkillNames.has(skill.toLowerCase())).length;
            const matchPercentage = Math.min(100, Math.max(20, Math.round((matchCount / job.skills.length) * 100)));

            return { ...job, matchPercentage };
          })
          .sort((a, b) => b.matchPercentage - a.matchPercentage);

        setJobs(updatedJobs);

        const updatedPath = mockUserProfile.learningPath.map((item) => {
          const matchCount = item.resources.reduce((c, resource) => {
            const text = `${resource.title} ${resource.provider ?? ''} ${resource.url} ${resource?.title ?? ''}`.toLowerCase();
            return c + (Array.from(detectedSkillNames).some((skill) => text.includes(skill)) ? 1 : 0);
          }, 0);
          return {
            ...item,
            priority: matchCount > 0 ? 'high' : item.priority,
          };
        });

        setLearningPath(updatedPath);
      }

      if (skills.length > 0 || aiTech.length > 0 || aiSoft.length > 0) {
        setAnalysisMessage(`Extracted ${skills.length} raw skills and ${aiTech.length + aiSoft.length} AI skills from your links.`);
      } else {
        setAnalysisMessage('No skills were detected in the provided links yet. Try adding GitHub/LinkedIn/portfolio/dev.to links.');
      }
    } catch (err) {
      setAnalysisError('Failed to analyze profile. Please try again.');
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };
  interface Skill {
  name: string;
  confidence: number;
  source: string;
}

  useEffect(() => {
    let ignore = false;

    const loadPreferences = async () => {
      try {
        const response = await fetch('/api/session/preferences', { cache: 'no-store' });

        if (!response.ok) {
          throw new Error('Unable to fetch preferences from session API.');
        }

        const payload = (await response.json()) as {
          theme?: ThemeMode;
          links?: Partial<SocialLinks>;
          expandedJobs?: boolean;
        };

        if (ignore) {
          return;
        }

        if (payload.theme === 'dark' || payload.theme === 'light') {
          setTheme(payload.theme);
        }

        if (typeof payload.expandedJobs === 'boolean') {
          setExpandedJobs(payload.expandedJobs);
        }

        // Keep social inputs blank on load; users enter links manually each session.
      } catch {
        const savedTheme = window.localStorage.getItem('theme-mode');

        if (savedTheme === 'dark' || savedTheme === 'light') {
          setTheme(savedTheme);
        } else {
          // Default to light mode on first visit
          setTheme('light');
        }
      } finally {
        if (!ignore) {
          setPreferencesLoaded(true);
        }
      }
    };

    void loadPreferences();

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('theme-mode', theme);
  }, [theme]);

  useEffect(() => {
    if (!preferencesLoaded) {
      return;
    }

    const requestController = new AbortController();
    const saveTimer = window.setTimeout(() => {
      void fetch('/api/session/preferences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          theme,
          links,
          expandedJobs,
        }),
        signal: requestController.signal,
      }).catch((error: unknown) => {
        // Ignore expected cancellation during rapid updates/unmount.
        if (error instanceof DOMException && error.name === 'AbortError') {
          return;
        }

        console.error('Failed to save preferences', error);
      });
    }, 250);

    return () => {
      window.clearTimeout(saveTimer);
      requestController.abort();
    };
  }, [theme, links, expandedJobs, preferencesLoaded]);

  const handleThemeToggle = () => {
    if (theme === 'dark') {
      setThemeTransition('to-light');
      window.setTimeout(() => setTheme('light'), 260);
      window.setTimeout(() => setThemeTransition(null), 900);
      return;
    }

    setTheme('dark');
    setThemeTransition('to-dark');
    window.setTimeout(() => setThemeTransition('dark-moon'), 560);
    window.setTimeout(() => setThemeTransition(null), 1080);
  };

  return (
    <>
      <AnimatePresence>
        {themeTransition === 'to-light' && (
          <motion.div
            key="sun-expand"
            className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.28 }}
          >
            <motion.div
              className="absolute h-20 w-20 rounded-full bg-amber-200/35"
              initial={{ scale: 0.2 }}
              animate={{ scale: 25 }}
              transition={{ duration: 0.72, ease: 'easeInOut' }}
            />
            <motion.div
              className="rounded-full border border-amber-200/70 bg-amber-100/30 p-5"
              initial={{ opacity: 0.65, scale: 0.4 }}
              animate={{ opacity: 0.18, scale: 1.75 }}
              transition={{ duration: 0.66, ease: 'easeOut' }}
            >
              <Sun className="h-12 w-12 text-amber-50" />
            </motion.div>
          </motion.div>
          
        )}
      </AnimatePresence>

      <AnimatePresence>
        {themeTransition === 'to-dark' && (
          <motion.div
            key="light-shrink"
            className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              className="h-[100vmax] w-[100vmax] rounded-full bg-amber-100/35"
              initial={{ scale: 1.4 }}
              animate={{ scale: 0 }}
              transition={{ duration: 0.52, ease: 'easeInOut' }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {themeTransition === 'dark-moon' && (
          <motion.div
            key="moon-glow"
            className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.22 }}
          >
            <motion.div
              className="rounded-full border border-indigo-200/60 bg-indigo-200/20 p-5"
              initial={{ scale: 0.72, opacity: 0 }}
              animate={{ scale: 1.18, opacity: 0.55 }}
              exit={{ scale: 1.28, opacity: 0 }}
              transition={{ duration: 0.42 }}
            >
              <Moon className="h-11 w-11 text-indigo-100" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-4 py-4 sm:px-8 lg:gap-10 lg:py-8">
        <nav className="glass-panel sticky top-3 z-40 flex items-center gap-3 rounded-2xl px-3 py-2 sm:gap-4 sm:px-4">
          <a
            href="#overview"
            className="inline-flex shrink-0 items-center gap-3 rounded-2xl px-1 py-1 transition hover:bg-white/50"
            aria-label="Go to Next-Gen Skillforge overview"
          >
            <img
              src="/next-gen-skillforge-logo.svg"
              alt="Next-Gen Skillforge logo"
              className="h-11 w-auto shrink-0"
            />
            <span className="flex flex-col leading-tight">
              <span className="text-sm font-semibold text-slate-900 sm:text-base">Next-Gen Skillforge</span>
            </span>
          </a>

          <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto sm:gap-3">
            {navSections.map((section) => (
              <a
                key={section.id}
                href={`#${section.id}`}
                className="shrink-0 rounded-lg px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wide text-slate-700 transition hover:bg-white/70 hover:text-emerald-800"
              >
                {section.label}
              </a>
            ))}
          </div>

          <button
            type="button"
            onClick={handleThemeToggle}
            className="inline-flex shrink-0 items-center gap-2 rounded-xl btn-accent px-3 py-2 text-xs font-semibold uppercase tracking-wide"
            aria-label="Toggle theme"
          >
            {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </nav>

      <motion.section
        id="overview"
        initial="hidden"
        animate="visible"
        variants={animationVariant}
        transition={{ duration: 0.45 }}
        className="glass-panel relative overflow-hidden rounded-3xl p-6 sm:p-8 lg:p-10"
      >
        <div className="pointer-events-none absolute -right-16 -top-16 h-44 w-44 rounded-full bg-emerald-200/40 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-20 -left-16 h-48 w-48 rounded-full bg-amber-200/40 blur-2xl" />

        <p className="mb-2 inline-flex items-center gap-2 rounded-full chip px-4 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-900">
          <Sparkles className="h-3.5 w-3.5" />
          Portfolio Intelligence Dashboard
        </p>
        <h1 className="heading-gradient max-w-3xl text-3xl font-bold sm:text-4xl lg:text-5xl">
          Build your next role strategy with AI-driven skill, fit, and growth insights.
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-slate-700 sm:text-base">
          Connect your professional profiles, evaluate your skills, inspect targeted job recommendations,
          and get a learning path tailored to your current experience and industry demand.
        </p>
      </motion.section>

        <motion.section
        id="links"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={animationVariant}
        transition={{ duration: 0.45, delay: 0.05 }}
        className="glass-panel rounded-3xl p-6 sm:p-8"
      >
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-2xl font-semibold text-slate-900">
            <UserCircle2 className="h-6 w-6 text-emerald-700" />
            Professional Link Inputs
          </h2>
          <button
            type="button"
            onClick={handleAnalyzeProfile}
            disabled={analyzing}
            className="rounded-xl bg-emerald-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:bg-gray-400"
          >
            {analyzing ? 'Analyzing...' : 'Analyze Profile'}
          </button>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
           <LinkInput
            label="GitHub"
            value={links.github}
            onChange={(value) => updateLink('github', value)}
            error={linkErrors.github}
            icon={<Github className="h-4 w-4" />}
          />
          <LinkInput
            label="LinkedIn"
            value={links.linkedin}
            onChange={(value) => updateLink('linkedin', value)}
            error={linkErrors.linkedin}
            icon={<Linkedin className="h-4 w-4" />}
          />
          <LinkInput
            label="Resume"
            value={links.resume}
            onChange={(value) => updateLink('resume', value)}
            error={linkErrors.resume}
            icon={<BookOpen className="h-4 w-4" />}
          />
          <LinkInput
            label="Portfolio"
            value={links.portfolio}
            onChange={(value) => updateLink('portfolio', value)}
            error={linkErrors.portfolio}
            icon={<ExternalLink className="h-4 w-4" />}
          />
          <LinkInput
            label="Twitter"
            value={links.twitter}
            onChange={(value) => updateLink('twitter', value)}
            error={linkErrors.twitter}
            icon={<ExternalLink className="h-4 w-4" />}
          />
          <LinkInput
            label="Dev.to"
            value={links.devto}
            onChange={(value) => updateLink('devto', value)}
            error={linkErrors.devto}
            icon={<ExternalLink className="h-4 w-4" />}
          />
        </div>
         {/* Error Display */}
        {analysisError && <p className="text-red-500 mt-4">{analysisError}</p>}
        {!analysisError && analysisMessage && (
          <p className="text-emerald-600 mt-4">{analysisMessage}</p>
        )}

        {/* Display Extracted Skills */}
        {extractedSkills.length > 0 && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-semibold mb-3 text-slate-900">AI-Extracted Skills</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {extractedSkills.map((skill, index) => (
                <div key={index} className="p-3 bg-white border border-gray-200 rounded-lg">
                  <p className="font-medium text-slate-900">{skill.name}</p>
                  <p className="text-sm text-slate-600">Confidence: {(skill.confidence * 100).toFixed(0)}%</p>
                  <p className="text-xs text-slate-500">From: {skill.source}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      

      </motion.section>

      <motion.section
        id="skills"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={animationVariant}
        transition={{ duration: 0.45, delay: 0.08 }}
        className="grid gap-6 lg:grid-cols-2"
      >
        <RadarPanel
          title="Technical Skill Spider Graph"
          subtitle={`Average score ${averageTechnical}%`}
          data={technicalSkillsData}
          stroke="#8B5CF6"
          fill="#8B5CF6"
        />
        <RadarPanel
          title="Experience & Soft Skills Spider Graph"
          subtitle={`Average score ${averageSoft}%`}
          data={softSkillsData}
          stroke="#F97316"
          fill="#F97316"
        />
      </motion.section>

      <motion.section
        id="jobs"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={animationVariant}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="glass-panel rounded-3xl p-6 sm:p-8"
      >
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <h2 className="flex items-center gap-2 text-2xl font-semibold text-slate-900">
            <Briefcase className="h-6 w-6 text-emerald-700" />
            Job Recommendation System
          </h2>
          <button
            type="button"
            onClick={() => setExpandedJobs((prev) => !prev)}
            className="rounded-xl btn-accent px-4 py-2 text-sm font-semibold"
          >
            {expandedJobs ? 'Show Top 3' : 'Expand to Top 5'}
          </button>
        </div>

        <div className="grid gap-4">
          {jobsToShow.map((job) => (
            <article key={job.id} className="rounded-2xl border border-emerald-100 bg-white/80 p-5 shadow-sm">
              <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">{job.title}</h3>
                  <p className="text-sm text-slate-600">
                    {job.company} · {job.location}
                  </p>
                </div>
                <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800">
                  {job.matchPercentage}% match
                </span>
              </div>
              <p className="mb-2 text-sm text-slate-700">{job.description}</p>
              <div className="mb-3 flex flex-wrap gap-2">
                {job.skills.map((skill) => (
                  <span
                    key={`${job.id}-${skill}`}
                    className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-700"
                  >
                    {skill}
                  </span>
                ))}
              </div>
              <p className="text-sm font-medium text-slate-800">
                {job.type} · {job.salary}
              </p>
            </article>
          ))}
        </div>
      </motion.section>

      <motion.section
        id="summary"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={animationVariant}
        transition={{ duration: 0.45, delay: 0.14 }}
        className="grid gap-6 lg:grid-cols-5"
      >
        <article className="glass-panel rounded-3xl p-6 lg:col-span-3">
          <h2 className="mb-3 flex items-center gap-2 text-2xl font-semibold text-slate-900">
            <Sparkles className="h-6 w-6 text-emerald-700" />
            AI Summary: Skills, Experience, Industry Relevance
          </h2>
          <p className="text-sm leading-6 text-slate-700">{aiSummary || mockUserProfile.aiSummary.overview}</p>

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-emerald-800">Strengths</h3>
              <ul className="space-y-2 text-sm text-slate-700">
                {(aiRecommendations.length > 0 ? aiRecommendations : mockUserProfile.aiSummary.strengths).map((item) => (
                  <li key={item} className="rounded-xl bg-emerald-50 px-3 py-2">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-orange-700">Growth Gaps</h3>
              <ul className="space-y-2 text-sm text-slate-700">
                {(aiGaps.length > 0 ? aiGaps : mockUserProfile.aiSummary.gaps).map((item) => (
                  <li key={item} className="rounded-xl bg-orange-50 px-3 py-2">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </article>

        <article className="glass-panel rounded-3xl p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-slate-900">Industry Relevance</h3>
          <div className="mt-4 rounded-2xl border border-emerald-200 bg-white p-4">
            <p className="text-3xl font-bold text-emerald-800">
              {industryRelevanceScore}%
            </p>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-emerald-100">
              <div
                className="h-full rounded-full bg-emerald-600"
                style={{ width: `${industryRelevanceScore}%` }}
              />
            </div>
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-700">{industryInsights}</p>
          <div className="mt-4 flex flex-wrap gap-2">
            {(topSkills.length > 0 ? topSkills : mockUserProfile.aiSummary.topSkills).map((skill) => (
              <span key={skill} className="rounded-full chip px-3 py-1 text-xs font-semibold text-emerald-800">
                {skill}
              </span>
            ))}
          </div>
        </article>
      </motion.section>

      <motion.section
        id="learning"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={animationVariant}
        transition={{ duration: 0.45, delay: 0.2 }}
        className="glass-panel rounded-3xl p-6 sm:p-8"
      >
        <h2 className="mb-5 flex items-center gap-2 text-2xl font-semibold text-slate-900">
          <Lightbulb className="h-6 w-6 text-amber-600" />
          AI Learning Path Suggestions
        </h2>

        <div className="grid gap-4 lg:grid-cols-2">
          {learningPath.map((path) => (
            <article key={path.id} className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-lg font-semibold text-slate-900">{path.topic}</h3>
                <span
                  className={cn(
                    'rounded-full border px-3 py-1 text-xs font-semibold uppercase',
                    priorityClass[path.priority],
                  )}
                >
                  {path.priority}
                </span>
              </div>

              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Estimated time: {path.timeEstimate}
              </p>
              <p className="mb-4 text-sm leading-6 text-slate-700">{path.explanation}</p>

              <div className="space-y-2">
                {path.resources.map((resource) => (
                  <a
                    key={`${path.id}-${resource.title}`}
                    href={resource.url}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-start justify-between gap-3 rounded-xl border border-slate-200 px-3 py-2 text-sm transition hover:border-emerald-300 hover:bg-emerald-50"
                  >
                    <span className="font-medium text-slate-800">
                      {resource.title}
                      {resource.provider ? ` · ${resource.provider}` : ''}
                    </span>
                    <span className="shrink-0 text-xs font-semibold text-emerald-700">
                      {resourceLabel[resource.type]}
                      {resource.free ? ' · Free' : ''}
                    </span>
                  </a>
                ))}
              </div>
            </article>
          ))}
        </div>
      </motion.section>
      
      </main>
    </>
  );
}

function RadarPanel({
  title,
  subtitle,
  data,
  stroke,
  fill,
}: {
  title: string;
  subtitle: string;
  data: { skill: string; value: number; fullMark: number }[];
  stroke: string;
  fill: string;
}) {
  return (
    <article className="glass-panel rounded-3xl p-6 sm:p-8">
      <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
      <p className="mb-4 mt-1 text-sm text-slate-600">{subtitle}</p>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="72%">
            <PolarGrid stroke="var(--grid)" />
            <PolarAngleAxis dataKey="skill" tick={{ fill: 'var(--text-primary)', fontSize: 12 }} />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--card)',
                borderColor: 'var(--line)',
                color: 'var(--text-primary)',
                boxShadow: '0 8px 22px rgba(0,0,0,0.25)',
                borderRadius: '0.65rem',
              }}
              labelStyle={{ color: 'var(--text-secondary)', fontWeight: 600 }}
              itemStyle={{ color: 'var(--text-primary)' }}
            />
            <Radar
              dataKey="value"
              stroke={stroke}
              fill={fill}
              fillOpacity={0.35}
              strokeWidth={2}
              dot={{ fill: stroke, strokeWidth: 2, r: 4 }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function LinkInput({
  label,
  value,
  onChange,
  icon,
  error,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  icon: React.ReactNode;
  error?: string;
}) {
  return (
    <label className="space-y-1.5">
      <span className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </span>
      <span className="relative block">
        <span
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2"
          style={{ color: 'var(--text-muted)' }}
        >
          {icon}
        </span>
        <input
          type="url"
          autoComplete="off"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={`Enter ${label} URL`}
          style={{
            backgroundColor: 'var(--input-bg)',
            borderColor: value ? 'var(--accent)' : 'var(--line)',
            color: 'var(--text-primary)',
          }}
          className="w-full rounded-xl border px-10 py-2.5 text-sm outline-none transition placeholder:text-[color:var(--text-muted)] focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/30"
        />
      </span>
      {error ? <p className="mt-1 text-xs font-medium text-rose-300">{error}</p> : null}
    </label>
  );
}
