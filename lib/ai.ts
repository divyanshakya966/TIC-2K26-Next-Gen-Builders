import type { SocialLinks } from './types';

export type AIProvider = 'openai' | 'azure_openai' | 'anthropic' | 'huggingface' | 'none';

export interface AIDetectedSkill {
  name: string;
  score: number; // 0-100
  confidence: number; // 0-1
  type: 'technical' | 'soft' | 'other';
  source: string;
}

export interface AIJobRecommendation {
  title: string;
  company: string;
  type: 'Full-time' | 'Internship (Paid)' | 'Internship (Unpaid)' | 'Contract';
  salary?: string;
  location: string;
  skills: string[];
  description: string;
  matchPercentage: number;
  fitReason: string;
}

export interface AIProfileAnalysis {
  technicalSkills: AIDetectedSkill[];
  softSkills: AIDetectedSkill[];
  summary: string;
  strengths: string[];
  gaps: string[];
  recommendations: string[];
  jobRecommendations: AIJobRecommendation[];
  industryRelevanceScore: number;
  atsScore: number;
  atsFeedback: string[];
  industryInsights: string;
  topSkills: string[];
  rawText: string;
}

const defaultProvider: AIProvider = 'openai';

export function getAIProvider(): AIProvider {
  if (process.env.OPENAI_API_KEY) return 'openai';
  if (process.env.AZURE_OPENAI_API_KEY) return 'azure_openai';
  if (process.env.ANTHROPIC_API_KEY) return 'anthropic';
  if (process.env.HF_API_TOKEN) return 'huggingface';
  return 'none';
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeSkillName(name: string): string {
  return name
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[^a-zA-Z0-9 ._+-]/g, '')
    .replace(/\bjs\b/i, 'JavaScript')
    .replace(/\bts\b/i, 'TypeScript')
    .replace(/\breactjs\b/i, 'React')
    .replace(/\bvuejs\b/i, 'Vue.js')
    .replace(/\bnextjs\b/i, 'Next.js')
    .replace(/\bnodejs\b/i, 'Node.js');
}

function extractJsonObject(output: string): Record<string, unknown> | null {
  const trimmed = output.trim();
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = (fenceMatch?.[1] ?? trimmed).trim();

  const firstBrace = candidate.indexOf('{');
  const lastBrace = candidate.lastIndexOf('}');

  if (firstBrace < 0 || lastBrace < 0 || lastBrace <= firstBrace) {
    return null;
  }

  const jsonString = candidate.slice(firstBrace, lastBrace + 1);

  try {
    const parsed = JSON.parse(jsonString);
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function mapSkillItems(items: unknown, type: AIDetectedSkill['type']): AIDetectedSkill[] {
  if (!Array.isArray(items)) return [];

  return items
    .map((item) => {
      if (!item || typeof item !== 'object') return null;

      const skill = item as Record<string, unknown>;
      if (typeof skill.name !== 'string' || !skill.name.trim()) return null;

      const confidence = clamp(Number(skill.confidence) || 0.5, 0, 1);
      const numericScore = Number(skill.score);
      const score = clamp(Number.isFinite(numericScore) ? numericScore / 100 : confidence, 0, 1) * 100;

      return {
        name: normalizeSkillName(skill.name),
        score: Math.round(score),
        confidence,
        type,
        source: typeof skill.source === 'string' && skill.source.trim() ? skill.source : 'AI',
      };
    })
    .filter(Boolean) as AIDetectedSkill[];
}

async function callOpenAI(prompt: string): Promise<string> {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('OPENAI_API_KEY missing');

  const model = process.env.OPENAI_MODEL || 'gpt-4o-mini';
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.05,
      max_tokens: 900,
      messages: [
        { role: 'system', content: 'You are a helpful AI that extracts skills and confidence scores from professional profile text.' },
        {
          role: 'user',
          content: prompt,
        },
      ],
    }),
  });

  if (!response.ok) {
    const errBody = await response.text();
    throw new Error(`OpenAI error ${response.status} ${errBody}`);
  }

  const data = await response.json();
  return data?.choices?.[0]?.message?.content ?? '';
}

async function callAzureOpenAI(prompt: string): Promise<string> {
  const key = process.env.AZURE_OPENAI_API_KEY;
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
  const deployment = process.env.AZURE_OPENAI_DEPLOYMENT || 'gpt-4o-mini';
  if (!key || !endpoint) throw new Error('Azure OpenAI env keys missing');

  const response = await fetch(`${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=2024-12-01`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'api-key': key,
    },
    body: JSON.stringify({
      messages: [
        { role: 'system', content: 'You are a helpful AI that extracts skills and confidence scores from professional profile text.' },
        { role: 'user', content: prompt },
      ],
      temperature: 0.05,
      max_tokens: 900,
    }),
  });

  if (!response.ok) {
    const errBody = await response.text();
    throw new Error(`Azure OpenAI error ${response.status} ${errBody}`);
  }

  const data = await response.json();
  return data?.choices?.[0]?.message?.content ?? '';
}

async function callAnthropic(prompt: string): Promise<string> {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) throw new Error('ANTHROPIC_API_KEY missing');

  const response = await fetch('https://api.anthropic.com/v1/complete', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: 'claude-3.5-sonic',
      prompt: `\n\nHuman: ${prompt}\n\nAssistant:`,
      max_tokens_to_sample: 900,
      temperature: 0.05,
      stop_sequences: ['\n\nHuman:'],
    }),
  });

  if (!response.ok) {
    const errBody = await response.text();
    throw new Error(`Anthropic error ${response.status} ${errBody}`);
  }

  const data = await response.json();
  return data?.completion ?? '';
}

async function callHuggingFace(prompt: string): Promise<string> {
  const token = process.env.HF_API_TOKEN;
  if (!token) throw new Error('HF_API_TOKEN missing');

  // Use a modern instruction-tuned model instead of GPT-2 (from 2019)
  // Mistral-7B-Instruct can follow JSON instructions reliably
  const model = process.env.HF_MODEL || 'mistralai/Mistral-7B-Instruct-v0.1';

  // Wrap prompt with instruction tags that modern models expect
  const formattedPrompt = `[INST] ${prompt} [/INST]`;

  const response = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      inputs: formattedPrompt,
      parameters: {
        max_new_tokens: 1200, // Slightly higher for complex JSON with job recommendations
        temperature: 0.05,
        return_full_text: false, // Critical: don't return the original prompt in the response
      },
    }),
  });

  if (!response.ok) {
    const errBody = await response.text();
    throw new Error(`HuggingFace error ${response.status} ${errBody}`);
  }

  const data = await response.json();

  // Extract the generated text (handles both array and object responses)
  let generatedText = '';
  if (Array.isArray(data) && data[0]?.generated_text) {
    generatedText = data[0].generated_text;
  } else if (data.generated_text) {
    generatedText = data.generated_text;
  }

  // Strip the instruction tags from the response if they're included
  // This handles cases where the model echoes back the tags
  const cleanedText = generatedText
    .replace(/^\[INST\]\s*/, '')
    .replace(/\s*\[\/INST\]\s*/, '')
    .trim();

  return cleanedText;
}

export async function callModel(prompt: string): Promise<string> {
  const provider = getAIProvider() || defaultProvider;

  if (provider === 'openai') return await callOpenAI(prompt);
  if (provider === 'azure_openai') return await callAzureOpenAI(prompt);
  if (provider === 'anthropic') return await callAnthropic(prompt);
  if (provider === 'huggingface') return await callHuggingFace(prompt);

  throw new Error('No AI provider configured. Provide OPENAI_API_KEY/AZURE_OPENAI_API_KEY/ANTHROPIC_API_KEY/HF_API_TOKEN.');
}

export function parseSkillsFromAI(output: string): AIDetectedSkill[] {
  const parsed = extractJsonObject(output);
  if (!parsed) return [];

  if (Array.isArray(parsed.skills)) {
    return parsed.skills
      .map((item) => {
        if (!item || typeof item !== 'object') return null;

        const skill = item as Record<string, unknown>;
        if (typeof skill.name !== 'string' || !skill.name.trim()) return null;

        const type = skill.type === 'soft' ? 'soft' : skill.type === 'technical' ? 'technical' : 'other';
        const confidence = clamp(Number(skill.confidence) || 0.5, 0, 1);
        const numericScore = Number(skill.score);
        const score = clamp(Number.isFinite(numericScore) ? numericScore / 100 : confidence, 0, 1) * 100;

        return {
          name: normalizeSkillName(skill.name),
          score: Math.round(score),
          confidence,
          type,
          source: typeof skill.source === 'string' && skill.source.trim() ? skill.source : 'AI',
        };
      })
      .filter(Boolean) as AIDetectedSkill[];
  }

  return [...mapSkillItems(parsed.technicalSkills, 'technical'), ...mapSkillItems(parsed.softSkills, 'soft')];
}

export async function analyzeProfileText(rawText: string): Promise<AIProfileAnalysis> {
  const prompt = `Analyze the profile text and return ONLY valid JSON with this exact shape:
{
  "technicalSkills": [{"name":"string","confidence":0.0,"score":0,"source":"string"}],
  "softSkills": [{"name":"string","confidence":0.0,"score":0,"source":"string"}],
  "summary":"string",
  "gaps":["string"],
  "recommendations":["string"],
  "atsScore": 0,
  "atsFeedback": ["string"],
  "jobRecommendations": [
    {
      "title":"string",
      "company":"string",
      "type":"Full-time|Internship (Paid)|Internship (Unpaid)|Contract",
      "salary":"string or null",
      "location":"string",
      "skills":["string"],
      "description":"string",
      "matchPercentage":80,
      "fitReason":"string"
    }
  ]
}

Rules:
- Do not wrap the response in markdown fences.
- Do not include any extra keys or commentary.
- confidence must be between 0 and 1.
- score must be between 0 and 100.
- Generate 5-8 job recommendations based on the user's specific skills and experience level.
- Include a mix of full-time roles, paid internships, and unpaid internships.
- Tailor all job recommendations to match the user's actual skills detected in their profile.
- matchPercentage should reflect how well the job aligns with the user's skills (0-100).
- atsScore must be between 0 and 100 and estimate resume ATS-readiness (keyword coverage, role alignment, measurable impact statements, and formatting clarity).
- atsFeedback should contain 3-6 concise, actionable points to improve ATS score.
- location can be "Remote", "Hybrid", or a specific city.
- salary should be a realistic range or null for internships.

Profile text:
${rawText}`;

  const aiResponse = await callModel(prompt);
  const parsed = extractJsonObject(aiResponse);
  const skills = parseSkillsFromAI(aiResponse);

  const technicalSkills = Array.isArray(parsed?.technicalSkills)
    ? mapSkillItems(parsed.technicalSkills, 'technical')
    : skills.filter((s) => s.type === 'technical');
  const softSkills = Array.isArray(parsed?.softSkills)
    ? mapSkillItems(parsed.softSkills, 'soft')
    : skills.filter((s) => s.type === 'soft');

  let summary = '';
  let gaps: string[] = [];
  let recommendations: string[] = [];
  let atsScore = 0;
  let atsFeedback: string[] = [];
  let jobRecommendations: AIJobRecommendation[] = [];

  if (parsed) {
    summary = typeof parsed.summary === 'string' ? parsed.summary : '';
    gaps = Array.isArray(parsed.gaps) ? parsed.gaps.slice(0, 6).map(String) : [];
    recommendations = Array.isArray(parsed.recommendations) ? parsed.recommendations.slice(0, 6).map(String) : [];
    atsScore = Math.min(100, Math.max(0, Number(parsed.atsScore) || 0));
    atsFeedback = Array.isArray(parsed.atsFeedback) ? parsed.atsFeedback.slice(0, 6).map(String) : [];
    
    if (Array.isArray(parsed.jobRecommendations)) {
      jobRecommendations = parsed.jobRecommendations
        .slice(0, 8)
        .map((job: any, index: number) => ({
          title: typeof job.title === 'string' ? job.title : `Position ${index + 1}`,
          company: typeof job.company === 'string' ? job.company : 'Company',
          type: ['Full-time', 'Internship (Paid)', 'Internship (Unpaid)', 'Contract'].includes(job.type) ? job.type : 'Full-time',
          salary: job.salary ? String(job.salary) : undefined,
          location: typeof job.location === 'string' ? job.location : 'Remote',
          skills: Array.isArray(job.skills) ? job.skills.map(String).slice(0, 5) : [],
          description: typeof job.description === 'string' ? job.description : '',
          matchPercentage: Math.min(100, Math.max(0, Number(job.matchPercentage) || 50)),
          fitReason: typeof job.fitReason === 'string' ? job.fitReason : 'Good match for your skills',
        }))
        .filter((job: any) => job.title && job.company);
    }
  } else {
    summary = aiResponse.slice(0, 600);
  }

  const technicalAvg = technicalSkills.length
    ? technicalSkills.reduce((sum, skill) => sum + skill.score, 0) / technicalSkills.length
    : 0;
  const softAvg = softSkills.length ? softSkills.reduce((sum, skill) => sum + skill.score, 0) / softSkills.length : 0;
  const industryRelevanceScore = Math.round((technicalAvg + softAvg) / 2);

  const topSkills = [...technicalSkills, ...softSkills]
    .sort((a, b) => b.score - a.score)
    .slice(0, 6)
    .map((skill) => skill.name);

  const industryInsights = `Technical average ${Math.round(technicalAvg)}, soft average ${Math.round(softAvg)}. Prioritize upskilling in ${topSkills.slice(0, 3).join(', ')}.`;

  const computedAtsScore = atsScore > 0
    ? Math.round(atsScore)
    : Math.round((technicalAvg * 0.55) + (softAvg * 0.2) + (topSkills.length >= 5 ? 15 : topSkills.length * 3));

  const normalizedAtsScore = Math.max(0, Math.min(100, computedAtsScore));

  const normalizedAtsFeedback = atsFeedback.length > 0
    ? atsFeedback
    : [
        'Tailor resume keywords to match target role descriptions.',
        'Highlight measurable project outcomes (metrics, impact, scale).',
        'Use clear section headers and consistent role-based skill grouping.',
      ];

  return {
    technicalSkills,
    softSkills,
    summary,
    strengths: topSkills.slice(0, 5).map((skill) => `Strong signal in ${skill}`),
    gaps,
    recommendations,
    jobRecommendations,
    industryRelevanceScore,
    atsScore: normalizedAtsScore,
    atsFeedback: normalizedAtsFeedback,
    industryInsights,
    topSkills,
    rawText,
  };
}
