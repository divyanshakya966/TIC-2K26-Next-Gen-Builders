import { NextResponse } from 'next/server';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

export const runtime = 'nodejs';

const MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024;

const RESUME_SKILL_KEYWORDS = [
  'react',
  'javascript',
  'typescript',
  'node',
  'node.js',
  'next.js',
  'python',
  'java',
  'sql',
  'postgresql',
  'mongodb',
  'docker',
  'kubernetes',
  'aws',
  'azure',
  'git',
  'graphql',
  'rest',
  'api',
  'tailwind',
  'html',
  'css',
  'testing',
  'jest',
  'playwright',
  'ci/cd',
  'devops',
];

function normalizeSkillName(skill: string): string {
  return skill
    .replace(/\./g, ' ')
    .replace(/\//g, ' ')
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function detectSkills(text: string): string[] {
  const lowered = text.toLowerCase();
  const found = new Set<string>();

  for (const keyword of RESUME_SKILL_KEYWORDS) {
    if (lowered.includes(keyword.toLowerCase())) {
      found.add(normalizeSkillName(keyword));
    }
  }

  return Array.from(found).slice(0, 16);
}

function cleanExtractedText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

async function extractPdfText(buffer: Buffer): Promise<string> {
  const pdfParse = require('pdf-parse/lib/pdf-parse.js') as (
    dataBuffer: Buffer,
    options?: Record<string, unknown>,
  ) => Promise<{ text: string }>;
  const parsed = await pdfParse(buffer);
  return cleanExtractedText(parsed.text || '');
}

async function extractDocxText(buffer: Buffer): Promise<string> {
  const mammothModule = await import('mammoth');
  const mammoth = mammothModule.default;
  const parsed = await mammoth.extractRawText({ buffer });
  return cleanExtractedText(parsed.value || '');
}

async function extractPlainText(buffer: Buffer): Promise<string> {
  return cleanExtractedText(buffer.toString('utf8'));
}

function inferExtension(fileName: string): string {
  const trimmed = fileName.trim().toLowerCase();
  const index = trimmed.lastIndexOf('.');
  return index >= 0 ? trimmed.slice(index) : '';
}

function extractErrorMessage(error: unknown): string {
  if (error instanceof Error && typeof error.message === 'string' && error.message.trim()) {
    return error.message.trim();
  }

  return 'Unknown parser error';
}

function getResumeParseError(extension: string, mimeType: string, error: unknown): string {
  const parserMessage = extractErrorMessage(error);

  if (extension === '.pdf' || mimeType.includes('pdf')) {
    return `Could not read this PDF. If it is scanned, encrypted, or image-only, export it as text-based PDF or DOCX and try again. (${parserMessage})`;
  }

  if (extension === '.docx' || mimeType.includes('wordprocessingml.document')) {
    return `Could not read this DOCX file. Try opening it in Word/Google Docs and re-saving as DOCX or PDF. (${parserMessage})`;
  }

  if (extension === '.txt' || mimeType.startsWith('text/')) {
    return `Could not read this text file. Ensure it is UTF-8 encoded and not empty. (${parserMessage})`;
  }

  return `Resume parsing failed for this file. (${parserMessage})`;
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const uploadedResume = formData.get('resume');

    if (!(uploadedResume instanceof File)) {
      return NextResponse.json({ error: 'Please upload a resume file.' }, { status: 400 });
    }

    const file = uploadedResume;

    if (file.size === 0) {
      return NextResponse.json({ error: 'Uploaded file is empty.' }, { status: 400 });
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return NextResponse.json({ error: 'Resume is too large. Keep file size under 8MB.' }, { status: 400 });
    }

    const fileName = file.name || 'resume';
    const extension = inferExtension(fileName);
    const mimeType = (file.type || '').toLowerCase();

    const byteArray = await file.arrayBuffer();
    const buffer = Buffer.from(byteArray);

    let extractedText = '';

    if (mimeType.includes('msword') || extension === '.doc') {
      return NextResponse.json(
        { error: 'Legacy .doc files are not supported yet. Please re-save as .docx or PDF and upload again.' },
        { status: 400 },
      );
    }

    try {
      if (mimeType.includes('pdf') || extension === '.pdf') {
        extractedText = await extractPdfText(buffer);
      } else if (
        mimeType.includes('wordprocessingml.document')
        || extension === '.docx'
      ) {
        extractedText = await extractDocxText(buffer);
      } else if (mimeType.startsWith('text/') || extension === '.txt' || !mimeType) {
        extractedText = await extractPlainText(buffer);
      } else {
        return NextResponse.json(
          { error: 'Unsupported file type. Upload PDF, DOCX, or TXT.' },
          { status: 400 },
        );
      }
    } catch (parseError) {
      return NextResponse.json(
        { error: getResumeParseError(extension, mimeType, parseError) },
        { status: 422 },
      );
    }

    if (!extractedText || extractedText.length < 40) {
      return NextResponse.json(
        { error: 'Could not extract enough readable content from this resume.' },
        { status: 422 },
      );
    }

    const skills = detectSkills(extractedText);

    return NextResponse.json({
      fileName,
      resumeText: extractedText.slice(0, 18000),
      detectedSkills: skills,
      extractedChars: extractedText.length,
    });
  } catch (error) {
    console.error('Resume import failed:', error);
    const genericMessage = extractErrorMessage(error);
    return NextResponse.json(
      { error: `Failed to parse uploaded resume. ${genericMessage}` },
      { status: 500 },
    );
  }
}
