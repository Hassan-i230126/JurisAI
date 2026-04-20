import { defaultSchema } from 'rehype-sanitize'

export function formatTimestamp(value) {
  const date = value instanceof Date ? value : new Date(value)
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  return `${hours}:${minutes}`
}

export function distanceToRelevance(distance) {
  if (typeof distance !== 'number' || Number.isNaN(distance)) {
    return 'medium'
  }
  if (distance < 0.5) {
    return 'high'
  }
  if (distance <= 0.9) {
    return 'medium'
  }
  return 'low'
}

export function formatCitation(rawCitation, index = 0) {
  if (!rawCitation) {
    return {
      id: `citation-${index}`,
      title: 'Unknown source',
      type: 'statute',
      relevance: 'medium',
      distance: null,
    }
  }

  if (typeof rawCitation === 'object') {
    const title = rawCitation.title || rawCitation.citation || `Source ${index + 1}`
    const type = rawCitation.doc_type || inferCitationType(title)
    const distance =
      typeof rawCitation.distance === 'number' ? rawCitation.distance : null

    return {
      id: rawCitation.id || `citation-${index}`,
      title,
      type,
      relevance: distanceToRelevance(distance),
      distance,
    }
  }

  const parsed = parseDistanceToken(rawCitation)

  return {
    id: `citation-${index}`,
    title: parsed.title,
    type: inferCitationType(parsed.title),
    relevance: distanceToRelevance(parsed.distance),
    distance: parsed.distance,
  }
}

export function formatCitations(citations = []) {
  return citations.map((entry, index) => formatCitation(entry, index))
}

function parseDistanceToken(text) {
  const match = text.match(/(?:distance|dist)\s*[:=]\s*([0-9]*\.?[0-9]+)/i)
  if (!match) {
    return { title: text, distance: null }
  }

  const distance = Number.parseFloat(match[1])
  const title = text.replace(match[0], '').replace(/[|,-]\s*$/, '').trim()
  return {
    title: title || text,
    distance: Number.isNaN(distance) ? null : distance,
  }
}

function inferCitationType(citation) {
  const lowered = citation.toLowerCase()
  if (lowered.includes('judgment') || lowered.includes('ca-') || lowered.includes('supreme')) {
    return 'judgment'
  }
  return 'statute'
}

export const markdownSanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [...(defaultSchema.attributes?.code || []), ['className']],
    span: [...(defaultSchema.attributes?.span || []), ['className']],
  },
}
