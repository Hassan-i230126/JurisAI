/**
 * Juris AI — API configuration
 *
 * In local dev, VITE_BACKEND_URL is empty so all fetch calls use relative
 * paths like /api/... which Vite's proxy forwards to localhost:8000.
 *
 * On Vercel, set VITE_BACKEND_URL to your ngrok / Cloudflare Tunnel URL
 * (e.g. https://abc123.ngrok-free.app) in the Vercel project's
 * Environment Variables dashboard. No trailing slash.
 */
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, '') ?? ''

/**
 * Build a full API URL that works both locally and on Vercel.
 * @param {string} path - e.g. '/api/health' or '/api/clients'
 */
export function apiUrl(path) {
  return `${BACKEND_URL}${path}`
}
