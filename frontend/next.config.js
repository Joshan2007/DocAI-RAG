/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // BACKEND_URL must be set as an environment variable in Vercel:
    //   e.g.  https://docai-backend.onrender.com
    // In local dev it falls back to localhost:8000 automatically.
    const backendBase =
      process.env.BACKEND_URL ||
      (process.env.NODE_ENV === 'production'
        ? '' // no fallback in prod — BACKEND_URL must be set
        : 'http://127.0.0.1:8000');

    if (!backendBase) return []; // graceful no-op until BACKEND_URL is set

    return [
      {
        source: '/api/:path*',
        destination: `${backendBase}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
