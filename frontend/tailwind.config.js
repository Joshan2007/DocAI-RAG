/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: "#FAF9F5",
          panel: "#F3F2ED",
          card: "#FFFFFF",
          border: "#E5E4DE",
          hover: "#ECEAE4",
          terracotta: "#CC785C",
          "terracotta-dark": "#B8664B",
          text: "#1F1E1D",
          muted: "#6B6963",
          light: "#94928B",
        },
      },
      fontFamily: {
        serif: ["Charter", "Bitstream Charter", "Sitka Text", "Cambria", "serif"],
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [],
};
