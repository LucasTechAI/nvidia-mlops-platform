import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        nvidia: {
          DEFAULT: "#76B900",
          dark: "#5a8f00",
          light: "#9ED700",
          50: "#f3ffe0",
          100: "#e4ffc2",
          200: "#c5ff85",
          300: "#a5f049",
          400: "#8dd622",
          500: "#76B900",
          600: "#5a8f00",
          700: "#446d00",
          800: "#365600",
          900: "#2d4800",
        },
        surface: {
          DEFAULT: "#0E1117",
          card: "#1a1c24",
          hover: "#262730",
          border: "#2d3040",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "Monaco", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
