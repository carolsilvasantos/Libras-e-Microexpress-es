/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#c8ff00",
        cyber: {
          bg: "#121212",
          surface: "#1e1e1e",
          accent: "#c8ff00"
        }
      }
    },
  },
  plugins: [],
}
