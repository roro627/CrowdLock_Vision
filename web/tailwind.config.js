/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        accent: '#1dd1a1',
        accent2: '#ff9f43'
      }
    }
  },
  plugins: []
};
