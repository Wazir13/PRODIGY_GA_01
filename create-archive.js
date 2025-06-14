const fs = require('fs');
const path = require('path');

// Create a simple script to help organize files for download
const projectStructure = {
  'package.json': 'Root package.json file',
  'index.html': 'Main HTML file',
  'vite.config.ts': 'Vite configuration',
  'tailwind.config.js': 'Tailwind CSS configuration',
  'postcss.config.js': 'PostCSS configuration',
  'tsconfig.json': 'TypeScript configuration',
  'tsconfig.app.json': 'TypeScript app configuration',
  'tsconfig.node.json': 'TypeScript node configuration',
  'eslint.config.js': 'ESLint configuration',
  'src/main.tsx': 'React entry point',
  'src/App.tsx': 'Main App component',
  'src/index.css': 'Global styles',
  'src/vite-env.d.ts': 'Vite environment types',
  'src/components/Header.tsx': 'Header component',
  'src/components/TabNavigation.tsx': 'Tab navigation component',
  'src/components/OverviewTab.tsx': 'Overview tab component',
  'src/components/DatasetTab.tsx': 'Dataset management tab',
  'src/components/TrainingTab.tsx': 'Training configuration tab',
  'src/components/PlaygroundTab.tsx': 'Text generation playground',
  'src/components/EvaluationTab.tsx': 'Model evaluation tab',
  'src/components/ResourcesTab.tsx': 'Resources and documentation tab'
};

console.log('GPT-2 Fine-tuning Project Structure:');
console.log('=====================================');
Object.entries(projectStructure).forEach(([file, description]) => {
  console.log(`${file.padEnd(35)} - ${description}`);
});

console.log('\nTo recreate this project:');
console.log('1. Create the directory structure shown above');
console.log('2. Copy each file content from the web interface');
console.log('3. Run "npm install" to install dependencies');
console.log('4. Run "npm run dev" to start the development server');