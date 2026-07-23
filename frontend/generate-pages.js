const fs = require('fs');
const path = require('path');

const pages = [
  { route: '(auth)/login', name: 'Login' },
  { route: '(auth)/signup', name: 'Signup' },
  { route: '(auth)/forgot-password', name: 'ForgotPassword' },
  { route: '(dashboard)', name: 'Dashboard' },
  { route: '(dashboard)/disease', name: 'DiseaseDetection' },
  { route: '(dashboard)/iot', name: 'IoTDashboard' },
  { route: '(dashboard)/weather', name: 'Weather' },
  { route: '(dashboard)/assistant', name: 'Assistant' },
  { route: '(dashboard)/history', name: 'History' },
  { route: '(dashboard)/settings', name: 'Settings' },
  { route: '(dashboard)/profile', name: 'Profile' }
];

const basePath = path.join(__dirname, 'src', 'app');

pages.forEach(page => {
  const fileContent = `export default function ${page.name}Page() {\n  return (\n    <div className="p-6">\n      <h1 className="text-2xl font-bold">${page.name}</h1>\n      <p className="text-muted-foreground mt-2">Mock data view.</p>\n    </div>\n  );\n}\n`;
  const filePath = path.join(basePath, page.route, 'page.tsx');
  fs.writeFileSync(filePath, fileContent);
});

console.log('Generated placeholder pages.');
