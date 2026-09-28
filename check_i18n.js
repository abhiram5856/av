const fs = require('fs');
const content = fs.readFileSync('frontend/src/lib/i18n.tsx', 'utf-8');

// Match everything between en: { and the next root key
const enMatch = content.match(/en:\s*{([\s\S]*?)},\s*te:/);
const teMatch = content.match(/te:\s*{([\s\S]*?)},\s*hi:/);
const hiMatch = content.match(/hi:\s*{([\s\S]*?)}/);

const extractKeys = (str) => {
    let keys = [];
    const regex = /"([^"]+)":/g;
    let match;
    while ((match = regex.exec(str)) !== null) {
        keys.push(match[1]);
    }
    return keys;
};

const enKeys = extractKeys(enMatch[1]);
const teKeys = extractKeys(teMatch[1]);
const hiKeys = extractKeys(hiMatch[1]);

const missingTe = enKeys.filter(k => !teKeys.includes(k));
const missingHi = enKeys.filter(k => !hiKeys.includes(k));

console.log('Missing TE:', missingTe);
console.log('Missing HI:', missingHi);
