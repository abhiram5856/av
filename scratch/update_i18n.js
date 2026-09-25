const fs = require('fs');
const path = 'c:\\\\Users\\\\ABHIRAM MODUKURU\\\\OneDrive\\\\Desktop\\\\AgriVision-AI\\\\frontend\\\\src\\\\lib\\\\i18n.tsx';
let content = fs.readFileSync(path, 'utf8');

// The existing file has these strings which we need to replace or we can just replace the whole English block up to "nav.dashboard"
const enReplacements = {
    '"intelligence.demo_analytics": "DEMO ANALYTICS"': '"intelligence.demo_analytics": "DEMO ANALYTICS — SIMULATED DATA"',
    '"intelligence.disease_trends": "Disease Trends"': '"intelligence.disease_trends": "Recorded Observation Trend"',
    '"intelligence.geographic_heatmap": "Geographic Disease Heatmap"': '"intelligence.geographic_heatmap": "Observation Density by Region"',
    '"intelligence.simulated_distribution": "Simulated disease distribution"': '"intelligence.simulated_distribution": "Illustrative simulated geographic distribution"',
    '"intelligence.crop_disease_matrix": "Crop × Disease Matrix"': '"intelligence.crop_disease_matrix": "Crops × Disease Observations"',
    '"intelligence.environmental_association": "Environmental Association"': '"intelligence.environmental_association": "Recorded Environmental Patterns"',
    '"intelligence.regional_hotspots": "Disease Hotspots"': '"intelligence.regional_hotspots": "Observation Hotspots"',
    '"intelligence.rd_insights": "R&D Intelligence"': '"intelligence.rd_insights": "R&D Exploration Signals"'
};

const teReplacements = {
    '"intelligence.demo_analytics": "డెమో విశ్లేషణలు"': '"intelligence.demo_analytics": "డెమో విశ్లేషణలు — అనుకరించిన డేటా"',
    '"intelligence.disease_trends": "వ్యాధి పోకడలు"': '"intelligence.disease_trends": "నమోదైన పరిశీలన ట్రెండ్"',
    '"intelligence.geographic_heatmap": "భౌగోళిక వ్యాధి హీట్‌మ్యాప్"': '"intelligence.geographic_heatmap": "ప్రాంతం వారీగా పరిశీలన సాంద్రత"',
    '"intelligence.simulated_distribution": "అనుకరించిన వ్యాధి పంపిణీ"': '"intelligence.simulated_distribution": "వివరణాత్మక అనుకరించిన భౌగోళిక పంపిణీ"',
    '"intelligence.crop_disease_matrix": "పంట × వ్యాధి మాతృక"': '"intelligence.crop_disease_matrix": "పంటలు × వ్యాధి పరిశీలనలు"',
    '"intelligence.environmental_association": "పర్యావరణ అనుబంధం"': '"intelligence.environmental_association": "నమోదైన పర్యావరణ నమూనాలు"',
    '"intelligence.regional_hotspots": "వ్యాధి హాట్‌స్పాట్‌లు"': '"intelligence.regional_hotspots": "పరిశీలన హాట్‌స్పాట్‌లు"',
    '"intelligence.rd_insights": "R&D ఇంటెలిజెన్స్"': '"intelligence.rd_insights": "R&D అన్వేషణ సంకేతాలు"'
};

const hiReplacements = {
    '"intelligence.demo_analytics": "डेमो एनालिटिक्स"': '"intelligence.demo_analytics": "डेमो एनालिटिक्स — सिम्युलेटेड डेटा"',
    '"intelligence.disease_trends": "रोग के रुझान"': '"intelligence.disease_trends": "दर्ज किया गया अवलोकन रुझान"',
    '"intelligence.geographic_heatmap": "भौगोलिक रोग हीटमैप"': '"intelligence.geographic_heatmap": "क्षेत्र के अनुसार अवलोकन घनत्व"',
    '"intelligence.simulated_distribution": "सिम्युलेटेड रोग वितरण"': '"intelligence.simulated_distribution": "चित्रात्मक सिम्युलेटेड भौगोलिक वितरण"',
    '"intelligence.crop_disease_matrix": "फसल × रोग मैट्रिक्स"': '"intelligence.crop_disease_matrix": "फसलें × रोग अवलोकन"',
    '"intelligence.environmental_association": "पर्यावरण संघ"': '"intelligence.environmental_association": "दर्ज किए गए पर्यावरणीय पैटर्न"',
    '"intelligence.regional_hotspots": "रोग हॉटस्पॉट"': '"intelligence.regional_hotspots": "अवलोकन हॉटस्पॉट"',
    '"intelligence.rd_insights": "R&D इंटेलिजेंस"': '"intelligence.rd_insights": "R&D अन्वेषण संकेत"'
};

for (const [oldStr, newStr] of Object.entries(enReplacements)) {
    content = content.replace(oldStr, newStr);
}
for (const [oldStr, newStr] of Object.entries(teReplacements)) {
    content = content.replace(oldStr, newStr);
}
for (const [oldStr, newStr] of Object.entries(hiReplacements)) {
    content = content.replace(oldStr, newStr);
}

const addEN = `
    "intelligence.simulated_source": "Source: Simulated observations",
    "intelligence.data_coverage": "Data Coverage",
    "intelligence.date_range": "Date Range",
    "intelligence.demo_dataset_notice": "Demonstration dataset — not population surveillance.",
    "intelligence.potential_area": "Potential area for further field investigation.",
    "nav.dashboard": "Dashboard",`;

const addTE = `
    "intelligence.simulated_source": "మూలం: అనుకరించిన పరిశీలనలు",
    "intelligence.data_coverage": "డేటా కవరేజ్",
    "intelligence.date_range": "తేదీ పరిధి",
    "intelligence.demo_dataset_notice": "ప్రదర్శన డేటాసెట్ — జనాభా పర్యవేక్షణ కాదు.",
    "intelligence.potential_area": "మరింత క్షేత్రస్థాయి విచారణకు సంభావ్య ప్రాంతం.",
    "nav.dashboard": "à°¡à± à°¯à°¾à°·à± â€Œà°¬à±‹à°°à± à°¡à± ",`; // NOTE: We can just use the literal replacement

const addHI = `
    "intelligence.simulated_source": "स्रोत: सिम्युलेटेड अवलोकन",
    "intelligence.data_coverage": "डेटा कवरेज",
    "intelligence.date_range": "तिथि सीमा",
    "intelligence.demo_dataset_notice": "प्रदर्शन डेटासेट — जनसंख्या निगरानी नहीं।",
    "intelligence.potential_area": "आगे की क्षेत्रीय जांच के लिए संभावित क्षेत्र।",
    "nav.dashboard": "डैशबोर्ड",`;

content = content.replace('"nav.dashboard": "Dashboard",', addEN);
content = content.replace('"nav.dashboard": "à°¡à± à°¯à°¾à°·à± â€Œà°¬à±‹à°°à± à°¡à± ",', addTE);
content = content.replace('"nav.dashboard": "डैशबोर्ड",', addHI);

fs.writeFileSync(path, content, 'utf8');
console.log('Successfully updated translations');
