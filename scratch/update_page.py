import re
import os

filepath = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\frontend\src\app\dashboard\disease\page.tsx"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
content = content.replace(
    'import { SEVERITY_MAP, getSeverityFromUrgency, type SeverityLevel } from "@/lib/design-tokens";',
    'import { CONCERN_MAP, getConcernLevel, type ConcernLevel } from "@/lib/design-tokens";'
)

# 2. Update DiagnosisResult interface
old_interface = """interface DiagnosisResult {
  disease: string;
  confidence: number;
  severity: string;
  severityLevel: SeverityLevel;
  affectedArea: string;
  symptoms: string;
  causes: string;
  treatment: string;
  preventative: string;
  raw_b64: string;
  context_hash: string;
  weather_summary: string;
  weather_temp: string;
  weather_humidity: string;
  root_causes: string[];
  evidence_reasoning: string;
  has_mismatch: boolean;
}"""

new_interface = """interface DiagnosisResult {
  disease: string;
  confidence: number;
  concernScore: number;
  concernLevelStr: string;
  concernLevel: ConcernLevel;
  environmentalConflict: boolean;
  contributingFactors: string[];
  limitingFactors: string[];
  visualEvidenceLevel: string;
  environmentalCompat: string;
  growthStageSelected: string;
  affectedArea: string;
  treatment: string;
  preventative: string;
  raw_b64: string;
  context_hash: string;
  weather_summary: string;
  weather_temp: string;
  weather_humidity: string;
  weather_soil_ph: string;
  root_causes: string[];
  evidence_reasoning: string;
  isHealthy: boolean;
  isLowConfidence: boolean;
}"""

content = content.replace(old_interface, new_interface)

# 3. Add growthStage state
content = content.replace(
    'const [humidity, setHumidity] = useState<string>("60.0");',
    'const [humidity, setHumidity] = useState<string>("60.0");\n  const [growthStage, setGrowthStage] = useState<string>("Unknown");'
)

# 4. Update handleAnalyze formData
content = content.replace(
    'formData.append("longitude", longitude);',
    'formData.append("longitude", longitude);\n      formData.append("growth_stage", growthStage);'
)

# 5. Update setResult block
old_set_result = """      const urgency = res.severity_assessment?.urgency || "Moderate";
      const severityLevel = getSeverityFromUrgency(urgency);

      setResult({
        disease: res.prediction.disease
          .replace(/_/g, " ")
          .replace(/\\b\\w/g, (c: string) => c.toUpperCase()),
        confidence: Math.round(res.prediction.confidence * 10) / 10,
        severity: urgency,
        severityLevel,
        affectedArea: `${Math.round(res.ai_context.gradcam.coverage * 100)}%`,
        symptoms: "Concentric spots, leaf lesions with chlorotic halos, leaf wilting.",
        causes: res.root_cause_analysis?.ranked_causes?.[0]?.cause_label || "Environmental stress combined with pathogen infection.",
        treatment: res.ai_context.knowledge.context_text_block || "Apply appropriate fungicide and manage moisture levels.",
        preventative: "Ensure proper sanitation, rotate crops yearly, and use drip irrigation.",
        raw_b64: res.gradcam_heatmap_b64,
        context_hash: res.context_hash,
        weather_summary: `${res.ai_context.weather.temp_avg}°C avg, ${res.ai_context.weather.humidity_avg}% humidity`,
        weather_temp: `${res.ai_context.weather.temp_avg}°C`,
        weather_humidity: `${res.ai_context.weather.humidity_avg}%`,
        root_causes: [res.root_cause_analysis?.ranked_causes?.[0]?.cause_label || "Environmental stress"],
        evidence_reasoning: res.root_cause_analysis?.ranked_causes?.[0]?.reasoning_sentence || "Visual and environmental evidence are consistent.",
        has_mismatch: res.root_cause_analysis?.conflicts_detected?.length > 0 || Math.random() > 0.8, // Fallback random mismatch for demo if empty
      });"""

new_set_result = """      const concernData = res.concern || {};
      const envData = res.environment || {};
      const diagData = res.diagnosis || res.prediction || {};
      const visData = res.visual_evidence || {};
      const ragData = res.recommendation?.rag_response || res.root_cause_analysis || {};

      const concernLvlStr = concernData.level || "Unknown";
      const concernLvl = getConcernLevel(concernLvlStr);

      setResult({
        disease: diagData.display_name || diagData.disease.replace(/_/g, " ").replace(/\\b\\w/g, (c: string) => c.toUpperCase()),
        confidence: Math.round((diagData.confidence || 0) * 10) / 10,
        concernScore: concernData.score || 0,
        concernLevelStr: concernLvlStr,
        concernLevel: concernLvl,
        environmentalConflict: !!concernData.environmental_conflict,
        contributingFactors: concernData.contributing_factors || [],
        limitingFactors: concernData.limiting_factors || [],
        visualEvidenceLevel: visData.evidence_level || "Unknown",
        environmentalCompat: envData.compatibility || "Unknown",
        growthStageSelected: res.growth_stage?.selected_stage || "Unknown",
        affectedArea: `${Math.round((visData.attention_indicator || 0) * 100)}%`,
        treatment: res.recommendation?.retrieved_sources?.length ? (res.ai_context?.knowledge?.context_text_block || "") : "Apply appropriate fungicide and manage moisture levels.",
        preventative: "Ensure proper sanitation, rotate crops yearly, and use drip irrigation.",
        raw_b64: res.gradcam_heatmap_b64 || visData.heatmap_b64,
        context_hash: res.context_hash,
        weather_summary: `${envData.temperature || 25}°C avg, ${envData.humidity || 60}% humidity`,
        weather_temp: `${envData.temperature || 25}°C`,
        weather_humidity: `${envData.humidity || 60}%`,
        weather_soil_ph: envData.soil_ph ? envData.soil_ph.toString() : "Not available",
        root_causes: [ragData.ranked_causes?.[0]?.cause_label || "Environmental stress"],
        evidence_reasoning: ragData.ranked_causes?.[0]?.reasoning_sentence || "Visual and environmental evidence are consistent.",
        isHealthy: !!diagData.is_healthy,
        isLowConfidence: !!res.is_low_confidence
      });"""

content = content.replace(old_set_result, new_set_result)

# 6. Change pdf download logic (just small cleanups)
content = content.replace(
    'severity: result.severity,',
    'severity: result.concernLevelStr,'
)
content = content.replace(
    'severity_score: parseFloat(result.affectedArea),',
    'severity_score: result.concernScore,'
)
content = content.replace(
    'urgency: result.severity,',
    'urgency: result.concernLevelStr,'
)

# 7. Update text sharing
content = content.replace(
    'severity:* ${result.severity} Risk',
    'Concern Score:* ${result.concernScore}/100 (${result.concernLevelStr})'
)
content = content.replace(
    'Severity: ${result.severity}.',
    'Concern Score: ${result.concernScore}/100.'
)

# 8. UI OVERHAUL: Replace the entire result section
# We'll use regex to find the start of the result section and replace it.
# The result section starts with `{result && (`
old_ui_start = "{result && ("
old_ui_end = "      </div>\n    </div>\n  );\n}"

# Because the file is large, I'll extract everything before {result && ( and then append the new UI.
parts = content.split("{result && (")
if len(parts) == 2:
    prefix = parts[0]
    # We also need to add the Growth Stage dropdown in the form area.
    # Look for the file upload button area
    upload_area = """              </Button>
            </div>
            
            <p className="text-xs text-muted-foreground mt-4 text-center">"""
    
    new_upload_area = """              </Button>
            </div>
            
            <div className="mt-6">
              <label className="block text-sm font-medium mb-2">Growth Stage</label>
              <select 
                className="w-full p-2 rounded-md bg-background border border-border"
                value={growthStage}
                onChange={(e) => setGrowthStage(e.target.value)}
              >
                <option value="Unknown">Unknown</option>
                <option value="Seedling">Seedling</option>
                <option value="Vegetative">Vegetative</option>
                <option value="Flowering">Flowering</option>
                <option value="Fruiting">Fruiting</option>
                <option value="Harvest">Harvest</option>
              </select>
            </div>
            
            <p className="text-xs text-muted-foreground mt-4 text-center">"""
    
    prefix = prefix.replace(upload_area, new_upload_area)

    new_ui = """{result && (
        <div ref={resultRef} className="mt-8 animate-fade-in">
          
          {result.isLowConfidence && (
            <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start space-x-3 text-red-600 dark:text-red-400">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Low Confidence Warning</p>
                <p className="text-sm">Model confidence is below 60%. Please consult an agronomist or agricultural expert for verification before taking action.</p>
              </div>
            </div>
          )}

          {result.environmentalConflict && (
            <div className="mb-6 p-4 rounded-xl bg-orange-500/10 border border-orange-500/20 flex items-start space-x-3 text-orange-600 dark:text-orange-400">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Visual and environmental evidence are not fully consistent.</p>
                <p className="text-sm">The detected disease is not typically supported by current environmental conditions. Please verify with an expert.</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Left Column: Concern Score & Visual Evidence */}
            <div className="lg:col-span-1 space-y-6">
              <div className="glass-panel p-6 text-center">
                <div 
                  className={`inline-flex items-center justify-center p-4 rounded-full mb-4 ${CONCERN_MAP[result.concernLevel].bgColor}`}
                >
                  <AlertTriangle className={`h-8 w-8 ${CONCERN_MAP[result.concernLevel].textColor}`} />
                </div>
                <h2 className="text-4xl font-bold mb-2">{result.concernScore} <span className="text-xl text-muted-foreground">/ 100</span></h2>
                <Badge 
                  variant="outline" 
                  className={`mb-4 ${CONCERN_MAP[result.concernLevel].textColor} ${CONCERN_MAP[result.concernLevel].borderColor}`}
                >
                  {result.concernLevelStr}
                </Badge>
                <h3 className="text-xl font-semibold">{result.disease}</h3>
                <p className="text-sm text-muted-foreground mt-2 mb-4">
                  Concern Score combines visual and available environmental evidence. It is a decision-support indicator, not a biological severity percentage.
                </p>
                
                <div className="flex flex-col gap-2">
                  <Button variant="outline" className="w-full justify-center" onClick={handleDownloadPDF}>
                    <FileDown className="mr-2 h-4 w-4" /> Download Report
                  </Button>
                </div>
              </div>

              <div className="glass-panel p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold flex items-center"><Camera className="mr-2 h-4 w-4" /> Visual Attention Evidence</h3>
                  <Button variant="ghost" size="sm" onClick={() => setShowHeatmap(!showHeatmap)}>
                    {showHeatmap ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
                <div className="relative rounded-xl overflow-hidden aspect-square border border-border/50">
                  {showHeatmap && result.raw_b64 ? (
                    <img src={result.raw_b64} alt="Grad-CAM" className="w-full h-full object-cover" />
                  ) : (
                    <img src={preview!} alt="Original" className="w-full h-full object-cover" />
                  )}
                  <div className="absolute bottom-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-xs text-white">
                    Model Confidence: {result.confidence}%
                  </div>
                </div>
                <div className="mt-4 flex justify-between text-sm">
                  <span className="text-muted-foreground">Visual Evidence Level:</span>
                  <span className="font-medium">{result.visualEvidenceLevel}</span>
                </div>
              </div>
            </div>

            {/* Right Column: Environmental, Factors, RAG */}
            <div className="lg:col-span-2 space-y-6">
              
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Thermometer className="h-5 w-5 mb-2 text-orange-500" />
                  <span className="text-sm text-muted-foreground">Temp (Simulated)</span>
                  <span className="font-medium mt-1">{result.weather_temp}</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Droplets className="h-5 w-5 mb-2 text-blue-500" />
                  <span className="text-sm text-muted-foreground">Humidity (Simulated)</span>
                  <span className="font-medium mt-1">{result.weather_humidity}</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <CloudRain className="h-5 w-5 mb-2 text-cyan-500" />
                  <span className="text-sm text-muted-foreground">Soil Moisture</span>
                  <span className="font-medium mt-1">Not available</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Wind className="h-5 w-5 mb-2 text-emerald-500" />
                  <span className="text-sm text-muted-foreground">Soil pH</span>
                  <span className="font-medium mt-1">{result.weather_soil_ph}</span>
                </div>
              </div>

              <div className="glass-panel p-6">
                <h3 className="font-semibold flex items-center mb-4 text-lg">Context Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground block mb-1">Environmental Compatibility:</span>
                    <Badge variant="secondary">{result.environmentalCompat}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground block mb-1">Growth Stage:</span>
                    <Badge variant="secondary">{result.growthStageSelected}</Badge>
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  {result.contributingFactors.length > 0 && (
                    <div>
                      <h4 className="font-medium text-emerald-600 dark:text-emerald-400 mb-2 flex items-center"><AlertTriangle className="h-4 w-4 mr-1" /> Contributing Factors</h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                        {result.contributingFactors.map((f, i) => <li key={i}>{f}</li>)}
                      </ul>
                    </div>
                  )}
                  
                  {result.limitingFactors.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border">
                      <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-2 flex items-center"><X className="h-4 w-4 mr-1" /> Limiting Factors</h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                        {result.limitingFactors.map((f, i) => <li key={i}>{f}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              </div>

              <div className="glass-panel p-6 border-l-4 border-l-primary relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                  <FileText className="h-32 w-32" />
                </div>
                <h3 className="text-lg font-bold mb-4">AI Agronomist Recommendation</h3>
                
                {result.isHealthy ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p>The crop appears healthy. Maintain standard agricultural practices and monitor local weather conditions.</p>
                  </div>
                ) : (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p className="whitespace-pre-line">{result.treatment}</p>
                    {result.isLowConfidence && (
                      <p className="mt-4 text-red-500 font-semibold">Please verify with a specialist.</p>
                    )}
                  </div>
                )}
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
"""
    final_content = prefix + new_ui
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print("Updated successfully")
else:
    print("Could not find the split point.")
