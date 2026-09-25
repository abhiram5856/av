"use client";

import { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  FileImage,
  AlertTriangle,
  Camera,
  Loader2,
  Volume2,
  FileDown,
  RotateCcw,
  Thermometer,
  Droplets,
  Wind,
  CloudRain,
  Eye,
  EyeOff,
  X,
  FileText,
  Share2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useTranslation } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import { API_BASE_URL } from "@/lib/api-client";
import { CONCERN_MAP, getConcernLevel, type ConcernLevel } from "@/lib/design-tokens";

// â”€â”€â”€ Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface DiagnosisResult {
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
}

// â”€â”€â”€ Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export default function DiseaseDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Live Weather States
  const [latitude, setLatitude] = useState<string>("17.3850"); // Default Hyderabad
  const [longitude, setLongitude] = useState<string>("78.4867");
  const [locationSource, setLocationSource] = useState<"live" | "demo">("demo");
  const [temperature, setTemperature] = useState<string>("25.0");
  const [humidity, setHumidity] = useState<string>("60.0");
  const [weatherSource, setWeatherSource] = useState<"live" | "demo">("demo");
  const [weatherTimestamp, setWeatherTimestamp] = useState<string | null>(null);
  const [growthStage, setGrowthStage] = useState<string>("Unknown");
  const [weatherLoading, setWeatherLoading] = useState(true);
  
  const { t } = useTranslation();
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  const language = useAppStore((state) => state.language);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to results
  useEffect(() => {
    if (result && resultRef.current) {
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 300);
    }
  }, [result]);

  // Fetch Live Weather on Mount
  useEffect(() => {
    const fetchWeather = async (lat: number, lon: number, isLiveLocation: boolean) => {
      // Very basic caching using sessionStorage
      const cacheKey = `weather_${lat.toFixed(2)}_${lon.toFixed(2)}`;
      const cached = sessionStorage.getItem(cacheKey);
      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          // 30 min cache
          if (Date.now() - parsed.timestamp < 30 * 60 * 1000) {
            setLatitude(lat.toString());
            setLongitude(lon.toString());
            setLocationSource(isLiveLocation ? "live" : "demo");
            setTemperature(parsed.temperature);
            setHumidity(parsed.humidity);
            setWeatherSource(isLiveLocation ? "live" : "demo");
            setWeatherTimestamp(new Date(parsed.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
            setWeatherLoading(false);
            return;
          }
        } catch (e) {
          // ignore cache errors
        }
      }

      try {
        setLatitude(lat.toString());
        setLongitude(lon.toString());
        setLocationSource(isLiveLocation ? "live" : "demo");
        const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`);
        const data = await res.json();
        if (data.current) {
          const temp = data.current.temperature_2m.toString();
          const hum = data.current.relative_humidity_2m.toString();
          setTemperature(temp);
          setHumidity(hum);
          setWeatherSource(isLiveLocation ? "live" : "demo");
          const now = Date.now();
          setWeatherTimestamp(new Date(now).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
          sessionStorage.setItem(cacheKey, JSON.stringify({ temperature: temp, humidity: hum, timestamp: now }));
        }
      } catch (err) {
        console.error("Failed to fetch live weather", err);
      } finally {
        setWeatherLoading(false);
      }
    };

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => fetchWeather(position.coords.latitude, position.coords.longitude, true),
        (error) => {
          console.warn("Geolocation denied or failed, using defaults.", error);
          fetchWeather(17.385, 78.4867, false);
        }
      );
    } else {
      fetchWeather(17.385, 78.4867, false);
    }
  }, []);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setShowHeatmap(false);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(selectedFile);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("temperature", temperature);
      formData.append("humidity", humidity);
      formData.append("ph_level", "6.5"); // pH remains hardcoded or optional
      formData.append("latitude", latitude);
      formData.append("longitude", longitude);
      formData.append("growth_stage", growthStage);
      formData.append("user_id", "usr_farmer_ui");

      const response = await fetch(`${API_BASE_URL}/api/v1/diagnose/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Diagnosis failed (${response.status})`);
      }

      const res = await response.json();
      
      const concernData = res.concern || {};
      const envData = res.environment || {};
      const diagData = res.diagnosis || res.prediction || {};
      const visData = res.visual_evidence || {};
      const ragData = res.recommendation?.rag_response || res.root_cause_analysis || {};

      const concernLvlStr = concernData.level || "Unknown";
      const concernLvl = getConcernLevel(concernLvlStr);

      setResult({
        disease: diagData.display_name || diagData.disease.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()),
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
        weather_summary: `${envData.temperature || 25}Â°C avg, ${envData.humidity || 60}% humidity`,
        weather_temp: `${envData.temperature || 25}Â°C`,
        weather_humidity: `${envData.humidity || 60}%`,
        weather_soil_ph: envData.soil_ph ? envData.soil_ph.toString() : "Not available",
        root_causes: [ragData.ranked_causes?.[0]?.cause_label || "Environmental stress"],
        evidence_reasoning: ragData.ranked_causes?.[0]?.reasoning_sentence || "Visual and environmental evidence are consistent.",
        isHealthy: !!diagData.is_healthy,
        isLowConfidence: !!res.is_low_confidence
      });

    } catch (err) {
      console.error(err);
      setError("Unable to connect to the backend. Please ensure the AgriVision ML server is running.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!result) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: preview,
          heatmap_b64: result.raw_b64,
          disease: result.disease,
          confidence: result.confidence,
          severity_score: result.concernScore,
          urgency: result.concernLevelStr,
          weather_summary: result.weather_summary,
          rag_recommendations: result.treatment,
          context_hash: result.context_hash,
        }),
      });

      if (!response.ok) throw new Error("Failed to generate report PDF");

      const res = await response.json();
      window.open(`${API_BASE_URL}${res.download_url}`, "_blank");
    } catch (err) {
      console.error(err);
      setError("Could not generate PDF report.");
    }
  };

  const handleWhatsAppShare = () => {
    if (!result) return;
    
    const text = `*AgriVision AI Diagnosis Report*\n\n*Disease:* ${result.disease}\n*Concern Score:* ${result.concernScore}/100 (${result.concernLevelStr})\n*Confidence:* ${result.confidence}%\n\n*Treatment Recommended:*\n${result.treatment}\n\n*Preventative Measures:*\n${result.preventative}`;
    const encodedText = encodeURIComponent(text);
    window.open(`https://wa.me/?text=${encodedText}`, "_blank");
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setShowHeatmap(false);
  };

  const speakResult = () => {
    if (!result || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();

    const text = `Disease detected: ${result.disease}. Confidence: ${result.confidence} percent. Concern Score: ${result.concernScore}/100. Recommended treatment: ${result.treatment}`;
    const utterance = new SpeechSynthesisUtterance(text);
    if (language === "hi") utterance.lang = "hi-IN";
    else if (language === "te") utterance.lang = "te-IN";
    else utterance.lang = "en-US";

    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto w-full">
      {/* â”€â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <div className="border-b pb-4">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {t("nav.disease")}
          </h1>
          {isDemoMode && (
            <Badge variant="destructive" className="ml-2 animate-pulse">
              {t("demo_mode")}
            </Badge>
          )}
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mt-1 gap-2">
          <p className="text-sm text-muted-foreground">
            {t("disease.upload_hint")}
          </p>
          <div className="flex gap-2 flex-wrap">
            <Badge variant="outline" className="text-xs bg-muted/30">
              {locationSource === "live" ? (
                <span className="flex items-center text-emerald-600 dark:text-emerald-400">{t("location.live")}</span>
              ) : (
                <span className="flex items-center text-orange-600 dark:text-orange-400">{t("location.demo")}</span>
              )}
            </Badge>
            <Badge variant="outline" className="text-xs bg-muted/30">
              {weatherLoading ? (
                <span className="flex items-center"><Loader2 className="h-3 w-3 mr-1 animate-spin" /> {t("disease.locating")}</span>
              ) : (
                <span className="flex items-center" title={`${t("weather.updated")} ${weatherTimestamp || 'unknown'}`}>
                  <Thermometer className="h-3 w-3 mr-1" /> {temperature}Â°C | {humidity}% RH
                  {weatherSource === "live" ? (
                    <span className="ml-2 text-emerald-600 dark:text-emerald-400">({t("weather.live")})</span>
                  ) : (
                    <span className="ml-2 text-orange-600 dark:text-orange-400">({t("weather.demo")})</span>
                  )}
                </span>
              )}
            </Badge>
          </div>
        </div>
      </div>

      {/* â”€â”€â”€ Upload Section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {!result && (
        <div className="bg-card rounded-lg border shadow-sm overflow-hidden">
          {!preview ? (
            /* Drop Zone */
            <div
              className="border-2 border-dashed border-border rounded-lg flex flex-col items-center justify-center p-12 sm:p-20 text-center cursor-pointer hover:bg-muted/50 transition-colors m-4"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload image"
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
              }}
            >
              <div className="bg-muted p-4 rounded-full mb-4">
                <UploadCloud className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-medium mb-1">
                {t("disease.upload")}
              </h3>
              <p className="text-sm text-muted-foreground mb-6">
                {t("disease.drag_drop")} (JPEG, PNG, WEBP up to 10MB)
              </p>
              <div className="flex items-center gap-3">
                <Button
                  variant="default"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  <Camera className="mr-2 h-4 w-4" />
                  {t("disease.camera")}
                </Button>
                <Button
                  variant="secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  <FileImage className="mr-2 h-4 w-4" />
                  {t("disease.upload")}
                </Button>
              </div>
              <input
                ref={fileInputRef}
                id="file-upload"
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
              />
            </div>
          ) : (
            /* Image Preview */
            <div className="p-4 space-y-4">
              <div className="relative rounded-lg overflow-hidden border bg-black/5 aspect-video flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-contain"
                />

                {/* Analyzing Overlay */}
                {analyzing && (
                  <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm">
                    <Loader2 className="h-8 w-8 text-primary animate-spin mb-4" />
                    <span className="font-medium">{t("disease.analyzing_image")}</span>
                  </div>
                )}
              </div>

              {/* File Info + Actions */}
              <div className="flex items-center justify-between">
                <div className="text-sm text-muted-foreground flex items-center gap-2">
                  <FileImage className="h-4 w-4" />
                  <span className="truncate max-w-[250px]">{file?.name}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={resetForm}
                  disabled={analyzing}
                >
                  <X className="mr-2 h-4 w-4" /> {t("disease.change_image")}
                </Button>
              </div>

              {/* Analyze Button */}
              <Button
                className="w-full"
                disabled={!file || analyzing || !!result}
                onClick={handleAnalyze}
              >
                {analyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t("disease.analyzing")}
                  </>
                ) : (
                  t("disease.analyze")
                )}
              </Button>

              {/* Error Display */}
              {error && (
                <div className="bg-destructive/10 border border-destructive/20 text-destructive rounded-md p-3 text-sm flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  {error}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* â”€â”€â”€ Results Section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {result && (
        <div ref={resultRef} className="mt-8 animate-fade-in">
          
          {result.isLowConfidence && (
            <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start space-x-3 text-red-600 dark:text-red-400">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">{t("results.low_confidence_title")}</p>
                <p className="text-sm">{t("results.low_confidence_desc")}</p>
              </div>
            </div>
          )}

          {result.environmentalConflict && (
            <div className="mb-6 p-4 rounded-xl bg-orange-500/10 border border-orange-500/20 flex items-start space-x-3 text-orange-600 dark:text-orange-400">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">{t("results.env_conflict_title")}</p>
                <p className="text-sm">{t("results.env_conflict_desc")}</p>
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
                  {t("results.concern_score_desc")}
                </p>
                
                <div className="flex flex-col gap-2">
                  <Button variant="outline" className="w-full justify-center" onClick={handleDownloadPDF}>
                    <FileDown className="mr-2 h-4 w-4" /> {t("disease.download_report")}
                  </Button>
                </div>
              </div>

              <div className="glass-panel p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold flex items-center"><Camera className="mr-2 h-4 w-4" /> {t("disease.visual_attention")}</h3>
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
                    {t("disease.model_confidence")}: {result.confidence}%
                  </div>
                </div>
                <div className="mt-4 flex justify-between text-sm">
                  <span className="text-muted-foreground">{t("disease.visual_evidence_level")}:</span>
                  <span className="font-medium">{result.visualEvidenceLevel}</span>
                </div>
              </div>
            </div>

            {/* Right Column: Environmental, Factors, RAG */}
            <div className="lg:col-span-2 space-y-6">
              
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Thermometer className="h-5 w-5 mb-2 text-orange-500" />
                  <span className="text-sm text-muted-foreground">{t("disease.temp_simulated")}</span>
                  <span className="font-medium mt-1">{result.weather_temp}</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Droplets className="h-5 w-5 mb-2 text-blue-500" />
                  <span className="text-sm text-muted-foreground">{t("disease.humidity_simulated")}</span>
                  <span className="font-medium mt-1">{result.weather_humidity}</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <CloudRain className="h-5 w-5 mb-2 text-cyan-500" />
                  <span className="text-sm text-muted-foreground">{t("disease.soil_moisture")}</span>
                  <span className="font-medium mt-1">{t("disease.not_available")}</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Wind className="h-5 w-5 mb-2 text-emerald-500" />
                  <span className="text-sm text-muted-foreground">{t("disease.soil_ph")}</span>
                  <span className="font-medium mt-1">{result.weather_soil_ph}</span>
                </div>
              </div>

              <div className="glass-panel p-6">
                <h3 className="font-semibold flex items-center mb-4 text-lg">{t("disease.context_summary")}</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground block mb-1">{t("disease.env_compatibility")}:</span>
                    <Badge variant="secondary">{result.environmentalCompat}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground block mb-1">{t("disease.growth_stage")}:</span>
                    <Badge variant="secondary">{result.growthStageSelected}</Badge>
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  {result.contributingFactors.length > 0 && (
                    <div>
                      <h4 className="font-medium text-emerald-600 dark:text-emerald-400 mb-2 flex items-center"><AlertTriangle className="h-4 w-4 mr-1" /> {t("disease.contributing_factors")}</h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                        {result.contributingFactors.map((f, i) => <li key={i}>{f}</li>)}
                      </ul>
                    </div>
                  )}
                  
                  {result.limitingFactors.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border">
                      <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-2 flex items-center"><X className="h-4 w-4 mr-1" /> {t("disease.limiting_factors")}</h4>
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
                <h3 className="text-lg font-bold mb-4">{t("disease.ai_recommendation")}</h3>
                
                {result.isHealthy ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p>{t("results.healthy_desc")}</p>
                  </div>
                ) : (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p className="whitespace-pre-line">{result.treatment}</p>
                    {result.isLowConfidence && (
                      <p className="mt-4 text-red-500 font-semibold">{t("results.verify_specialist")}</p>
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

