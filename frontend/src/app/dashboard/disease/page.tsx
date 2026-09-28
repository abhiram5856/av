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
import imageCompression from 'browser-image-compression';
import { runOfflineInference, loadModel, isModelLoaded } from "@/lib/offline-inference";

// ─── Types ──────────────────────────────────────────────────────────────────

interface DiagnosisResult {
  // VISION
  disease: string;
  confidence: number;
  isHealthy: boolean;
  isLowConfidence: boolean;
  lowConfidenceWarning: string | null;
  lookalikes: string[];
  
  // VISUAL EVIDENCE
  raw_b64: string;
  affectedArea: string;
  localizationConfidence: string;
  qualityWarning: string | null;
  
  // ENVIRONMENT
  weather_temp: string;
  weather_humidity: string;
  weather_soil_ph: string;
  environmentalCompat: string;
  environmentalExplanation: string;
  
  // KNOWLEDGE
  diseaseType: string;
  knowledgeCompleteness: string;
  distinctivePattern: string;
  prevention: string;
  chemicalControl: string;
  
  // DECISION
  actionPriority: string;
  concernLevel: ConcernLevel;
  whatToDoNow: string;
  monitor: string;
  expertReview: string;
  rescanInstructions: string | null;
  
  // Legacy
  context_hash: string;
  weather_summary: string;
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
  const [latitude, setLatitude] = useState<string | null>(null);
  const [longitude, setLongitude] = useState<string | null>(null);
  const [locationSource, setLocationSource] = useState<"live" | "user" | "unavailable">("unavailable");
  const [temperature, setTemperature] = useState<string>("N/A");
  const [humidity, setHumidity] = useState<string>("N/A");
  const [weatherSource, setWeatherSource] = useState<"live" | "unavailable">("unavailable");
  const [weatherTimestamp, setWeatherTimestamp] = useState<string | null>(null);
  const [growthStage, setGrowthStage] = useState<string>("Unknown");
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [announcement, setAnnouncement] = useState("");
  
  // Offline ML
  const [offlineModelReady, setOfflineModelReady] = useState(false);
  const [offlineKnowledge, setOfflineKnowledge] = useState<any>(null);

  const { t } = useTranslation();
  
  const speak = (msg: string) => setAnnouncement(msg);
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

  // Load offline model & knowledge in background
  useEffect(() => {
    async function initOffline() {
      try {
        await loadModel();
        setOfflineModelReady(isModelLoaded());
        const res = await fetch("/data/offline_knowledge.json");
        if (res.ok) {
          const kb = await res.json();
          setOfflineKnowledge(kb);
        }
      } catch (err) {
        console.error("Failed to initialize offline model", err);
      }
    }
    initOffline();
  }, []);

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
            setLocationSource(isLiveLocation ? "live" : "user");
            setTemperature(parsed.temperature);
            setHumidity(parsed.humidity);
            setWeatherSource("live");
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
        setLocationSource(isLiveLocation ? "live" : "user");
        const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`);
        const data = await res.json();
        if (data.current) {
          const temp = data.current.temperature_2m.toString();
          const hum = data.current.relative_humidity_2m.toString();
          setTemperature(temp);
          setHumidity(hum);
          setWeatherSource("live");
          const now = Date.now();
          setWeatherTimestamp(new Date(now).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
          sessionStorage.setItem(cacheKey, JSON.stringify({ temperature: temp, humidity: hum, timestamp: now }));
        }
      } catch (err) {
        console.error("Failed to fetch live weather", err);
        setWeatherSource("unavailable");
      } finally {
        setWeatherLoading(false);
      }
    };

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => fetchWeather(position.coords.latitude, position.coords.longitude, true),
        (error) => {
          console.warn("Geolocation denied or failed, location unavailable.", error);
          setWeatherLoading(false);
          setLocationSource("unavailable");
          setWeatherSource("unavailable");
          speak("Location access denied. Weather data unavailable.");
        }
      );
    } else {
      setWeatherLoading(false);
      setLocationSource("unavailable");
      setWeatherSource("unavailable");
      speak("Location services unavailable.");
    }
  }, []);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileSelect = async (selectedFile: File) => {
    setError(null);
    if (!selectedFile.type.startsWith("image/")) {
      setError("That file is not a supported image. Please upload a JPEG, PNG, or WEBP.");
      speak("Upload failed. Unsupported image type.");
      return;
    }
    
    // Check network before even compressing, unless offline model is ready
    if (!navigator.onLine && !offlineModelReady) {
      setError("You are offline. Connect once to download the offline model.");
      speak("You are offline. Connect once to download the offline model.");
      return;
    }
    
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Photo is too large (max 10MB). Please choose another image or take a new photo.");
      speak("Upload failed. Photo is too large.");
      return;
    }
    
    setAnalyzing(true); // Re-use analyzing state for "Preparing photo..."
    speak("Preparing photo for upload...");
    try {
      // 1. Client-side compression for low bandwidth
      const options = {
        maxSizeMB: 1,
        maxWidthOrHeight: 1200,
        useWebWorker: true,
        fileType: "image/jpeg"
      };
      const compressedFile = await imageCompression(selectedFile, options);
      setFile(compressedFile);
      setResult(null);
      setShowHeatmap(false);
      
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(compressedFile);
    } catch (err) {
      console.error("Compression failed:", err);
      setError("Failed to prepare photo for upload.");
      speak("Failed to prepare photo for upload.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    if (!navigator.onLine && !offlineModelReady) {
      setError("You are offline. Connect once to download the offline model.");
      speak("You are offline. Connect once to download the offline model.");
      return;
    }

    setAnalyzing(true);
    setError(null);
    speak("Analyzing image. Please wait.");

    try {
      // --- OFFLINE INFERENCE PATH ---
      if (!navigator.onLine && offlineModelReady) {
        const offlineResult = await runOfflineInference(file);
        
        const kb = offlineKnowledge ? offlineKnowledge[offlineResult.disease] : null;
        const diseaseType = kb?.disease_type || "Unknown";
        
        let cLevel: ConcernLevel = "Medium";
        if (offlineResult.isHealthy) cLevel = "Low";
        else if (offlineResult.confidence < 0.4) cLevel = "Low";
        else cLevel = "High";

        const actionPri = cLevel === "High" ? "ATTENTION" : (cLevel === "Medium" ? "MONITOR" : "LOW");

        setResult({
          disease: offlineResult.disease.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
          confidence: offlineResult.confidence,
          isHealthy: offlineResult.isHealthy,
          isLowConfidence: offlineResult.isLowConfidence,
          lowConfidenceWarning: offlineResult.isLowConfidence ? "Insufficient visual evidence for a reliable diagnosis." : null,
          lookalikes: [],
          
          raw_b64: "", // No Grad-CAM offline
          affectedArea: "Unknown",
          localizationConfidence: "Unknown",
          qualityWarning: null,
          
          weather_temp: "Unavailable",
          weather_humidity: "Unavailable",
          weather_soil_ph: "Not measured",
          environmentalCompat: "Unknown",
          environmentalExplanation: "",
          
          diseaseType: diseaseType,
          knowledgeCompleteness: kb ? "Offline Knowledge" : "Unavailable",
          distinctivePattern: kb?.distinctive_pattern || "Unavailable",
          prevention: kb?.prevention || "Unavailable",
          chemicalControl: kb?.chemical_control || "Unavailable",
          
          actionPriority: actionPri,
          concernLevel: cLevel,
          whatToDoNow: kb?.treatment || "Offline diagnosis limits specific guidance.",
          monitor: "Monitor plant condition closely.",
          expertReview: kb?.expert_review || "Consult local agronomist if symptoms persist.",
          rescanInstructions: offlineResult.isLowConfidence ? "Please retake the photo." : null,
          
          context_hash: "offline-" + Date.now(),
          weather_summary: "Weather unavailable"
        });
        setAnalyzing(false);
        return;
      }

      // --- ONLINE INFERENCE PATH ---

      formData.append("image", file);
      formData.append("temperature", temperature);
      formData.append("humidity", humidity);
      if (latitude && longitude) {
        formData.append("latitude", latitude);
        formData.append("longitude", longitude);
      }
      formData.append("growth_stage", growthStage);
      // NOTE: user_id is resolved server-side from the Bearer token, not from the form.

      // The diagnosis endpoint uses raw fetch (not fetchFromAPI) because it sends FormData.
      // We must manually attach the Supabase JWT.
      const { createClient } = await import('@/utils/supabase/client');
      const supabase = createClient();
      const { data: { session } } = await supabase.auth.getSession();
      const authHeaders: HeadersInit = session?.access_token
        ? { 'Authorization': `Bearer ${session.access_token}`, 'Bypass-Tunnel-Reminder': 'true' }
        : { 'Bypass-Tunnel-Reminder': 'true' };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(`${API_BASE_URL}/api/v1/diagnose/`, {
        method: "POST",
        body: formData,
        headers: authHeaders,
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Diagnosis failed (${response.status})`);
      }

      const res = await response.json();
      
      const v = res.VISION || {};
      const e = res.ENVIRONMENT || {};
      const k = res.KNOWLEDGE || {};
      const d = res.DECISION || {};
      const vis = v.visual_evidence || {};

      const actionPri = d.action_priority || "LOW";
      let cLevel: ConcernLevel = "Low";
      if (actionPri === "ATTENTION") cLevel = "High";
      else if (actionPri === "MONITOR") cLevel = "Medium";

      setResult({
        disease: v.display_name || v.prediction || "Unknown",
        confidence: v.confidence || 0,
        isHealthy: !!v.is_healthy,
        isLowConfidence: v.prediction === "Unknown",
        lowConfidenceWarning: v.low_confidence_warning || null,
        lookalikes: v.differential_conditions || [],
        
        raw_b64: vis.heatmap_b64 || "",
        affectedArea: `${Math.round((vis.attention_indicator || 0) * 100)}%`,
        localizationConfidence: vis.localization_confidence || "Unknown",
        qualityWarning: vis.quality_warning || null,
        
        weather_temp: e.temperature != null ? `${e.temperature}°C` : "Unavailable",
        weather_humidity: e.humidity != null ? `${e.humidity}%` : "Unavailable",
        weather_soil_ph: e.soil_ph != null ? e.soil_ph.toString() : "Not measured",
        environmentalCompat: e.compatibility || "Unknown",
        environmentalExplanation: e.explanation || "",
        
        diseaseType: k.disease_specific_knowledge?.disease_type || "Unknown",
        knowledgeCompleteness: k.knowledge_completeness || "Unknown",
        distinctivePattern: k.disease_specific_knowledge?.distinctive_pattern || "",
        prevention: k.prevention || "",
        chemicalControl: k.chemical_control || "",
        
        actionPriority: actionPri,
        concernLevel: cLevel,
        whatToDoNow: d.what_to_do_now || "",
        monitor: d.monitor || "",
        expertReview: d.expert_review || "",
        rescanInstructions: d.rescan_instructions || null,
        
        context_hash: res.request_id || "",
        weather_summary: e.temperature != null
          ? `${e.temperature}°C avg, ${e.humidity ?? "—"}% humidity`
          : "Weather unavailable"
      });

    } catch (err: any) {
      console.error(err);
      if (err.name === 'AbortError') {
        setError("Analysis timed out. Please check your connection and try again.");
      } else {
        setError("Unable to complete diagnosis. The service may be temporarily unavailable or your connection dropped. Please try again.");
      }
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
          severity_score: result.actionPriority === "ATTENTION" ? 80 : 20,
          urgency: result.actionPriority,
          weather_summary: result.weather_summary,
          rag_recommendations: result.whatToDoNow,
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
      <div aria-live="polite" className="sr-only" aria-atomic="true">
        {announcement}
      </div>
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <div className="border-b pb-4">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {t("nav.disease")}
          </h1>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mt-1 gap-2">
          <p className="text-sm text-muted-foreground">
            {t("disease.upload_hint")}
          </p>
          <div className="flex gap-2 flex-wrap">
            <Badge variant="outline" className="text-xs bg-muted/30">
              {locationSource === "live" ? (
                <span className="flex items-center text-emerald-600 dark:text-emerald-400">Live Location</span>
              ) : locationSource === "user" ? (
                <span className="flex items-center text-blue-600 dark:text-blue-400">User Location</span>
              ) : (
                <span className="flex items-center text-orange-600 dark:text-orange-400">Location Unavailable</span>
              )}
            </Badge>
            <Badge variant="outline" className="text-xs bg-muted/30">
              {weatherLoading ? (
                <span className="flex items-center"><Loader2 className="h-3 w-3 mr-1 animate-spin" /> Locating...</span>
              ) : weatherSource === "unavailable" ? (
                <span className="flex items-center text-orange-600 dark:text-orange-400">Weather Unavailable</span>
              ) : (
                <span className="flex items-center" title={`${t("weather.updated")} ${weatherTimestamp || 'unknown'}`}>
                  <Thermometer className="h-3 w-3 mr-1" /> {temperature}°C | {humidity}% RH
                  {weatherSource === "live" ? (
                    <span className="ml-2 text-emerald-600 dark:text-emerald-400">(Live)</span>
                  ) : (
                    <span className="ml-2 text-orange-600 dark:text-orange-400">(Unavailable)</span>
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

          {result.qualityWarning && (
            <div className="mb-6 p-4 rounded-xl bg-orange-500/10 border border-orange-500/20 flex items-start space-x-3 text-orange-600 dark:text-orange-400">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Quality Warning: Background Focus</p>
                <p className="text-sm">{result.qualityWarning}</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Left Column: VISION & DECISION */}
            <div className="lg:col-span-1 space-y-6">
              <div className="glass-panel p-6 text-center">
                <div 
                  className={`inline-flex items-center justify-center p-4 rounded-full mb-4 ${CONCERN_MAP[result.concernLevel].bgColor}`}
                >
                  <AlertTriangle className={`h-8 w-8 ${CONCERN_MAP[result.concernLevel].textColor}`} />
                </div>
                <Badge 
                  variant="outline" 
                  className={`mb-4 ${CONCERN_MAP[result.concernLevel].textColor} ${CONCERN_MAP[result.concernLevel].borderColor}`}
                >
                  {result.actionPriority}
                </Badge>
                <h3 className="text-xl font-semibold">{result.disease}</h3>
                <p className="text-sm font-medium mt-2 mb-4">
                  {result.isLowConfidence 
                    ? "UNCERTAIN" 
                    : result.confidence >= 85 
                      ? "HIGHER CONFIDENCE" 
                      : "MODERATE CONFIDENCE"}
                </p>
                
                <div className="flex flex-col gap-2">
                  <Button variant="outline" className="w-full justify-center" onClick={handleDownloadPDF}>
                    <FileDown className="mr-2 h-4 w-4" /> {t("disease.download_report")}
                  </Button>
                </div>
              </div>

              <div className="glass-panel p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold flex items-center"><Camera className="mr-2 h-4 w-4" /> Visual Evidence</h3>
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
                </div>
                <div className="mt-4 flex justify-between text-sm">
                  <span className="text-muted-foreground">Affected Area:</span>
                  <span className="font-medium">{result.affectedArea}</span>
                </div>
                <div className="mt-2 flex justify-between text-sm">
                  <span className="text-muted-foreground">Localization:</span>
                  <span className="font-medium">{result.localizationConfidence}</span>
                </div>
              </div>
            </div>

            {/* Right Column: ENVIRONMENT & KNOWLEDGE */}
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
                  <span className="font-medium mt-1">N/A</span>
                </div>
                <div className="glass-panel p-4 flex flex-col items-center justify-center text-center">
                  <Wind className="h-5 w-5 mb-2 text-emerald-500" />
                  <span className="text-sm text-muted-foreground">{t("disease.soil_ph")}</span>
                  <span className="font-medium mt-1">{result.weather_soil_ph}</span>
                </div>
              </div>

              <div className="glass-panel p-6">
                <h3 className="font-semibold flex items-center mb-4 text-lg">Environmental Context</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground block mb-1">Compatibility:</span>
                    <Badge variant="secondary">{result.environmentalCompat}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground block mb-1">Explanation:</span>
                    <span className="text-muted-foreground">{result.environmentalExplanation}</span>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-6 border-l-4 border-l-primary relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                  <FileText className="h-32 w-32" />
                </div>
                <h3 className="text-lg font-bold mb-4">Agronomic Decision Support</h3>
                
                {result.isHealthy ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <p>{result.whatToDoNow}</p>
                  </div>
                ) : (
                  <div className="prose prose-sm dark:prose-invert max-w-none space-y-4">
                    {result.rescanInstructions && (
                      <div className="p-3 bg-blue-500/10 text-blue-700 dark:text-blue-300 rounded border border-blue-500/20">
                        <strong>Rescan Needed:</strong> {result.rescanInstructions}
                      </div>
                    )}
                    <div>
                      <strong className="block text-emerald-600 dark:text-emerald-400">What to do now:</strong>
                      <p>{result.whatToDoNow || "Consult local extension services."}</p>
                    </div>
                    <div>
                      <strong className="block text-orange-600 dark:text-orange-400">Monitoring:</strong>
                      <p>{result.monitor}</p>
                    </div>
                    <div>
                      <strong className="block text-blue-600 dark:text-blue-400">Prevention:</strong>
                      <p>{result.prevention}</p>
                    </div>
                    {result.chemicalControl && (
                      <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded border border-slate-200 dark:border-slate-800">
                        <strong className="block text-purple-600 dark:text-purple-400 mb-2">Chemical Control / Pesticide Safety:</strong>
                        <p className="mb-3">{result.chemicalControl}</p>
                        <p className="text-xs text-muted-foreground italic border-t pt-2 border-border/50">
                          <strong>Note:</strong> Chemical treatment depends on the crop, disease, locality, product label, and local agricultural recommendations. Follow the product label and local agricultural authority guidance. Consult an agriculture officer or qualified agronomist before application.
                        </p>
                      </div>
                    )}
                    {result.expertReview && (
                      <div className="text-red-500 font-semibold mt-4">
                        {result.expertReview}
                      </div>
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

