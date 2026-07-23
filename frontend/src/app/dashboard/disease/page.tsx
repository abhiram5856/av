"use client";

import { useState, useRef } from "react";
import { UploadCloud, FileImage, ShieldAlert, CheckCircle, ArrowRight, Volume2, Camera, Loader2, Sparkles, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { motion, AnimatePresence } from "framer-motion";
import { useTranslation } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";

export default function DiseaseDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const { t } = useTranslation();
  const language = useAppStore((state) => state.language);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleAnalyze = () => {
    if (!file) return;
    
    setAnalyzing(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setAnalyzing(false);
          setResult({
            disease: "Tomato Early Blight",
            confidence: 94.5,
            severity: "High",
            affectedArea: "32%",
            time: "1.2s",
            symptoms: "Dark, concentric rings on older leaves. Yellowing of surrounding tissue.",
            causes: "Fungus Alternaria solani. Thrives in warm, humid conditions.",
            treatment: "Remove affected leaves immediately to prevent spread. Apply a copper-based fungicide. Ensure plants have adequate spacing for air circulation.",
            preventative: "Rotate crops yearly. Use drip irrigation instead of overhead watering to keep foliage dry."
          });
          return 100;
        }
        return prev + 5;
      });
    }, 150);
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setProgress(0);
  };

  const speakResult = () => {
    if (!result || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    
    const textToSpeak = `Disease detected: ${result.disease}. Confidence: ${result.confidence} percent. Severity: ${result.severity}. Recommended treatment: ${result.treatment}`;
    
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    // Rough language matching
    if (language === "hi") utterance.lang = "hi-IN";
    else if (language === "te") utterance.lang = "te-IN";
    else utterance.lang = "en-US";
    
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex flex-col gap-8 max-w-6xl mx-auto">
      <div>
        <h1 className="text-4xl font-extrabold tracking-tight mb-2">{t("nav.disease")}</h1>
        <p className="text-muted-foreground text-lg">Upload an image of a plant leaf for instant AI analysis and explainable Grad-CAM results.</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <Card className="border-border shadow-sm">
          <CardHeader>
            <CardTitle>{t("disease.upload")}</CardTitle>
            <CardDescription>Drag & drop, browse, or capture from your camera.</CardDescription>
          </CardHeader>
          <CardContent>
            {!preview ? (
              <div 
                className="border-2 border-dashed border-primary/20 bg-primary/5 rounded-2xl flex flex-col items-center justify-center p-16 text-center cursor-pointer hover:bg-primary/10 transition-colors"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="bg-background shadow-sm rounded-full p-4 mb-4">
                  <UploadCloud className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-lg font-semibold">{t("disease.drag_drop")} or click</h3>
                <p className="text-sm text-muted-foreground mt-2">JPEG, PNG, WEBP (Max 10MB)</p>
                <div className="flex items-center gap-2 mt-6">
                  <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                    Browse Files
                  </Button>
                  <Button variant="outline" size="sm" onClick={(e) => e.stopPropagation()}>
                    <Camera className="h-4 w-4 mr-2" /> {t("disease.camera")}
                  </Button>
                </div>
                <input 
                  ref={fileInputRef}
                  id="file-upload" 
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden border bg-muted aspect-square sm:aspect-video flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Leaf preview" className="w-full h-full object-cover" />
                  
                  {/* Grad-CAM Heatmap Overlay (Mock) */}
                  <AnimatePresence>
                    {result && (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.65 }}
                        className="absolute inset-0 z-10 pointer-events-none"
                        style={{
                          background: "radial-gradient(circle at 45% 55%, rgba(255,0,0,0.8) 0%, rgba(255,165,0,0.6) 25%, rgba(0,255,0,0.2) 60%, transparent 100%)",
                          mixBlendMode: "overlay"
                        }}
                      />
                    )}
                  </AnimatePresence>
                  
                  {/* Scanning Overlay */}
                  {analyzing && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-background/50 backdrop-blur-sm">
                      <Loader2 className="h-10 w-10 text-primary animate-spin mb-4" />
                      <div className="bg-background px-4 py-2 rounded-full shadow-sm border font-medium">
                        Analyzing... {progress}%
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center font-medium bg-muted px-3 py-1.5 rounded-md">
                    <FileImage className="mr-2 h-4 w-4 text-primary" />
                    <span className="truncate max-w-[200px]">{file?.name}</span>
                  </div>
                  <Button variant="ghost" size="sm" onClick={resetForm} disabled={analyzing}>
                    Change
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button 
              className="w-full h-12 text-base shadow-lg" 
              disabled={!file || analyzing || !!result} 
              onClick={handleAnalyze}
            >
              {analyzing ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  {t("disease.analyzing")} {progress}%
                </>
              ) : result ? (
                <><CheckCircle className="mr-2 h-5 w-5" /> Analysis Complete</>
              ) : (
                <><Sparkles className="mr-2 h-5 w-5" /> Run AI Analysis</>
              )}
            </Button>
          </CardFooter>
          {analyzing && (
            <div className="px-6 pb-6">
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </Card>

        {/* Results Panel */}
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <Card className="border shadow-md h-full overflow-hidden relative">
                <div className="absolute top-0 right-0 p-4">
                  <Button variant="outline" size="icon" className="rounded-full" onClick={speakResult} title="Read Aloud">
                    <Volume2 className="h-5 w-5" />
                  </Button>
                </div>
                
                <CardHeader className="pb-4 border-b bg-muted/10">
                  <Badge variant={result.severity === "High" ? "destructive" : "default"} className="w-fit mb-3">
                    {result.severity} Risk
                  </Badge>
                  <CardTitle className="text-2xl">{result.disease}</CardTitle>
                  <div className="flex flex-wrap gap-4 mt-2">
                    <div className="flex flex-col">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">{t("results.confidence")}</span>
                      <span className="text-lg font-bold text-green-500">{result.confidence}%</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">Affected Area</span>
                      <span className="text-lg font-bold">{result.affectedArea}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">Inference Time</span>
                      <span className="text-lg font-bold">{result.time}</span>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-6 pt-6">
                  <div className="space-y-2 border p-4 rounded-xl bg-card">
                    <h4 className="font-semibold flex items-center text-sm">
                      <AlertTriangle className="mr-2 h-4 w-4 text-orange-500" /> Symptoms
                    </h4>
                    <p className="text-sm text-muted-foreground pl-6 leading-relaxed">
                      {result.symptoms}
                    </p>
                  </div>

                  <div className="space-y-2 border p-4 rounded-xl bg-card">
                    <h4 className="font-semibold flex items-center text-sm">
                      <ShieldAlert className="mr-2 h-4 w-4 text-destructive" /> Possible Causes
                    </h4>
                    <p className="text-sm text-muted-foreground pl-6 leading-relaxed">
                      {result.causes}
                    </p>
                  </div>

                  <div className="space-y-2 border border-primary/20 bg-primary/5 p-4 rounded-xl">
                    <h4 className="font-semibold flex items-center text-sm text-primary">
                      <CheckCircle className="mr-2 h-4 w-4" /> {t("results.treatment")}
                    </h4>
                    <p className="text-sm text-foreground pl-6 leading-relaxed">
                      {result.treatment}
                    </p>
                  </div>
                  
                  <Button variant="outline" className="w-full mt-4">Download PDF Report</Button>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="h-full"
            >
              <Card className="flex flex-col items-center justify-center text-center p-8 text-muted-foreground h-full min-h-[500px] border-dashed border-2 bg-muted/20">
                <div className="bg-background p-6 rounded-full shadow-sm mb-6 border">
                  <Sparkles className="h-10 w-10 text-primary/40" />
                </div>
                <h3 className="text-xl font-bold text-foreground mb-2">Waiting for Image</h3>
                <p className="text-sm max-w-sm mb-8">
                  Upload an image of a leaf to instantly receive a detailed AI analysis, Grad-CAM heatmap, and actionable treatment recommendations.
                </p>
                <div className="grid grid-cols-2 gap-4 w-full max-w-sm opacity-50">
                  <div className="bg-background border rounded-lg p-3 text-left">
                    <div className="h-2 w-12 bg-muted rounded mb-2"></div>
                    <div className="h-2 w-full bg-muted rounded"></div>
                  </div>
                  <div className="bg-background border rounded-lg p-3 text-left">
                    <div className="h-2 w-12 bg-muted rounded mb-2"></div>
                    <div className="h-2 w-full bg-muted rounded"></div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
