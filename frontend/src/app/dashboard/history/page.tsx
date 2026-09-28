"use client";

import { useEffect, useState } from "react";
import { Clock, ArrowRight, Activity, Thermometer, Droplets, FlaskConical, AlertTriangle, LineChart as LineChartIcon } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { fetchFromAPI } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useTranslation } from "@/lib/i18n";
import { LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts";

// Match the python Pydantic Schema we created in history.py
interface DiagnosisResponse {
  id: string;
  disease_name: string;
  confidence: number;
  concern_level: string | null;
  temperature: number | null;
  humidity: number | null;
  ph_level: number | null;
  created_at: string;
  // We can skip gradcam_heatmap and root_cause_json here for a simplified list view
}

export default function HistoryPage() {
  const { t } = useTranslation();
  const [diagnoses, setDiagnoses] = useState<DiagnosisResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadHistory() {
      try {
        const data = await fetchFromAPI("/api/v1/history/");
        setDiagnoses(data);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "An unknown error occurred";
        setError("Failed to load history: " + message);
        setDiagnoses([]);
      } finally {
        setLoading(false);
      }
    }
    loadHistory();
  }, []);

  const getConcernColor = (severity: string | null) => {
    switch (severity?.toLowerCase()) {
      case "critical": return "text-destructive bg-destructive/10";
      case "high": return "text-orange-600 bg-orange-100 dark:bg-orange-900/30";
      case "medium": return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30";
      default: return "text-green-600 bg-green-100 dark:bg-green-900/30";
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto w-full">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <div className="border-b pb-4 mb-2 flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {t("history.title")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t("history.subtitle")}
          </p>
        </div>
        <Link href="/dashboard/disease">
          <Button variant="default" size="sm">
            {t("dashboard.new_diagnosis")}
          </Button>
        </Link>
      </div>

      {/* ─── Loading / Error ────────────────────────────────────────────── */}
      {loading && (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      )}
      
      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-md border border-destructive/20 flex items-center">
          <AlertTriangle className="mr-2 h-5 w-5" />
          {error}
        </div>
      )}

      {/* ─── Empty State ──────────────────────────────────────────────────── */}
      {!loading && !error && diagnoses.length === 0 && (
        <div className="flex flex-col items-center justify-center p-12 text-center bg-card border rounded-lg shadow-sm">
          <Activity className="h-16 w-16 text-muted-foreground mb-4 opacity-30" />
          <h2 className="text-xl font-semibold mb-2">No diagnoses yet.</h2>
          <p className="text-muted-foreground mb-6 max-w-sm">
            Your saved plant assessments will appear here. Start by analyzing a new plant photo.
          </p>
          <Link href="/dashboard/disease">
            <Button variant="default">Diagnose a Plant</Button>
          </Link>
        </div>
      )}

      {/* ─── Data List ──────────────────────────────────────────────────── */}
      {!loading && diagnoses.length > 0 && (
        <div className="space-y-6">
          {/* Farm Health Timeline */}
          <Card className="border-border/40 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <LineChartIcon className="h-5 w-5 text-green-600" />
                Farm Health Timeline
              </CardTitle>
              <CardDescription>Concern levels of your recent scans</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={diagnoses.slice().reverse().map(d => ({
                    date: new Date(d.created_at).toLocaleDateString(),
                    concern: d.disease_name.toLowerCase().includes('healthy') ? 0 : d.confidence * 100,
                    name: d.disease_name
                  }))}>
                    <XAxis dataKey="date" tick={{fontSize: 12}} hide />
                    <YAxis domain={[0, 100]} hide />
                    <RechartsTooltip 
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload;
                          return (
                            <div className="bg-background border shadow-sm p-2 rounded text-sm">
                              <p className="font-bold">{d.date}</p>
                              <p className="capitalize">{d.name.replace(/_/g, " ")}</p>
                              <p className="text-xs text-muted-foreground mt-1">
                                {d.concern === 0 ? "Healthy" : `Concern: ${d.concern.toFixed(0)}%`}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line type="monotone" dataKey="concern" stroke="#16a34a" strokeWidth={3} dot={{r: 4, fill: "#16a34a"}} activeDot={{r: 6}} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
          {diagnoses.map((item) => (
            <Card key={item.id} className="overflow-hidden hover:border-primary/50 transition-colors">
              <CardHeader className="pb-3 bg-muted/30">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-lg capitalize">
                      {item.disease_name.replace(/_/g, " ")}
                    </CardTitle>
                    <CardDescription className="flex items-center mt-1">
                      <Clock className="mr-1 h-3 w-3" />
                      {new Date(item.created_at).toLocaleDateString()} at {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </CardDescription>
                  </div>
                  {item.concern_level && (
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getConcernColor(item.concern_level)}`}>
                      {item.concern_level}
                    </span>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pt-4 pb-4">
                <div className="grid grid-cols-2 gap-y-3 text-sm">
                  <div className="flex items-center text-muted-foreground">
                    <Activity className="mr-2 h-4 w-4" />
                    <span>AI Confidence: <strong className="text-foreground">{(item.confidence * 100).toFixed(1)}%</strong></span>
                  </div>
                  {item.temperature !== null && (
                    <div className="flex items-center text-muted-foreground">
                      <Thermometer className="mr-2 h-4 w-4" />
                      <span>Temp: <strong className="text-foreground">{item.temperature}°C</strong></span>
                    </div>
                  )}
                  {item.humidity !== null && (
                    <div className="flex items-center text-muted-foreground">
                      <Droplets className="mr-2 h-4 w-4" />
                      <span>Humidity: <strong className="text-foreground">{item.humidity}%</strong></span>
                    </div>
                  )}
                  {item.ph_level !== null && (
                    <div className="flex items-center text-muted-foreground">
                      <FlaskConical className="mr-2 h-4 w-4" />
                      <span>pH Level: <strong className="text-foreground">{item.ph_level}</strong></span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        </div>
      )}

      {/* ─── Empty State ────────────────────────────────────────────────── */}
      {!loading && !error && diagnoses.length === 0 && (
        <div className="flex flex-col items-center justify-center text-center py-24 border rounded-lg bg-card shadow-sm">
          <div className="bg-muted p-4 rounded-full mb-4">
            <Clock className="h-6 w-6 text-muted-foreground" />
          </div>
          <h2 className="text-lg font-medium mb-2">{t("history.no_history")}</h2>
          <p className="text-sm text-muted-foreground max-w-sm mb-6">
            {t("dashboard.no_scans_desc")}
          </p>
          <Link href="/dashboard/disease">
            <Button variant="outline">
              {t("dashboard.new_diagnosis")}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}
