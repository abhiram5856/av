"use client";

import { useEffect, useState } from "react";
import { fetchFromAPI } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useTranslation } from "@/lib/i18n";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import { AlertTriangle } from "lucide-react";

interface DiagnosisResponse {
  id: string;
  disease_name: string;
  confidence: number;
  concern_level: string | null;
  concern_score: number | null;
  created_at: string;
}

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6'];

export default function AnalyticsPage() {
  const { t } = useTranslation();
  const [diagnoses, setDiagnoses] = useState<DiagnosisResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        // Fetch all history
        const data = await fetchFromAPI("/api/v1/history/");
        setDiagnoses(data);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "An unknown error occurred";
        setError(message + " - Showing offline demo data.");
        setDiagnoses([
          { id: "1", disease_name: "early_blight", confidence: 0.92, concern_level: "High Concern", concern_score: 85, created_at: new Date(Date.now() - 86400000 * 1).toISOString() },
          { id: "2", disease_name: "healthy", confidence: 0.98, concern_level: "Low Concern", concern_score: 10, created_at: new Date(Date.now() - 86400000 * 2).toISOString() },
          { id: "3", disease_name: "late_blight", confidence: 0.88, concern_level: "Critical Attention Required", concern_score: 95, created_at: new Date(Date.now() - 86400000 * 3).toISOString() },
          { id: "4", disease_name: "early_blight", confidence: 0.85, concern_level: "Moderate Concern", concern_score: 55, created_at: new Date(Date.now() - 86400000 * 4).toISOString() },
          { id: "5", disease_name: "powdery_mildew", confidence: 0.90, concern_level: "High Concern", concern_score: 75, created_at: new Date(Date.now() - 86400000 * 5).toISOString() },
        ]);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Process Data for Pie Chart (Disease Distribution)
  const diseaseCounts = diagnoses.reduce((acc, curr) => {
    const name = curr.disease_name.replace(/_/g, " ");
    acc[name] = (acc[name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(diseaseCounts).map(([name, value]) => ({
    name,
    value
  })).sort((a, b) => b.value - a.value); // Sort by highest count

  // Process Data for Line Chart (Concern Score over time)
  // We'll group by date (YYYY-MM-DD)
  const timelineDataMap = diagnoses.reduce((acc, curr) => {
    const dateStr = new Date(curr.created_at).toLocaleDateString();
    if (!acc[dateStr]) {
      acc[dateStr] = { date: dateStr, totalConcern: 0, count: 0 };
    }
    
    let sevNum = 0;
    const sev = curr.concern_level?.toLowerCase();
    if (sev === "critical") sevNum = 90;
    else if (sev === "high") sevNum = 75;
    else if (sev === "medium") sevNum = 50;
    else if (sev === "low") sevNum = 25;
    else if (sev === "healthy") sevNum = 10;
    else sevNum = curr.concern_score ?? 0;

    acc[dateStr].totalConcern += sevNum;
    acc[dateStr].count += 1;
    return acc;
  }, {} as Record<string, { date: string, totalConcern: number, count: number }>);

  // Convert to array and sort chronologically
  const timelineData = Object.values(timelineDataMap)
    .map(item => ({
      date: item.date,
      avgConcern: Math.round(item.totalConcern / item.count)
    }))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto w-full pb-20 lg:pb-8">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <div className="border-b pb-4 mb-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          {t("analytics.title")}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t("analytics.subtitle")}
        </p>
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

      {/* ─── Charts ──────────────────────────────────────────────────────── */}
      {!loading && diagnoses.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          
          {/* Pie Chart */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>{t("analytics.distribution")}</CardTitle>
              <CardDescription>Breakdown of diagnosed plant conditions</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px] flex justify-center items-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent = 0 }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Line Chart */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>{t("analytics.timeline")}</CardTitle>
              <CardDescription>Average concern score progression</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={timelineData}
                  margin={{ top: 5, right: 30, left: -20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tick={{fontSize: 12}} />
                  <YAxis domain={[0, 100]} tick={{fontSize: 12}} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="avgConcern" 
                    stroke="#f59e0b" 
                    strokeWidth={3}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

        </div>
      )}

      {/* ─── Empty State ────────────────────────────────────────────────── */}
      {!loading && !error && diagnoses.length === 0 && (
        <div className="flex flex-col items-center justify-center text-center py-24 border rounded-lg bg-card shadow-sm">
          <h2 className="text-lg font-medium mb-2">{t("history.no_history")}</h2>
          <p className="text-sm text-muted-foreground max-w-sm mb-6">
            {t("dashboard.no_scans_desc")}
          </p>
        </div>
      )}
    </div>
  );
}
