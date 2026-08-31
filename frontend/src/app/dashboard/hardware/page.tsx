"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";
import { Activity, Droplets, ThermometerSun, Wifi, AlertTriangle, CloudSun } from "lucide-react";
import { ElementType } from "react";
import { API_BASE_URL } from "@/lib/api-client";
import { useTranslation } from "@/lib/i18n";

interface GaugeProps {
  value: number;
  max: number;
  label: string;
  color: string;
  icon: ElementType;
}

interface SensorData {
  is_simulated?: boolean;
  moisture: number;
  temperature?: number;
  temp?: number;
  humidity?: number;
}

interface HistoryPoint {
  time: string;
  moisture: number;
  humidity: number;
  temp: number;
}

interface RiskData {
  is_heuristic?: boolean;
  fungal_risk: string;
  fungal_risk_reason: string;
  irrigation_status: string;
  irrigation_reason: string;
  overall_health: string;
}

// Mock animated gauge component using Recharts
const Gauge = ({ value, max, label, color, icon: Icon }: GaugeProps) => {
  const data = [{ name: label, value: value, fill: color }];

  return (
    <div className="flex flex-col items-center justify-center p-6 glass-panel shape-blob-1 hover:shape-blob-2 transition-all duration-1000 relative">
      <div className="h-40 w-40">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="80%"
            outerRadius="100%"
            barSize={10}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis type="number" domain={[0, max]} angleAxisId={0} tick={false} />
            <RadialBar
              background
              dataKey="value"
              cornerRadius={5}
              fill={color}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/3 flex flex-col items-center text-center">
        <Icon className="h-5 w-5 mb-1" style={{ color }} />
        <span className="text-2xl font-bold">{value}</span>
        <span className="text-xs text-muted-foreground uppercase tracking-widest">{label}</span>
      </div>
    </div>
  );
};

export default function HardwareDashboard() {
  const { t } = useTranslation();
  const [sensors, setSensors] = useState<SensorData | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [risk, setRisk] = useState<RiskData | null>(null);
  const [status, setStatus] = useState(t("monitoring.status.connecting"));

  const fetchData = useCallback(async () => {
    try {
      // Provide dummy auth headers since verify_token is used in backend
      const headers = { Authorization: "Bearer dummy_token" };
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const [latestRes, historyRes, riskRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/iot/latest`, { headers, signal: controller.signal }),
        fetch(`${API_BASE_URL}/api/v1/iot/readings`, { headers, signal: controller.signal }),
        fetch(`${API_BASE_URL}/api/v1/predict-risk/`, { headers, signal: controller.signal })
      ]);

      clearTimeout(timeoutId);

      if (latestRes.ok) setSensors(await latestRes.json() as SensorData);
      if (historyRes.ok) {
        const histData = await historyRes.json() as Array<{ timestamp: string; soil_moisture: number; humidity: number; temperature: number }>;
        // format for chart
        setHistory(histData.reverse().map((d) => ({
          time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          moisture: d.soil_moisture,
          humidity: d.humidity,
          temp: d.temperature
        })));
      }
      if (riskRes.ok) setRisk(await riskRes.json() as RiskData);
      setStatus(t("monitoring.status.connected"));
    } catch (e) {
      console.error(e);
      setStatus(t("monitoring.status.offline"));
      
      // Provide dummy data to prevent infinite loading screen
      setSensors({
        is_simulated: true,
        moisture: 45.2,
        temperature: 28.5,
        humidity: 62.1,
      });
      setHistory(
        Array.from({ length: 24 }).map((_, i) => ({
          time: `${i}:00`,
          moisture: 40 + Math.random() * 10,
          humidity: 60 + Math.random() * 5,
          temp: 25 + Math.random() * 5,
        }))
      );
      setRisk({
        is_heuristic: true,
        fungal_risk: "Moderate",
        fungal_risk_reason: "Humidity levels are slightly elevated.",
        irrigation_status: "Adequate",
        irrigation_reason: "Soil moisture is above 40%.",
        overall_health: "Good",
      });
    }
  }, [t]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async data fetch: setState occurs after async resolution, not synchronously
    void fetchData();
    const interval = setInterval(() => { void fetchData(); }, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  if (!sensors) {
    return <div className="p-8 text-center animate-pulse">Connecting to Field Nodes...</div>;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col gap-8 max-w-6xl mx-auto w-full pb-20 lg:pb-8"
    >
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between border-b pb-4 border-border/50 gap-4">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-gradient pb-1">
            {t("monitoring.title")}
          </h1>
          <p className="text-sm text-muted-foreground mt-2 flex items-center gap-2">
            {t("monitoring.subtitle")}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-3 bg-secondary/30 backdrop-blur px-4 py-2 rounded-full border border-border/50">
            <Wifi className="h-4 w-4 text-emerald-500 animate-pulse" />
            <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">{status}</span>
            <span className="text-xs text-muted-foreground ml-2">Node: Alpha-1</span>
          </div>
          {sensors?.is_simulated && (
            <span className="text-[10px] uppercase tracking-widest bg-amber-500/20 text-amber-600 px-2 py-1 rounded-sm">Simulated Data</span>
          )}
        </div>
      </div>

      {/* Predictive Risk & Environmental Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-2">
        {/* Risk Panel */}
        <div className="lg:col-span-1 glass-panel p-6 flex flex-col gap-4 border-amber-500/20 shadow-lg">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2 text-amber-500">
              <AlertTriangle className="h-5 w-5" /> {t("monitoring.predictive_risk")}
            </h2>
            {risk?.is_heuristic && (
              <span className="text-[10px] uppercase tracking-widest bg-blue-500/10 text-blue-500 px-2 py-1 rounded-sm border border-blue-500/20">Heuristic</span>
            )}
          </div>
          {risk ? (
            <div className="space-y-4">
              <div>
                <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Fungal Risk</p>
                <p className={`text-lg font-bold ${risk.fungal_risk !== 'Low' ? 'text-amber-500' : 'text-emerald-500'}`}>
                  {risk.fungal_risk}
                </p>
                <p className="text-xs text-muted-foreground mt-1">{risk.fungal_risk_reason}</p>
              </div>
              <div className="border-t pt-3">
                <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Irrigation Need</p>
                <p className={`text-lg font-bold ${risk.irrigation_status !== 'Adequate' ? 'text-blue-500' : 'text-emerald-500'}`}>
                  {risk.irrigation_status}
                </p>
                <p className="text-xs text-muted-foreground mt-1">{risk.irrigation_reason}</p>
              </div>
              <div className="border-t pt-3">
                <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Overall Health</p>
                <p className="text-lg font-bold">{risk.overall_health}</p>
              </div>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">Loading predictive risk models...</div>
          )}
        </div>

        {/* Historical Graph */}
        <div className="lg:col-span-2 glass-panel p-6 shadow-lg">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" /> {t("monitoring.telemetry")}
          </h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <Line type="monotone" dataKey="moisture" stroke="#3b82f6" strokeWidth={3} dot={false} name="Soil Moisture (%)" />
                <Line type="monotone" dataKey="humidity" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Humidity (%)" />
                <Line type="monotone" dataKey="temp" stroke="#f59e0b" strokeWidth={2} dot={false} name="Temp (°C)" />
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" opacity={0.2} />
                <XAxis dataKey="time" tick={{fontSize: 12}} tickMargin={10} />
                <YAxis tick={{fontSize: 12}} />
                <Tooltip contentStyle={{ borderRadius: '8px', backgroundColor: 'rgba(0,0,0,0.8)', border: 'none', color: '#fff' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Environmental Metrics */}
      <div className="mt-4">
        <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
          <ThermometerSun className="h-5 w-5 text-orange-500" /> {t("monitoring.environmental_status")}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Soil Moisture */}
          <div className="glass-panel p-6 flex items-center justify-between shape-pill group hover:shape-blob-1 transition-all duration-700">
            <div className="flex items-center gap-4">
              <div className="bg-blue-500/20 p-4 rounded-full group-hover:scale-110 transition-transform">
                <Droplets className="h-8 w-8 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground font-medium uppercase tracking-wider">Soil Moisture</p>
                <p className="text-3xl font-bold font-mono">{Math.round(sensors.moisture)}%</p>
              </div>
            </div>
            <div className="text-right">
              <span className={`text-xs px-2 py-1 rounded-full ${sensors.moisture > 40 ? 'bg-emerald-500/20 text-emerald-600' : 'bg-red-500/20 text-red-600'}`}>
                {sensors.moisture > 40 ? 'Optimal' : 'Low'}
              </span>
            </div>
          </div>

          {/* Temperature & Humidity */}
          <div className="glass-panel p-6 flex flex-col justify-center shape-pill group hover:shape-blob-3 transition-all duration-700">
            <div className="flex justify-between items-center mb-4">
               <div className="flex items-center gap-3">
                 <div className="bg-orange-500/20 p-3 rounded-full"><ThermometerSun className="h-6 w-6 text-orange-500" /></div>
                 <div>
                   <p className="text-sm text-muted-foreground font-medium uppercase">Ambient Temp</p>
                   <p className="text-2xl font-bold font-mono">{sensors.temperature?.toFixed(1) || sensors.temp?.toFixed(1)}°C</p>
                 </div>
               </div>
            </div>
            <div className="flex justify-between items-center pt-4 border-t border-border/50">
               <div className="flex items-center gap-3">
                 <div className="bg-indigo-500/20 p-3 rounded-full"><CloudSun className="h-6 w-6 text-indigo-500" /></div>
                 <div>
                   <p className="text-sm text-muted-foreground font-medium uppercase">Humidity</p>
                   <p className="text-2xl font-bold font-mono">{sensors.humidity?.toFixed(1)}%</p>
                 </div>
               </div>
            </div>
          </div>

        </div>
      </div>

    </motion.div>
  );
}
