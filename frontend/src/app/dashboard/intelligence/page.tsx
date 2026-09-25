"use client";

import { useState, useMemo, useEffect } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  ScatterChart, Scatter, ZAxis, BarChart, Bar, Legend
} from "recharts";
import { AlertTriangle, Filter, Map, Activity, BarChart3, Database, Info, Calendar } from "lucide-react";
import { useTranslation } from "@/lib/i18n";
import { generateIntelligenceData, IntelligenceObservation } from "@/lib/mock-intelligence-data";

export default function IntelligenceDashboard() {
  const { t } = useTranslation();
  const [data, setData] = useState<IntelligenceObservation[]>([]);
  const [loading, setLoading] = useState(true);

  const [selectedCrop, setSelectedCrop] = useState<string>("All");
  const [selectedDisease, setSelectedDisease] = useState<string>("All");
  const isDemo = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';

  useEffect(() => {
    // Simulate fetching data
    setTimeout(() => {
      const generated = generateIntelligenceData(3000);
      setData(generated);
      setLoading(false);
    }, 1000);
  }, []);

  // Filtered Data
  const filteredData = useMemo(() => {
    return data.filter(d => 
      (selectedCrop === "All" || d.crop === selectedCrop) &&
      (selectedDisease === "All" || d.disease === selectedDisease)
    );
  }, [data, selectedCrop, selectedDisease]);

  // Derived Metrics
  const uniqueCrops = useMemo(() => Array.from(new Set(data.map(d => d.crop))), [data]);
  const uniqueDiseases = useMemo(() => Array.from(new Set(data.map(d => d.disease))), [data]);
  const uniqueRegions = useMemo(() => Array.from(new Set(filteredData.map(d => d.region))), [filteredData]);

  const dateRange = useMemo(() => {
    if (data.length === 0) return "";
    const sorted = [...data].sort((a,b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    const start = new Date(sorted[0].timestamp).toLocaleDateString();
    const end = new Date(sorted[sorted.length-1].timestamp).toLocaleDateString();
    return `${start} - ${end}`;
  }, [data]);

  // Temporal Data (Group by Month)
  const trendData = useMemo(() => {
    const monthlyMap: Record<string, number> = {};
    filteredData.forEach(d => {
      const month = new Date(d.timestamp).toISOString().slice(0, 7); // YYYY-MM
      monthlyMap[month] = (monthlyMap[month] || 0) + 1;
    });
    return Object.keys(monthlyMap).sort().map(month => ({
      month,
      observations: monthlyMap[month]
    }));
  }, [filteredData]);

  // Environmental Association (Humidity vs Temperature Density)
  const envData = useMemo(() => {
    const bins: Record<string, { temp: string, count: number }> = {};
    filteredData.forEach(d => {
      if (d.isHealthy) return; // Only map diseases
      const tBin = Math.floor(d.temperature / 5) * 5;
      const hBin = Math.floor(d.humidity / 10) * 10;
      const key = `${tBin}-${hBin}`;
      if (!bins[key]) {
        bins[key] = { temp: `${tBin}-${tBin+5}°C, ${hBin}-${hBin+10}% RH`, count: 0 };
      }
      bins[key].count += 1;
    });
    return Object.values(bins).sort((a,b) => b.count - a.count).slice(0, 5); // top 5 conditions
  }, [filteredData]);

  // Crop x Disease Matrix
  const matrixData = useMemo(() => {
    const map: Record<string, Record<string, number>> = {};
    filteredData.forEach(d => {
      if (!map[d.crop]) map[d.crop] = {};
      map[d.crop][d.disease] = (map[d.crop][d.disease] || 0) + 1;
    });
    return map;
  }, [filteredData]);

  // Hotspots Ranking
  const hotspots = useMemo(() => {
    const regionMap: Record<string, { count: number, totalConf: number }> = {};
    filteredData.forEach(d => {
      if (d.isHealthy) return; // Only count diseases for hotspots
      const key = `${d.region} - ${d.disease}`;
      if (!regionMap[key]) regionMap[key] = { count: 0, totalConf: 0 };
      regionMap[key].count += 1;
      regionMap[key].totalConf += d.confidence;
    });
    return Object.entries(regionMap)
      .map(([key, val]) => ({
        key,
        region: key.split(" - ")[0],
        disease: key.split(" - ")[1],
        count: val.count,
        avgConfidence: Math.round(val.totalConf / val.count)
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Top 5
  }, [filteredData]);

  // Automated R&D Insights (using explicit simulated language)
  const insights = useMemo(() => {
    const msgs = [];
    if (hotspots.length > 0) {
      msgs.push(`High simulated observation concentration: ${hotspots[0].disease} in ${hotspots[0].region} (${hotspots[0].count} cases).`);
    }
    if (envData.length > 0) {
      msgs.push(`Recorded environmental pattern: High observation frequency under ${envData[0].temp}.`);
    }
    if (uniqueRegions.length > 0 && uniqueRegions.length <= 3) {
      msgs.push(`Research gap: Current filtered observations are concentrated in only ${uniqueRegions.length} region(s).`);
    }
    return msgs;
  }, [hotspots, envData, uniqueRegions]);


  if (loading) {
    return <div className="p-12 text-center animate-pulse text-muted-foreground">Loading R&D Intelligence Data...</div>;
  }

  return (
    <div className="flex flex-col gap-6 w-full pb-20 lg:pb-8">
      
      {/* SIMULATED DATA PROMINENT HEADER */}
      {isDemo && (
        <div className="bg-amber-500/15 border-l-4 border-amber-500 p-4 rounded-r-xl shadow-sm">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-6 w-6 text-amber-600 dark:text-amber-500 shrink-0 mt-0.5" />
            <div>
              <h2 className="text-lg font-bold text-amber-800 dark:text-amber-400 tracking-tight">
                {t("intelligence.demo_analytics")}
              </h2>
              <p className="text-amber-700/80 dark:text-amber-200/80 text-sm mt-1 leading-relaxed max-w-4xl">
                {t("intelligence.data_disclaimer")}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end border-b pb-4 border-border/50 gap-4 mt-2">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            {t("intelligence.title")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t("intelligence.subtitle")}
          </p>
        </div>

        {/* Global Filters */}
        <div className="flex items-center gap-3 bg-card p-2 rounded-lg border shadow-sm">
          <Filter className="h-4 w-4 text-muted-foreground ml-2" />
          <select 
            value={selectedCrop} 
            onChange={e => setSelectedCrop(e.target.value)}
            className="text-sm bg-transparent border-none focus:ring-0 cursor-pointer text-foreground font-medium"
          >
            <option value="All">All Crops</option>
            {uniqueCrops.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <div className="h-4 w-px bg-border mx-1"></div>
          <select 
            value={selectedDisease} 
            onChange={e => setSelectedDisease(e.target.value)}
            className="text-sm bg-transparent border-none focus:ring-0 cursor-pointer text-foreground font-medium"
          >
            <option value="All">All Diseases</option>
            {uniqueDiseases.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
      </div>

      {/* KPI Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: t("intelligence.total_observations"), val: filteredData.length, icon: Activity },
          { label: t("intelligence.crops_monitored"), val: selectedCrop === "All" ? uniqueCrops.length : 1, icon: Filter },
          { label: t("intelligence.diseases_observed"), val: selectedDisease === "All" ? uniqueDiseases.length : 1, icon: AlertTriangle },
          { label: t("intelligence.regions_covered"), val: uniqueRegions.length, icon: Map },
        ].map((kpi, i) => (
          <div key={i} className="glass-panel p-4 flex flex-col justify-center border border-border/40">
            <div className="flex items-center gap-2 mb-2 text-muted-foreground">
              <kpi.icon className="h-4 w-4" />
              <span className="text-xs uppercase tracking-wider font-semibold truncate">{kpi.label}</span>
            </div>
            <span className="text-2xl font-bold">{kpi.val}</span>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Heatmap (Scatter) */}
        <div className="lg:col-span-2 glass-panel p-6 border border-border/40">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Map className="h-5 w-5 text-primary" /> {t("intelligence.geographic_heatmap")}
            </h2>
            {isDemo && <span className="text-xs px-2 py-1 bg-muted rounded-md text-muted-foreground">{t("intelligence.simulated_source")}</span>}
          </div>
          <p className="text-xs text-muted-foreground mb-4">{t("intelligence.simulated_distribution")} based on recorded coordinates.</p>
          <div className="h-[300px] w-full bg-slate-100 dark:bg-slate-900/50 rounded-xl border border-border/50 relative overflow-hidden">
             <ResponsiveContainer width="100%" height="100%">
               <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                 <XAxis type="number" dataKey="longitude" domain={['auto', 'auto']} name="Longitude" hide />
                 <YAxis type="number" dataKey="latitude" domain={['auto', 'auto']} name="Latitude" hide />
                 <ZAxis type="number" dataKey="concernScore" range={[20, 400]} />
                 <Tooltip 
                   cursor={{ strokeDasharray: '3 3' }} 
                   content={({ active, payload }) => {
                     if (active && payload && payload.length) {
                       const d = payload[0].payload;
                       return (
                         <div className="bg-background border shadow-lg p-3 rounded-lg text-sm">
                           <p className="font-bold text-foreground">{d.region}</p>
                           <p className="text-muted-foreground">{d.crop} - {d.disease}</p>
                           <div className="flex justify-between items-center mt-2 pt-2 border-t border-border/50 text-xs">
                             <span className="text-muted-foreground">Confidence:</span>
                             <span className="font-semibold text-foreground">{d.confidence}%</span>
                           </div>
                           {isDemo && <p className="text-[10px] text-amber-600 mt-1 italic">{t("intelligence.simulated_source")}</p>}
                         </div>
                       );
                     }
                     return null;
                   }}
                 />
                 <Scatter name="Observations" data={filteredData} fill="#ef4444" opacity={0.6} />
               </ScatterChart>
             </ResponsiveContainer>
          </div>
        </div>

        {/* Data Coverage (New Panel) */}
        <div className="lg:col-span-1 glass-panel p-6 border border-border/40 flex flex-col">
          <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
            <Database className="h-5 w-5 text-indigo-500" /> {t("intelligence.data_coverage")}
          </h2>
          <div className="flex-1 space-y-4">
             <div className="flex justify-between items-center p-3 bg-muted/40 rounded-lg">
               <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <Activity className="h-4 w-4" /> Total Records
               </div>
               <span className="font-semibold text-foreground">{data.length.toLocaleString()}</span>
             </div>
             <div className="flex justify-between items-center p-3 bg-muted/40 rounded-lg">
               <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <Calendar className="h-4 w-4" /> {t("intelligence.date_range")}
               </div>
               <span className="font-semibold text-foreground text-xs">{dateRange}</span>
             </div>
             <div className="flex justify-between items-center p-3 bg-muted/40 rounded-lg">
               <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <Filter className="h-4 w-4" /> Crops Matrix
               </div>
               <span className="font-semibold text-foreground">{uniqueCrops.length}</span>
             </div>
          </div>
          {isDemo && (
             <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 text-blue-700 dark:text-blue-400 text-xs rounded-lg flex items-start gap-2">
               <Info className="h-4 w-4 shrink-0 mt-0.5" />
               <p>{t("intelligence.demo_dataset_notice")}</p>
             </div>
          )}
        </div>

      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
        
        {/* Temporal Trends */}
        <div className="lg:col-span-1 xl:col-span-2 glass-panel p-6 border border-border/40">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Activity className="h-5 w-5 text-indigo-500" /> {t("intelligence.disease_trends")}
            </h2>
          </div>
          <p className="text-xs text-muted-foreground mb-4">Simulated monthly observations</p>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="month" tick={{fontSize: 10}} tickFormatter={v => v.split('-')[1]} />
                <YAxis tick={{fontSize: 10}} />
                <Tooltip 
                   content={({ active, payload, label }) => {
                     if (active && payload && payload.length) {
                       return (
                         <div className="bg-background border shadow-lg p-2 rounded text-sm">
                           <p className="font-bold">{label}</p>
                           <p>Observations: {payload[0].value}</p>
                           {isDemo && <p className="text-[10px] text-amber-600 mt-1 italic">{t("intelligence.simulated_source")}</p>}
                         </div>
                       );
                     }
                     return null;
                   }}
                />
                <Line type="monotone" dataKey="observations" stroke="#6366f1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Environmental Association */}
        <div className="lg:col-span-1 xl:col-span-2 glass-panel p-6 border border-border/40">
           <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
             <BarChart3 className="h-5 w-5 text-emerald-500" /> {t("intelligence.environmental_association")}
           </h2>
           <p className="text-xs text-muted-foreground mb-4">Based on simulated observation conditions.</p>
           <div className="h-[250px] w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={envData} layout="vertical" margin={{ top: 0, right: 0, bottom: 0, left: 10 }}>
                 <XAxis type="number" hide />
                 <YAxis type="category" dataKey="temp" tick={{fontSize: 10}} width={120} />
                 <Tooltip 
                   content={({ active, payload }) => {
                     if (active && payload && payload.length) {
                       const d = payload[0].payload;
                       return (
                         <div className="bg-background border shadow-lg p-2 rounded text-sm">
                           <p className="font-bold">{d.temp}</p>
                           <p>Count: {d.count}</p>
                           {isDemo && <p className="text-[10px] text-amber-600 mt-1 italic">{t("intelligence.simulated_source")}</p>}
                         </div>
                       );
                     }
                     return null;
                   }}
                 />
                 <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* R&D Intelligence (Text Signals) */}
        <div className="xl:col-span-1 glass-panel p-6 border-l-4 border-l-amber-500 border-t border-r border-b border-border/40 flex flex-col">
           <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
             <AlertTriangle className="h-5 w-5 text-amber-500" /> {t("intelligence.rd_insights")}
           </h2>
           <div className="space-y-3 flex-1">
             {insights.map((msg, idx) => (
               <div key={idx} className="p-3 bg-amber-500/10 text-amber-800 dark:text-amber-200 text-sm rounded-lg border border-amber-500/20 leading-relaxed">
                 {msg}
               </div>
             ))}
             {insights.length === 0 && <p className="text-sm text-muted-foreground">Adjust filters to see insights.</p>}
           </div>
           <div className="mt-4 pt-4 border-t border-border/50">
             <p className="text-xs text-muted-foreground italic flex items-start gap-1.5">
               <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
               {t("intelligence.potential_area")}
             </p>
           </div>
        </div>

        {/* Crop x Disease Matrix (Compact Table) */}
        <div className="xl:col-span-1 glass-panel p-6 border border-border/40 overflow-x-auto">
           <div className="flex justify-between items-center mb-4">
             <h2 className="text-lg font-semibold flex items-center gap-2">
               <Database className="h-5 w-5 text-blue-500" /> {t("intelligence.crop_disease_matrix")}
             </h2>
           </div>
           <div className="w-full max-h-[300px] overflow-y-auto pr-2">
             <table className="w-full text-sm text-left">
               <thead className="text-xs text-muted-foreground uppercase bg-muted/50 sticky top-0 z-10 backdrop-blur-md">
                 <tr>
                   <th className="px-3 py-2 rounded-l-lg">Crop / Disease</th>
                   <th className="px-3 py-2 text-right rounded-r-lg">Count</th>
                 </tr>
               </thead>
               <tbody>
                 {Object.entries(matrixData).map(([crop, diseases]) => (
                   <React.Fragment key={crop}>
                     <tr className="bg-muted/20 border-b border-border/50">
                       <td colSpan={2} className="px-3 py-2 font-bold text-foreground text-xs">{crop}</td>
                     </tr>
                     {Object.entries(diseases).map(([disease, count]) => (
                       <tr key={disease} className="border-b border-border/30 last:border-0 hover:bg-muted/30">
                         <td className="px-3 py-2 text-muted-foreground pl-6 flex items-center gap-2">
                           <div className="w-1.5 h-1.5 rounded-full bg-blue-500 opacity-50"></div>
                           {disease}
                         </td>
                         <td className="px-3 py-2 text-right text-foreground font-medium">{count}</td>
                       </tr>
                     ))}
                   </React.Fragment>
                 ))}
               </tbody>
             </table>
           </div>
        </div>

        {/* Hotspots Table */}
        <div className="xl:col-span-1 glass-panel p-6 border border-border/40 overflow-x-auto">
           <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
             <Map className="h-5 w-5 text-orange-500" /> {t("intelligence.regional_hotspots")}
           </h2>
           <div className="w-full max-h-[300px] overflow-y-auto pr-2">
             <table className="w-full text-sm text-left">
               <thead className="text-xs text-muted-foreground uppercase bg-muted/50 sticky top-0 z-10 backdrop-blur-md">
                 <tr>
                   <th className="px-3 py-2 rounded-l-lg">Region</th>
                   <th className="px-3 py-2">Disease</th>
                   <th className="px-3 py-2 text-right">Obs.</th>
                 </tr>
               </thead>
               <tbody>
                 {hotspots.map((hs, i) => (
                   <tr key={i} className="border-b border-border/50 last:border-0 hover:bg-muted/30">
                     <td className="px-3 py-3 font-medium text-foreground">{hs.region}</td>
                     <td className="px-3 py-3 text-muted-foreground text-xs">{hs.disease}</td>
                     <td className="px-3 py-3 text-right text-foreground font-medium">{hs.count}</td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
        </div>

      </div>

    </div>
  );
}
