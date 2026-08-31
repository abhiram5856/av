"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { 
  Users, Activity, Database, AlertTriangle, ShieldCheck, 
  MapPin, Settings, Server, ChevronRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const MOCK_REGIONS = [
  { region: "North Zone", scans: 1240, highRisk: 420 },
  { region: "South Zone", scans: 3100, highRisk: 150 },
  { region: "East Zone", scans: 850, highRisk: 300 },
  { region: "West Zone", scans: 2200, highRisk: 890 },
];

export default function AdminDashboard() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading global stats
    const timer = setTimeout(() => setLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-background">
      
      {/* Sidebar */}
      <aside className="w-64 border-r bg-card/50 backdrop-blur-xl hidden md:flex flex-col">
        <div className="p-6 border-b">
          <h2 className="text-xl font-bold text-gradient flex items-center gap-2">
            <ShieldCheck className="text-primary h-6 w-6" /> NOVA Admin
          </h2>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <Button variant="secondary" className="w-full justify-start">
            <Activity className="mr-3 h-4 w-4" /> Global Overview
          </Button>
          <Button variant="ghost" className="w-full justify-start">
            <Users className="mr-3 h-4 w-4" /> User Management
          </Button>
          <Button variant="ghost" className="w-full justify-start">
            <Database className="mr-3 h-4 w-4" /> RAG Knowledge Base
          </Button>
          <Button variant="ghost" className="w-full justify-start text-orange-500 hover:text-orange-600 hover:bg-orange-500/10">
            <AlertTriangle className="mr-3 h-4 w-4" /> Outbreak Alerts
          </Button>
        </nav>
        <div className="p-4 border-t">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center">
              <Server className="h-4 w-4 text-primary" />
            </div>
            <div className="text-sm">
              <p className="font-medium">System Status</p>
              <p className="text-emerald-500 text-xs">All Systems Operational</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto relative">
        {/* Background Blob */}
        <div className="absolute top-[-20%] right-[-10%] w-[50vw] h-[50vw] bg-primary/5 shape-blob-2 mix-blend-multiply blur-3xl opacity-50 z-[-1] pointer-events-none" />

        <div className="max-w-6xl mx-auto space-y-8">
          
          <header className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Global Command Center</h1>
              <p className="text-muted-foreground mt-1">System-wide crop health monitoring and RAG management.</p>
            </div>
            <Button>
              <Settings className="mr-2 h-4 w-4" /> Platform Settings
            </Button>
          </header>

          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="glass-panel">
              <CardContent className="p-6">
                <p className="text-sm font-medium text-muted-foreground mb-1">Total Active Farmers</p>
                <p className="text-3xl font-bold">14,209</p>
                <p className="text-xs text-emerald-500 mt-2 flex items-center">+12% this month</p>
              </CardContent>
            </Card>
            <Card className="glass-panel">
              <CardContent className="p-6">
                <p className="text-sm font-medium text-muted-foreground mb-1">Total Scans Analyzed</p>
                <p className="text-3xl font-bold">142,850</p>
                <p className="text-xs text-emerald-500 mt-2 flex items-center">+5.2k today</p>
              </CardContent>
            </Card>
            <Card className="glass-panel border-orange-500/20">
              <CardContent className="p-6">
                <p className="text-sm font-medium text-muted-foreground mb-1">Critical Outbreaks</p>
                <p className="text-3xl font-bold text-orange-500">24</p>
                <p className="text-xs text-orange-500 mt-2 flex items-center">Active regional alerts</p>
              </CardContent>
            </Card>
            <Card className="glass-panel">
              <CardContent className="p-6">
                <p className="text-sm font-medium text-muted-foreground mb-1">Avg Model Confidence</p>
                <p className="text-3xl font-bold">94.2%</p>
                <p className="text-xs text-emerald-500 mt-2 flex items-center">Stable</p>
              </CardContent>
            </Card>
          </div>

          {/* Regional Risk Map (Chart) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2 glass-panel">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="h-5 w-5 text-primary" /> Regional Risk Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={MOCK_REGIONS} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
                    <XAxis dataKey="region" tick={{fontSize: 12}} />
                    <YAxis tick={{fontSize: 12}} />
                    <Tooltip cursor={{fill: 'transparent'}} />
                    <Bar dataKey="scans" name="Total Scans" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="highRisk" name="High Risk Scans" fill="#f97316" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* RAG System Status */}
            <Card className="glass-panel">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-purple-500" /> RAG Knowledge Base
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center pb-4 border-b">
                    <div>
                      <p className="font-medium text-sm">Indexed Documents</p>
                      <p className="text-xs text-muted-foreground">Agricultural papers</p>
                    </div>
                    <span className="font-mono bg-secondary px-2 py-1 rounded text-sm">4,192</span>
                  </div>
                  <div className="flex justify-between items-center pb-4 border-b">
                    <div>
                      <p className="font-medium text-sm">Vector Store</p>
                      <p className="text-xs text-muted-foreground">FAISS CPU</p>
                    </div>
                    <span className="text-xs text-emerald-500 flex items-center">
                      <div className="h-2 w-2 rounded-full bg-emerald-500 mr-2 animate-pulse"></div> Healthy
                    </span>
                  </div>
                  <Button className="w-full mt-2" variant="outline">
                    Update Vector Index <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

        </div>
      </main>
    </div>
  );
}
