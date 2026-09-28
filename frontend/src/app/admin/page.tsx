"use client";

import { useState, useEffect } from "react";
import { 
  Users, Activity, Database, AlertTriangle, ShieldCheck, 
  Settings, Server, ChevronRight, BarChart2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AdminDashboard() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Replace with real admin stats API when available
    const timer = setTimeout(() => setLoading(false), 300);
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
          <h2 className="text-xl font-bold flex items-center gap-2">
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
      <main className="flex-1 p-8 overflow-y-auto">
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

          {/* Platform Status — no fabricated KPIs */}
          <Card className="border-border/50">
            <CardContent className="p-8 flex flex-col items-center justify-center text-center gap-3 min-h-[160px]">
              <BarChart2 className="h-10 w-10 text-muted-foreground/40" />
              <p className="font-semibold text-muted-foreground">Platform analytics not yet available.</p>
              <p className="text-sm text-muted-foreground/70">
                Real-time statistics will appear here once the admin analytics API is connected.
              </p>
            </CardContent>
          </Card>

          {/* Regional Risk — removed MOCK_REGIONS */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2 border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  Regional Risk Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center min-h-[240px] gap-3 text-center">
                <p className="text-sm font-medium text-muted-foreground">
                  Regional analytics are not available yet.
                </p>
                <p className="text-xs text-muted-foreground/70">
                  More recorded observations are needed to display this view.
                </p>
              </CardContent>
            </Card>

            {/* RAG System Status */}
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
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
