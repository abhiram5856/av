"use client";

import { Camera, CloudSun, Clock, Upload, Search, FileText } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { useTranslation } from "@/lib/i18n";

export default function DashboardPage() {
  const { t } = useTranslation();
  const [firstName, setFirstName] = useState("User");

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (data.user?.user_metadata?.first_name) {
        setFirstName(data.user.user_metadata.first_name);
      }
    });
  }, []);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return t("dashboard.greeting.morning");
    if (hour < 17) return t("dashboard.greeting.afternoon");
    return t("dashboard.greeting.evening");
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto w-full">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b pb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {getGreeting()}, {firstName}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t("dashboard.overview")}
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            href="/dashboard/disease"
            className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-primary-foreground bg-primary rounded-md shadow hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 transition-colors"
          >
            <Camera className="h-4 w-4" />
            {t("dashboard.new_diagnosis")}
          </Link>
        </div>
      </div>

      {/* ─── Quick Actions ────────────────────────────────────────────────── */}
      <div className="grid sm:grid-cols-3 gap-4">
        {[
          {
            title: t("dashboard.upload_image"),
            description: t("dashboard.upload_desc"),
            icon: Upload,
            href: "/dashboard/disease",
          },
          {
            title: t("dashboard.view_history"),
            description: t("dashboard.history_desc"),
            icon: FileText,
            href: "/dashboard/history",
          },
          {
            title: t("dashboard.ask_assistant"),
            description: t("dashboard.assistant_desc"),
            icon: Search,
            href: "/dashboard/assistant",
          },
        ].map((action) => (
          <Link
            key={action.title}
            href={action.href}
            className="group flex flex-col gap-2 p-5 rounded-lg border bg-card hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            <action.icon className="h-5 w-5 text-muted-foreground group-hover:text-foreground mb-2" />
            <h3 className="font-medium text-sm">{action.title}</h3>
            <p className="text-xs text-muted-foreground">{action.description}</p>
          </Link>
        ))}
      </div>

      {/* ─── Main Content Grid ────────────────────────────────────────────── */}
      <div className="grid lg:grid-cols-3 gap-6">
        
        {/* Recent Activity (Spans 2 cols on lg) */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <h2 className="text-lg font-medium">{t("dashboard.recent_scans")}</h2>
          <div className="flex-1 rounded-lg border bg-card p-8 flex flex-col items-center justify-center text-center min-h-[300px]">
            <Clock className="h-8 w-8 text-muted-foreground mb-4" />
            <h3 className="font-medium mb-1">{t("dashboard.no_recent_scans")}</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-sm">
              {t("dashboard.no_scans_desc")}
            </p>
            <Link
              href="/dashboard/disease"
              className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium border rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              {t("dashboard.start_scan")}
            </Link>
          </div>
        </div>

        {/* Sidebar info (Weather, etc.) */}
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium">{t("dashboard.local_weather")}</h2>
              <Link href="/dashboard/weather" className="text-sm text-primary hover:underline">
                {t("dashboard.details")}
              </Link>
            </div>
            <div className="rounded-lg border bg-card p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-3xl font-semibold mb-1">28°C</div>
                  <div className="text-sm font-medium text-muted-foreground">{t("dashboard.clear_sunny")}</div>
                </div>
                <CloudSun className="h-10 w-10 text-orange-500" />
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm pt-4 border-t">
                <div className="flex flex-col">
                  <span className="text-muted-foreground">{t("dashboard.humidity")}</span>
                  <span className="font-medium">45%</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">{t("dashboard.wind")}</span>
                  <span className="font-medium">12 km/h</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
      </div>
    </div>
  );
}
