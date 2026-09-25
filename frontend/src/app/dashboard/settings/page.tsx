"use client";

import { useTheme } from "next-themes";
import { useAppStore } from "@/lib/store";
import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { useTranslation } from "@/lib/i18n";

export default function SettingsPage() {
  const { t } = useTranslation();
  const { theme, setTheme } = useTheme();
  const { language, setLanguage, largeFont, setLargeFont, highContrast, setHighContrast } = useAppStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- hydration guard: runs once after mount, no cascading renders
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto w-full">
      <div className="border-b pb-4 mb-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          {t("settings.title")}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t("settings.subtitle")}
        </p>
      </div>

      <div className="space-y-8">
        
        {/* Appearance */}
        <section className="space-y-4">
          <h2 className="text-lg font-medium">{t("settings.appearance")}</h2>
          <div className="border rounded-lg bg-card p-4 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <Label className="text-base">{t("settings.theme")}</Label>
                <p className="text-sm text-muted-foreground">{t("settings.theme_desc")}</p>
              </div>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
                className="h-10 px-3 rounded-md border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary w-full sm:w-48"
              >
                <option value="light">{t("settings.theme_light")}</option>
                <option value="dark">{t("settings.theme_dark")}</option>
                <option value="system">{t("settings.theme_system")}</option>
              </select>
            </div>
            
            <div className="pt-4 border-t flex items-center justify-between">
              <div>
                <Label className="text-base cursor-pointer" onClick={() => setHighContrast(!highContrast)}>{t("settings.high_contrast")}</Label>
                <p className="text-sm text-muted-foreground">{t("settings.high_contrast_desc")}</p>
              </div>
              <button
                role="switch"
                aria-checked={highContrast}
                onClick={() => setHighContrast(!highContrast)}
                className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                  highContrast ? "bg-primary" : "bg-muted"
                }`}
              >
                <div
                  className={`h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
                    highContrast ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
            </div>

            <div className="pt-4 border-t flex items-center justify-between">
              <div>
                <Label className="text-base cursor-pointer" onClick={() => setLargeFont(!largeFont)}>{t("settings.large_text")}</Label>
                <p className="text-sm text-muted-foreground">{t("settings.large_text_desc")}</p>
              </div>
              <button
                role="switch"
                aria-checked={largeFont}
                onClick={() => setLargeFont(!largeFont)}
                className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                  largeFont ? "bg-primary" : "bg-muted"
                }`}
              >
                <div
                  className={`h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
                    largeFont ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
            </div>
          </div>
        </section>

        {/* Language */}
        <section className="space-y-4">
          <h2 className="text-lg font-medium">{t("settings.localization")}</h2>
          <div className="border rounded-lg bg-card p-4 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <Label className="text-base">{t("settings.language")}</Label>
                <p className="text-sm text-muted-foreground">{t("settings.language_desc")}</p>
              </div>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as "en" | "te" | "hi")}
                className="h-10 px-3 rounded-md border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary w-full sm:w-48"
              >
                <option value="en">English</option>
                <option value="te">తెలుగు (Telugu)</option>
                <option value="hi">हिंदी (Hindi)</option>
              </select>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
