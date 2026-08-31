"use client";

import { Leaf, Settings, Sun, Moon } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { User as SupabaseUser } from "@supabase/supabase-js";
import Link from "next/link";
import { useTheme } from "next-themes";
import { useTranslation } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";

export function AppNavbar() {
  const { t } = useTranslation();
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (data.user) setUser(data.user);
      setMounted(true);
    });
  }, []);

  const language = useAppStore((state) => state.language);
  const setLanguage = useAppStore((state) => state.setLanguage);

  const firstName = user?.user_metadata?.first_name || "User";
  const initials = firstName.charAt(0).toUpperCase();

  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between px-4 md:px-6 lg:px-8 max-w-6xl mx-auto w-full">
        {/* Logo */}
        <Link
          href="/dashboard"
          className="flex items-center gap-2.5 font-bold text-xl text-foreground hover:text-primary transition-colors"
        >
          <div className="bg-primary/10 p-1.5 rounded-xl">
            <Leaf className="h-5 w-5 text-primary" />
          </div>
          <span className="tracking-tight">NOVA</span>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center gap-1 text-sm font-medium">
          {[
            { href: "/dashboard", label: t("nav.home") },
            { href: "/dashboard/disease", label: t("nav.disease") },
            { href: "/dashboard/weather", label: t("nav.weather") },
            { href: "/dashboard/hardware", label: t("nav.monitoring") },
            { href: "/dashboard/analytics", label: t("nav.analytics") },
            { href: "/dashboard/assistant", label: t("nav.assistant") },
            { href: "/dashboard/history", label: t("nav.history") },
          ].map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className="px-3 py-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              {label}
            </Link>
          ))}
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-2">
          {/* Theme Toggle */}
          {mounted && (
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="p-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </button>
          )}

          {/* Language Switcher */}
          {mounted && (
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as "en" | "te" | "hi")}
              className="hidden sm:block p-1.5 rounded-md text-sm border-0 bg-transparent text-muted-foreground focus:ring-0 hover:text-foreground cursor-pointer font-medium"
              aria-label="Select Language"
            >
              <option value="en">EN</option>
              <option value="te">తెలుగు</option>
              <option value="hi">हिंदी</option>
            </select>
          )}

          {/* Settings */}
          <Link
            href="/dashboard/settings"
            className="p-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-colors hidden lg:flex"
            aria-label="Settings"
          >
            <Settings className="h-5 w-5" />
          </Link>

          {/* User Avatar */}
          <Link href="/dashboard/profile">
            <Avatar className="h-9 w-9 border-2 border-primary/20 hover:border-primary/40 transition-colors">
              <AvatarImage src="" alt={firstName} />
              <AvatarFallback className="bg-primary/10 text-primary font-bold text-sm">
                {initials}
              </AvatarFallback>
            </Avatar>
          </Link>
        </div>
      </div>
    </header>
  );
}
