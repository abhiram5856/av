"use client";

import { useAppStore } from "@/lib/store";
import { I18nProvider } from "@/lib/i18n";
import { ThemeProvider } from "@/components/theme-provider";
import { useEffect } from "react";

export function AppProviders({ children }: { children: React.ReactNode }) {
  const { largeFont, highContrast } = useAppStore();

  useEffect(() => {
    const html = document.documentElement;
    if (largeFont) html.classList.add("large-font");
    else html.classList.remove("large-font");

    if (highContrast) html.classList.add("high-contrast");
    else html.classList.remove("high-contrast");
  }, [largeFont, highContrast]);

  return (
    <I18nProvider>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        {children}
      </ThemeProvider>
    </I18nProvider>
  );
}
