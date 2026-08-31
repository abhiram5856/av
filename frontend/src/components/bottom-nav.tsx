"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, ScanLine, CloudSun, MessageCircle, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useTranslation } from "@/lib/i18n";

const navItems = [
  { name: "nav.home", href: "/dashboard", icon: Home },
  { name: "nav.scan", href: "/dashboard/disease", icon: ScanLine },
  { name: "nav.weather", href: "/dashboard/weather", icon: CloudSun },
  { name: "nav.assistant", href: "/dashboard/assistant", icon: MessageCircle },
  { name: "nav.settings", href: "/dashboard/settings", icon: Settings },
];

export function BottomNav() {
  const pathname = usePathname();
  const { t } = useTranslation();

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-t lg:hidden pb-safe">
      <nav
        className="flex justify-around items-center h-16 max-w-lg mx-auto px-2"
        role="navigation"
        aria-label="Main navigation"
      >
        {navItems.map((item) => {
          const isActive =
            item.href === "/dashboard"
              ? pathname === "/dashboard"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "relative flex flex-col items-center justify-center min-w-[48px] min-h-[48px] gap-0.5 text-[11px] font-medium transition-colors rounded-xl",
                isActive
                  ? "text-primary"
                  : "text-muted-foreground active:text-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              {isActive && (
                <motion.div
                  layoutId="bottom-nav-indicator"
                  className="absolute -top-1 w-5 h-1 bg-primary rounded-full"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <item.icon
                className={cn("h-5 w-5", isActive && "text-primary")}
                strokeWidth={isActive ? 2.5 : 2}
              />
              <span>{t(item.name)}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
