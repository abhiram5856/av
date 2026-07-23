"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Stethoscope, CloudSun, User } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { name: "Home", href: "/dashboard", icon: Home },
  { name: "Scan", href: "/dashboard/disease", icon: Stethoscope },
  { name: "Weather", href: "/dashboard/weather", icon: CloudSun },
  { name: "Profile", href: "/dashboard/profile", icon: User },
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 bg-background border-t pb-safe">
      <nav className="flex justify-around items-center h-16 max-w-md mx-auto px-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex flex-col items-center justify-center w-16 h-full gap-1 text-xs font-medium transition-colors",
                isActive
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <item.icon className={cn("h-6 w-6", isActive && "fill-primary/20")} />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
