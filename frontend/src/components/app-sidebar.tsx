"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Stethoscope, 
  Radio, 
  CloudSun, 
  Bot, 
  Settings, 
  History
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Disease Detection", href: "/dashboard/disease", icon: Stethoscope },
  { name: "IoT Monitoring", href: "/dashboard/iot", icon: Radio },
  { name: "Weather Forecast", href: "/dashboard/weather", icon: CloudSun },
  { name: "AI Assistant", href: "/dashboard/assistant", icon: Bot },
  { name: "History", href: "/dashboard/history", icon: History },
  { name: "Settings", href: "/dashboard/settings", icon: Settings },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-50 w-64 border-r bg-sidebar md:flex hidden flex-col">
      <div className="flex h-16 shrink-0 items-center border-b px-6">
        <div className="flex items-center gap-2 font-semibold text-lg text-sidebar-foreground">
          <Stethoscope className="h-6 w-6 text-primary" />
          AgriVision AI
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-1 overflow-y-auto p-4">
        <nav className="grid gap-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
