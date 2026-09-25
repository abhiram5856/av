"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertOctagon, X, MapPin } from "lucide-react";
import { usePathname } from "next/navigation";
import { useTranslation } from "@/lib/i18n";

export function RegionalAlert() {
  const { t } = useTranslation();
  const [visible, setVisible] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    // Only simulate alert on the main dashboard home page for demo purposes
    if (pathname === "/dashboard") {
      const timer = setTimeout(() => {
        setVisible(true);
      }, 5000); // Alert pops up after 5 seconds

      return () => clearTimeout(timer);
    }
  }, [pathname]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="fixed bottom-20 md:bottom-6 right-4 md:right-6 z-50 max-w-sm w-[calc(100vw-2rem)]"
        >
          <div className="glass-panel border-orange-500/30 bg-background/80 p-4 relative shadow-orange-500/10">
            <button 
              onClick={() => setVisible(false)}
              className="absolute top-2 right-2 p-1 text-muted-foreground hover:text-foreground rounded-full hover:bg-muted"
            >
              <X className="h-4 w-4" />
            </button>
            
            <div className="flex gap-3 items-start">
              <div className="bg-orange-500/20 p-2 rounded-full mt-1 shrink-0">
                <AlertOctagon className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <h4 className="font-semibold text-orange-500 flex items-center gap-1">
                  {t("regional.outbreak_title")}
                </h4>
                <p className="text-sm font-medium mt-1">{t("regional.late_blight_detected")}</p>
                <p className="text-xs text-muted-foreground mt-1 flex items-center">
                  <MapPin className="h-3 w-3 mr-1" /> {t("regional.distance")}
                </p>
                <div className="mt-3 text-xs bg-orange-500/10 text-orange-600 p-2 rounded border border-orange-500/20">
                  <span className="font-semibold">{t("regional.recommended_action")}</span> {t("regional.fungicide_advice")}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
