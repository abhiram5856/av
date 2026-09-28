"use client";

import Link from "next/link";
import { Camera, ArrowRight, ScanLine, CheckCircle, BarChart3, Sprout, ShieldCheck } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { motion } from "framer-motion";
import { useTranslation } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import { Globe } from "lucide-react";

export default function MarketingPage() {
  const { t } = useTranslation();
  const { language, setLanguage } = useAppStore();

  return (
    <div className="bg-white min-h-screen text-slate-900 overflow-x-hidden font-sans relative">
      
      {/* Language Switcher */}
      <div className="absolute top-6 right-6 z-50 flex items-center gap-2 bg-white px-3 py-1.5 rounded-sm border border-slate-200 shadow-sm">
        <Globe className="h-4 w-4 text-slate-500" />
        <select 
          value={language}
          onChange={(e) => setLanguage(e.target.value as "en" | "te" | "hi")}
          className="bg-transparent text-sm font-medium text-slate-700 focus:outline-none cursor-pointer"
        >
          <option value="en">English</option>
          <option value="te">తెలుగు</option>
          <option value="hi">हिंदी</option>
        </select>
      </div>

      {/* ─── Minimal Hero Section ─────────────────────────── */}
      <section className="relative pt-32 pb-24 lg:pt-48 lg:pb-32 px-6">
        <div className="max-w-5xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="inline-flex items-center gap-2 mb-6">
              <Sprout className="h-5 w-5 text-green-700" />
              <span className="text-sm font-semibold text-green-700 tracking-wide uppercase">{t("marketing.subtitle")}</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 mb-6 leading-[1.1]">
              {t("marketing.headline_1")} <br className="hidden md:block"/> <span className="text-green-700">{t("marketing.headline_2")}</span>
            </h1>
            
            <p className="text-lg md:text-xl text-slate-600 max-w-2xl font-normal mb-12">
              {t("marketing.desc")}
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4">
              {/* Farmer Path */}
              <Link
                href="/dashboard/disease"
                className="flex items-center justify-center gap-2 w-full sm:w-auto px-8 py-4 bg-green-700 text-white font-semibold text-base hover:bg-green-800 transition-colors rounded-sm"
              >
                <Camera className="h-5 w-5" />
                {t("marketing.btn_farmer")}
              </Link>

              {/* Agritech Path */}
              <Link
                href="/login"
                className="flex items-center justify-center gap-2 w-full sm:w-auto px-8 py-4 bg-slate-100 text-slate-900 font-semibold text-base hover:bg-slate-200 transition-colors border border-slate-200 rounded-sm"
              >
                <BarChart3 className="h-5 w-5 text-slate-500" />
                {t("marketing.btn_agritech")}
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ─── Value Proposition Cards ────────────────────────────── */}
      <section className="px-6 py-24 bg-slate-50 border-t border-slate-200">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: ScanLine,
                title: t("marketing.val1_title"),
                desc: t("marketing.val1_desc"),
                color: "text-emerald-600",
                bg: "bg-emerald-50"
              },
              {
                icon: ShieldCheck,
                title: t("marketing.val2_title"),
                desc: t("marketing.val2_desc"),
                color: "text-blue-600",
                bg: "bg-blue-50"
              },
              {
                icon: BarChart3,
                title: t("marketing.val3_title"),
                desc: t("marketing.val3_desc"),
                color: "text-amber-600",
                bg: "bg-amber-50"
              },
            ].map(({ icon: Icon, title, desc, color, bg }, i) => (
              <div 
                key={i} 
                className="p-8 bg-white border border-slate-200 rounded-sm"
              >
                <div className={`h-12 w-12 rounded-sm ${bg} flex items-center justify-center mb-6 border border-slate-100`}>
                  <Icon className={`h-6 w-6 ${color}`} />
                </div>
                <h3 className="text-xl font-bold mb-3 text-slate-900">{title}</h3>
                <p className="text-slate-600 font-normal leading-relaxed">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Bottom CTA ─────────────────────────────────────────────────── */}
      <section className="px-6 py-24 bg-slate-900 border-t border-slate-800">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-8 text-white">
            {t("marketing.cta")}
          </h2>
          <div className="flex flex-col sm:flex-row items-start gap-4">
            <Link
              href="/dashboard/disease"
              className="px-8 py-4 bg-white text-slate-900 font-semibold text-base hover:bg-slate-100 transition-colors rounded-sm flex items-center gap-2"
            >
              {t("marketing.cta_btn")}
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
