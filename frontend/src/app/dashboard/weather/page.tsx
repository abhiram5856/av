"use client";

import { Cloud, CloudRain, CloudSun, Droplets, MapPin, Sun, Wind, Thermometer } from "lucide-react";
import { motion } from "framer-motion";
import { staggerContainer, staggerItem } from "@/lib/design-tokens";
import { useTranslation } from "@/lib/i18n";

const forecastData = [
  { day: "Today", temp: "28°C", condition: "Sunny", icon: Sun, color: "text-orange-500", bg: "bg-orange-500/10" },
  { day: "Tomorrow", temp: "26°C", condition: "Partly Cloudy", icon: CloudSun, color: "text-blue-400", bg: "bg-blue-400/10" },
  { day: "Wednesday", temp: "24°C", condition: "Rain", icon: CloudRain, color: "text-blue-500", bg: "bg-blue-500/10" },
  { day: "Thursday", temp: "22°C", condition: "Heavy Rain", icon: CloudRain, color: "text-blue-600", bg: "bg-blue-600/10" },
  { day: "Friday", temp: "25°C", condition: "Cloudy", icon: Cloud, color: "text-gray-500", bg: "bg-gray-500/10" },
];

export default function WeatherPage() {
  const { t } = useTranslation();

  return (
    <motion.div
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
      className="flex flex-col gap-6 max-w-5xl mx-auto"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">{t("weather.title")}</h1>
          <p className="text-muted-foreground mt-2 text-base">{t("weather.subtitle")}</p>
        </div>
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground bg-muted/50 px-4 py-2 rounded-full w-fit">
          <MapPin className="h-4 w-4" />
          <span>Hyderabad, India</span>
        </div>
      </motion.div>

      {/* Current Conditions */}
      <div className="grid gap-6 md:grid-cols-3">
        <motion.div variants={staggerItem} className="md:col-span-2">
          <div className="bg-gradient-to-br from-blue-500 to-blue-700 text-white rounded-3xl shadow-lg overflow-hidden relative p-8">
            <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
              <Sun className="h-48 w-48" />
            </div>
            <p className="text-blue-100 text-sm font-medium mb-4">{t("weather.current")}</p>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 relative z-10">
              <div className="flex items-center gap-6">
                <Sun className="h-20 w-20 text-yellow-300 drop-shadow-md shrink-0" />
                <div>
                  <div className="text-6xl font-extrabold tracking-tighter">28°C</div>
                  <p className="text-xl font-medium text-blue-100 mt-1">{t("dashboard.clear_sunny")}</p>
                  <p className="text-sm text-blue-200">{t("weather.feels_like")} 30°C</p>
                </div>
              </div>
              <div className="flex gap-8">
                {[
                  { icon: Droplets, value: "45%", label: t("weather.humidity") },
                  { icon: Wind, value: "12 km/h", label: t("weather.wind") },
                  { icon: Thermometer, value: "30°C", label: t("weather.feels_like") },
                ].map(({ icon: Icon, value, label }) => (
                  <div key={label} className="flex flex-col items-center gap-1.5">
                    <Icon className="h-5 w-5 text-blue-200" />
                    <span className="text-sm font-bold">{value}</span>
                    <span className="text-xs text-blue-200">{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Agri Insights */}
        <motion.div variants={staggerItem}>
          <div className="bg-card rounded-3xl border border-border/50 shadow-sm p-6 h-full flex flex-col gap-4">
            <h3 className="font-bold text-base">{t("weather.tips")}</h3>
            <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-2xl flex-1">
              <h4 className="font-semibold text-green-700 dark:text-green-400 text-sm mb-1.5">
                ✓ {t("weather.favorable")}
              </h4>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {t("weather.favorable_desc")}
              </p>
            </div>
            <div className="bg-yellow-500/10 border border-yellow-500/20 p-4 rounded-2xl flex-1">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 text-sm mb-1.5">
                ⚠ {t("weather.upcoming_rain")}
              </h4>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {t("weather.upcoming_rain_desc")}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* 5-Day Forecast */}
      <motion.div variants={staggerItem}>
        <div className="bg-card rounded-3xl border border-border/50 shadow-sm p-6">
          <h3 className="font-bold text-base mb-5">{t("weather.forecast")}</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
            {forecastData.map((day, index) => (
              <motion.div
                key={index}
                variants={staggerItem}
                className="flex flex-col items-center justify-center p-5 rounded-2xl border bg-muted/20 hover:bg-muted/40 transition-colors"
              >
                <span className="text-sm font-medium text-muted-foreground mb-3">
                  {t(`weather.${day.day.toLowerCase()}`)}
                </span>
                <div className={`${day.bg} p-3 rounded-xl mb-3`}>
                  <day.icon className={`h-8 w-8 ${day.color}`} />
                </div>
                <span className="text-2xl font-extrabold mb-1">{day.temp}</span>
                <span className="text-xs font-medium text-center text-muted-foreground">
                  {t(`weather.${day.condition.toLowerCase().replace(' ', '_')}`)}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
