"use client";

import { Camera, CloudSun, TrendingUp, Users, Leaf, Activity } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function DashboardPage() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const item = {
    hidden: { opacity: 0, scale: 0.95 },
    show: { opacity: 1, scale: 1, transition: { type: "spring", stiffness: 100 } },
  };

  return (
    <div className="relative flex flex-col gap-8 pt-4 pb-20 overflow-hidden">
      
      {/* Dynamic Background */}
      <div className="absolute -top-40 -right-40 w-[500px] h-[500px] bg-primary/10 rounded-full blur-3xl -z-10 pointer-events-none"></div>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="px-2">
        <h1 className="text-4xl font-extrabold tracking-tight text-foreground">Welcome Back</h1>
        <p className="text-muted-foreground mt-2 text-lg font-medium flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary animate-pulse" />
          All systems nominal. Ready to scan.
        </p>
      </motion.div>

      {/* Bento Grid Layout */}
      <motion.div 
        variants={container} 
        initial="hidden" 
        animate="show" 
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        
        {/* Main Featured Action (Takes up 2 columns on desktop) */}
        <motion.div variants={item} className="md:col-span-2">
          <Link
            href="/dashboard/disease"
            className="group flex flex-col justify-between h-full p-8 rounded-[2rem] glass-panel-hover bg-gradient-to-br from-primary/90 to-emerald-500 overflow-hidden relative"
          >
            {/* Background Icon */}
            <Camera className="absolute -right-8 -bottom-8 h-64 w-64 text-white/10 group-hover:scale-110 transition-transform duration-500" />
            
            <div className="mb-12">
              <div className="inline-flex items-center justify-center p-4 rounded-2xl bg-white/20 backdrop-blur-md mb-6">
                <ScanLine className="h-10 w-10 text-white" />
              </div>
              <h2 className="text-3xl font-extrabold text-white mb-2">Scan Crop</h2>
              <p className="text-white/90 text-lg font-medium max-w-sm">
                Detect diseases instantly using your camera and the NOVA XAI engine.
              </p>
            </div>
            
            <div className="flex items-center text-white font-bold group-hover:translate-x-2 transition-transform">
              Start Diagnosis &rarr;
            </div>
          </Link>
        </motion.div>

        {/* Weather Card */}
        <motion.div variants={item}>
          <Link
            href="/dashboard/weather"
            className="group flex flex-col justify-between h-full p-6 rounded-[2rem] glass-panel-hover bg-card"
          >
            <div>
              <div className="inline-flex items-center justify-center p-3 rounded-xl bg-orange-500/10 text-orange-500 mb-4 group-hover:bg-orange-500 group-hover:text-white transition-colors">
                <CloudSun className="h-8 w-8" />
              </div>
              <h2 className="text-xl font-bold text-foreground">Weather</h2>
              <p className="text-muted-foreground mt-2 font-medium">Check local forecasts and severe alerts.</p>
            </div>
          </Link>
        </motion.div>

        {/* Market Prices Card */}
        <motion.div variants={item}>
          <Link
            href="/dashboard/market"
            className="group flex flex-col justify-between h-full p-6 rounded-[2rem] glass-panel-hover bg-card"
          >
            <div>
              <div className="inline-flex items-center justify-center p-3 rounded-xl bg-blue-500/10 text-blue-500 mb-4 group-hover:bg-blue-500 group-hover:text-white transition-colors">
                <TrendingUp className="h-8 w-8" />
              </div>
              <h2 className="text-xl font-bold text-foreground">Market Prices</h2>
              <p className="text-muted-foreground mt-2 font-medium">Live crop pricing and trends.</p>
            </div>
          </Link>
        </motion.div>

        {/* Ask Expert Card */}
        <motion.div variants={item}>
          <Link
            href="/dashboard/assistant"
            className="group flex flex-col justify-between h-full p-6 rounded-[2rem] glass-panel-hover bg-card border-primary/20"
          >
            <div>
              <div className="inline-flex items-center justify-center p-3 rounded-xl bg-primary/10 text-primary mb-4 group-hover:bg-primary group-hover:text-white transition-colors">
                <Users className="h-8 w-8" />
              </div>
              <h2 className="text-xl font-bold text-foreground">NOVA Assistant</h2>
              <p className="text-muted-foreground mt-2 font-medium">Chat with the multilingual AI agronomy expert.</p>
            </div>
          </Link>
        </motion.div>

        {/* History Card (Takes up 1 col) */}
        <motion.div variants={item}>
          <Link
            href="/dashboard/history"
            className="group flex flex-col justify-between h-full p-6 rounded-[2rem] glass-panel-hover bg-card"
          >
            <div>
              <div className="inline-flex items-center justify-center p-3 rounded-xl bg-purple-500/10 text-purple-500 mb-4 group-hover:bg-purple-500 group-hover:text-white transition-colors">
                <Leaf className="h-8 w-8" />
              </div>
              <h2 className="text-xl font-bold text-foreground">History</h2>
              <p className="text-muted-foreground mt-2 font-medium">Review past crop diagnoses.</p>
            </div>
          </Link>
        </motion.div>

      </motion.div>
    </div>
  );
}

// Ensure ScanLine is imported for the main card
function ScanLine(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <line x1="7" x2="17" y1="12" y2="12" />
    </svg>
  )
}
