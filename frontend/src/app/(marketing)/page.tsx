"use client";

import Link from "next/link";
import { Leaf, ScanLine, ShieldCheck } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { motion } from "framer-motion";

export default function MarketingPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 100 } },
  };

  return (
    <div className="relative flex flex-col items-center justify-center min-h-[90vh] px-6 text-center overflow-hidden">
      
      {/* Decorative Background Elements */}
      <div className="absolute top-1/4 left-0 w-72 h-72 bg-primary/20 rounded-full blur-3xl -z-10 animate-pulse"></div>
      <div className="absolute bottom-1/4 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl -z-10"></div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-3xl mx-auto glass-panel p-8 md:p-12 rounded-3xl"
      >
        <motion.div variants={itemVariants} className="flex justify-center mb-8">
          <div className="relative p-6 bg-gradient-to-br from-primary to-emerald-400 rounded-full shadow-lg shadow-primary/30">
            <ScanLine className="h-16 w-16 text-primary-foreground absolute inset-0 m-auto animate-pulse" />
            <Leaf className="h-16 w-16 text-primary-foreground/30" />
          </div>
        </motion.div>
        
        <motion.h1 variants={itemVariants} className="text-5xl md:text-7xl font-extrabold tracking-tighter text-foreground mb-6">
          Meet <span className="text-gradient">NOVA</span>
        </motion.h1>
        
        <motion.p variants={itemVariants} className="text-xl md:text-2xl text-muted-foreground mb-10 font-medium leading-relaxed">
          The ultimate Neural Optimized Vision Assistant. Detect diseases instantly, estimate severity, and get multilingual treatment plans right from your pocket.
        </motion.p>

        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link 
            href="/dashboard" 
            className={buttonVariants({ 
              size: "lg", 
              className: "w-full sm:w-auto h-16 px-10 rounded-2xl text-xl font-bold shadow-xl shadow-primary/20 hover:scale-105 transition-transform" 
            })}
          >
            Open Dashboard
          </Link>
          <Link 
            href="/login" 
            className={buttonVariants({ 
              variant: "outline",
              size: "lg", 
              className: "w-full sm:w-auto h-16 px-10 rounded-2xl text-xl font-bold bg-background/50 hover:bg-background/80 hover:scale-105 transition-transform border-2" 
            })}
          >
            Log In
          </Link>
        </motion.div>

        <motion.div variants={itemVariants} className="mt-12 flex items-center justify-center gap-2 text-sm text-muted-foreground font-medium">
          <ShieldCheck className="h-5 w-5 text-primary" />
          <span>Powered by Explainable AI (Grad-CAM) & Offline PWA Ready</span>
        </motion.div>

      </motion.div>
    </div>
  );
}
