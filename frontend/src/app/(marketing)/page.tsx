"use client";

import Link from "next/link";
import { Camera, ImagePlus, ArrowRight, Leaf, ScanLine, CheckCircle } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { motion } from "framer-motion";

export default function MarketingPage() {
  return (
    <div>
      {/* ─── Hero Section (Farmer-Friendly) ─────────────────────────── */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-4 pt-20 pb-24 overflow-hidden bg-green-900">
        
        {/* Simple Background */}
        <div className="absolute inset-0 z-0 overflow-hidden bg-[url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-40">
          <div className="absolute inset-0 bg-black/60 z-10" />
        </div>

        {/* Hero Content */}
        <div className="relative z-20 max-w-3xl mx-auto text-center mt-12 sm:mt-0">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight text-white leading-tight mb-6 drop-shadow-2xl">
              Got a sick plant?
            </h1>
            <p className="text-xl sm:text-3xl text-zinc-100 leading-snug mb-12 max-w-2xl mx-auto drop-shadow-md font-medium">
              Take a photo of the leaf. We will tell you exactly what's wrong and how to fix it immediately.
            </p>
            <div className="flex flex-col items-center justify-center gap-6">
              <Link
                href="/dashboard/disease"
                className={buttonVariants({
                  size: "lg",
                  className: "w-full sm:w-auto h-20 px-12 rounded-full text-2xl font-bold gap-3 shadow-2xl hover:scale-105 transition-transform bg-green-500 text-white hover:bg-green-400 border-4 border-green-600",
                })}
              >
                <Camera className="h-8 w-8" />
                Tap to Check Leaf
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ─── How It Works (Ultra Simple) ────────────────────────────── */}
      <section id="how-it-works" className="px-6 py-24 bg-white relative z-20 text-slate-900">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold tracking-tight mb-4 text-green-800">
              How it works
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-12">
            {[
              {
                num: "1",
                icon: Camera,
                title: "Take a Photo",
                desc: "Point your phone camera at the sick leaf.",
              },
              {
                num: "2",
                icon: ScanLine,
                title: "Wait 2 Seconds",
                desc: "Our system identifies the disease automatically.",
              },
              {
                num: "3",
                icon: CheckCircle,
                title: "Get the Solution",
                desc: "Read exact instructions on what to spray or do next.",
              },
            ].map(({ num, icon: Icon, title, desc }) => (
              <div key={num} className="flex flex-col items-center text-center p-8 rounded-3xl bg-slate-50 border-2 border-slate-200 shadow-sm">
                <div className="h-24 w-24 bg-green-100 rounded-full flex items-center justify-center mb-6 text-green-700 shadow-inner">
                  <Icon className="h-12 w-12" />
                </div>
                <h3 className="text-3xl font-bold mb-4 text-slate-800">{title}</h3>
                <p className="text-xl text-slate-600 leading-relaxed font-medium">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Bottom CTA ─────────────────────────────────────────────────── */}
      <section className="px-6 py-24 bg-green-800 relative z-20">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-8 text-white">
            Ready to save your harvest?
          </h2>
          <Link
            href="/dashboard/disease"
            className={buttonVariants({
              size: "lg",
              className: "w-full sm:w-auto h-20 px-12 rounded-full text-2xl font-bold gap-3 shadow-2xl hover:scale-105 transition-transform bg-white text-green-800 hover:bg-slate-100",
            })}
          >
            Start Now — It's Free
            <ArrowRight className="h-6 w-6" />
          </Link>
          
          <div className="mt-16 pt-8 border-t border-green-700/50">
            <p className="text-green-100 text-lg">
              Are you an agricultural company or farm manager? <br/>
              <Link href="/dashboard" className="text-white font-bold underline hover:text-green-300 mt-2 inline-block">
                Enter the Corporate Dashboard
              </Link>
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
