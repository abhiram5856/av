"use client";

import Link from "next/link";
import { Camera, ImagePlus, ArrowRight, Leaf, ScanLine, CheckCircle } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { motion } from "framer-motion";

export default function MarketingPage() {
  return (
    <div>
      {/* ─── Hero with Video Background ─────────────────────────────────── */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-6 pt-20 pb-24 overflow-hidden">
        
        {/* Video Background Container */}
        <div className="absolute inset-0 z-0 overflow-hidden bg-black">
          {/* We use a large min-width/min-height with aspect-ratio math to ensure the iframe covers the whole screen without letterboxing */}
          <iframe
            className="absolute top-1/2 left-1/2 w-[100vw] h-[56.25vw] min-h-[100vh] min-w-[177.77vh] -translate-x-1/2 -translate-y-1/2 opacity-70 pointer-events-none"
            src="https://www.youtube.com/embed/g1L2mC_uUqI?autoplay=1&mute=1&controls=0&loop=1&playlist=g1L2mC_uUqI&showinfo=0&rel=0&iv_load_policy=3&modestbranding=1&playsinline=1"
            allow="autoplay; encrypted-media"
            title="Farming Drone View"
            tabIndex={-1}
          />
          {/* Gradient overlays to blend the video into the rest of the page and ensure text readability */}
          <div className="absolute inset-0 bg-black/40 z-10" />
          <div className="absolute inset-0 bg-gradient-to-b from-background/30 via-transparent to-background z-10" />
        </div>

        {/* Hero Content */}
        <div className="relative z-20 max-w-4xl mx-auto text-center mt-12 sm:mt-0">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            <h1 className="text-5xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.05] mb-6 drop-shadow-xl">
              Crop diagnosis, instantly.
            </h1>
            <p className="text-lg sm:text-xl text-zinc-200 leading-relaxed mb-10 max-w-2xl mx-auto drop-shadow-md">
              NOVA uses computer vision to identify plant diseases from a single photo.
              Upload a leaf image, and get a diagnosis with severity assessment and
              treatment recommendations.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/dashboard/disease"
                className={buttonVariants({
                  size: "lg",
                  className: "h-14 px-8 rounded-full text-base font-bold gap-2 shadow-xl hover:scale-105 transition-transform bg-primary text-primary-foreground hover:bg-primary/90",
                })}
              >
                <Camera className="h-5 w-5" />
                Start diagnosis
              </Link>
              <Link
                href="#how-it-works"
                className={buttonVariants({
                  variant: "outline",
                  size: "lg",
                  className: "h-14 px-8 rounded-full text-base font-bold gap-2 shadow-xl hover:scale-105 transition-transform bg-background/20 backdrop-blur-md border-white/20 text-white hover:bg-white/20 hover:text-white",
                })}
              >
                How it works
                <ArrowRight className="h-5 w-5" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ─── How It Works ───────────────────────────────────────────────── */}
      <section id="how-it-works" className="px-6 py-24 bg-premium-gradient relative z-20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold tracking-tight mb-4">
              How it works
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Three simple steps to healthier crops and smarter farming decisions.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-12">
            {[
              {
                num: "01",
                icon: Camera,
                title: "Upload a photo",
                desc: "Take a picture of the affected leaf with your phone, or upload one from your gallery.",
              },
              {
                num: "02",
                icon: ScanLine,
                title: "Get a diagnosis",
                desc: "Our model analyzes the image, identifies the disease, and estimates how severe it is.",
              },
              {
                num: "03",
                icon: CheckCircle,
                title: "See recommendations",
                desc: "You'll get a clear treatment plan, the root cause, and a downloadable PDF report.",
              },
            ].map(({ num, icon: Icon, title, desc }) => (
              <div key={num} className="flex flex-col items-center text-center animate-slide-up glass-panel-hover p-6 rounded-3xl">
                <div className="h-16 w-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-6 text-primary">
                  <Icon className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-bold mb-3">{title}</h3>
                <p className="text-base text-muted-foreground leading-relaxed">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── What you get ───────────────────────────────────────────────── */}
      <section id="features" className="px-6 py-24 bg-background border-y relative z-20">
        <div className="max-w-5xl mx-auto animate-fade-in">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold tracking-tight mb-4">
              What you get
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Every diagnosis includes these outputs — no extra setup needed.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-6">
            {[
              {
                title: "Disease identification",
                desc: "Trained on thousands of images across multiple crop types. Gives you the disease name with a confidence score.",
              },
              {
                title: "Root cause analysis",
                desc: "TRACE-RCE looks beyond symptoms to find underlying causes — weather, soil conditions, pest pressure.",
              },
              {
                title: "Visual explanation",
                desc: "GradCAM heatmaps show exactly which parts of the leaf the model focused on. No black box.",
              },
              {
                title: "PDF reports",
                desc: "Download a full report with the image, heatmap, diagnosis, and recommendations. Useful for record-keeping.",
              },
            ].map(({ title, desc }) => (
              <div
                key={title}
                className="p-8 rounded-3xl glass-panel glass-panel-hover"
              >
                <h3 className="text-xl font-bold mb-3">{title}</h3>
                <p className="text-base text-muted-foreground leading-relaxed">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Bottom CTA ─────────────────────────────────────────────────── */}
      <section className="px-6 py-24 bg-premium-gradient relative z-20">
        <div className="max-w-3xl mx-auto text-center animate-slide-up">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-6">
            Ready to save your harvest?
          </h2>
          <p className="text-lg text-muted-foreground mb-10">
            Create a free account in seconds. Upload a photo and see the results instantly.
          </p>
          <Link
            href="/signup"
            className={buttonVariants({
              size: "lg",
              className: "h-14 px-10 rounded-full text-lg font-bold gap-3 shadow-xl hover:scale-105 transition-transform",
            })}
          >
            Create free account
            <ArrowRight className="h-5 w-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}
