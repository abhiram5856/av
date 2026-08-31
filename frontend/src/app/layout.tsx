import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppProviders } from "@/components/app-providers";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "NOVA — AI Crop Disease Diagnosis",
  description:
    "Instantly diagnose plant diseases with AI. Upload a photo of your crop and get disease identification, severity assessment, root cause analysis, and treatment recommendations.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "NOVA",
  },
  openGraph: {
    title: "NOVA — AI Crop Disease Diagnosis",
    description: "Instantly diagnose plant diseases with AI. Upload a photo of your crop and get disease identification, severity assessment, root cause analysis, and treatment recommendations.",
    url: "https://nova-agrivision.app",
    siteName: "NOVA",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "NOVA AI Crop Disease Diagnosis",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "NOVA — AI Crop Disease Diagnosis",
    description: "Instantly diagnose plant diseases with AI. Upload a photo of your crop and get disease identification, severity assessment, root cause analysis, and treatment recommendations.",
    images: ["/og-image.jpg"],
  },
};

export const viewport: Viewport = {
  themeColor: "#16a34a",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col relative overflow-x-hidden">
        
        {/* Animated Background Orbs */}
        <div className="fixed top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-primary/10 shape-blob-1 mix-blend-multiply blur-3xl opacity-60 z-[-1] pointer-events-none" />
        <div className="fixed bottom-[-10%] right-[-10%] w-[50vw] h-[50vw] bg-emerald-400/10 shape-blob-2 mix-blend-multiply blur-3xl opacity-60 z-[-1] pointer-events-none animate-float" />
        <div className="fixed top-[40%] left-[60%] w-[25vw] h-[25vw] bg-teal-400/5 shape-blob-1 mix-blend-multiply blur-3xl opacity-50 z-[-1] pointer-events-none" style={{ animationDelay: '2s' }} />

        <AppProviders>
          {children}
        </AppProviders>
      </body>
    </html>
  );
}
