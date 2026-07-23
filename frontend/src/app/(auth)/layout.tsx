import { ReactNode } from "react";
import { Leaf } from "lucide-react";
import Link from "next/link";

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="flex flex-col justify-center items-center p-8">
        <div className="w-full max-w-[400px]">
          <Link href="/" className="flex items-center gap-2 mb-12 w-fit">
            <div className="bg-primary/10 p-2 rounded-lg">
              <Leaf className="h-6 w-6 text-primary" />
            </div>
            <span className="font-bold text-xl tracking-tight">AgriVision AI</span>
          </Link>
          {children}
        </div>
      </div>
      <div className="hidden lg:flex flex-col justify-center items-center bg-muted relative overflow-hidden">
        <div className="absolute inset-0 bg-primary/5" />
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-40 mix-blend-overlay" />
        <div className="z-10 max-w-lg p-12 text-center">
          <h2 className="text-4xl font-bold tracking-tight mb-4 text-foreground/90">
            Empowering Modern Agriculture
          </h2>
          <p className="text-lg text-muted-foreground">
            Leverage artificial intelligence to monitor crop health, predict weather impacts, and optimize your farm's yield in real-time.
          </p>
        </div>
      </div>
    </div>
  );
}
