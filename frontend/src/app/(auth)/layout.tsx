import { ReactNode } from "react";
import Link from "next/link";

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-muted/20 px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <Link href="/" className="inline-block font-bold text-2xl tracking-tight">
            NOVA
          </Link>
        </div>
        <div className="bg-card border rounded-lg shadow-sm p-6 sm:p-8">
          {children}
        </div>
      </div>
    </div>
  );
}
