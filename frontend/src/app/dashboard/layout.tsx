import { AppNavbar } from "@/components/app-navbar";
import { BottomNav } from "@/components/bottom-nav";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen bg-muted/20 pb-20">
      <div className="flex flex-1 flex-col mx-auto w-full max-w-md bg-background shadow-2xl relative min-h-screen">
        <AppNavbar />
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          {children}
        </main>
        <BottomNav />
      </div>
    </div>
  );
}
