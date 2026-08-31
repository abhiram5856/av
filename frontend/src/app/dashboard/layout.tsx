import { AppNavbar } from "@/components/app-navbar";
import { BottomNav } from "@/components/bottom-nav";
import { RegionalAlert } from "@/components/regional-alert";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-muted/20">
      <AppNavbar />
      <main className="flex-1 w-full max-w-6xl mx-auto px-4 md:px-6 lg:px-8 py-6 pb-24 lg:pb-8">
        {children}
      </main>
      <BottomNav />
      <RegionalAlert />
    </div>
  );
}
