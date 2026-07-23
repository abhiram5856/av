"use client";

import { Leaf } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useAppStore } from "@/lib/store";
import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { User as SupabaseUser } from "@supabase/supabase-js";
import Link from "next/link";

export function AppNavbar() {
  const [user, setUser] = useState<SupabaseUser | null>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (data.user) setUser(data.user);
    });
  }, []);

  const firstName = user?.user_metadata?.first_name || "User";
  const initials = firstName.charAt(0);

  return (
    <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b bg-background px-4 shadow-sm md:px-6 lg:px-8 max-w-md mx-auto w-full">
      <Link href="/dashboard" className="flex items-center gap-2 font-bold text-xl text-primary">
        <Leaf className="h-6 w-6" />
        AgriVision
      </Link>
      
      <div className="flex items-center gap-4">
        <Link href="/dashboard/profile">
          <Avatar className="h-9 w-9 border-2 border-primary/20">
            <AvatarImage src="" alt={firstName} />
            <AvatarFallback className="bg-primary/10 text-primary font-bold">{initials}</AvatarFallback>
          </Avatar>
        </Link>
      </div>
    </header>
  );
}
