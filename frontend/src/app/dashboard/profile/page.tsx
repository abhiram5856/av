"use client";

import { useEffect, useState } from "react";
import { User, Mail, Calendar, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { createClient } from "@/utils/supabase/client";
import { signOutAction } from "@/app/actions/auth";

export default function ProfilePage() {
  const [user, setUser] = useState<{
    email: string;
    firstName: string;
    lastName: string;
    createdAt: string;
  } | null>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (data.user) {
        setUser({
          email: data.user.email || "",
          firstName: data.user.user_metadata?.first_name || "User",
          lastName: data.user.user_metadata?.last_name || "",
          createdAt: data.user.created_at || new Date().toISOString(),
        });
      }
    });
  }, []);

  const fullName = user
    ? `${user.firstName} ${user.lastName}`.trim()
    : "Loading...";

  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto w-full">
      <div className="border-b pb-4 mb-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Profile Settings
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage your account information and preferences.
        </p>
      </div>

      <div className="border rounded-lg bg-card shadow-sm p-6 space-y-8">
        
        {/* Profile Info Form (Read-only for now) */}
        <div className="space-y-4">
          <h2 className="text-lg font-medium">Personal Information</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Full Name</label>
              <div className="flex items-center p-2.5 border rounded-md bg-muted/30">
                <User className="h-4 w-4 mr-3 text-muted-foreground" />
                <span className="text-sm">{fullName}</span>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Email Address</label>
              <div className="flex items-center p-2.5 border rounded-md bg-muted/30">
                <Mail className="h-4 w-4 mr-3 text-muted-foreground" />
                <span className="text-sm">{user?.email || "—"}</span>
              </div>
            </div>
          </div>
          <div className="space-y-2">
             <label className="text-sm font-medium text-foreground">Account Created</label>
              <div className="flex items-center p-2.5 border rounded-md bg-muted/30 max-w-sm">
                <Calendar className="h-4 w-4 mr-3 text-muted-foreground" />
                <span className="text-sm">
                  {user?.createdAt
                    ? new Date(user.createdAt).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                      })
                    : "—"}
                </span>
              </div>
          </div>
        </div>

        <div className="pt-6 border-t">
           <h2 className="text-lg font-medium text-destructive mb-4">Danger Zone</h2>
           <form action={signOutAction}>
             <Button
               type="submit"
               variant="destructive"
               className="w-full sm:w-auto"
             >
               <LogOut className="h-4 w-4 mr-2" />
               Sign Out
             </Button>
           </form>
        </div>
      </div>
    </div>
  );
}
