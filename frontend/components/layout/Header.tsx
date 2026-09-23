"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Menu, X, CheckCircle2, AlertCircle } from "lucide-react";
import { api } from "@/lib/api";

export function Header() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .getHealth()
      .then((res) => setIsOnline(res.status === "ok" && res.model_loaded))
      .catch(() => setIsOnline(false));
  }, []);

  const navItems = [
    { label: "Overview", href: "/" },
    { label: "Drug Explorer", href: "/drugs" },
    { label: "Scenario Lab", href: "/scenarios" },
    { label: "Methodology", href: "/methodology" },
  ];

  return (
    <header className="h-16 bg-slate-900/80 border-b border-slate-800 px-4 md:px-8 flex items-center justify-between sticky top-0 z-40 backdrop-blur-md">
      <div className="flex items-center space-x-3 md:hidden">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-sky-400" />
          <span className="font-bold text-slate-100">MedCascade</span>
        </div>
      </div>

      <div className="hidden md:flex items-center space-x-2 text-xs text-slate-400">
        <span>Backend API Status:</span>
        {isOnline === null ? (
          <span className="inline-flex items-center text-slate-500">Checking...</span>
        ) : isOnline ? (
          <span className="inline-flex items-center text-emerald-400 space-x-1 font-medium bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
            <CheckCircle2 className="w-3 h-3 mr-1" /> FastAPI Online
          </span>
        ) : (
          <span className="inline-flex items-center text-amber-400 space-x-1 font-medium bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
            <AlertCircle className="w-3 h-3 mr-1" /> API Disconnected
          </span>
        )}
      </div>

      <div className="flex items-center space-x-3 text-xs text-slate-400">
        <span className="bg-slate-800 px-2.5 py-1 rounded border border-slate-700 font-mono text-[11px] text-slate-300">
          v0.1.0-release
        </span>
      </div>

      {mobileOpen && (
        <div className="absolute top-16 left-0 w-full bg-slate-900 border-b border-slate-800 p-4 md:hidden space-y-2 z-50">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`block px-3 py-2 rounded-md text-sm ${
                pathname === item.href
                  ? "bg-sky-500/10 text-sky-400 font-medium"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}
